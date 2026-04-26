//! Arena: load every `iter_*.bin` checkpoint, play random 4-seat free-for-all matches with raw
//! policy sampling (T=1), and fit Plackett-Luce ratings on the Elo scale.

#![recursion_limit = "256"]

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::path::PathBuf;
use std::sync::Mutex;

use az::{AZNet, Game};
use burn::module::Module;
use burn::record::{BinFileRecorder, FullPrecisionSettings};
use burn::tensor::{Tensor, TensorData, backend::Backend};
use catan::{ACT, CatanGame, CatanNet, CatanNetConfig, OBS, PLAYERS, backend};
use clap::Parser;
use rand::seq::SliceRandom;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

const GAMES: usize = 2000;
const THREADS: usize = 8;
const MAX_PLIES: usize = 2500;
const TEMPERATURE: f32 = 1.0;
const FIT_STEPS: usize = 2000;
const FIT_LR: f32 = 0.05;

type B = backend::Inner;

#[derive(Parser, Debug)]
#[command(name = "arena")]
struct Args {
    #[arg(long, default_value = "./checkpoints", value_name = "DIR")]
    checkpoints: PathBuf,
}

struct Checkpoint {
    iteration: usize,
    net: CatanNet<B>,
}

struct GameRecord {
    seats: [usize; PLAYERS],
    winner: Option<usize>,
}

fn main() {
    let cli = Args::parse();
    eprintln!("backend: {}", backend::NAME);

    let device = <B as Backend>::Device::default();
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    let net_config = CatanNetConfig {
        act_size: ACT,
        num_players: PLAYERS,
        branch_dim: 64,
        trunk_hidden: 384,
        num_blocks: 4,
    };

    let mut checkpoints: Vec<Checkpoint> = std::fs::read_dir(&cli.checkpoints)
        .unwrap_or_else(|e| panic!("read_dir {}: {e}", cli.checkpoints.display()))
        .filter_map(|res| res.ok())
        .filter_map(|entry| {
            let path = entry.path();
            let name = path.file_name()?.to_str()?;
            let iteration = name
                .strip_suffix(".bin")
                .unwrap_or(name)
                .strip_prefix("iter_")?
                .parse::<usize>()
                .ok()?;
            let net = CatanNet::<B>::new(&net_config, &device)
                .load_file(&path, &recorder, &device)
                .unwrap_or_else(|e| panic!("load {}: {e}", path.display()));
            Some(Checkpoint { iteration, net })
        })
        .collect();
    checkpoints.sort_by_key(|c| c.iteration);

    assert!(
        checkpoints.len() >= PLAYERS,
        "need >= {PLAYERS} checkpoints, found {}",
        checkpoints.len(),
    );
    eprintln!(
        "loaded {} checkpoints (iters {}..={}); playing {GAMES} games across {THREADS} threads",
        checkpoints.len(),
        checkpoints.first().unwrap().iteration,
        checkpoints.last().unwrap().iteration,
    );

    let records: Mutex<Vec<GameRecord>> = Mutex::new(Vec::with_capacity(GAMES));
    std::thread::scope(|s| {
        for tid in 0..THREADS {
            let records = &records;
            let device = device.clone();
            let nets: Vec<CatanNet<B>> = checkpoints.iter().map(|c| c.net.clone()).collect();
            s.spawn(move || {
                let mut rng = ChaCha8Rng::seed_from_u64(tid as u64);
                for g in (tid..GAMES).step_by(THREADS) {
                    let seats = sample_seats(nets.len(), &mut rng);
                    let winner = play_game(&nets, &seats, &device, &mut rng);
                    records
                        .lock()
                        .unwrap()
                        .push(GameRecord { seats, winner });
                    if g.is_multiple_of(50) {
                        eprintln!("thread {tid}: game {g}");
                    }
                }
            });
        }
    });

    let records = records.into_inner().unwrap();
    let elos = fit_elo(&records, checkpoints.len());
    let (games_played, wins) = tally(&records, checkpoints.len());
    let draws = records.iter().filter(|g| g.winner.is_none()).count();
    eprintln!(
        "arena: {} games complete ({} draws / max-plies cutoffs)",
        records.len(),
        draws,
    );

    println!(
        "{:>8} {:>7} {:>6} {:>9} {:>9}",
        "iter", "games", "wins", "win_rate", "elo"
    );
    for (k, ck) in checkpoints.iter().enumerate() {
        let g = games_played[k];
        let w = wins[k];
        let wr = if g > 0 { w as f32 / g as f32 } else { 0.0 };
        println!(
            "{:>8} {:>7} {:>6} {:>9.4} {:>+9.1}",
            ck.iteration, g, w, wr, elos[k],
        );
    }
}

fn sample_seats(n: usize, rng: &mut ChaCha8Rng) -> [usize; PLAYERS] {
    let mut all: Vec<usize> = (0..n).collect();
    all.partial_shuffle(rng, PLAYERS);
    [all[0], all[1], all[2], all[3]]
}

fn play_game(
    nets: &[CatanNet<B>],
    seats: &[usize; PLAYERS],
    device: &<B as Backend>::Device,
    rng: &mut ChaCha8Rng,
) -> Option<usize> {
    let mut game = CatanGame::new();
    for _ in 0..MAX_PLIES {
        if game.is_terminal() {
            for seat in 0..PLAYERS {
                let player = <CatanGame as Game<ACT, OBS, PLAYERS>>::index_to_player(seat);
                if game.result(player) > 0.5 {
                    return Some(seat);
                }
            }
            return None;
        }

        let acting = <CatanGame as Game<ACT, OBS, PLAYERS>>::current_player(&game);
        let seat = <CatanGame as Game<ACT, OBS, PLAYERS>>::player_to_index(acting);
        let net = &nets[seats[seat]];

        let obs = game.observe();
        let tensor = Tensor::<B, 2>::from_data(TensorData::new(obs.to_vec(), [1, OBS]), device);
        let (policy, _value) = net.forward(tensor);
        let logits: Vec<f32> = policy.into_data().to_vec().expect("policy to vec");

        let legal = game.legal_actions();
        debug_assert!(!legal.is_empty());
        let action = sample_action(&logits, &legal, rng);
        debug_assert!(legal.contains(&action));
        game.apply(action);
    }
    None
}

fn sample_action(
    logits: &[f32],
    legal: &[catan_sim::Action],
    rng: &mut ChaCha8Rng,
) -> catan_sim::Action {
    let inv_t = 1.0 / TEMPERATURE;
    let scaled: Vec<f32> = legal.iter().map(|a| logits[a.to_index()] * inv_t).collect();
    let max = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scaled.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();

    let r: f32 = rng.random_range(0.0..1.0);
    let mut cum = 0.0;
    for (i, &e) in exps.iter().enumerate() {
        cum += e / sum;
        if r < cum {
            return legal[i];
        }
    }
    *legal.last().unwrap()
}

fn tally(records: &[GameRecord], n: usize) -> (Vec<u32>, Vec<u32>) {
    let mut games_played = vec![0u32; n];
    let mut wins = vec![0u32; n];
    for g in records {
        for s in g.seats {
            games_played[s] += 1;
        }
        if let Some(w) = g.winner {
            wins[g.seats[w]] += 1;
        }
    }
    (games_played, wins)
}

/// Fit Plackett-Luce strengths by first-place-only MLE, then anchor earliest
/// iter to zero and rescale to Elo (`400/ln(10)` per logit unit).
fn fit_elo(records: &[GameRecord], n: usize) -> Vec<f32> {
    let scoring: Vec<&GameRecord> = records.iter().filter(|g| g.winner.is_some()).collect();
    let mut r = vec![0.0f32; n];

    for _ in 0..FIT_STEPS {
        let mut grad = vec![0.0f32; n];
        for g in &scoring {
            let logits: [f32; PLAYERS] = std::array::from_fn(|i| r[g.seats[i]]);
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: [f32; PLAYERS] = std::array::from_fn(|i| (logits[i] - max).exp());
            let sum: f32 = exps.iter().sum();
            for i in 0..PLAYERS {
                grad[g.seats[i]] -= exps[i] / sum;
            }
            grad[g.seats[g.winner.unwrap()]] += 1.0;
        }
        for k in 0..n {
            r[k] += FIT_LR * grad[k];
        }
        let mean: f32 = r.iter().sum::<f32>() / n as f32;
        for v in &mut r {
            *v -= mean;
        }
    }

    let r0 = r[0];
    let elo_scale: f32 = 400.0 / 10.0_f32.ln();
    r.iter().map(|v| elo_scale * (v - r0)).collect()
}
