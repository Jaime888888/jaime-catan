//! Arena: load every `iter_*.bin` checkpoint, play 4-seat free-for-all matches with raw policy
//! sampling (T=1), and maintain online Plackett-Luce ratings on the Elo scale.

#![recursion_limit = "256"]

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::fmt::Write as _;
use std::io::Write as _;
use std::path::PathBuf;

use az::{AZNet, Game};
use burn::module::Module;
use burn::record::{BinFileRecorder, FullPrecisionSettings};
use burn::tensor::{Tensor, TensorData, backend::Backend};
use catan::{ACT, CatanGame, CatanNet, CatanNetConfig, OBS, PLAYERS, backend};
use clap::Parser;
use rand::seq::index;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

const GAMES: usize = 100;
const MAX_PLIES: usize = 20_000;
const TEMPERATURE: f32 = 1.0;
const LR: f32 = 0.05;
const SEED: u64 = 0;
/// When true, print the board periodically as games are played. Pair with a small `GAMES`.
const DISPLAY: bool = true;
/// Render every Nth ply (always also renders the initial and terminal states).
const DISPLAY_EVERY: usize = 32;
/// Sleep between rendered frames in display mode, in milliseconds. Set to 0 for no pause.
const DISPLAY_DELAY_MS: u64 = 0;

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

/// Hides the terminal cursor for the lifetime of the guard. Restored on drop, so
/// the cursor reappears on normal exit and panic unwinds (but not on SIGKILL).
struct CursorGuard;

impl CursorGuard {
    fn hide() -> Self {
        let stdout = std::io::stdout();
        let mut handle = stdout.lock();
        let _ = handle.write_all(b"\x1b[?25l");
        let _ = handle.flush();
        Self
    }
}

impl Drop for CursorGuard {
    fn drop(&mut self) {
        let stdout = std::io::stdout();
        let mut handle = stdout.lock();
        let _ = handle.write_all(b"\x1b[?25h");
        let _ = handle.flush();
    }
}

fn main() {
    let cli = Args::parse();
    eprintln!("backend: {}", backend::NAME);

    let _cursor_guard = DISPLAY.then(CursorGuard::hide);

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

    let n = checkpoints.len();
    assert!(n >= PLAYERS, "need >= {PLAYERS} checkpoints, found {n}");
    eprintln!(
        "loaded {n} checkpoints (iters {}..={}); playing {GAMES} games",
        checkpoints.first().unwrap().iteration,
        checkpoints.last().unwrap().iteration,
    );

    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let mut strengths = vec![0.0f32; n];
    let mut games_played = vec![0u32; n];
    let mut wins = vec![0u32; n];
    let mut cutoffs = 0u32;

    let elo_scale = 400.0 / 10.0_f32.ln();

    for g in 0..GAMES {
        let seats = sample_seats(n, &mut rng);
        for &s in &seats {
            games_played[s] += 1;
        }

        match play_game(&checkpoints, &seats, &device, &mut rng, g) {
            Some(w) => {
                wins[seats[w]] += 1;
                pl_update(&mut strengths, &seats, w);
            }
            None => cutoffs += 1,
        }

        if g.is_multiple_of(50) {
            eprintln!("game {g}/{GAMES} (cutoffs so far: {cutoffs})");
        }
    }

    eprintln!("arena: {GAMES} games complete ({cutoffs} hit {MAX_PLIES}-ply cutoff)");

    let r0 = strengths[0];
    let mut ranked: Vec<usize> = (0..n).collect();
    ranked.sort_by(|&a, &b| strengths[b].partial_cmp(&strengths[a]).unwrap());

    println!(
        "{:>5} {:>8} {:>7} {:>6} {:>9} {:>9}",
        "rank", "iter", "games", "wins", "win_rate", "elo"
    );
    for (rank, &k) in ranked.iter().enumerate() {
        let ck = &checkpoints[k];
        let g = games_played[k];
        let w = wins[k];
        let wr = if g > 0 { w as f32 / g as f32 } else { 0.0 };
        let elo = elo_scale * (strengths[k] - r0);
        println!(
            "{:>5} {:>8} {:>7} {:>6} {:>9.4} {:>+9.1}",
            rank + 1,
            ck.iteration,
            g,
            w,
            wr,
            elo,
        );
    }
}

/// Sample `PLAYERS` distinct checkpoint indices uniformly from `0..n`.
fn sample_seats(n: usize, rng: &mut ChaCha8Rng) -> [usize; PLAYERS] {
    let idxs = index::sample(rng, n, PLAYERS);
    std::array::from_fn(|i| idxs.index(i))
}

/// Online Plackett-Luce gradient step on first-place log-likelihood.
/// `seats[w]` is the global checkpoint index of the winner.
fn pl_update(strengths: &mut [f32], seats: &[usize; PLAYERS], w: usize) {
    let logits: [f32; PLAYERS] = std::array::from_fn(|i| strengths[seats[i]]);
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: [f32; PLAYERS] = std::array::from_fn(|i| (logits[i] - max).exp());
    let sum: f32 = exps.iter().sum();
    for i in 0..PLAYERS {
        let p = exps[i] / sum;
        let grad = if i == w { 1.0 - p } else { -p };
        strengths[seats[i]] += LR * grad;
    }
}

fn play_game(
    checkpoints: &[Checkpoint],
    seats: &[usize; PLAYERS],
    device: &<B as Backend>::Device,
    rng: &mut ChaCha8Rng,
    game_idx: usize,
) -> Option<usize> {
    let mut game = CatanGame::new();
    if DISPLAY {
        render(&game, game_idx, checkpoints, seats, 0, None);
    }
    let mut last_action = None;
    for ply in 0..MAX_PLIES {
        if game.is_terminal() {
            if DISPLAY {
                render(&game, game_idx, checkpoints, seats, ply, last_action);
            }
            for seat in 0..PLAYERS {
                let player = <CatanGame as Game<ACT, OBS, PLAYERS>>::index_to_player(seat);
                if game.result(player) > 0.5 {
                    return Some(seat);
                }
            }
            unreachable!("terminal game with no winner");
        }

        let acting = <CatanGame as Game<ACT, OBS, PLAYERS>>::current_player(&game);
        let seat = <CatanGame as Game<ACT, OBS, PLAYERS>>::player_to_index(acting);
        let net = &checkpoints[seats[seat]].net;

        let obs = game.observe();
        let tensor = Tensor::<B, 2>::from_data(TensorData::new(obs.to_vec(), [1, OBS]), device);
        let (policy, _value) = net.forward(tensor);
        let logits: Vec<f32> = policy.into_data().to_vec().expect("policy to vec");

        let legal = game.legal_actions();
        debug_assert!(!legal.is_empty());
        let action = sample_action(&logits, &legal, rng);
        debug_assert!(legal.contains(&action));
        game.apply(action);
        last_action = Some(action);

        if DISPLAY && (ply + 1).is_multiple_of(DISPLAY_EVERY) {
            render(&game, game_idx, checkpoints, seats, ply + 1, last_action);
        }
    }
    None
}

fn render(
    game: &CatanGame,
    game_idx: usize,
    checkpoints: &[Checkpoint],
    seats: &[usize; PLAYERS],
    ply: usize,
    last_action: Option<catan_sim::Action>,
) {
    let mut buf = String::with_capacity(4096);
    writeln!(buf, "arena game {game_idx}/{GAMES}  ·  ply {ply}").unwrap();
    for (s, &k) in seats.iter().enumerate() {
        writeln!(buf, "  seat {s} = iter {:>5}", checkpoints[k].iteration).unwrap();
    }
    if let Some(a) = last_action {
        writeln!(buf, "last action: {a:?}").unwrap();
    }
    writeln!(buf).unwrap();
    write!(buf, "{}", game.as_sim()).unwrap();

    // Overwrite the previous frame in place: home cursor (no clear → no flash),
    // append \x1b[K before every newline so any wider trailing content from the
    // previous frame on the same line is erased, then \x1b[J at the end to wipe
    // any extra rows below in case this frame is shorter than the last.
    let body = buf.replace('\n', "\x1b[K\n");
    let frame = format!("\x1b[H{body}\x1b[K\x1b[J");

    let stdout = std::io::stdout();
    let mut handle = stdout.lock();
    handle.write_all(frame.as_bytes()).ok();
    handle.flush().ok();

    if DISPLAY_DELAY_MS > 0 {
        std::thread::sleep(std::time::Duration::from_millis(DISPLAY_DELAY_MS));
    }
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
