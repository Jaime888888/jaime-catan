#![recursion_limit = "256"]

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use az::{Game, IterationStats, ResNet, ResNetConfig, SelfPlayStepInfo, TrainConfig, train};
use burn::backend::Autodiff;
use burn::tensor::backend::Backend;
use catan_sim::{
    ACTION_SPACE_SIZE, Action, ObsEdge, ObsVertex, Phase, PlayerId, PlayerRelation, Port, Turn,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use wandb::{BackendOptions, DataValue, LogData, Run, RunInfo, WandB};

#[cfg(target_os = "macos")]
mod backend {
    pub type Inner = burn::backend::Metal;
    pub const NAME: &str = "Metal";
}

#[cfg(all(not(target_os = "macos"), feature = "cuda"))]
mod backend {
    pub type Inner = burn::backend::Cuda;
    pub const NAME: &str = "CUDA";
}

#[cfg(all(
    not(target_os = "macos"),
    not(feature = "cuda"),
    any(target_os = "linux", target_os = "windows"),
))]
mod backend {
    pub type Inner = burn::backend::Vulkan;
    pub const NAME: &str = "Vulkan";
}

#[cfg(all(
    not(target_os = "macos"),
    not(feature = "cuda"),
    not(target_os = "linux"),
    not(target_os = "windows"),
))]
mod backend {
    pub type Inner = burn::backend::NdArray;
    pub const NAME: &str = "NdArray (CPU)";
}

const ACT: usize = ACTION_SPACE_SIZE;
const PLAYERS: usize = 4;

// Observation float layout:
//   tiles:    19 × (terrain_onehot[6] + number + robber)           = 152
//   vertices: 54 × (building_level + is_mine + opp_relation)      = 162
//   edges:    72 × (has_road + is_mine + opp_relation)            = 216
//   harbors:   9 × kind_onehot[6]                                 =  54
//   self:     resources[5] + dev_cards[5] + 4 scalars + ports[6]  =  20
//   others:    3 × 5 scalars                                      =  15
//   meta:     turn_number + dev_remaining + bank[5]               =   7
const OBS: usize = 152 + 162 + 216 + 54 + 20 + 15 + 7;

fn catan_phase_bucket(phase: &Phase) -> u8 {
    match phase {
        Phase::SetupSettlement { .. } | Phase::SetupRoad { .. } => 0,
        Phase::PreRoll | Phase::ChanceRoll => 1,
        Phase::Discard { .. } => 2,
        Phase::MoveRobber | Phase::Steal { .. } => 3,
        Phase::Main => 4,
        Phase::RoadBuilding { .. } => 5,
        Phase::GameOver { .. } => 6,
    }
}

/// Coarse action families for self-play logging / EMA (11 buckets).
const ACTION_LOG_BUCKETS: usize = 11;

fn action_log_bucket(a: Action) -> usize {
    match a {
        Action::EndTurn => 0,
        Action::BankTrade { .. } => 1,
        Action::PlaceSettlement(_) => 2,
        Action::PlaceRoad(_) => 3,
        Action::BuildCity(_) => 4,
        Action::BuyDevelopmentCard => 5,
        Action::PlayKnight => 6,
        Action::PlayRoadBuilding | Action::PlayYearOfPlenty(_, _) | Action::PlayMonopoly(_) => 7,
        Action::RollDice => 8,
        Action::MoveRobber(_) | Action::StealFrom(_) | Action::StealFromNone => 9,
        Action::DiscardResource(_) => 10,
    }
}

const ACTION_BUCKET_LABELS: [&str; ACTION_LOG_BUCKETS] = [
    "end", "bank", "set", "road", "city", "bdev", "knight", "odev", "roll", "rob", "disc",
];

fn encode_relation(r: PlayerRelation) -> (f32, f32) {
    match r {
        PlayerRelation::Self_ => (1.0, 0.0),
        PlayerRelation::Clockwise1 => (0.0, 1.0 / 3.0),
        PlayerRelation::Clockwise2 => (0.0, 2.0 / 3.0),
        PlayerRelation::Clockwise3 => (0.0, 1.0),
    }
}

#[derive(Clone)]
struct CatanGame {
    inner: catan_sim::Game,
    rng: ChaCha8Rng,
}

impl CatanGame {
    pub(crate) fn as_sim(&self) -> &catan_sim::Game {
        &self.inner
    }

    fn new() -> Self {
        let mut rng = ChaCha8Rng::from_rng(&mut rand::rng());
        let mut game = Self {
            inner: catan_sim::Game::new(&mut rng),
            rng,
        };
        game.resolve_chance();
        game
    }

    fn resolve_chance(&mut self) {
        while matches!(self.inner.phase, Phase::ChanceRoll) {
            match self.inner.turn() {
                Turn::Chance(ct) => ct.resolve_random(&mut self.rng),
                _ => unreachable!(),
            }
        }
    }
}

impl Game<ACT, OBS, PLAYERS> for CatanGame {
    type Action = Action;
    type Player = PlayerId;

    fn legal_actions(&self) -> Vec<Action> {
        self.inner.action_mask().actions().collect()
    }

    fn apply(&mut self, action: Action) {
        match self.inner.turn() {
            Turn::Player(pt) => pt.apply(action, &mut self.rng).expect("illegal action"),
            Turn::Terminal => panic!("apply called on terminal game"),
            Turn::Chance(_) => unreachable!("chance should be auto-resolved"),
        }
        self.resolve_chance();
    }

    fn is_terminal(&self) -> bool {
        matches!(self.inner.phase, Phase::GameOver { .. })
    }

    fn current_player(&self) -> PlayerId {
        self.inner.acting_player()
    }

    fn result(&self, player: PlayerId) -> f32 {
        match self.inner.winner() {
            Some(w) if w == player => 1.0,
            _ => 0.0,
        }
    }

    fn action_to_index(action: Action) -> usize {
        action.to_index()
    }

    fn index_to_action(index: usize) -> Action {
        Action::from_index(index)
    }

    fn player_to_index(player: PlayerId) -> usize {
        player.idx()
    }

    fn index_to_player(index: usize) -> PlayerId {
        PlayerId(index as u8)
    }

    fn observe(&self) -> [f32; OBS] {
        let obs = self.inner.observe(self.inner.acting_player());
        let mut out = [0.0f32; OBS];
        let mut i = 0;

        for tile in &obs.tiles {
            out[i + tile.terrain as u8 as usize] = 1.0;
            i += 6;
            out[i] = tile.number as f32 / 12.0;
            out[i + 1] = tile.has_robber as u8 as f32;
            i += 2;
        }

        for &v in &obs.vertices {
            let (level, mine, opp) = match v {
                ObsVertex::Empty => (0.0, 0.0, 0.0),
                ObsVertex::Settlement(r) => {
                    let (m, o) = encode_relation(r);
                    (0.5, m, o)
                }
                ObsVertex::City(r) => {
                    let (m, o) = encode_relation(r);
                    (1.0, m, o)
                }
            };
            out[i] = level;
            out[i + 1] = mine;
            out[i + 2] = opp;
            i += 3;
        }

        for &e in &obs.edges {
            let (road, mine, opp) = match e {
                ObsEdge::Empty => (0.0, 0.0, 0.0),
                ObsEdge::Road(rel) => {
                    let (m, o) = encode_relation(rel);
                    (1.0, m, o)
                }
            };
            out[i] = road;
            out[i + 1] = mine;
            out[i + 2] = opp;
            i += 3;
        }

        for &harbor in &obs.harbors {
            let port_idx = match harbor {
                Port::ThreeToOne => 0,
                Port::TwoToOne(r) => 1 + r as usize,
            };
            out[i + port_idx] = 1.0;
            i += 6;
        }

        let s = &obs.self_player;
        for j in 0..5 {
            out[i + j] = s.resources[j] as f32 / 19.0;
        }
        i += 5;
        for j in 0..5 {
            out[i + j] = s.dev_cards[j] as f32 / 5.0;
        }
        i += 5;
        out[i] = s.played_knights as f32 / 14.0;
        i += 1;
        out[i] = s.has_longest_road as u8 as f32;
        i += 1;
        out[i] = s.longest_road_length as f32 / 15.0;
        i += 1;
        out[i] = s.has_largest_army as u8 as f32;
        i += 1;
        out[i] = s.has_three_to_one_port as u8 as f32;
        i += 1;
        for j in 0..5 {
            out[i + j] = s.two_to_one_ports[j] as u8 as f32;
        }
        i += 5;

        for o in &obs.other_players {
            out[i] = o.total_resource_cards as f32 / 19.0;
            i += 1;
            out[i] = o.total_dev_cards as f32 / 25.0;
            i += 1;
            out[i] = o.played_knights as f32 / 14.0;
            i += 1;
            out[i] = o.has_longest_road as u8 as f32;
            i += 1;
            out[i] = o.has_largest_army as u8 as f32;
            i += 1;
        }

        let m = &obs.meta;
        out[i] = m.turn_number as f32 / 500.0;
        i += 1;
        out[i] = m.dev_cards_remaining as f32 / 25.0;
        i += 1;
        for j in 0..5 {
            out[i + j] = m.resource_bank[j] as f32 / 19.0;
        }
        i += 5;

        debug_assert_eq!(i, OBS);
        out
    }
}

fn main() {
    type B = Autodiff<backend::Inner>;

    eprintln!("backend: {}", backend::NAME);

    let device = <B as Backend>::Device::default();

    let net_config = ResNetConfig {
        obs_size: OBS,
        act_size: ACT,
        num_players: PLAYERS,
        num_blocks: 4,
        hidden_dim: 128,
    };
    let net = ResNet::<B>::new(&net_config, &device);

    let config = TrainConfig {
        iterations: 500,
        games_per_iteration: 256,
        replay_capacity: 2 * 1024 * 1024,
        batch_size: 256,
        training_steps_per_iteration: 512,
        max_simulations: 576,
        sims_per_eval: 48,
        learning_rate: 2e-4,
        max_plies_per_game: Some(2500),
        c_puct: 1.5,
        temperature: 1.0,
        root_dirichlet_epsilon: 0.25,
        root_dirichlet_alpha: 0.3,
        ..Default::default()
    };

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()
        .expect("create tokio runtime");
    let handle = runtime.handle().clone();

    let wandb_run: Option<Arc<Run>> = std::env::var("WANDB_API_KEY").ok().and_then(|api_key| {
        let info = RunInfo::new("catan-az")
            .build()
            .expect("build wandb RunInfo");
        let (tx, rx) = std::sync::mpsc::channel();
        handle.spawn(async move {
            let wandb = WandB::new(BackendOptions::new(api_key));
            let _ = tx.send(wandb.new_run(info).await);
        });
        match rx.recv().expect("wandb init task dropped") {
            Ok(run) => {
                eprintln!("wandb: run initialized");
                Some(Arc::new(run))
            }
            Err(e) => {
                eprintln!("wandb: failed to initialize run ({e}); continuing without logging");
                None
            }
        }
    });

    // Self-play observability. Player threads increment atomics per step;
    // the single-threaded `on_iteration` closure drains them at iteration
    // boundaries and updates an EMA over action-family fractions.
    const CATAN_PHASE_BUCKETS: usize = 7;
    const EMA_ALPHA: f32 = 0.1;

    let action_counts: [AtomicU64; ACTION_LOG_BUCKETS] = std::array::from_fn(|_| AtomicU64::new(0));
    let phase_counts: [AtomicU64; CATAN_PHASE_BUCKETS] = std::array::from_fn(|_| AtomicU64::new(0));
    let step_total = AtomicU64::new(0);
    let mut action_ema = [0.0f32; ACTION_LOG_BUCKETS];

    let on_step = |info: &SelfPlayStepInfo<CatanGame, ACT, OBS, PLAYERS>| {
        let ab = action_log_bucket(info.action);
        let pb = catan_phase_bucket(&info.game.as_sim().phase) as usize;
        action_counts[ab].fetch_add(1, Ordering::Relaxed);
        phase_counts[pb].fetch_add(1, Ordering::Relaxed);
        step_total.fetch_add(1, Ordering::Relaxed);
    };

    let on_iter = |stats: &IterationStats| {
        let total = step_total.swap(0, Ordering::Relaxed).max(1) as f32;
        let action_frac: [f32; ACTION_LOG_BUCKETS] =
            std::array::from_fn(|i| action_counts[i].swap(0, Ordering::Relaxed) as f32 / total);
        let phase_frac: [f32; CATAN_PHASE_BUCKETS] =
            std::array::from_fn(|i| phase_counts[i].swap(0, Ordering::Relaxed) as f32 / total);

        for i in 0..ACTION_LOG_BUCKETS {
            action_ema[i] = EMA_ALPHA * action_frac[i] + (1.0 - EMA_ALPHA) * action_ema[i];
        }

        let action_str = ACTION_BUCKET_LABELS
            .iter()
            .zip(action_ema.iter())
            .map(|(l, f)| format!("{l}={f:.2}"))
            .collect::<Vec<_>>()
            .join(" ");
        let phase_str = phase_frac
            .iter()
            .enumerate()
            .map(|(i, f)| format!("p{i}={f:.2}"))
            .collect::<Vec<_>>()
            .join(" ");

        let wall = stats.elapsed_secs.max(1e-6);
        let avg_batch = if stats.gpu_dispatches > 0 {
            stats.gpu_leaves as f32 / stats.gpu_dispatches as f32
        } else {
            0.0
        };
        let leaves_per_sec = stats.gpu_leaves as f32 / wall;
        let samples_per_sec = stats.new_samples as f32 / wall;

        eprintln!(
            "az iter {}: samples={} ({:.0}/s) buf={} loss=(p={:.4} v={:.4} t={:.4}) {:.2}s | gpu[disp={} avg_bs={:.0} leaves={} leaves/s={:.0}] | actions[{}] | phases[{}]",
            stats.iteration,
            stats.new_samples,
            samples_per_sec,
            stats.buffer_size,
            stats.policy_loss,
            stats.value_loss,
            stats.total_loss,
            stats.elapsed_secs,
            stats.gpu_dispatches,
            avg_batch,
            stats.gpu_leaves,
            leaves_per_sec,
            action_str,
            phase_str,
        );

        if let Some(run) = wandb_run.as_ref() {
            let mut data: HashMap<String, DataValue> = HashMap::new();
            data.insert("_step".into(), (stats.iteration as i64).into());
            data.insert("new_samples".into(), (stats.new_samples as u64).into());
            data.insert("samples_per_sec".into(), (samples_per_sec as f64).into());
            data.insert("buffer_size".into(), (stats.buffer_size as u64).into());
            data.insert("elapsed_secs".into(), (stats.elapsed_secs as f64).into());
            data.insert("loss/policy".into(), (stats.policy_loss as f64).into());
            data.insert("loss/value".into(), (stats.value_loss as f64).into());
            data.insert("loss/total".into(), (stats.total_loss as f64).into());
            data.insert("gpu/dispatches".into(), stats.gpu_dispatches.into());
            data.insert("gpu/leaves".into(), stats.gpu_leaves.into());
            data.insert("gpu/avg_batch".into(), (avg_batch as f64).into());
            data.insert("gpu/leaves_per_sec".into(), (leaves_per_sec as f64).into());
            for (label, frac) in ACTION_BUCKET_LABELS.iter().zip(action_ema.iter()) {
                data.insert(format!("action/{label}"), (*frac as f64).into());
            }
            for (i, frac) in phase_frac.iter().enumerate() {
                data.insert(format!("phase/p{i}"), (*frac as f64).into());
            }

            let run = Arc::clone(run);
            let log_data: LogData = data.into();
            handle.spawn(async move {
                run.log(log_data).await;
            });
        }
    };

    let _trained = train::<B, _, CatanGame, ACT, OBS, PLAYERS>(
        net,
        CatanGame::new,
        config,
        device,
        on_iter,
        &on_step,
    );

    drop(wandb_run);
    runtime.shutdown_timeout(std::time::Duration::from_secs(15));
}
