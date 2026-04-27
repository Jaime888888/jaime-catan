#![recursion_limit = "256"]

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use az::{IterationStats, SelfPlayStepInfo, TrainConfig, train};
use burn::backend::Autodiff;
use burn::module::Module;
use burn::record::{BinFileRecorder, FullPrecisionSettings};
use burn::tensor::backend::Backend;
use catan::{
    ACT, ACTION_BUCKET_LABELS, ACTION_LOG_BUCKETS, CatanGame, CatanNet, CatanNetConfig, OBS,
    PLAYERS, action_log_bucket, backend, catan_phase_bucket,
};
use clap::Parser;
use wandb::{BackendOptions, DataValue, LogData, Run, RunInfo, WandB};

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value = "./checkpoints", value_name = "DIR")]
    checkpoint_dir: PathBuf,

    #[arg(long, value_name = "PATH")]
    load: Option<PathBuf>,
}

fn main() {
    type B = Autodiff<backend::Inner>;

    let cli = Args::parse();

    eprintln!("backend: {}", backend::NAME);

    let device = <B as Backend>::Device::default();

    let net_config = CatanNetConfig {
        act_size: ACT,
        num_players: PLAYERS,
        branch_dim: 64,
        trunk_hidden: 384,
        num_blocks: 4,
    };

    let checkpoint_dir = cli.checkpoint_dir;
    std::fs::create_dir_all(&checkpoint_dir).expect("create checkpoint dir");
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();

    let net = if let Some(path) = cli.load {
        let n = path
            .file_name()
            .expect("file name")
            .to_str()
            .expect("file name to string")
            .strip_prefix("iter_")
            .and_then(|s| s.strip_suffix(".bin"))
            .expect("expected iter_NNNNN.bin")
            .parse::<usize>()
            .expect("iter to usize");

        eprintln!(
            "resume: loading {} (completed iter index {n}, next global index {})",
            path.display(),
            n + 1
        );
        CatanNet::<B>::new(&net_config, &device)
            .load_file(&path, &recorder, &device)
            .unwrap_or_else(|e| panic!("load checkpoint {}: {e}", path.display()))
    } else {
        CatanNet::<B>::new(&net_config, &device)
    };

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
    const PHASE_BUCKET_LABELS: [&str; CATAN_PHASE_BUCKETS] = [
        "setup",
        "preroll",
        "discard",
        "robber",
        "main",
        "roadbuilding",
        "gameover",
    ];
    const EMA_ALPHA: f32 = 0.1;

    let action_counts: [AtomicU64; ACTION_LOG_BUCKETS] = std::array::from_fn(|_| AtomicU64::new(0));
    let phase_counts: [AtomicU64; CATAN_PHASE_BUCKETS] = std::array::from_fn(|_| AtomicU64::new(0));
    let step_total = AtomicU64::new(0);
    let mut action_ema: Option<[f32; ACTION_LOG_BUCKETS]> = None;

    let on_step = |info: &SelfPlayStepInfo<CatanGame, ACT, OBS, PLAYERS>| {
        let ab = action_log_bucket(info.action);
        let pb = catan_phase_bucket(&info.game.as_sim().phase) as usize;
        action_counts[ab].fetch_add(1, Ordering::Relaxed);
        phase_counts[pb].fetch_add(1, Ordering::Relaxed);
        step_total.fetch_add(1, Ordering::Relaxed);
    };

    const CHECKPOINT_EVERY: usize = 10;

    let on_iter = |stats: &IterationStats, net: &CatanNet<B>| {
        let total = step_total.swap(0, Ordering::Relaxed).max(1) as f32;
        let action_frac: [f32; ACTION_LOG_BUCKETS] =
            std::array::from_fn(|i| action_counts[i].swap(0, Ordering::Relaxed) as f32 / total);
        let phase_frac: [f32; CATAN_PHASE_BUCKETS] =
            std::array::from_fn(|i| phase_counts[i].swap(0, Ordering::Relaxed) as f32 / total);

        let ema = action_ema.get_or_insert(action_frac);
        for i in 0..ACTION_LOG_BUCKETS {
            ema[i] = EMA_ALPHA * action_frac[i] + (1.0 - EMA_ALPHA) * ema[i];
        }

        let action_str = ACTION_BUCKET_LABELS
            .iter()
            .zip(ema.iter())
            .map(|(l, f)| format!("{l}={f:.2}"))
            .collect::<Vec<_>>()
            .join(" ");
        let phase_str = PHASE_BUCKET_LABELS
            .iter()
            .zip(phase_frac.iter())
            .map(|(l, f)| format!("{l}={f:.2}"))
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
            "az iter {}: samples={} ({:.0}/s) buf={} loss=(p={:.4} v={:.4} t={:.4}) {:.2}s | gpu[disp={} avg_bs={:.0} leaves={} leaves/s={:.0}] | games[fin={} cut={} plies mean={:.0} p50={} p95={} max={}] | actions[{}] | phases[{}]",
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
            stats.finished_games,
            stats.cutoff_games,
            stats.finished_plies_mean,
            stats.finished_plies_p50,
            stats.finished_plies_p95,
            stats.finished_plies_max,
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
            data.insert("games/finished".into(), (stats.finished_games as u64).into());
            data.insert("games/cutoff".into(), (stats.cutoff_games as u64).into());
            data.insert(
                "games/plies_mean".into(),
                (stats.finished_plies_mean as f64).into(),
            );
            data.insert(
                "games/plies_p50".into(),
                (stats.finished_plies_p50 as u64).into(),
            );
            data.insert(
                "games/plies_p95".into(),
                (stats.finished_plies_p95 as u64).into(),
            );
            data.insert(
                "games/plies_max".into(),
                (stats.finished_plies_max as u64).into(),
            );
            for (label, frac) in ACTION_BUCKET_LABELS.iter().zip(action_frac.iter()) {
                data.insert(format!("action/{label}"), (*frac as f64).into());
            }
            for (label, frac) in PHASE_BUCKET_LABELS.iter().zip(phase_frac.iter()) {
                data.insert(format!("phase/{label}"), (*frac as f64).into());
            }

            let run = Arc::clone(run);
            let log_data: LogData = data.into();
            handle.spawn(async move {
                run.log(log_data).await;
            });
        }

        if stats.iteration.is_multiple_of(CHECKPOINT_EVERY) {
            let path = checkpoint_dir.join(format!("iter_{:05}", stats.iteration));
            match net.clone().save_file(&path, &recorder) {
                Ok(()) => eprintln!("checkpoint: saved {}", path.display()),
                Err(e) => eprintln!("checkpoint: failed to save {}: {e}", path.display()),
            }
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
