use std::collections::VecDeque;
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use flume::RecvTimeoutError;
use rand::RngExt;

use burn::{
    Tensor,
    module::{AutodiffModule, Module},
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::Backend,
    tensor::{ElementConversion, TensorData, activation::log_softmax, backend::AutodiffBackend},
};

pub mod mcts;
pub mod net;

pub use mcts::{ActionDistribution, ActionPolicy, Leaf, Tree};
pub use net::{ResNet, ResNetConfig};

const NUM_EVALUATORS: usize = 2;

pub trait Game<const ACT: usize, const OBS: usize, const PLAYERS: usize>:
    Clone + Send + Sync
{
    type Action: Copy + Eq + Hash + Debug + Send + Sync;
    type Player: Copy + Eq + Hash + Debug + Send + Sync;

    fn legal_actions(&self) -> Vec<Self::Action>;
    fn apply(&mut self, action: Self::Action);
    fn is_terminal(&self) -> bool;
    fn current_player(&self) -> Self::Player;
    fn result(&self, player: Self::Player) -> f32;
    fn action_to_index(action: Self::Action) -> usize;
    fn index_to_action(index: usize) -> Self::Action;
    fn player_to_index(player: Self::Player) -> usize;
    fn index_to_player(index: usize) -> Self::Player;
    fn observe(&self) -> [f32; OBS];
}

pub trait AZNet<B: Backend>: Module<B> + Send {
    fn forward(&self, obs: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>);
}

#[derive(Clone)]
pub struct Sample<const ACT: usize, const OBS: usize, const PLAYERS: usize> {
    pub observation: [f32; OBS],
    pub policy_target: [f32; ACT],
    pub value_target: [f32; PLAYERS],
}

#[derive(Clone)]
pub struct TrainConfig {
    pub iterations: usize,

    pub games_per_iteration: usize,
    pub batch_size: usize,
    pub replay_capacity: usize,
    pub training_steps_per_iteration: usize,
    pub learning_rate: f64,

    pub max_simulations: usize,
    pub sims_per_eval: usize,
    pub c_puct: f32,
    pub temperature: f32,
    pub root_dirichlet_epsilon: f32,
    pub root_dirichlet_alpha: f32,
    pub max_plies_per_game: Option<usize>,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            iterations: 100,
            games_per_iteration: 100,
            batch_size: 64,
            replay_capacity: 50_000,
            training_steps_per_iteration: 100,
            learning_rate: 1e-3,
            max_simulations: 800,
            sims_per_eval: 8,
            c_puct: 1.5,
            temperature: 1.0,
            root_dirichlet_epsilon: 0.25,
            root_dirichlet_alpha: 0.3,
            max_plies_per_game: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct IterationStats {
    pub iteration: usize,
    pub new_samples: usize,
    pub buffer_size: usize,
    pub policy_loss: f32,
    pub value_loss: f32,
    pub total_loss: f32,
    pub elapsed_secs: f32,
    /// Total GPU eval dispatches across all evaluators this iteration.
    pub gpu_dispatches: u64,
    /// Total leaves (rows in the forward pass) across all evaluators this iteration.
    pub gpu_leaves: u64,
}

/// Passed to `on_self_play_step` once per ply, from the player thread that
/// owns the game. Fired after MCTS and action selection, before `apply`, so
/// `game` reflects the state the action was chosen from. Called concurrently
/// across player threads; the closure must be `Sync`.
pub struct SelfPlayStepInfo<'a, G, const ACT: usize, const OBS: usize, const PLAYERS: usize>
where
    G: Game<ACT, OBS, PLAYERS>,
{
    pub iteration: usize,
    pub game_idx: usize,
    pub ply: usize,
    pub game: &'a G,
    pub action: G::Action,
    pub policy: &'a ActionPolicy<ACT>,
}

struct EvalRequest<const ACT: usize, const PLAYERS: usize> {
    obs: Vec<f32>,
    bs: usize,
    reply: oneshot::Sender<Vec<([f32; ACT], [f32; PLAYERS])>>,
}

pub fn train<B, N, G, const ACT: usize, const OBS: usize, const PLAYERS: usize>(
    mut net: N,
    game_factory: impl Fn() -> G + Sync,
    config: TrainConfig,
    device: B::Device,
    mut on_iteration: impl FnMut(&IterationStats),
    on_self_play_step: impl Fn(&SelfPlayStepInfo<G, ACT, OBS, PLAYERS>) + Sync,
) -> N
where
    B: AutodiffBackend,
    N: AZNet<B> + AutodiffModule<B> + Clone,
    N::InnerModule: AZNet<B::InnerBackend> + 'static,
    G: Game<ACT, OBS, PLAYERS>,
{
    let on_self_play_step = &on_self_play_step;
    let replay_buffer = Arc::new(Mutex::new(ReplayBuffer::<ACT, OBS, PLAYERS>::new(
        config.replay_capacity,
    )));

    let mut optim = AdamConfig::new().init::<B, N>();
    let mut rng = rand::rng();

    for iteration in 0..config.iterations {
        let iter_start = Instant::now();
        let new_samples = Arc::new(AtomicUsize::new(0));
        let gpu_dispatches = Arc::new(AtomicU64::new(0));
        let gpu_leaves = Arc::new(AtomicU64::new(0));

        std::thread::scope(|s| {
            let evaluators = (0..NUM_EVALUATORS)
                .map(|i| {
                    let (tx, rx) = flume::unbounded::<EvalRequest<ACT, PLAYERS>>();

                    let live = Arc::new(AtomicUsize::new(
                        config.games_per_iteration / NUM_EVALUATORS
                            + (i < config.games_per_iteration % NUM_EVALUATORS) as usize,
                    ));

                    let net_clone = net.clone().valid();
                    let device_clone = device.clone();
                    let live_clone = live.clone();
                    let gpu_dispatches_clone = gpu_dispatches.clone();
                    let gpu_leaves_clone = gpu_leaves.clone();
                    s.spawn(move || {
                        let mut requests = Vec::new();

                        while let Ok(first) = rx.recv() {
                            let mut flat = Vec::new();
                            flat.extend_from_slice(&first.obs);
                            requests.push(first);

                            loop {
                                if requests.len() >= live_clone.load(Ordering::Relaxed) {
                                    break;
                                }

                                match rx.recv_deadline(Instant::now() + Duration::from_micros(200))
                                {
                                    Ok(req) => {
                                        flat.extend_from_slice(&req.obs);
                                        requests.push(req);
                                    }
                                    Err(RecvTimeoutError::Disconnected) => break,
                                    Err(RecvTimeoutError::Timeout) => continue,
                                }
                            }

                            let bs_total: usize = requests.iter().map(|r| r.bs).sum();
                            gpu_dispatches_clone.fetch_add(1, Ordering::Relaxed);
                            gpu_leaves_clone.fetch_add(bs_total as u64, Ordering::Relaxed);

                            let tensor: Tensor<B::InnerBackend, 2> = Tensor::from_data(
                                TensorData::new(flat, [bs_total, OBS]),
                                &device_clone,
                            );

                            let (policy, values) = net_clone.forward(tensor);

                            let pol = policy.into_data().to_vec().expect("policy to vec");
                            let val = values.into_data().to_vec().expect("value to vec");

                            let mut row = 0usize;
                            for req in requests.drain(..) {
                                let mut out = Vec::with_capacity(req.bs);
                                for i in 0..req.bs {
                                    let r = row + i;
                                    let mut p = [0.0f32; ACT];
                                    p.copy_from_slice(&pol[r * ACT..(r + 1) * ACT]);
                                    let mut v = [0.0f32; PLAYERS];
                                    v.copy_from_slice(&val[r * PLAYERS..(r + 1) * PLAYERS]);
                                    out.push((p, v));
                                }
                                let _ = req.reply.send(out);
                                row += req.bs;
                            }
                        }
                    });

                    (tx, live)
                })
                .collect::<Vec<_>>();

            for game_idx in 0..config.games_per_iteration {
                let (tx, live) = evaluators[game_idx % NUM_EVALUATORS].clone();
                let mut game = game_factory();

                let replay_buffer_clone = replay_buffer.clone();
                let new_samples_clone = new_samples.clone();

                s.spawn(move || {
                    struct LiveGuard(Arc<AtomicUsize>);

                    impl Drop for LiveGuard {
                        fn drop(&mut self) {
                            self.0.fetch_sub(1, Ordering::Relaxed);
                        }
                    }

                    let _live_guard = LiveGuard(live.clone());

                    let mut history = Vec::new();
                    let mut plies = 0usize;
                    let mut rng = rand::rng();

                    let mut tree = Tree::<ACT, PLAYERS>::default();
                    let mut leaves = Vec::with_capacity(config.sims_per_eval);

                    loop {
                        for _ in 0..config.max_simulations.div_ceil(config.sims_per_eval) {
                            let mut obs_flat = Vec::with_capacity(config.sims_per_eval * OBS);
                            for _ in 0..config.sims_per_eval {
                                if let Some(leaf) = tree.select::<G, OBS>(&game, config.c_puct) {
                                    obs_flat.extend_from_slice(&leaf.state.observe());
                                    leaves.push(leaf);
                                }
                            }
                            if leaves.is_empty() {
                                break;
                            }

                            let (reply_tx, reply_rx) = oneshot::channel();

                            if tx
                                .send(EvalRequest {
                                    obs: obs_flat,
                                    bs: leaves.len(),
                                    reply: reply_tx,
                                })
                                .is_err()
                            {
                                return;
                            }

                            let Ok(results) = reply_rx.recv() else {
                                return;
                            };

                            for (leaf, (policy, values)) in leaves.iter().zip(results) {
                                tree.expand::<G, OBS>(
                                    leaf,
                                    &policy,
                                    values,
                                    config.root_dirichlet_epsilon,
                                    config.root_dirichlet_alpha,
                                );
                            }

                            leaves.clear();
                        }

                        let obs = game.observe();
                        let policy = tree.action_distribution().policy(config.temperature);
                        tree.clear();

                        if policy.0.iter().all(|&p| p == 0.0) {
                            return;
                        }

                        let action = G::index_to_action(policy.pick(&mut rng));
                        on_self_play_step(&SelfPlayStepInfo {
                            iteration,
                            game_idx,
                            ply: plies,
                            game: &game,
                            action,
                            policy: &policy,
                        });

                        game.apply(action);
                        history.push((obs, policy, game.current_player()));
                        plies += 1;

                        if game.is_terminal() {
                            let value_target =
                                std::array::from_fn(|p| game.result(G::index_to_player(p)));

                            new_samples_clone.fetch_add(history.len(), Ordering::Relaxed);

                            let mut buffer = replay_buffer_clone.lock().unwrap();
                            buffer.extend(history.into_iter().map(|(obs, pol, _)| Sample {
                                observation: obs,
                                policy_target: pol.0,
                                value_target,
                            }));

                            return;
                        } else if config.max_plies_per_game.is_some_and(|c| plies >= c) {
                            return;
                        }
                    }
                });
            }
        });

        let mut last_policy_loss = 0.0f32;
        let mut last_value_loss = 0.0f32;
        let mut last_total_loss = 0.0f32;

        let buffer = replay_buffer.lock().unwrap();
        if buffer.len() >= config.batch_size {
            for _ in 0..config.training_steps_per_iteration {
                let batch = buffer.sample_batch(config.batch_size, &mut rng);
                let bs = batch.len();

                let obs_flat: Vec<f32> = batch
                    .iter()
                    .flat_map(|s| s.observation.iter().copied())
                    .collect();
                let obs = Tensor::<B, 2>::from_data(TensorData::new(obs_flat, [bs, OBS]), &device);

                let tp_flat: Vec<f32> = batch
                    .iter()
                    .flat_map(|s| s.policy_target.iter().copied())
                    .collect();
                let target_policy =
                    Tensor::<B, 2>::from_data(TensorData::new(tp_flat, [bs, ACT]), &device);

                let tv_flat: Vec<f32> = batch
                    .iter()
                    .flat_map(|s| s.value_target.iter().copied())
                    .collect();
                let target_value =
                    Tensor::<B, 2>::from_data(TensorData::new(tv_flat, [bs, PLAYERS]), &device);

                let (policy_logits, value_pred) = net.forward(obs);

                let log_probs = log_softmax(policy_logits, 1);
                let policy_loss = (target_policy * log_probs).sum().neg() / (bs as f32);

                let value_diff = value_pred - target_value;
                let value_loss = (value_diff.clone() * value_diff).mean();

                let loss = policy_loss.clone() + value_loss.clone();

                last_policy_loss = policy_loss.into_scalar().elem::<f32>();
                last_value_loss = value_loss.into_scalar().elem::<f32>();
                last_total_loss = loss.clone().into_scalar().elem::<f32>();

                let raw_grads = loss.backward();
                let grads = GradientsParams::from_grads(raw_grads, &net);
                net = optim.step(config.learning_rate, net, grads);
            }
        }

        on_iteration(&IterationStats {
            iteration: iteration + 1,
            new_samples: new_samples.load(Ordering::Relaxed),
            buffer_size: buffer.len(),
            policy_loss: last_policy_loss,
            value_loss: last_value_loss,
            total_loss: last_total_loss,
            elapsed_secs: iter_start.elapsed().as_secs_f32(),
            gpu_dispatches: gpu_dispatches.load(Ordering::Relaxed),
            gpu_leaves: gpu_leaves.load(Ordering::Relaxed),
        });
    }

    net
}

struct ReplayBuffer<const ACT: usize, const OBS: usize, const PLAYERS: usize> {
    buf: VecDeque<Sample<ACT, OBS, PLAYERS>>,
    capacity: usize,
}

impl<const ACT: usize, const OBS: usize, const PLAYERS: usize> ReplayBuffer<ACT, OBS, PLAYERS> {
    fn new(capacity: usize) -> Self {
        Self {
            buf: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    fn extend(&mut self, samples: impl IntoIterator<Item = Sample<ACT, OBS, PLAYERS>>) {
        for sample in samples {
            if self.buf.len() >= self.capacity {
                self.buf.pop_front();
            }
            self.buf.push_back(sample);
        }
    }

    fn len(&self) -> usize {
        self.buf.len()
    }

    fn sample_batch(
        &self,
        batch_size: usize,
        rng: &mut impl rand::Rng,
    ) -> Vec<Sample<ACT, OBS, PLAYERS>> {
        let n = self.buf.len();
        (0..batch_size)
            .map(|_| self.buf[rng.random_range(0..n)].clone())
            .collect()
    }
}
