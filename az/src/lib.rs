use std::collections::VecDeque;
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::mpsc;
use std::time::Instant;

use rand::RngExt;
use rayon::prelude::*;

use burn::{
    Tensor,
    module::{AutodiffModule, Module},
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::Backend,
    tensor::{ElementConversion, TensorData, activation::log_softmax, backend::AutodiffBackend},
};

pub mod mcts;
pub mod net;

pub use mcts::{ActionDistribution, Leaf, Tree};
pub use net::{ResNet, ResNetConfig};

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

fn evaluate_batch<B, N, const ACT: usize, const OBS: usize, const PLAYERS: usize>(
    net: &N,
    device: &B::Device,
    observations: &[[f32; OBS]],
) -> Vec<([f32; ACT], [f32; PLAYERS])>
where
    B: Backend,
    N: AZNet<B>,
{
    let bs = observations.len();
    assert!(bs > 0, "evaluate_batch: empty observations");
    let flat: Vec<f32> = observations
        .iter()
        .flat_map(|o| o.iter().copied())
        .collect();
    let tensor: Tensor<B, 2> = Tensor::from_data(TensorData::new(flat, [bs, OBS]), device);
    let (policy_logits, values) = net.forward(tensor);

    let policy_data: Vec<f32> = policy_logits.into_data().to_vec().expect("policy to vec");
    let value_data: Vec<f32> = values.into_data().to_vec().expect("value to vec");

    (0..bs)
        .map(|i| {
            let mut policy = [0.0f32; ACT];
            policy.copy_from_slice(&policy_data[i * ACT..(i + 1) * ACT]);
            let mut vals = [0.0f32; PLAYERS];
            vals.copy_from_slice(&value_data[i * PLAYERS..(i + 1) * PLAYERS]);
            (policy, vals)
        })
        .collect()
}

#[derive(Clone)]
pub struct TrainConfig {
    pub iterations: usize,

    pub games_per_iteration: usize,
    pub batch_size: usize,
    pub replay_capacity: usize,
    pub training_steps_per_iteration: usize,
    pub learning_rate: f64,

    pub max_simulations: u32,
    pub sims_per_eval: u32,
    pub c_puct: f32,
    pub temperature: f32,
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
}

pub fn train_loop<B, N, G, const ACT: usize, const OBS: usize, const PLAYERS: usize>(
    mut net: N,
    game_factory: impl Fn() -> G,
    config: TrainConfig,
    device: B::Device,
    mut on_iteration: impl FnMut(&IterationStats),
) -> N
where
    B: AutodiffBackend,
    N: AZNet<B> + AutodiffModule<B> + Clone,
    N::InnerModule: AZNet<B::InnerBackend> + 'static,
    G: Game<ACT, OBS, PLAYERS>,
{
    let mut buffer = ReplayBuffer::<ACT, OBS, PLAYERS>::new(config.replay_capacity);
    let mut optim = AdamConfig::new().init::<B, N>();
    let mut rng = rand::rng();

    for iteration in 0..config.iterations {
        let iter_start = Instant::now();
        let n = config.games_per_iteration;

        let (mut games, mut trees): (Vec<_>, Vec<_>) =
            (0..n).map(|_| (game_factory(), Tree::new())).unzip();
        let mut histories: Vec<Vec<_>> = (0..n).map(|_| Vec::new()).collect();
        let mut samples: Vec<Sample<ACT, OBS, PLAYERS>> = Vec::new();

        let (tx_obs, rx_obs) = mpsc::channel::<Vec<[f32; OBS]>>();
        let (tx_results, rx_results) = mpsc::channel::<Vec<([f32; ACT], [f32; PLAYERS])>>();

        let gpu_net = net.clone().valid();
        let gpu_dev = device.clone();
        let gpu_thread = std::thread::spawn(move || {
            while let Ok(obs) = rx_obs.recv() {
                let results = evaluate_batch::<B::InnerBackend, N::InnerModule, ACT, OBS, PLAYERS>(
                    &gpu_net, &gpu_dev, &obs,
                );
                tx_results.send(results).expect("results receiver alive");
            }
        });

        while !games.is_empty() {
            let spe = config.sims_per_eval as usize;
            let outer_rounds = (config.max_simulations as usize).div_ceil(spe);
            let n_games = games.len();
            let c_puct = config.c_puct;

            let mut prev: Option<Vec<(usize, Leaf<G>)>> = None;

            for round in 0..=outer_rounds {
                let (pending, obs) = if round < outer_rounds {
                    let batched: Vec<Vec<_>> = trees
                        .par_iter_mut()
                        .zip(games.par_iter())
                        .map(|(tree, game)| {
                            (0..spe)
                                .filter_map(|_| {
                                    tree.select::<G, OBS>(game, c_puct).map(|leaf| {
                                        let obs = leaf.state.observe();
                                        (leaf, obs)
                                    })
                                })
                                .collect()
                        })
                        .collect();
                    let mut pending = Vec::new();
                    let mut obs = Vec::new();
                    for (i, group) in batched.into_iter().enumerate() {
                        for (leaf, o) in group {
                            pending.push((i, leaf));
                            obs.push(o);
                        }
                    }
                    (pending, obs)
                } else {
                    (Vec::new(), Vec::new())
                };

                let pending_and_evals = prev.take().map(|prev_pending| {
                    let evals = rx_results.recv().expect("gpu thread alive");
                    (prev_pending, evals)
                });

                if round < outer_rounds && !obs.is_empty() {
                    tx_obs.send(obs).expect("gpu thread alive");
                    prev = Some(pending);
                }

                if let Some((prev_pending, evals)) = pending_and_evals {
                    debug_assert_eq!(
                        prev_pending.len(),
                        evals.len(),
                        "MCTS pending leaves must match network batch size"
                    );
                    let mut groups: Vec<Vec<_>> = (0..n_games).map(|_| Vec::new()).collect();
                    for ((i, leaf), (policy, values)) in prev_pending.into_iter().zip(evals) {
                        groups[i].push((leaf, policy, values));
                    }
                    trees
                        .par_iter_mut()
                        .zip(groups.into_par_iter())
                        .for_each(|(tree, group)| {
                            for (ref leaf, ref policy, values) in group {
                                tree.expand::<G, OBS>(leaf, policy, values);
                            }
                        });
                }
            }

            // Pick moves from the search distributions.
            let move_data: Vec<_> = games
                .par_iter()
                .zip(trees.par_iter())
                .map(|(game, tree)| {
                    let dist = tree.action_distribution();
                    let obs = game.observe();
                    let policy = dist.policy(config.temperature);
                    let player = game.current_player();
                    let has_root_visits = dist.visits.iter().any(|&v| v > 0);
                    let action = if has_root_visits {
                        G::index_to_action(dist.pick(config.temperature, &mut rand::rng()))
                    } else {
                        // If search produced no root visits for this game,
                        // fall back to a random legal action instead of
                        // deterministically picking index 0.
                        let legal = game.legal_actions();
                        debug_assert!(
                            !legal.is_empty(),
                            "non-terminal game must have legal actions"
                        );
                        let mut rng = rand::rng();
                        legal[rng.random_range(0..legal.len())]
                    };
                    (obs, policy, player, action, has_root_visits)
                })
                .collect();

            for (i, (obs, policy, player, action, _has_root_visits)) in
                move_data.into_iter().enumerate()
            {
                histories[i].push((obs, policy, player));
                games[i].apply(action);
            }

            // Retire finished games and reset trees for ongoing ones.
            let mut i = games.len();
            while i > 0 {
                i -= 1;
                let is_terminal = games[i].is_terminal();
                if is_terminal {
                    let game = games.swap_remove(i);
                    trees.swap_remove(i);
                    let game_history = histories.swap_remove(i);
                    let result: [f32; PLAYERS] =
                        std::array::from_fn(|p| game.result(G::index_to_player(p)));
                    samples.extend(game_history.into_iter().map(|(obs, pol, _player)| Sample {
                        observation: obs,
                        policy_target: pol,
                        value_target: result,
                    }));
                } else {
                    trees[i] = Tree::new();
                }
            }

        }

        drop(tx_obs);
        gpu_thread.join().expect("gpu thread panicked");

        let num_samples = samples.len();
        buffer.extend(samples);

        // Training phase.
        let mut last_policy_loss = 0.0f32;
        let mut last_value_loss = 0.0f32;
        let mut last_total_loss = 0.0f32;

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
            new_samples: num_samples,
            buffer_size: buffer.len(),
            policy_loss: last_policy_loss,
            value_loss: last_value_loss,
            total_loss: last_total_loss,
            elapsed_secs: iter_start.elapsed().as_secs_f32(),
        });
    }

    net
}

// ── Replay buffer ────────────────────────────────────────────────────────────

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
