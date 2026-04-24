use std::collections::VecDeque;
use std::fmt::Debug;
use std::hash::Hash;
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

pub fn evaluate_batch<B, N, const ACT: usize, const OBS: usize, const PLAYERS: usize>(
    net: &N,
    device: &B::Device,
    observations: &[[f32; OBS]],
) -> Vec<([f32; ACT], [f32; PLAYERS])>
where
    B: Backend,
    N: AZNet<B>,
{
    let bs = observations.len();
    assert!(bs > 0, "evaluate_batch called with empty observations");

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
            let mut values = [0.0f32; PLAYERS];
            values.copy_from_slice(&value_data[i * PLAYERS..(i + 1) * PLAYERS]);
            (policy, values)
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
    N::InnerModule: AZNet<B::InnerBackend>,
    G: Game<ACT, OBS, PLAYERS>,
{
    let mut buffer = ReplayBuffer::<ACT, OBS, PLAYERS>::new(config.replay_capacity);
    let mut optim = AdamConfig::new().init::<B, N>();
    let mut rng = rand::rng();

    for iteration in 0..config.iterations {
        let iter_start = Instant::now();
        let inference_net = net.clone().valid();
        let n = config.games_per_iteration;

        let (mut games, mut trees): (Vec<_>, Vec<_>) =
            (0..n).map(|_| (game_factory(), Tree::new())).unzip();
        let mut histories = (0..n).map(|_| Vec::new()).collect::<Vec<_>>();
        let mut samples: Vec<Sample<ACT, OBS, PLAYERS>> = Vec::new();

        while !games.is_empty() {
            for _ in 0..config.max_simulations {
                let leaves: Vec<Option<(usize, Leaf<G>, [f32; OBS])>> = trees
                    .par_iter_mut()
                    .zip(games.par_iter())
                    .enumerate()
                    .map(|(i, (tree, game))| {
                        tree.select::<G, OBS>(game, config.c_puct).map(|leaf| {
                            let obs = leaf.state.observe();
                            (i, leaf, obs)
                        })
                    })
                    .collect();

                let mut pending: Vec<(usize, Leaf<G>)> = Vec::new();
                let mut pending_obs: Vec<[f32; OBS]> = Vec::new();
                for entry in leaves.into_iter().flatten() {
                    let (i, leaf, obs) = entry;
                    pending_obs.push(obs);
                    pending.push((i, leaf));
                }

                if !pending_obs.is_empty() {
                    let evals = evaluate_batch::<B::InnerBackend, N::InnerModule, ACT, OBS, PLAYERS>(
                        &inference_net,
                        &device,
                        &pending_obs,
                    );

                    for ((i, leaf), (policy, values)) in pending.into_iter().zip(evals) {
                        trees[i].expand::<G, OBS>(&leaf, &policy, values);
                    }
                }
            }

            let move_data: Vec<_> = games
                .par_iter()
                .zip(trees.par_iter())
                .map(|(game, tree)| {
                    let dist = tree.action_distribution();
                    let obs = game.observe();
                    let policy = dist.policy(config.temperature);
                    let player = game.current_player();
                    let action =
                        G::index_to_action(dist.pick(config.temperature, &mut rand::rng()));
                    (obs, policy, player, action)
                })
                .collect();

            for (i, (obs, policy, player, action)) in move_data.into_iter().enumerate() {
                histories[i].push((obs, policy, player));
                games[i].apply(action);
            }

            let mut i = games.len();
            while i > 0 {
                i -= 1;
                if games[i].is_terminal() {
                    let game = games.swap_remove(i);
                    trees.swap_remove(i);
                    let result: [f32; PLAYERS] =
                        std::array::from_fn(|p| game.result(G::index_to_player(p)));
                    samples.extend(histories.swap_remove(i).into_iter().map(
                        |(obs, pol, _player)| Sample {
                            observation: obs,
                            policy_target: pol,
                            value_target: result,
                        },
                    ));
                } else {
                    trees[i] = Tree::new();
                }
            }
        }

        let num_samples = samples.len();
        buffer.extend(samples);

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

                let grads = loss.backward();
                let grads = GradientsParams::from_grads(grads, &net);
                net = optim.step(config.learning_rate, net, grads);
            }
        }

        let stats = IterationStats {
            iteration: iteration + 1,
            new_samples: num_samples,
            buffer_size: buffer.len(),
            policy_loss: last_policy_loss,
            value_loss: last_value_loss,
            total_loss: last_total_loss,
            elapsed_secs: iter_start.elapsed().as_secs_f32(),
        };
        on_iteration(&stats);
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
