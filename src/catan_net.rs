//! Catan-specific policy–value network: splits the flat observation vector into the same regions
//! as `observe()` in `main.rs`, encodes each with its own linear layer, concatenates, then runs a
//! residual MLP trunk. [`AZNet::forward`](az::AZNet) still receives `[batch, OBS]`; all structure is
//! internal to this module.

use az::AZNet;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig, Relu};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

pub const OBS: usize = TILES + VERTICES + EDGES + HARBORS + SELF_PLAYER + OTHER_PLAYERS + META;

const TILES: usize = 19 * (6 + 2);
const VERTICES: usize = 54 * 3;
const EDGES: usize = 72 * 3;
const HARBORS: usize = 9 * 6;
const SELF_PLAYER: usize = 5 + 5 + 1 + 1 + 1 + 1 + 1 + 5;
const OTHER_PLAYERS: usize = 3 * 5;
const META: usize = 1 + 1 + 5;

const _OBS_LAYOUT: () = assert!(OBS == 626);

const BRANCHES: usize = 7;

#[derive(Clone, Debug)]
pub struct CatanNetConfig {
    pub act_size: usize,
    pub num_players: usize,
    pub branch_dim: usize,
    pub trunk_hidden: usize,
    pub num_blocks: usize,
}

#[derive(Module, Debug)]
struct FcResBlock<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    relu: Relu,
}

impl<B: Backend> FcResBlock<B> {
    fn new(dim: usize, device: &B::Device) -> Self {
        Self {
            fc1: LinearConfig::new(dim, dim).init(device),
            fc2: LinearConfig::new(dim, dim).init(device),
            relu: Relu::new(),
        }
    }

    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let residual = x.clone();
        let x = self.fc1.forward(x);
        let x = self.relu.forward(x);
        let x = self.fc2.forward(x);
        self.relu.forward(x + residual)
    }
}

#[derive(Module, Debug)]
pub struct CatanNet<B: Backend> {
    enc_tiles: Linear<B>,
    enc_vertices: Linear<B>,
    enc_edges: Linear<B>,
    enc_harbors: Linear<B>,
    enc_self: Linear<B>,
    enc_others: Linear<B>,
    enc_meta: Linear<B>,
    fusion: Linear<B>,
    blocks: Vec<FcResBlock<B>>,
    policy_head: Linear<B>,
    value_fc: Linear<B>,
    value_out: Linear<B>,
    relu: Relu,
}

impl<B: Backend> CatanNet<B> {
    pub fn new(config: &CatanNetConfig, device: &B::Device) -> Self {
        let d = config.branch_dim;
        let blocks = (0..config.num_blocks)
            .map(|_| FcResBlock::new(config.trunk_hidden, device))
            .collect();

        Self {
            enc_tiles: LinearConfig::new(TILES, d).init(device),
            enc_vertices: LinearConfig::new(VERTICES, d).init(device),
            enc_edges: LinearConfig::new(EDGES, d).init(device),
            enc_harbors: LinearConfig::new(HARBORS, d).init(device),
            enc_self: LinearConfig::new(SELF_PLAYER, d).init(device),
            enc_others: LinearConfig::new(OTHER_PLAYERS, d).init(device),
            enc_meta: LinearConfig::new(META, d).init(device),
            fusion: LinearConfig::new(BRANCHES * d, config.trunk_hidden).init(device),
            blocks,
            policy_head: LinearConfig::new(config.trunk_hidden, config.act_size).init(device),
            value_fc: LinearConfig::new(config.trunk_hidden, config.trunk_hidden).init(device),
            value_out: LinearConfig::new(config.trunk_hidden, config.num_players).init(device),
            relu: Relu::new(),
        }
    }
}

impl<B: Backend> AZNet<B> for CatanNet<B> {
    fn forward(&self, obs: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let mut start = 0usize;
        let mut encode_branch = |enc: &Linear<B>, obs: Tensor<B, 2>, len: usize| {
            let x = obs.slice_dim(1, start..start + len);
            let x = self.relu.forward(enc.forward(x));
            start += len;
            x
        };

        let t = encode_branch(&self.enc_tiles, obs.clone(), TILES);
        let v = encode_branch(&self.enc_vertices, obs.clone(), VERTICES);
        let e = encode_branch(&self.enc_edges, obs.clone(), EDGES);
        let h = encode_branch(&self.enc_harbors, obs.clone(), HARBORS);
        let s = encode_branch(&self.enc_self, obs.clone(), SELF_PLAYER);
        let o = encode_branch(&self.enc_others, obs.clone(), OTHER_PLAYERS);
        let m = encode_branch(&self.enc_meta, obs, META);

        let fused = Tensor::cat(vec![t, v, e, h, s, o, m], 1);

        let mut x = self.relu.forward(self.fusion.forward(fused));
        for block in &self.blocks {
            x = block.forward(x);
        }

        let policy_logits = self.policy_head.forward(x.clone());
        let v = self.relu.forward(self.value_fc.forward(x));
        let value = self.value_out.forward(v);

        (policy_logits, value)
    }
}
