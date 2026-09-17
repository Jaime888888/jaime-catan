use burn::module::Module;
use burn::nn::{Linear, LinearConfig, Relu};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

use crate::AZNet;

#[derive(Module, Debug)]
struct ResBlock<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    relu: Relu,
}

impl<B: Backend> ResBlock<B> {
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

#[derive(Clone)]
pub struct ResNetConfig {
    pub obs_size: usize,
    pub act_size: usize,
    pub num_players: usize,
    pub num_blocks: usize,
    pub hidden_dim: usize,
}

impl Default for ResNetConfig {
    fn default() -> Self {
        Self {
            obs_size: 64,
            act_size: 9,
            num_players: 2,
            num_blocks: 4,
            hidden_dim: 128,
        }
    }
}

#[derive(Module, Debug)]
pub struct ResNet<B: Backend> {
    input_proj: Linear<B>,
    blocks: Vec<ResBlock<B>>,
    policy_head: Linear<B>,
    value_fc: Linear<B>,
    value_out: Linear<B>,
    relu: Relu,
}

impl<B: Backend> ResNet<B> {
    pub fn new(config: &ResNetConfig, device: &B::Device) -> Self {
        let blocks = (0..config.num_blocks)
            .map(|_| ResBlock::new(config.hidden_dim, device))
            .collect();

        Self {
            input_proj: LinearConfig::new(config.obs_size, config.hidden_dim).init(device),
            blocks,
            policy_head: LinearConfig::new(config.hidden_dim, config.act_size).init(device),
            value_fc: LinearConfig::new(config.hidden_dim, config.hidden_dim).init(device),
            value_out: LinearConfig::new(config.hidden_dim, config.num_players).init(device),
            relu: Relu::new(),
        }
    }
}

impl<B: Backend> AZNet<B> for ResNet<B> {
    fn forward(&self, obs: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let mut x = self.relu.forward(self.input_proj.forward(obs));
        for block in &self.blocks {
            x = block.forward(x);
        }

        let policy_logits = self.policy_head.forward(x.clone());

        let v = self.relu.forward(self.value_fc.forward(x));
        let value = self.value_out.forward(v);

        (policy_logits, value)
    }
}
