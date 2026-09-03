# Catan AlphaZero

An experimental Rust workspace for training and evaluating an AlphaZero-style agent for four-player Catan. The project combines a rules-based simulator, Monte Carlo Tree Search (MCTS), self-play, and a policy-value neural network built with [Burn](https://burn.dev/).

> **Project status:** Research prototype. It provides simulation, training, checkpointing, and model-arena tools; it is not a graphical or human-playable Catan client.

## Highlights

- Four-player Catan simulation with randomized boards and turn/chance handling
- Rules for setup, building, bank trades, development cards, discards, the robber, longest road, largest army, and victory
- Fixed 254-action space and 626-value observation encoding
- Parallel MCTS and self-play with replay-buffer training
- Catan-specific residual policy-value network
- Periodic checkpoint saving and optional Weights & Biases logging
- Arena evaluation with four-seat matches and online Plackett-Luce/Elo-style ratings
- Metal, Vulkan, and optional CUDA compute backends through Burn

## Workspace layout

| Path | Package | Purpose |
| --- | --- | --- |
| `sim/` | `catan-sim` | Board topology, game state, legal actions, transitions, observations, terminal display, and rule tests |
| `az/` | `az` | Reusable game trait, MCTS, self-play/training loop, replay samples, and policy-value network components |
| `src/` | `catan` | Catan-to-AlphaZero adapter, observation encoder, model definition, and training executable |
| `src/bin/arena.rs` | `arena` | Evaluates saved checkpoints against one another |
| `run.job` | — | Example Slurm job for CUDA training on an A100 node |
| `test.job` | — | Example shorter Slurm/CUDA training job |

## Requirements

- A current stable Rust toolchain with Cargo and Rust 2024 edition support
- One of the following Burn backends:
  - macOS: Metal
  - Linux or Windows: Vulkan by default
  - Linux or Windows: CUDA when built with `--features cuda`
- Optional: a [Weights & Biases](https://wandb.ai/) API key for experiment logging

Install Rust with [rustup](https://rustup.rs/) if it is not already available.

## Build and test

```bash
git clone https://github.com/Jaime888888/jaime-catan.git
cd jaime-catan
cargo build --release
cargo test --workspace
```

The workspace includes extensive simulator rule tests plus an end-to-end Tic-Tac-Toe test for the generic AlphaZero engine.

## Train an agent

Run training with the platform's default backend:

```bash
cargo run --release --bin catan -- \
  --checkpoint-dir ./checkpoints
```

Enable CUDA explicitly on a compatible Linux or Windows system:

```bash
cargo run --release --features cuda --bin catan -- \
  --checkpoint-dir ./checkpoints
```

Checkpoints are written every ten training iterations using names such as `iter_00010.bin`. Resume from a saved checkpoint with:

```bash
cargo run --release --bin catan -- \
  --checkpoint-dir ./checkpoints \
  --load ./checkpoints/iter_00010.bin
```

### Optional experiment logging

If `WANDB_API_KEY` is present, the trainer attempts to log metrics to a `catan-az` Weights & Biases run. Keep the key in your environment—never commit it:

```bash
export WANDB_API_KEY="your-key"
cargo run --release --bin catan -- --checkpoint-dir ./checkpoints
```

Without that variable, training continues without remote logging.

## Compare checkpoints

The arena loads every `iter_*.bin` checkpoint in a directory, samples four distinct checkpoints per match, and reports games, wins, win rate, and relative Elo-style strength:

```bash
cargo run --release --bin arena -- \
  --checkpoints ./checkpoints
```

At least four checkpoints are required. Arena constants such as game count, display behavior, maximum plies, and seed are currently configured in `src/bin/arena.rs`.

## Slurm jobs

`run.job` and `test.job` are cluster-specific examples. Before submitting them:

1. Update the Slurm account, partition, node constraints, memory, and module versions for your cluster.
2. Provide `WANDB_API_KEY` through the scheduler environment or your shell; do not place a real key in the repository.
3. Create or verify writable log and checkpoint locations.

Example submission:

```bash
sbatch run.job
```

## Generated data

The repository ignores common training outputs:

- `target/` — Rust build artifacts
- `checkpoints/` — saved model checkpoints
- `gladiators/` — generated evaluation data
- `flamegraph.svg` — profiling output
