use az::net::{ResNet, ResNetConfig};
use az::{Game, IterationStats, TrainConfig, train};
use burn::backend::{Autodiff, NdArray};

const ACT: usize = 9;
const OBS: usize = 27;
const PLAYERS: usize = 2;

#[derive(Clone)]
struct TicTacToe {
    board: [u8; 9],
    turn: u8,
    moves: u8,
}

impl TicTacToe {
    fn new() -> Self {
        Self {
            board: [0; 9],
            turn: 1,
            moves: 0,
        }
    }

    fn winner(&self) -> u8 {
        const LINES: [[usize; 3]; 8] = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8],
            [0, 4, 8],
            [2, 4, 6],
        ];
        for line in &LINES {
            let a = self.board[line[0]];
            if a != 0 && a == self.board[line[1]] && a == self.board[line[2]] {
                return a;
            }
        }
        0
    }
}

impl Game<ACT, OBS, PLAYERS> for TicTacToe {
    type Action = u8;
    type Player = u8;

    fn legal_actions(&self) -> Vec<u8> {
        (0..9).filter(|&i| self.board[i as usize] == 0).collect()
    }

    fn apply(&mut self, action: u8) {
        self.board[action as usize] = self.turn;
        self.turn = 3 - self.turn;
        self.moves += 1;
    }

    fn is_terminal(&self) -> bool {
        self.winner() != 0 || self.moves == 9
    }

    fn current_player(&self) -> u8 {
        self.turn
    }

    fn result(&self, player: u8) -> f32 {
        let w = self.winner();
        if w == 0 {
            0.5
        } else if w == player {
            1.0
        } else {
            0.0
        }
    }

    fn action_to_index(action: u8) -> usize {
        action as usize
    }
    fn index_to_action(index: usize) -> u8 {
        index as u8
    }

    fn player_to_index(player: u8) -> usize {
        (player - 1) as usize
    }
    fn index_to_player(index: usize) -> u8 {
        (index + 1) as u8
    }

    fn observe(&self) -> [f32; OBS] {
        let mut obs = [0.0f32; OBS];
        for i in 0..9 {
            match self.board[i] {
                0 => obs[i] = 1.0,
                1 => obs[9 + i] = 1.0,
                2 => obs[18 + i] = 1.0,
                _ => unreachable!(),
            }
        }
        obs
    }
}

#[test]
fn end_to_end() {
    type B = Autodiff<NdArray>;

    let device = Default::default();
    let net = ResNet::<B>::new(
        &ResNetConfig {
            obs_size: OBS,
            act_size: ACT,
            num_players: PLAYERS,
            num_blocks: 1,
            hidden_dim: 16,
        },
        &device,
    );

    let _trained = train::<B, _, TicTacToe, ACT, OBS, PLAYERS>(
        net,
        TicTacToe::new,
        TrainConfig {
            replay_capacity: 1000,
            ..Default::default()
        },
        device,
        |stats: &IterationStats| {
            eprintln!(
                "az iteration {}: {} new samples, buffer={}, policy_loss={:.4}, value_loss={:.4}, total_loss={:.4}, elapsed={:.2}s",
                stats.iteration,
                stats.new_samples,
                stats.buffer_size,
                stats.policy_loss,
                stats.value_loss,
                stats.total_loss,
                stats.elapsed_secs,
            )
        },
        |_| {},
    );
}
