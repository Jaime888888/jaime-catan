#![recursion_limit = "256"]

pub mod catan_net;

pub use catan_net::{CatanNet, CatanNetConfig, OBS};

use az::Game;
use catan_sim::{
    ACTION_SPACE_SIZE, Action, ObsEdge, ObsVertex, Phase, PlayerId, PlayerRelation, Port, Turn,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

pub const ACT: usize = ACTION_SPACE_SIZE;
pub const PLAYERS: usize = 4;

#[cfg(target_os = "macos")]
pub mod backend {
    pub type Inner = burn::backend::Metal;
    pub const NAME: &str = "Metal";
}

#[cfg(all(not(target_os = "macos"), feature = "cuda"))]
pub mod backend {
    pub type Inner = burn::backend::Cuda;
    pub const NAME: &str = "CUDA";
}

#[cfg(all(
    not(target_os = "macos"),
    not(feature = "cuda"),
    any(target_os = "linux", target_os = "windows"),
))]
pub mod backend {
    pub type Inner = burn::backend::Vulkan;
    pub const NAME: &str = "Vulkan";
}

#[cfg(all(
    not(target_os = "macos"),
    not(feature = "cuda"),
    not(target_os = "linux"),
    not(target_os = "windows"),
))]
pub mod backend {
    pub type Inner = burn::backend::NdArray;
    pub const NAME: &str = "NdArray (CPU)";
}

pub fn catan_phase_bucket(phase: &Phase) -> u8 {
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
pub const ACTION_LOG_BUCKETS: usize = 11;

pub fn action_log_bucket(a: Action) -> usize {
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

pub const ACTION_BUCKET_LABELS: [&str; ACTION_LOG_BUCKETS] = [
    "end", "bank", "settle", "road", "city", "bdev", "knight", "odev", "roll", "rob", "disc",
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
pub struct CatanGame {
    inner: catan_sim::Game,
    rng: ChaCha8Rng,
}

impl CatanGame {
    pub fn as_sim(&self) -> &catan_sim::Game {
        &self.inner
    }

    pub fn new() -> Self {
        let mut rng = ChaCha8Rng::from_rng(&mut rand::rng());
        Self::new_with_rng(&mut rng)
    }

    pub fn new_with_rng(rng: &mut ChaCha8Rng) -> Self {
        let mut game = Self {
            inner: catan_sim::Game::new(rng),
            rng: ChaCha8Rng::from_rng(rng),
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

impl Default for CatanGame {
    fn default() -> Self {
        Self::new()
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
