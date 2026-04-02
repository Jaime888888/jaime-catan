pub mod action;
pub mod board;
pub mod observation;
pub mod player;
pub mod transition;
pub mod types;

pub use action::{ACTION_SPACE_SIZE, Action, ActionMask};
pub use board::{Board, Resource, TOPO, Terrain, TileId};
pub use types::*;

use player::Player;
use rand::Rng;

// ---------------------------------------------------------------------------
// Phase
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Phase {
    SetupSettlement {
        player: PlayerId,
        round: u8,
    },
    SetupRoad {
        player: PlayerId,
        round: u8,
        settlement_vertex: u8,
    },
    PreRoll,
    ChanceRoll,
    Discard {
        remaining: PlayerMask,
        active: PlayerId,
    },
    MoveRobber,
    Steal {
        candidates: PlayerMask,
    },
    Main,
    RoadBuilding {
        roads_left: u8,
    },
    GameOver {
        winner: PlayerId,
    },
}

// ---------------------------------------------------------------------------
// Game
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Game {
    pub board: Board,
    pub players: [Player; 4],
    pub phase: Phase,
    pub current_player: PlayerId,
    pub turn_number: u16,
    pub dev_card_played_this_turn: bool,
    pub dev_cards_bought_this_turn: DevCardHand,
    pub has_rolled_this_turn: bool,
    pub longest_road_owner: Option<PlayerId>,
    pub longest_road_length: u8,
    pub largest_army_owner: Option<PlayerId>,
    pub largest_army_size: u8,
}

impl Game {
    pub fn new(rng: &mut impl Rng) -> Self {
        Self {
            board: Board::generate(rng),
            players: [
                Player::new(PlayerId::P0),
                Player::new(PlayerId::P1),
                Player::new(PlayerId::P2),
                Player::new(PlayerId::P3),
            ],
            phase: Phase::SetupSettlement {
                player: PlayerId::P0,
                round: 1,
            },
            current_player: PlayerId::P0,
            turn_number: 0,
            dev_card_played_this_turn: false,
            dev_cards_bought_this_turn: DevCardHand::EMPTY,
            has_rolled_this_turn: false,
            longest_road_owner: None,
            longest_road_length: 0,
            largest_army_owner: None,
            largest_army_size: 0,
        }
    }

    pub fn acting_player(&self) -> PlayerId {
        match &self.phase {
            Phase::SetupSettlement { player, .. } => *player,
            Phase::SetupRoad { player, .. } => *player,
            Phase::Discard { active, .. } => *active,
            _ => self.current_player,
        }
    }

    pub fn is_terminal(&self) -> bool {
        matches!(self.phase, Phase::GameOver { .. })
    }

    pub fn winner(&self) -> Option<PlayerId> {
        match self.phase {
            Phase::GameOver { winner } => Some(winner),
            _ => None,
        }
    }

    pub fn is_chance_node(&self) -> bool {
        matches!(self.phase, Phase::ChanceRoll)
    }
}
