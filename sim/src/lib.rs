pub mod action;
pub mod board;
pub mod display;
pub mod observation;
pub mod transition;

use core::fmt;

pub use action::{ACTION_SPACE_SIZE, Action, ActionMask, ActionMaskIter};
pub use board::{Board, Resource, TOPO, Terrain};
pub use observation::{OBSERVATION_LEN, Observation};
pub use transition::{ChanceTurn, InvalidAction, PlayerTurn, Turn};

use rand::Rng;

use crate::board::{DevCardHand, Port, ResourceBank};

macro_rules! id_type {
    ($name:ident) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name(pub u8);

        impl $name {
            pub fn idx(self) -> usize {
                self.0 as usize
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}{}", stringify!($name).trim_end_matches("Id"), self.0)
            }
        }
    };
}

id_type!(PlayerId);
id_type!(VertexId);
id_type!(EdgeId);
id_type!(TileId);

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
        remaining: [bool; 4],
        targets: [u8; 4],
        active: PlayerId,
    },
    MoveRobber,
    Steal {
        candidates: [bool; 4],
    },
    Main,
    RoadBuilding {
        roads_left: u8,
    },
    GameOver {
        winner: PlayerId,
    },
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TurnFlags {
    pub dev_card_played: bool,
    pub dev_cards_bought: DevCardHand,
    pub has_rolled: bool,
}

#[derive(Clone, Debug)]
pub struct Game {
    pub board: Board,
    pub players: [Player; 4],
    pub phase: Phase,
    pub turn_flags: TurnFlags,
    pub longest_road_len: [u8; 4],
    pub current_player: PlayerId,
    pub turn_number: u16,
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
            turn_flags: TurnFlags::default(),
            longest_road_len: [0; 4],
            current_player: PlayerId::P0,
            turn_number: 0,
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

    pub fn winner(&self) -> Option<PlayerId> {
        match self.phase {
            Phase::GameOver { winner } => Some(winner),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Player {
    pub id: PlayerId,
    pub resources: ResourceBank,
    pub dev_cards: DevCardHand,
    pub played_knights: u8,
    pub settlements_left: u8,
    pub cities_left: u8,
    pub roads_left: u8,
    pub has_three_to_one_port: bool,
    pub two_to_one_ports: [bool; 5],
}

impl Player {
    pub fn new(id: PlayerId) -> Self {
        Self {
            id,
            resources: ResourceBank([0; 5]),
            dev_cards: DevCardHand::EMPTY,
            played_knights: 0,
            settlements_left: 5,
            cities_left: 4,
            roads_left: 15,
            has_three_to_one_port: false,
            two_to_one_ports: [false; 5],
        }
    }

    pub fn trade_rate(&self, resource: Resource) -> u8 {
        if self.two_to_one_ports[resource as usize] {
            2
        } else if self.has_three_to_one_port {
            3
        } else {
            4
        }
    }

    pub fn update_ports(&mut self, port: Port) {
        match port {
            Port::ThreeToOne => self.has_three_to_one_port = true,
            Port::TwoToOne(r) => self.two_to_one_ports[r as usize] = true,
        }
    }
}

impl PlayerId {
    pub const P0: Self = Self(0);
    pub const P1: Self = Self(1);
    pub const P2: Self = Self(2);
    pub const P3: Self = Self(3);
    pub const ALL: [Self; 4] = [Self::P0, Self::P1, Self::P2, Self::P3];

    pub fn next(self) -> Self {
        Self((self.0 + 1) % 4)
    }

    pub fn prev(self) -> Self {
        Self((self.0 + 3) % 4)
    }
}
