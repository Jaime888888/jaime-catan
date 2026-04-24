use crate::board::{Edge, NUM_EDGES, NUM_PORTS, NUM_TILES, NUM_VERTICES, Port, Terrain, Vertex};
use crate::{Game, PlayerId, TileId};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlayerRelation {
    Self_ = 0,
    Clockwise1 = 1,
    Clockwise2 = 2,
    Clockwise3 = 3,
}

fn player_relative(observer: PlayerId, other: PlayerId) -> PlayerRelation {
    match (other.0 + 4 - observer.0) % 4 {
        0 => PlayerRelation::Self_,
        1 => PlayerRelation::Clockwise1,
        2 => PlayerRelation::Clockwise2,
        3 => PlayerRelation::Clockwise3,
        _ => unreachable!(),
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ObsVertex {
    Empty,
    Settlement(PlayerRelation),
    City(PlayerRelation),
}

#[derive(Clone, Copy, Debug)]
pub enum ObsEdge {
    Empty,
    Road(PlayerRelation),
}

#[derive(Clone, Copy, Debug)]
pub struct ObsTile {
    pub terrain: Terrain,
    pub number: u8,
    pub has_robber: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct ObsSelf {
    pub resources: [u8; 5],
    pub dev_cards: [u8; 5],
    pub played_knights: u8,
    pub has_longest_road: bool,
    pub longest_road_length: u8,
    pub has_largest_army: bool,
    pub has_three_to_one_port: bool,
    pub two_to_one_ports: [bool; 5],
}

#[derive(Clone, Copy, Debug)]
pub struct ObsOther {
    pub total_resource_cards: u8,
    pub total_dev_cards: u8,
    pub played_knights: u8,
    pub has_longest_road: bool,
    pub has_largest_army: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct ObsMeta {
    pub turn_number: u16,
    pub dev_cards_remaining: u8,
    pub resource_bank: [u8; 5],
}

#[derive(Clone, Debug)]
pub struct Observation {
    pub tiles: [ObsTile; NUM_TILES],
    pub vertices: [ObsVertex; NUM_VERTICES],
    pub edges: [ObsEdge; NUM_EDGES],
    pub harbors: [Port; NUM_PORTS],
    pub self_player: ObsSelf,
    pub other_players: [ObsOther; 3],
    pub meta: ObsMeta,
}

fn observe_vertex(game: &Game, perspective: PlayerId, v: usize) -> ObsVertex {
    match game.board.vertices[v] {
        Vertex::Empty => ObsVertex::Empty,
        Vertex::Settlement(p) => ObsVertex::Settlement(player_relative(perspective, p)),
        Vertex::City(p) => ObsVertex::City(player_relative(perspective, p)),
    }
}

impl Game {
    pub fn observe(&self, perspective: PlayerId) -> Observation {
        Observation {
            tiles: std::array::from_fn(|t| {
                let tile = &self.board.tiles[t];
                ObsTile {
                    terrain: tile.terrain,
                    number: tile.number,
                    has_robber: self.board.robber == TileId(t as u8),
                }
            }),
            vertices: std::array::from_fn(|v| observe_vertex(self, perspective, v)),
            edges: std::array::from_fn(|e| match self.board.edges[e] {
                Edge::Empty => ObsEdge::Empty,
                Edge::Road(p) => ObsEdge::Road(player_relative(perspective, p)),
            }),
            harbors: self.board.harbors,
            self_player: {
                let p = &self.players[perspective.idx()];
                ObsSelf {
                    resources: p.resources.0,
                    dev_cards: p.dev_cards.0,
                    played_knights: p.played_knights,
                    has_longest_road: self.longest_road_owner() == Some(perspective),
                    longest_road_length: self.longest_road_len[perspective.idx()],
                    has_largest_army: self.largest_army_owner() == Some(perspective),
                    has_three_to_one_port: p.has_three_to_one_port,
                    two_to_one_ports: p.two_to_one_ports,
                }
            },
            other_players: std::array::from_fn(|i| {
                let pid = PlayerId(((perspective.0 as usize + 1 + i) % 4) as u8);
                let op = &self.players[pid.idx()];
                ObsOther {
                    total_resource_cards: op.resources.total(),
                    total_dev_cards: op.dev_cards.total(),
                    played_knights: op.played_knights,
                    has_longest_road: self.longest_road_owner() == Some(pid),
                    has_largest_army: self.largest_army_owner() == Some(pid),
                }
            }),
            meta: ObsMeta {
                turn_number: self.turn_number,
                dev_cards_remaining: self.board.dev_card_deck.remaining(),
                resource_bank: self.board.bank.0,
            },
        }
    }
}
