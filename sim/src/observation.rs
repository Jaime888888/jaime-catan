use crate::board::{
    DevelopmentCard, Edge, NUM_EDGES, NUM_PORTS, NUM_TILES, NUM_VERTICES, Port, TOPO, Vertex,
};
use crate::{Game, PlayerId, TileId};
use std::mem::{align_of, size_of};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum PlayerRelation {
    Self_ = 0,
    Clockwise1 = 1,
    Clockwise2 = 2,
    Clockwise3 = 3,
}

fn rel(observer: PlayerId, other: PlayerId) -> PlayerRelation {
    match (other.0 + 4 - observer.0) % 4 {
        0 => PlayerRelation::Self_,
        1 => PlayerRelation::Clockwise1,
        2 => PlayerRelation::Clockwise2,
        3 => PlayerRelation::Clockwise3,
        _ => unreachable!(),
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ObsTile {
    pub terrain: u8,
    pub number: u8,
    pub has_robber: u8,
    pub adjacent_vertex_buildings: [u8; 6],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ObsHarbor {
    pub port_kind: u8,
    pub adjacent_vertex_a: u8,
    pub adjacent_vertex_b: u8,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ObsSelf {
    pub resources: [u8; 5],
    pub dev_cards: [u8; 5],
    pub played_knights: u8,
    pub victory_points: u8,
    pub has_longest_road: u8,
    pub longest_road_length: u8,
    pub has_largest_army: u8,
    pub settlements_left: u8,
    pub cities_left: u8,
    pub roads_left: u8,
    pub has_three_to_one_port: u8,
    pub two_to_one_ports: [u8; 5],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ObsOther {
    pub relation: u8,
    pub total_resource_cards: u8,
    pub total_dev_cards: u8,
    pub played_knights: u8,
    pub public_victory_points: u8,
    pub has_longest_road: u8,
    pub has_largest_army: u8,
    pub settlements_left: u8,
    pub cities_left: u8,
    pub roads_left: u8,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ObsMeta {
    pub turn_number: u16,
    pub dev_cards_remaining: u8,
    pub resource_bank: [u8; 5],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Observation {
    pub tiles: [ObsTile; NUM_TILES],
    pub vertices: [u8; NUM_VERTICES],
    pub edges: [u8; NUM_EDGES],
    pub harbors: [ObsHarbor; NUM_PORTS],
    pub self_player: ObsSelf,
    pub other_players: [ObsOther; 3],
    pub meta: ObsMeta,
}

pub const OBSERVATION_LEN: usize = size_of::<Observation>();

const _: () = assert!(OBSERVATION_LEN == 386);
const _: () = assert!(align_of::<Observation>() == 2);

impl Observation {
    pub fn as_bytes(&self) -> &[u8; OBSERVATION_LEN] {
        unsafe { &*std::ptr::from_ref(self).cast() }
    }
}

fn vertex_byte(game: &Game, perspective: PlayerId, v: usize) -> u8 {
    match game.board.vertices[v] {
        Vertex::Empty => 0,
        Vertex::Settlement(p) => 1 + rel(perspective, p) as u8,
        Vertex::City(p) => 5 + rel(perspective, p) as u8,
    }
}

fn road_byte(game: &Game, perspective: PlayerId, e: usize) -> u8 {
    match game.board.edges[e] {
        Edge::Empty => 0,
        Edge::Road(p) => 1 + rel(perspective, p) as u8,
    }
}

fn harbor_port_byte(p: Port) -> u8 {
    match p {
        Port::ThreeToOne => 0,
        Port::TwoToOne(r) => 1 + r as u8,
    }
}

impl Game {
    pub fn observe(&self, perspective: PlayerId) -> Observation {
        let topo = &*TOPO;

        Observation {
            tiles: std::array::from_fn(|t| {
                let tile = &self.board.tiles[t];
                ObsTile {
                    terrain: tile.terrain as u8,
                    number: tile.number,
                    has_robber: (self.board.robber == TileId(t as u8)) as u8,
                    adjacent_vertex_buildings: std::array::from_fn(|c| {
                        vertex_byte(self, perspective, topo.tile_vertices[t][c] as usize)
                    }),
                }
            }),
            vertices: std::array::from_fn(|v| vertex_byte(self, perspective, v)),
            edges: std::array::from_fn(|e| road_byte(self, perspective, e)),
            harbors: std::array::from_fn(|i| ObsHarbor {
                port_kind: harbor_port_byte(self.board.harbors[i]),
                adjacent_vertex_a: topo.port_vertices[i][0],
                adjacent_vertex_b: topo.port_vertices[i][1],
            }),
            self_player: {
                let p = &self.players[perspective.idx()];
                ObsSelf {
                    resources: p.resources.0,
                    dev_cards: p.dev_cards.0,
                    played_knights: p.played_knights,
                    victory_points: self.victory_points(perspective),
                    has_longest_road: (self.longest_road_owner() == Some(perspective)) as u8,
                    longest_road_length: self.longest_road_len[perspective.idx()],
                    has_largest_army: (self.largest_army_owner() == Some(perspective)) as u8,
                    settlements_left: p.settlements_left,
                    cities_left: p.cities_left,
                    roads_left: p.roads_left,
                    has_three_to_one_port: p.has_three_to_one_port as u8,
                    two_to_one_ports: std::array::from_fn(|i| p.two_to_one_ports[i] as u8),
                }
            },
            other_players: std::array::from_fn(|i| {
                let pid = PlayerId(((perspective.0 as usize + 1 + i) % 4) as u8);
                let op = &self.players[pid.idx()];
                ObsOther {
                    relation: rel(perspective, pid) as u8,
                    total_resource_cards: op.resources.total(),
                    total_dev_cards: op.dev_cards.total(),
                    played_knights: op.played_knights,
                    public_victory_points: self
                        .victory_points(pid)
                        .saturating_sub(op.dev_cards[DevelopmentCard::VictoryPoint]),
                    has_longest_road: (self.longest_road_owner() == Some(pid)) as u8,
                    has_largest_army: (self.largest_army_owner() == Some(pid)) as u8,
                    settlements_left: op.settlements_left,
                    cities_left: op.cities_left,
                    roads_left: op.roads_left,
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
