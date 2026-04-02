use crate::Game;
use crate::board::{
    DevelopmentCard, Edge, NUM_EDGES, NUM_PORTS, NUM_TILES, NUM_VERTICES, Port, ResourceBank, TOPO,
    Terrain, TileId, Vertex,
};
use crate::transition::longest_road;
use crate::types::{DevCardHand, PlayerId};

// ---------------------------------------------------------------------------
// Relative player identity (perspective-independent)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlayerRelation {
    Self_,
    Clockwise1,
    Clockwise2,
    Clockwise3,
}

fn player_relation(observer: PlayerId, other: PlayerId) -> PlayerRelation {
    match (other.0 + 4 - observer.0) % 4 {
        0 => PlayerRelation::Self_,
        1 => PlayerRelation::Clockwise1,
        2 => PlayerRelation::Clockwise2,
        3 => PlayerRelation::Clockwise3,
        _ => unreachable!(),
    }
}

// ---------------------------------------------------------------------------
// Board-level observations
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BuildingType {
    Settlement,
    City,
}

#[derive(Clone, Copy, Debug)]
pub struct VertexObs {
    pub building: Option<(BuildingType, PlayerRelation)>,
}

#[derive(Clone, Copy, Debug)]
pub struct EdgeObs {
    pub road: Option<PlayerRelation>,
}

#[derive(Clone, Debug)]
pub struct TileObs {
    pub terrain: Terrain,
    pub number: u8,
    pub has_robber: bool,
    pub vertex_buildings: [VertexObs; 6],
}

#[derive(Clone, Debug)]
pub struct HarborObs {
    pub port: Port,
    pub vertices: [u8; 2],
}

// ---------------------------------------------------------------------------
// Player observations
// ---------------------------------------------------------------------------

/// Full observation of the perspective player (private info visible).
#[derive(Clone, Debug)]
pub struct SelfPlayerObs {
    pub resources: ResourceBank,
    pub dev_cards: DevCardHand,
    pub played_knights: u8,
    pub victory_points: u8,
    pub has_longest_road: bool,
    pub longest_road_length: u8,
    pub has_largest_army: bool,
    pub settlements_left: u8,
    pub cities_left: u8,
    pub roads_left: u8,
    pub has_three_to_one_port: bool,
    pub two_to_one_ports: [bool; 5],
}

/// Observation of another player (only public info).
#[derive(Clone, Debug)]
pub struct OtherPlayerObs {
    pub relation: PlayerRelation,
    pub total_resource_cards: u8,
    pub total_dev_cards: u8,
    pub played_knights: u8,
    pub public_victory_points: u8,
    pub has_longest_road: bool,
    pub has_largest_army: bool,
    pub settlements_left: u8,
    pub cities_left: u8,
    pub roads_left: u8,
}

// ---------------------------------------------------------------------------
// Game-level metadata
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct GameMetaObs {
    pub turn_number: u16,
    pub dev_cards_remaining: u8,
    pub bank: ResourceBank,
}

// ---------------------------------------------------------------------------
// Full observation
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Observation {
    pub tiles: [TileObs; NUM_TILES],
    pub vertices: [VertexObs; NUM_VERTICES],
    pub edges: [EdgeObs; NUM_EDGES],
    pub harbors: [HarborObs; NUM_PORTS],
    pub self_player: SelfPlayerObs,
    pub other_players: [OtherPlayerObs; 3],
    pub meta: GameMetaObs,
}

impl Game {
    pub fn observe(&self, perspective: PlayerId) -> Observation {
        let topo = &*TOPO;

        // Tiles
        let tiles = std::array::from_fn(|t| {
            let tile = &self.board.tiles[t];
            let vertex_buildings = std::array::from_fn(|c| {
                let v = topo.tile_vertices[t][c] as usize;
                vertex_obs(self, perspective, v)
            });
            TileObs {
                terrain: tile.terrain,
                number: tile.number,
                has_robber: self.board.robber == TileId(t as u8),
                vertex_buildings,
            }
        });

        // All vertices
        let vertices = std::array::from_fn(|v| vertex_obs(self, perspective, v));

        // All edges
        let edges = std::array::from_fn(|e| {
            let road = match self.board.edges[e] {
                Edge::Road(p) => Some(player_relation(perspective, p)),
                Edge::Empty => None,
            };
            EdgeObs { road }
        });

        // Harbors
        let harbors = std::array::from_fn(|i| HarborObs {
            port: self.board.harbors[i],
            vertices: topo.port_vertices[i],
        });

        // Self player (full info)
        let p = &self.players[perspective.idx()];
        let self_player = SelfPlayerObs {
            resources: p.resources,
            dev_cards: p.dev_cards,
            played_knights: p.played_knights,
            victory_points: self.victory_points(perspective),
            has_longest_road: self.longest_road_owner == Some(perspective),
            longest_road_length: longest_road(&self.board, perspective),
            has_largest_army: self.largest_army_owner == Some(perspective),
            settlements_left: p.settlements_left,
            cities_left: p.cities_left,
            roads_left: p.roads_left,
            has_three_to_one_port: p.has_three_to_one_port,
            two_to_one_ports: p.two_to_one_ports,
        };

        // Other players (public info only, ordered clockwise from perspective)
        let other_players = std::array::from_fn(|i| {
            let pid = PlayerId(((perspective.0 as usize + 1 + i) % 4) as u8);
            let op = &self.players[pid.idx()];
            OtherPlayerObs {
                relation: player_relation(perspective, pid),
                total_resource_cards: op.resources.total(),
                total_dev_cards: op.dev_cards.total(),
                played_knights: op.played_knights,
                public_victory_points: self
                    .victory_points(pid)
                    .saturating_sub(op.dev_cards.get(DevelopmentCard::VictoryPoint)),
                has_longest_road: self.longest_road_owner == Some(pid),
                has_largest_army: self.largest_army_owner == Some(pid),
                settlements_left: op.settlements_left,
                cities_left: op.cities_left,
                roads_left: op.roads_left,
            }
        });

        let meta = GameMetaObs {
            turn_number: self.turn_number,
            dev_cards_remaining: self.board.dev_card_deck.remaining() as u8,
            bank: self.board.bank,
        };

        Observation {
            tiles,
            vertices,
            edges,
            harbors,
            self_player,
            other_players,
            meta,
        }
    }
}

fn vertex_obs(game: &Game, perspective: PlayerId, v: usize) -> VertexObs {
    let building = match game.board.vertices[v] {
        Vertex::Settlement(p) => Some((BuildingType::Settlement, player_relation(perspective, p))),
        Vertex::City(p) => Some((BuildingType::City, player_relation(perspective, p))),
        Vertex::Empty => None,
    };
    VertexObs { building }
}
