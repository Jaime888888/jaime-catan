use core::fmt;
use std::{
    array,
    cmp::Ordering,
    ops::{Add, Index, IndexMut},
    sync::LazyLock,
};

use rand::{Rng, seq::SliceRandom};

use crate::PlayerId;

pub const NUM_TILES: usize = 19;
pub const NUM_VERTICES: usize = 54;
pub const NUM_EDGES: usize = 72;
pub const NUM_PORTS: usize = 9;

pub struct Topology {
    pub tile_vertices: [[u8; 6]; NUM_TILES],
    pub edge_vertices: [[u8; 2]; NUM_EDGES],

    vertex_tiles: [[u8; 3]; NUM_VERTICES],
    vertex_tile_count: [u8; NUM_VERTICES],
    vertex_adjacent: [[u8; 3]; NUM_VERTICES],
    vertex_adj_count: [u8; NUM_VERTICES],
    vertex_edges: [[u8; 3]; NUM_VERTICES],
    vertex_edge_count: [u8; NUM_VERTICES],

    /// The 9 harbor positions on the frame, each a pair of vertex IDs.
    pub port_vertices: [[u8; 2]; NUM_PORTS],
    /// For each vertex, the harbor slot (0..8) it belongs to, or `None`.
    vertex_port: [Option<u8>; NUM_VERTICES],
}

impl Topology {
    pub fn vertex_tiles(&self, v: usize) -> &[u8] {
        &self.vertex_tiles[v][..self.vertex_tile_count[v] as usize]
    }

    pub fn vertex_neighbors(&self, v: usize) -> &[u8] {
        &self.vertex_adjacent[v][..self.vertex_adj_count[v] as usize]
    }

    pub fn vertex_edge_list(&self, v: usize) -> &[u8] {
        &self.vertex_edges[v][..self.vertex_edge_count[v] as usize]
    }

    pub fn port_type(&self, vertex: usize, harbors: &[Port; NUM_PORTS]) -> Option<Port> {
        self.vertex_port[vertex].map(|slot| harbors[slot as usize])
    }
}

impl Vertex {
    pub fn owner(self) -> Option<PlayerId> {
        match self {
            Vertex::Empty => None,
            Vertex::Settlement(p) | Vertex::City(p) => Some(p),
        }
    }

    pub fn is_empty(self) -> bool {
        matches!(self, Vertex::Empty)
    }
}

impl Edge {
    pub fn owner(self) -> Option<PlayerId> {
        match self {
            Edge::Empty => None,
            Edge::Road(p) => Some(p),
        }
    }

    pub fn is_empty(self) -> bool {
        matches!(self, Edge::Empty)
    }
}

impl Board {
    pub fn port_at(&self, v: usize) -> Option<Port> {
        TOPO.port_type(v, &self.harbors)
    }

    pub fn vertex_placement_valid(&self, v: usize) -> bool {
        if !self.vertices[v].is_empty() {
            return false;
        }
        for &adj in TOPO.vertex_neighbors(v) {
            if !self.vertices[adj as usize].is_empty() {
                return false;
            }
        }
        true
    }

    pub fn vertex_has_friendly_road(&self, v: usize, player: PlayerId) -> bool {
        for &e in TOPO.vertex_edge_list(v) {
            if self.edges[e as usize].owner() == Some(player) {
                return true;
            }
        }
        false
    }
}

pub static TOPO: LazyLock<Topology> = LazyLock::new(|| {
    /// Axial coordinates for the 19 hex tiles (pointy-top orientation).
    ///
    /// Layout (row by row, top to bottom):
    ///   Row r=-2: 3 hexes   (tiles 0-2)
    ///   Row r=-1: 4 hexes   (tiles 3-6)
    ///   Row r= 0: 5 hexes   (tiles 7-11)
    ///   Row r= 1: 4 hexes   (tiles 12-15)
    ///   Row r= 2: 3 hexes   (tiles 16-18)
    #[rustfmt::skip]
    const HEX_AXIAL: [(i8, i8); NUM_TILES] = [
                (0, -2), (1, -2), (2, -2),
            (-1, -1), (0, -1), (1, -1), (2, -1),
        (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0),
            (-2, 1), (-1, 1), (0, 1), (1, 1),
                (-2, 2), (-1, 2), (0, 2),
    ];

    /// For pointy-top hexes, vertex `k` of hex at axial (q, r) has a canonical key:
    ///   x_key = 2q + r + DX[k]
    ///   y_key = 3r     + DY[k]
    /// Two hex-corner pairs with identical keys are the same geometric vertex.
    const VERTEX_DX: [i8; 6] = [0, 1, 1, 0, -1, -1];
    const VERTEX_DY: [i8; 6] = [-2, -1, 1, 2, 1, -1];

    // ----- scratch buffers -----
    let mut vkeys: [(i8, i8); 120] = [(0, 0); 120];
    let mut num_v: usize = 0;

    let mut elist: [[u8; 2]; 150] = [[0; 2]; 150];
    let mut num_e: usize = 0;

    // ----- Step 1: vertices per tile -----
    let mut tile_vertices = [[0u8; 6]; NUM_TILES];
    for t in 0..NUM_TILES {
        let (q, r) = HEX_AXIAL[t];
        for c in 0..6 {
            let kx = 2 * q + r + VERTEX_DX[c];
            let ky = 3 * r + VERTEX_DY[c];
            let vid = find_or_insert(&mut vkeys, &mut num_v, (kx, ky));
            tile_vertices[t][c] = vid;
        }
    }
    assert_eq!(num_v, NUM_VERTICES);

    // ----- Step 2: edges per tile -----
    let mut tile_edges = [[0u8; 6]; NUM_TILES];
    for t in 0..NUM_TILES {
        for e in 0..6 {
            let v1 = tile_vertices[t][e];
            let v2 = tile_vertices[t][(e + 1) % 6];
            let (a, b) = if v1 < v2 { (v1, v2) } else { (v2, v1) };
            let eid = find_or_insert_pair(&mut elist, &mut num_e, [a, b]);
            tile_edges[t][e] = eid;
        }
    }
    assert_eq!(num_e, NUM_EDGES);

    // ----- Step 3: edge_vertices -----
    let mut edge_vertices = [[0u8; 2]; NUM_EDGES];
    for i in 0..NUM_EDGES {
        edge_vertices[i] = elist[i];
    }

    // ----- Step 4: vertex → tiles -----
    let mut vertex_tiles = [[0u8; 3]; NUM_VERTICES];
    let mut vertex_tile_count = [0u8; NUM_VERTICES];
    for t in 0..NUM_TILES {
        for c in 0..6 {
            let v = tile_vertices[t][c] as usize;
            let idx = vertex_tile_count[v] as usize;
            vertex_tiles[v][idx] = t as u8;
            vertex_tile_count[v] += 1;
        }
    }

    // ----- Step 5: edge → tiles (needed to identify coastal edges) -----
    let mut edge_tile_count = [0u8; NUM_EDGES];
    for t in 0..NUM_TILES {
        for e in 0..6 {
            let eid = tile_edges[t][e] as usize;
            edge_tile_count[eid] += 1;
        }
    }

    // ----- Step 6: vertex → edges -----
    let mut vertex_edges = [[0u8; 3]; NUM_VERTICES];
    let mut vertex_edge_count = [0u8; NUM_VERTICES];
    for e in 0..NUM_EDGES {
        for &v in &edge_vertices[e] {
            let vi = v as usize;
            let idx = vertex_edge_count[vi] as usize;
            vertex_edges[vi][idx] = e as u8;
            vertex_edge_count[vi] += 1;
        }
    }

    // ----- Step 7: vertex → adjacent vertices -----
    let mut vertex_adjacent = [[0u8; 3]; NUM_VERTICES];
    let mut vertex_adj_count = [0u8; NUM_VERTICES];
    for e in 0..NUM_EDGES {
        let v1 = edge_vertices[e][0] as usize;
        let v2 = edge_vertices[e][1] as usize;
        let i1 = vertex_adj_count[v1] as usize;
        vertex_adjacent[v1][i1] = v2 as u8;
        vertex_adj_count[v1] += 1;
        let i2 = vertex_adj_count[v2] as usize;
        vertex_adjacent[v2][i2] = v1 as u8;
        vertex_adj_count[v2] += 1;
    }

    // ----- Step 8: harbor positions -----
    // Sort coastal edges clockwise, then pick 9 fixed frame positions.
    let mut ce_with_angle: Vec<(u8, f64)> = Vec::new();
    for e in 0..NUM_EDGES {
        if edge_tile_count[e] == 1 {
            let v1 = edge_vertices[e][0] as usize;
            let v2 = edge_vertices[e][1] as usize;
            let mx = (vkeys[v1].0 as f64 + vkeys[v2].0 as f64) * 1.732_050_808;
            let my = vkeys[v1].1 as f64 + vkeys[v2].1 as f64;
            ce_with_angle.push((e as u8, f64::atan2(mx, -my)));
        }
    }
    ce_with_angle.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    assert_eq!(ce_with_angle.len(), 30);

    const PORT_COASTAL_INDICES: [usize; NUM_PORTS] = [0, 3, 7, 10, 14, 17, 20, 23, 27];
    let mut port_vertices = [[0u8; 2]; NUM_PORTS];
    let mut vertex_port = [None; NUM_VERTICES];
    for (slot, &cei) in PORT_COASTAL_INDICES.iter().enumerate() {
        let eid = ce_with_angle[cei].0 as usize;
        let verts = edge_vertices[eid];
        port_vertices[slot] = verts;
        vertex_port[verts[0] as usize] = Some(slot as u8);
        vertex_port[verts[1] as usize] = Some(slot as u8);
    }

    Topology {
        tile_vertices,
        edge_vertices,
        vertex_tiles,
        vertex_tile_count,
        vertex_adjacent,
        vertex_adj_count,
        vertex_edges,
        vertex_edge_count,
        port_vertices,
        vertex_port,
    }
});

fn find_or_insert(buf: &mut [(i8, i8); 120], count: &mut usize, key: (i8, i8)) -> u8 {
    for i in 0..*count {
        if buf[i] == key {
            return i as u8;
        }
    }
    let id = *count as u8;
    buf[*count] = key;
    *count += 1;
    id
}

fn find_or_insert_pair(buf: &mut [[u8; 2]; 150], count: &mut usize, pair: [u8; 2]) -> u8 {
    for i in 0..*count {
        if buf[i] == pair {
            return i as u8;
        }
    }
    let id = *count as u8;
    buf[*count] = pair;
    *count += 1;
    id
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TileId(pub u8);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Terrain {
    Desert,
    Hills,
    Forest,
    Mountains,
    Fields,
    Pasture,
}

impl Terrain {
    pub fn resource(self) -> Option<Resource> {
        match self {
            Terrain::Desert => None,
            Terrain::Hills => Some(Resource::Brick),
            Terrain::Forest => Some(Resource::Lumber),
            Terrain::Mountains => Some(Resource::Ore),
            Terrain::Fields => Some(Resource::Grain),
            Terrain::Pasture => Some(Resource::Wool),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tile {
    pub terrain: Terrain,
    pub number: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum Resource {
    Brick = 0,
    Lumber = 1,
    Ore = 2,
    Grain = 3,
    Wool = 4,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Port {
    ThreeToOne,
    TwoToOne(Resource),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Vertex {
    Empty,
    Settlement(PlayerId),
    City(PlayerId),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Edge {
    Empty,
    Road(PlayerId),
}

#[rustfmt::skip]
const STANDARD_NON_DESERT_TERRAINS: [Terrain; NUM_TILES - 1] = [
    Terrain::Hills, Terrain::Hills, Terrain::Hills,
    Terrain::Mountains, Terrain::Mountains, Terrain::Mountains,
    Terrain::Forest, Terrain::Forest, Terrain::Forest, Terrain::Forest,
    Terrain::Fields, Terrain::Fields, Terrain::Fields, Terrain::Fields,
    Terrain::Pasture, Terrain::Pasture, Terrain::Pasture, Terrain::Pasture,
];

const STANDARD_NON_ZERO_NUMBERS: [u8; NUM_TILES - 1] =
    [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12];

#[rustfmt::skip]
const STANDARD_PORTS: [Port; NUM_PORTS] = [
    Port::ThreeToOne, Port::ThreeToOne, Port::ThreeToOne, Port::ThreeToOne,
    Port::TwoToOne(Resource::Brick),
    Port::TwoToOne(Resource::Lumber),
    Port::TwoToOne(Resource::Ore),
    Port::TwoToOne(Resource::Grain),
    Port::TwoToOne(Resource::Wool),
];

#[derive(Clone, Debug)]
pub struct Board {
    pub tiles: [Tile; NUM_TILES],
    pub vertices: [Vertex; NUM_VERTICES],
    pub edges: [Edge; NUM_EDGES],
    pub robber: TileId,
    /// The 9 harbor tokens, shuffled each game. Index 0..8 corresponds to
    /// the harbor slot defined by `TOPO.port_vertices[i]`.
    pub harbors: [Port; NUM_PORTS],
    pub bank: ResourceBank,
    pub dev_card_deck: DevCardDeck,
}

impl Board {
    pub fn generate(rng: &mut impl Rng) -> Self {
        let mut numbers = STANDARD_NON_ZERO_NUMBERS;
        numbers.shuffle(rng);

        let mut tiles: [Tile; NUM_TILES] = array::from_fn({
            move |i| {
                if i == 0 {
                    Tile {
                        terrain: Terrain::Desert,
                        number: 0,
                    }
                } else {
                    let terrain = STANDARD_NON_DESERT_TERRAINS[i - 1];
                    let number = numbers[i - 1];
                    Tile { terrain, number }
                }
            }
        });
        tiles.shuffle(rng);

        let robber = TileId(
            tiles
                .iter()
                .position(|t| t.terrain == Terrain::Desert)
                .unwrap() as u8,
        );

        let mut harbors = STANDARD_PORTS;
        harbors.shuffle(rng);

        Board {
            tiles,
            vertices: [Vertex::Empty; NUM_VERTICES],
            edges: [Edge::Empty; NUM_EDGES],
            robber,
            harbors,
            bank: STANDARD_BANK,
            dev_card_deck: DevCardDeck::generate(rng),
        }
    }
}

const STANDARD_BANK: ResourceBank = ResourceBank([19; 5]);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct ResourceBank(pub [u8; 5]);

impl ResourceBank {
    pub fn single(r: Resource, n: u8) -> Self {
        let mut c = Self([0; 5]);
        c.0[r as usize] = n;
        c
    }

    pub fn total(self) -> u8 {
        self.0.iter().copied().sum()
    }

    pub fn can_afford(self, cost: ResourceBank) -> bool {
        (0..5).all(|i| self.0[i] >= cost.0[i])
    }

    pub fn checked_sub(mut self, other: ResourceBank) -> Option<Self> {
        if self < other {
            return None;
        }

        for i in 0..5 {
            self.0[i] -= other.0[i];
        }

        Some(self)
    }
}

impl Add<ResourceBank> for ResourceBank {
    type Output = ResourceBank;

    fn add(mut self, other: ResourceBank) -> Self::Output {
        for i in 0..5 {
            self.0[i] += other.0[i];
        }

        self
    }
}

impl PartialOrd for ResourceBank {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let mut ordering = Ordering::Equal;

        for i in 0..5 {
            if self.0[i] < other.0[i] {
                return Some(Ordering::Less);
            } else if self.0[i] > other.0[i] {
                ordering = Ordering::Greater;
            }
        }

        Some(ordering)
    }
}

impl Index<Resource> for ResourceBank {
    type Output = u8;
    fn index(&self, r: Resource) -> &u8 {
        &self.0[r as usize]
    }
}

impl IndexMut<Resource> for ResourceBank {
    fn index_mut(&mut self, r: Resource) -> &mut u8 {
        &mut self.0[r as usize]
    }
}

impl fmt::Display for ResourceBank {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[B:{} L:{} O:{} G:{} W:{}]",
            self.0[0], self.0[1], self.0[2], self.0[3], self.0[4]
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DevelopmentCard {
    Knight = 0,
    VictoryPoint = 1,
    RoadBuilding = 2,
    YearOfPlenty = 3,
    Monopoly = 4,
}

pub const NUM_DEV_CARDS: usize = 25;

#[derive(Clone, Debug)]
pub struct DevCardDeck([DevelopmentCard; NUM_DEV_CARDS], usize);

impl DevCardDeck {
    pub fn generate(rng: &mut impl Rng) -> Self {
        let mut deck = [DevelopmentCard::Knight; NUM_DEV_CARDS];
        deck.shuffle(rng);
        Self(deck, 0)
    }

    pub fn remaining(&self) -> usize {
        NUM_DEV_CARDS - self.1
    }

    pub fn draw(&mut self) -> Option<DevelopmentCard> {
        let DevCardDeck(cards, i) = self;

        if *i >= NUM_DEV_CARDS {
            return None;
        }

        let card = cards[*i];
        *i += 1;

        Some(card)
    }
}
