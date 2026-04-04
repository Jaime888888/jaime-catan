use std::{
    array,
    cmp::Ordering,
    collections::HashMap,
    ops::{Add, AddAssign, Index, IndexMut},
    sync::LazyLock,
};

use rand::{Rng, seq::SliceRandom};

use crate::{PlayerId, TileId};

pub const NUM_TILES: usize = 19;
pub const NUM_VERTICES: usize = 54;
pub const NUM_EDGES: usize = 72;
pub const NUM_PORTS: usize = 9;

#[derive(Clone, Copy)]
struct InlineVec<const N: usize> {
    data: [u8; N],
    len: u8,
}

impl<const N: usize> InlineVec<N> {
    const EMPTY: Self = Self {
        data: [0; N],
        len: 0,
    };

    fn push(&mut self, val: u8) {
        self.data[self.len as usize] = val;
        self.len += 1;
    }

    fn as_slice(&self) -> &[u8] {
        &self.data[..self.len as usize]
    }
}

pub struct Topology {
    pub tile_vertices: [[u8; 6]; NUM_TILES],
    pub edge_vertices: [[u8; 2]; NUM_EDGES],
    pub port_vertices: [[u8; 2]; NUM_PORTS],
    /// Canonical layout-plane coordinates `(vx, vy)` per vertex id (same keys as in `LazyLock` builder).
    pub vertex_pos: [(i8, i8); NUM_VERTICES],

    vertex_tiles: [InlineVec<3>; NUM_VERTICES],
    vertex_adjacent: [InlineVec<3>; NUM_VERTICES],
    vertex_edges: [InlineVec<3>; NUM_VERTICES],
    vertex_port: [Option<u8>; NUM_VERTICES],
}

impl Topology {
    pub fn vertex_tiles(&self, v: usize) -> &[u8] {
        self.vertex_tiles[v].as_slice()
    }

    pub fn vertex_neighbors(&self, v: usize) -> &[u8] {
        self.vertex_adjacent[v].as_slice()
    }

    pub fn vertex_edge_list(&self, v: usize) -> &[u8] {
        self.vertex_edges[v].as_slice()
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

pub static TOPO: LazyLock<Topology> = LazyLock::new(|| {
    /// Axial coordinates for the 19 hex tiles (pointy-top orientation).
    ///   Row r=-2: tiles 0-2,  r=-1: 3-6,  r=0: 7-11,  r=1: 12-15,  r=2: 16-18
    #[rustfmt::skip]
    const HEX_AXIAL: [(i8, i8); NUM_TILES] = [
                (0, -2), (1, -2), (2, -2),
            (-1, -1), (0, -1), (1, -1), (2, -1),
        (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0),
            (-2, 1), (-1, 1), (0, 1), (1, 1),
                (-2, 2), (-1, 2), (0, 2),
    ];

    /// For pointy-top hexes, vertex `k` of hex at axial (q, r) has canonical key:
    ///   (2q + r + DX[k],  3r + DY[k])
    /// Identical keys = same geometric vertex.
    const DX: [i8; 6] = [0, 1, 1, 0, -1, -1];
    const DY: [i8; 6] = [-2, -1, 1, 2, 1, -1];

    // Step 1: Deduplicate vertices via HashMap, build tile_vertices.
    let mut vertex_ids = HashMap::with_capacity(NUM_VERTICES);
    let tile_vertices: [[u8; 6]; NUM_TILES] = array::from_fn(|t| {
        let (q, r) = HEX_AXIAL[t];

        array::from_fn(|c| {
            let key = (2 * q + r + DX[c], 3 * r + DY[c]);
            let next = vertex_ids.len() as u8;
            *vertex_ids.entry(key).or_insert(next)
        })
    });
    assert_eq!(vertex_ids.len(), NUM_VERTICES);

    // Vertex positions needed later for harbor angle computation.
    let mut vertex_pos = [(0i8, 0i8); NUM_VERTICES];
    for (&key, &id) in &vertex_ids {
        vertex_pos[id as usize] = key;
    }

    // Step 2: Deduplicate edges, count tiles per edge (for coastal detection).
    let mut edge_ids = HashMap::with_capacity(NUM_EDGES);
    let mut edge_tile_count = [0u8; NUM_EDGES];
    for tv in &tile_vertices {
        for c in 0..6 {
            let (a, b) = (tv[c].min(tv[(c + 1) % 6]), tv[c].max(tv[(c + 1) % 6]));
            let next = edge_ids.len() as u8;
            let eid = *edge_ids.entry([a, b]).or_insert(next);
            edge_tile_count[eid as usize] += 1;
        }
    }
    assert_eq!(edge_ids.len(), NUM_EDGES);

    let mut edge_vertices = [[0u8; 2]; NUM_EDGES];
    for (&pair, &id) in &edge_ids {
        edge_vertices[id as usize] = pair;
    }

    // Step 3: Build all per-vertex lookups.
    let mut vertex_tiles = [InlineVec::<3>::EMPTY; NUM_VERTICES];
    for (t, tv) in tile_vertices.iter().enumerate() {
        for &v in tv {
            vertex_tiles[v as usize].push(t as u8);
        }
    }

    let mut vertex_edges = [InlineVec::<3>::EMPTY; NUM_VERTICES];
    let mut vertex_adjacent = [InlineVec::<3>::EMPTY; NUM_VERTICES];
    for (e, &[v1, v2]) in edge_vertices.iter().enumerate() {
        for (a, b) in [(v1, v2), (v2, v1)] {
            vertex_edges[a as usize].push(e as u8);
            vertex_adjacent[a as usize].push(b);
        }
    }

    // Step 4: Harbor positions — sort coastal edges clockwise, pick 9.
    let mut coastal = edge_vertices
        .iter()
        .enumerate()
        .filter(|&(e, _)| edge_tile_count[e] == 1)
        .map(|(e, &[v1, v2])| {
            let mx = (vertex_pos[v1 as usize].0 + vertex_pos[v2 as usize].0) as f64 * 1.732_050_808;
            let my = (vertex_pos[v1 as usize].1 + vertex_pos[v2 as usize].1) as f64;
            (e as u8, f64::atan2(mx, -my))
        })
        .collect::<Vec<_>>();
    coastal.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    assert_eq!(coastal.len(), 30);

    const PORT_PICK: [usize; NUM_PORTS] = [0, 3, 7, 10, 14, 17, 20, 23, 27];
    let mut port_vertices = [[0u8; 2]; NUM_PORTS];
    let mut vertex_port = [None; NUM_VERTICES];
    for (slot, &ci) in PORT_PICK.iter().enumerate() {
        let verts = edge_vertices[coastal[ci].0 as usize];
        port_vertices[slot] = verts;
        vertex_port[verts[0] as usize] = Some(slot as u8);
        vertex_port[verts[1] as usize] = Some(slot as u8);
    }

    Topology {
        tile_vertices,
        edge_vertices,
        port_vertices,
        vertex_pos,
        vertex_tiles,
        vertex_adjacent,
        vertex_edges,
        vertex_port,
    }
});

impl Board {
    pub fn port_at(&self, v: usize) -> Option<Port> {
        TOPO.port_type(v, &self.harbors)
    }

    pub fn vertex_placement_valid(&self, v: usize) -> bool {
        if !self.vertices[v].is_empty() {
            return false;
        }
        TOPO.vertex_neighbors(v)
            .iter()
            .all(|&adj| self.vertices[adj as usize].is_empty())
    }

    pub fn vertex_has_friendly_road(&self, v: usize, player: PlayerId) -> bool {
        TOPO.vertex_edge_list(v)
            .iter()
            .any(|&e| self.edges[e as usize].owner() == Some(player))
    }

    /// Whether `player` can extend a road to vertex `v`:
    /// they own a building there, or it's empty and they have a road to it.
    pub fn vertex_road_accessible(&self, v: usize, player: PlayerId) -> bool {
        match self.vertices[v].owner() {
            Some(owner) => owner == player,
            None => self.vertex_has_friendly_road(v, player),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Terrain {
    Desert = 0,
    Hills = 1,
    Forest = 2,
    Mountains = 3,
    Fields = 4,
    Pasture = 5,
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

impl Resource {
    pub const ALL: [Self; 5] = [
        Self::Brick,
        Self::Lumber,
        Self::Ore,
        Self::Grain,
        Self::Wool,
    ];
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

        let mut dev_card_deck = STANDARD_DEV_CARD_DECK;
        dev_card_deck.shuffle(rng);

        Board {
            tiles,
            vertices: [Vertex::Empty; NUM_VERTICES],
            edges: [Edge::Empty; NUM_EDGES],
            robber,
            harbors,
            bank: STANDARD_BANK,
            dev_card_deck: DevCardDeck(dev_card_deck, NUM_DEV_CARDS),
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DevelopmentCard {
    Knight = 0,
    VictoryPoint = 1,
    RoadBuilding = 2,
    YearOfPlenty = 3,
    Monopoly = 4,
}

const NUM_DEV_CARDS: usize = 25;

#[rustfmt::skip]
const STANDARD_DEV_CARD_DECK: [DevelopmentCard; NUM_DEV_CARDS] = [
    DevelopmentCard::Knight, DevelopmentCard::Knight, DevelopmentCard::Knight, 
    DevelopmentCard::Knight, DevelopmentCard::Knight, DevelopmentCard::Knight, 
    DevelopmentCard::Knight, DevelopmentCard::Knight, DevelopmentCard::Knight, 
    DevelopmentCard::Knight, DevelopmentCard::Knight, DevelopmentCard::Knight, 
    DevelopmentCard::Knight, DevelopmentCard::Knight, 
    DevelopmentCard::VictoryPoint, DevelopmentCard::VictoryPoint, DevelopmentCard::VictoryPoint,
    DevelopmentCard::VictoryPoint, DevelopmentCard::VictoryPoint, 
    DevelopmentCard::RoadBuilding, DevelopmentCard::RoadBuilding,
    DevelopmentCard::YearOfPlenty, DevelopmentCard::YearOfPlenty,
    DevelopmentCard::Monopoly, DevelopmentCard::Monopoly,
];

#[derive(Clone, Debug)]
pub struct DevCardDeck([DevelopmentCard; NUM_DEV_CARDS], usize);

impl DevCardDeck {
    pub fn draw(&mut self) -> Option<DevelopmentCard> {
        if self.1 == 0 {
            return None;
        }
        self.1 -= 1;
        Some(self.0[self.1])
    }

    pub fn remaining(&self) -> u8 {
        self.1 as u8
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DevCardHand(pub [u8; 5]);

impl DevCardHand {
    pub const EMPTY: Self = Self([0; 5]);

    pub fn checked_sub_assign(&mut self, c: DevelopmentCard) -> bool {
        let slot = &mut self.0[c as usize];

        if *slot > 0 {
            *slot -= 1;
            true
        } else {
            false
        }
    }

    pub fn total(&self) -> u8 {
        self.0.iter().sum()
    }

    pub fn has(&self, c: DevelopmentCard) -> bool {
        self.0[c as usize] > 0
    }
}

impl AddAssign<DevelopmentCard> for DevCardHand {
    fn add_assign(&mut self, c: DevelopmentCard) {
        self.0[c as usize] += 1;
    }
}

impl Index<DevelopmentCard> for DevCardHand {
    type Output = u8;
    fn index(&self, c: DevelopmentCard) -> &u8 {
        &self.0[c as usize]
    }
}

impl IndexMut<DevelopmentCard> for DevCardHand {
    fn index_mut(&mut self, c: DevelopmentCard) -> &mut u8 {
        &mut self.0[c as usize]
    }
}
