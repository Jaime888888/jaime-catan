use std::fmt;
use std::ops::{Index, IndexMut};

pub use crate::board::{DevelopmentCard, Port, Resource, ResourceBank, Terrain, TileId};

// ---------------------------------------------------------------------------
// PlayerId
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PlayerId(pub u8);

impl PlayerId {
    pub const P0: Self = Self(0);
    pub const P1: Self = Self(1);
    pub const P2: Self = Self(2);
    pub const P3: Self = Self(3);
    pub const ALL: [Self; 4] = [Self::P0, Self::P1, Self::P2, Self::P3];

    pub fn idx(self) -> usize {
        self.0 as usize
    }

    pub fn next(self) -> Self {
        Self((self.0 + 1) % 4)
    }

    pub fn prev(self) -> Self {
        Self((self.0 + 3) % 4)
    }
}

impl fmt::Display for PlayerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "P{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// VertexId / EdgeId
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VertexId(pub u8);

impl VertexId {
    pub fn idx(self) -> usize {
        self.0 as usize
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EdgeId(pub u8);

impl EdgeId {
    pub fn idx(self) -> usize {
        self.0 as usize
    }
}

// ---------------------------------------------------------------------------
// PlayerMask – compact bitmask over 4 players
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct PlayerMask(pub u8);

impl PlayerMask {
    pub const NONE: Self = Self(0);

    pub fn contains(self, player: PlayerId) -> bool {
        self.0 & (1 << player.0) != 0
    }

    pub fn insert(&mut self, player: PlayerId) {
        self.0 |= 1 << player.0;
    }

    pub fn remove(&mut self, player: PlayerId) {
        self.0 &= !(1 << player.0);
    }

    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub fn first(self) -> Option<PlayerId> {
        if self.0 == 0 {
            None
        } else {
            Some(PlayerId(self.0.trailing_zeros() as u8))
        }
    }

    pub fn count(self) -> u32 {
        self.0.count_ones()
    }

    pub fn iter(self) -> impl Iterator<Item = PlayerId> {
        (0u8..4)
            .filter(move |&i| self.0 & (1 << i) != 0)
            .map(PlayerId)
    }
}

// ---------------------------------------------------------------------------
// DevCardHand – counts of each dev card type in a player's hand
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct DevCardHand(pub [u8; 5]);

impl DevCardHand {
    pub const EMPTY: Self = Self([0; 5]);

    pub fn get(self, c: DevelopmentCard) -> u8 {
        self.0[c as usize]
    }

    pub fn add(&mut self, c: DevelopmentCard) {
        self.0[c as usize] += 1;
    }

    pub fn remove(&mut self, c: DevelopmentCard) -> bool {
        let slot = &mut self.0[c as usize];
        if *slot > 0 {
            *slot -= 1;
            true
        } else {
            false
        }
    }

    pub fn total(self) -> u8 {
        self.0.iter().copied().sum()
    }

    pub fn has(self, c: DevelopmentCard) -> bool {
        self.0[c as usize] > 0
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
