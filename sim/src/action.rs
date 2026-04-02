use crate::board::{Resource, TileId};
use crate::types::{EdgeId, PlayerId, VertexId};

const ALL_RESOURCES: [Resource; 5] = [
    Resource::Brick,
    Resource::Lumber,
    Resource::Ore,
    Resource::Grain,
    Resource::Wool,
];

/// Every action variant is a complete, atomic game action.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Action {
    PlaceSettlement(VertexId),
    PlaceRoad(EdgeId),
    BuildCity(VertexId),
    BuyDevelopmentCard,
    PlayKnight,
    PlayRoadBuilding,
    PlayYearOfPlenty(Resource, Resource),
    PlayMonopoly(Resource),
    RollDice,
    MoveRobber(TileId),
    StealFrom(PlayerId),
    StealFromNone,
    DiscardResource(Resource),
    BankTrade { give: Resource, receive: Resource },
    EndTurn,
}

// ---------------------------------------------------------------------------
// Flat action index  (total = 254)
// ---------------------------------------------------------------------------
//
//   0..54    PlaceSettlement(v)           54
//  54..126   PlaceRoad(e)                 72
// 126..180   BuildCity(v)                 54
//       180  BuyDevelopmentCard            1
//       181  PlayKnight                    1
//       182  PlayRoadBuilding              1
// 183..198   PlayYearOfPlenty(r1,r2)      15
// 198..203   PlayMonopoly(r)               5
//       203  RollDice                      1
// 204..223   MoveRobber(t)                19
// 223..227   StealFrom(p)                  4
//       227  StealFromNone                 1
// 228..233   DiscardResource(r)            5
// 233..253   BankTrade(give,recv)          20
//       253  EndTurn                       1
//                                  total 254

pub const ACTION_SPACE_SIZE: usize = 254;

impl Action {
    pub fn to_index(self) -> usize {
        match self {
            Action::PlaceSettlement(v) => v.idx(),
            Action::PlaceRoad(e) => 54 + e.idx(),
            Action::BuildCity(v) => 126 + v.idx(),
            Action::BuyDevelopmentCard => 180,
            Action::PlayKnight => 181,
            Action::PlayRoadBuilding => 182,
            Action::PlayYearOfPlenty(r1, r2) => 183 + yop_pair_index(r1, r2),
            Action::PlayMonopoly(r) => 198 + r as usize,
            Action::RollDice => 203,
            Action::MoveRobber(t) => 204 + t.0 as usize,
            Action::StealFrom(p) => 223 + p.idx(),
            Action::StealFromNone => 227,
            Action::DiscardResource(r) => 228 + r as usize,
            Action::BankTrade { give, receive } => 233 + bank_trade_index(give, receive),
            Action::EndTurn => 253,
        }
    }

    pub fn from_index(i: usize) -> Self {
        match i {
            0..54 => Action::PlaceSettlement(VertexId(i as u8)),
            54..126 => Action::PlaceRoad(EdgeId((i - 54) as u8)),
            126..180 => Action::BuildCity(VertexId((i - 126) as u8)),
            180 => Action::BuyDevelopmentCard,
            181 => Action::PlayKnight,
            182 => Action::PlayRoadBuilding,
            183..198 => {
                let (r1, r2) = yop_pair_from_index(i - 183);
                Action::PlayYearOfPlenty(r1, r2)
            }
            198..203 => Action::PlayMonopoly(ALL_RESOURCES[i - 198]),
            203 => Action::RollDice,
            204..223 => Action::MoveRobber(TileId((i - 204) as u8)),
            223..227 => Action::StealFrom(PlayerId((i - 223) as u8)),
            227 => Action::StealFromNone,
            228..233 => Action::DiscardResource(ALL_RESOURCES[i - 228]),
            233..253 => {
                let (g, r) = bank_trade_from_index(i - 233);
                Action::BankTrade {
                    give: g,
                    receive: r,
                }
            }
            253 => Action::EndTurn,
            _ => panic!("invalid action index {i}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Year-of-Plenty pair encoding (15 unordered pairs with repetition)
// ---------------------------------------------------------------------------

fn yop_pair_index(r1: Resource, r2: Resource) -> usize {
    let (a, b) = if (r1 as usize) <= (r2 as usize) {
        (r1 as usize, r2 as usize)
    } else {
        (r2 as usize, r1 as usize)
    };
    let row_offset = [0, 5, 9, 12, 14][a];
    row_offset + (b - a)
}

fn yop_pair_from_index(idx: usize) -> (Resource, Resource) {
    let (a, local) = if idx < 5 {
        (0, idx)
    } else if idx < 9 {
        (1, idx - 5)
    } else if idx < 12 {
        (2, idx - 9)
    } else if idx < 14 {
        (3, idx - 12)
    } else {
        (4, 0)
    };
    (ALL_RESOURCES[a], ALL_RESOURCES[a + local])
}

// ---------------------------------------------------------------------------
// BankTrade encoding (20 ordered pairs where give != receive)
// ---------------------------------------------------------------------------

fn bank_trade_index(give: Resource, receive: Resource) -> usize {
    let g = give as usize;
    let r = receive as usize;
    let r_adj = if r > g { r - 1 } else { r };
    g * 4 + r_adj
}

fn bank_trade_from_index(idx: usize) -> (Resource, Resource) {
    let g = idx / 4;
    let r_adj = idx % 4;
    let r = if r_adj >= g { r_adj + 1 } else { r_adj };
    (ALL_RESOURCES[g], ALL_RESOURCES[r])
}

// ---------------------------------------------------------------------------
// ActionMask – 254-bit mask stored in 4 × u64
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ActionMask {
    bits: [u64; 4],
}

impl ActionMask {
    pub const EMPTY: Self = Self { bits: [0; 4] };

    pub fn set(&mut self, idx: usize) {
        self.bits[idx / 64] |= 1u64 << (idx % 64);
    }

    pub fn clear(&mut self, idx: usize) {
        self.bits[idx / 64] &= !(1u64 << (idx % 64));
    }

    pub fn is_set(self, idx: usize) -> bool {
        self.bits[idx / 64] & (1u64 << (idx % 64)) != 0
    }

    pub fn is_empty(self) -> bool {
        self.bits == [0; 4]
    }

    pub fn count(self) -> u32 {
        self.bits.iter().map(|b| b.count_ones()).sum()
    }

    pub fn iter(self) -> ActionMaskIter {
        ActionMaskIter { mask: self, pos: 0 }
    }

    pub fn set_action(&mut self, action: Action) {
        self.set(action.to_index());
    }
}

pub struct ActionMaskIter {
    mask: ActionMask,
    pos: usize,
}

impl Iterator for ActionMaskIter {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        while self.pos < ACTION_SPACE_SIZE {
            let chunk = self.pos / 64;
            let bit = self.pos % 64;
            if chunk >= 4 {
                return None;
            }
            let word = self.mask.bits[chunk] >> bit;
            if word == 0 {
                self.pos = (chunk + 1) * 64;
                continue;
            }
            let tz = word.trailing_zeros() as usize;
            let idx = self.pos + tz;
            if idx >= ACTION_SPACE_SIZE {
                return None;
            }
            self.pos = idx + 1;
            return Some(idx);
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn action_index_roundtrip() {
        for i in 0..ACTION_SPACE_SIZE {
            let action = Action::from_index(i);
            assert_eq!(
                action.to_index(),
                i,
                "roundtrip failed for index {i}: {action:?}"
            );
        }
    }

    #[test]
    fn action_mask_basics() {
        let mut m = ActionMask::EMPTY;
        assert!(m.is_empty());
        m.set(0);
        m.set(253);
        m.set(100);
        assert_eq!(m.count(), 3);
        assert!(m.is_set(0));
        assert!(m.is_set(100));
        assert!(m.is_set(253));
        assert!(!m.is_set(1));
        let items: Vec<_> = m.iter().collect();
        assert_eq!(items, vec![0, 100, 253]);
    }
}
