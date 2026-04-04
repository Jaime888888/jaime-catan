use crate::board::Resource;
use crate::{EdgeId, PlayerId, TileId, VertexId};

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
            Action::PlaceSettlement(VertexId(vid)) => vid as usize,
            Action::PlaceRoad(EdgeId(eid)) => 54 + eid as usize,
            Action::BuildCity(VertexId(vid)) => 126 + vid as usize,
            Action::BuyDevelopmentCard => 180,
            Action::PlayKnight => 181,
            Action::PlayRoadBuilding => 182,
            Action::PlayYearOfPlenty(r1, r2) => 183 + yop_pair_index(r1, r2),
            Action::PlayMonopoly(r) => 198 + r as usize,
            Action::RollDice => 203,
            Action::MoveRobber(TileId(tid)) => 204 + tid as usize,
            Action::StealFrom(PlayerId(pid)) => 223 + pid as usize,
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
            198..203 => Action::PlayMonopoly(Resource::ALL[i - 198]),
            203 => Action::RollDice,
            204..223 => Action::MoveRobber(TileId((i - 204) as u8)),
            223..227 => Action::StealFrom(PlayerId((i - 223) as u8)),
            227 => Action::StealFromNone,
            228..233 => Action::DiscardResource(Resource::ALL[i - 228]),
            233..253 => {
                let (give, receive) = bank_trade_from_index(i - 233);
                Action::BankTrade { give, receive }
            }
            253 => Action::EndTurn,
            _ => panic!("invalid action index {i}"),
        }
    }
}

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
    (Resource::ALL[a], Resource::ALL[a + local])
}

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
    (Resource::ALL[g], Resource::ALL[r])
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ActionMask {
    bits: [u64; 4],
}

impl ActionMask {
    pub const EMPTY: Self = Self { bits: [0; 4] };

    pub fn set(&mut self, idx: usize) {
        self.bits[idx / 64] |= 1u64 << (idx % 64);
    }

    pub fn get(self, idx: usize) -> bool {
        self.bits[idx / 64] & (1u64 << (idx % 64)) != 0
    }

    pub fn count(self) -> u32 {
        self.bits.iter().map(|b| b.count_ones()).sum()
    }

    pub fn set_action(&mut self, action: Action) {
        self.set(action.to_index());
    }

    pub fn actions(self) -> ActionMaskIter {
        ActionMaskIter {
            bits: self.bits,
            pos: 0,
            remaining: self.count(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ActionMaskIter {
    bits: [u64; 4],
    pos: usize,
    remaining: u32,
}

impl Iterator for ActionMaskIter {
    type Item = Action;

    fn next(&mut self) -> Option<Self::Item> {
        while self.pos < ACTION_SPACE_SIZE {
            let chunk = self.pos / 64;
            let bit = self.pos % 64;
            if chunk >= 4 {
                return None;
            }
            let word = self.bits[chunk] >> bit;
            if word == 0 {
                self.pos = (chunk + 1) * 64;
                continue;
            }
            let tz = word.trailing_zeros() as usize;
            let idx = self.pos + tz;
            if idx >= ACTION_SPACE_SIZE {
                self.pos = idx + 1;
                continue;
            }
            self.pos = idx + 1;
            self.remaining -= 1;
            return Some(Action::from_index(idx));
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let r = self.remaining as usize;
        (r, Some(r))
    }
}

impl ExactSizeIterator for ActionMaskIter {}
