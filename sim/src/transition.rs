use rand::Rng;

use crate::action::{Action, ActionMask};
use crate::board::{
    DevelopmentCard, Edge, NUM_EDGES, NUM_TILES, NUM_VERTICES, Resource, ResourceBank, TOPO,
    Topology, Vertex,
};
use crate::{DevCardHand, EdgeId, Game, Phase, Player, PlayerId, TileId, VertexId};

const SETTLEMENT_COST: ResourceBank = ResourceBank([1, 1, 0, 1, 1]);
const CITY_COST: ResourceBank = ResourceBank([0, 0, 3, 2, 0]);
const ROAD_COST: ResourceBank = ResourceBank([1, 1, 0, 0, 0]);
const DEV_CARD_COST: ResourceBank = ResourceBank([0, 0, 1, 1, 1]);

const ALL_RESOURCES: [Resource; 5] = [
    Resource::Brick,
    Resource::Lumber,
    Resource::Ore,
    Resource::Grain,
    Resource::Wool,
];

// =========================================================================
// Legal action computation
// =========================================================================

impl Game {
    pub fn action_mask(&self) -> ActionMask {
        let mut mask = ActionMask::EMPTY;
        match &self.phase {
            Phase::SetupSettlement { .. } => {
                for v in 0..NUM_VERTICES {
                    if self.board.vertex_placement_valid(v) {
                        mask.set_action(Action::PlaceSettlement(VertexId(v as u8)));
                    }
                }
            }
            Phase::SetupRoad {
                settlement_vertex, ..
            } => {
                let topo = &*TOPO;
                for &e in topo.vertex_edge_list(*settlement_vertex as usize) {
                    if self.board.edges[e as usize].is_empty() {
                        mask.set_action(Action::PlaceRoad(EdgeId(e)));
                    }
                }
            }
            Phase::PreRoll => {
                mask.set_action(Action::RollDice);
                if self.can_play_dev_card(DevelopmentCard::Knight) {
                    mask.set_action(Action::PlayKnight);
                }
            }
            Phase::ChanceRoll => {}
            Phase::Discard {
                active: PlayerId(i),
                ..
            } => {
                let p = &self.players[*i as usize];
                for r in ALL_RESOURCES {
                    if p.resources[r] > 0 {
                        mask.set_action(Action::DiscardResource(r));
                    }
                }
            }
            Phase::MoveRobber => {
                for t in 0..NUM_TILES {
                    if TileId(t as u8) != self.board.robber {
                        mask.set_action(Action::MoveRobber(TileId(t as u8)));
                    }
                }
            }
            Phase::Steal { candidates } => {
                for (i, &is_candidate) in candidates.iter().enumerate() {
                    if is_candidate && self.players[i].resources.total() > 0 {
                        mask.set_action(Action::StealFrom(PlayerId(i as u8)));
                    }
                }
                if mask.count() == 0 {
                    mask.set_action(Action::StealFromNone);
                }
            }
            Phase::Main => self.legal_main_phase(&mut mask),
            Phase::RoadBuilding { .. } => {
                if self.players[self.current_player.idx()].roads_left > 0 {
                    self.add_legal_road_placements(self.current_player, &mut mask);
                }
            }
            Phase::GameOver { .. } => {}
        }
        mask
    }

    pub fn legal_actions(&self) -> Vec<Action> {
        self.action_mask().iter().map(Action::from_index).collect()
    }

    fn legal_main_phase(&self, mask: &mut ActionMask) {
        let pid = self.current_player;
        let player = &self.players[pid.idx()];

        if player.settlements_left > 0 && player.resources.can_afford(SETTLEMENT_COST) {
            for v in 0..NUM_VERTICES {
                if self.board.vertex_placement_valid(v)
                    && self.board.vertex_has_friendly_road(v, pid)
                {
                    mask.set_action(Action::PlaceSettlement(VertexId(v as u8)));
                }
            }
        }

        if player.roads_left > 0 && player.resources.can_afford(ROAD_COST) {
            self.add_legal_road_placements(pid, mask);
        }

        if player.cities_left > 0 && player.resources.can_afford(CITY_COST) {
            for v in 0..NUM_VERTICES {
                if let Vertex::Settlement(p) = self.board.vertices[v]
                    && p == pid
                {
                    mask.set_action(Action::BuildCity(VertexId(v as u8)));
                }
            }
        }

        if player.resources.can_afford(DEV_CARD_COST) && self.board.dev_card_deck.remaining() > 0 {
            mask.set_action(Action::BuyDevelopmentCard);
        }

        if self.can_play_dev_card(DevelopmentCard::Knight) {
            mask.set_action(Action::PlayKnight);
        }
        if self.can_play_dev_card(DevelopmentCard::RoadBuilding) && player.roads_left > 0 {
            mask.set_action(Action::PlayRoadBuilding);
        }
        if self.can_play_dev_card(DevelopmentCard::Monopoly) {
            for r in ALL_RESOURCES {
                mask.set_action(Action::PlayMonopoly(r));
            }
        }
        if self.can_play_dev_card(DevelopmentCard::YearOfPlenty) {
            for (i, &r1) in ALL_RESOURCES.iter().enumerate() {
                for &r2 in &ALL_RESOURCES[i..] {
                    if r1 == r2 {
                        if self.board.bank[r1] >= 2 {
                            mask.set_action(Action::PlayYearOfPlenty(r1, r2));
                        }
                    } else if self.board.bank[r1] >= 1 && self.board.bank[r2] >= 1 {
                        mask.set_action(Action::PlayYearOfPlenty(r1, r2));
                    }
                }
            }
        }

        for give in ALL_RESOURCES {
            let rate = player.trade_rate(give);
            if player.resources[give] >= rate {
                for recv in ALL_RESOURCES {
                    if recv != give && self.board.bank[recv] > 0 {
                        mask.set_action(Action::BankTrade {
                            give,
                            receive: recv,
                        });
                    }
                }
            }
        }

        mask.set_action(Action::EndTurn);
    }

    fn add_legal_road_placements(&self, player: PlayerId, mask: &mut ActionMask) {
        let topo = &*TOPO;
        for e in 0..NUM_EDGES {
            if !self.board.edges[e].is_empty() {
                continue;
            }
            let [v1, v2] = topo.edge_vertices[e];
            if self.vertex_accessible_for_road(player, v1 as usize)
                || self.vertex_accessible_for_road(player, v2 as usize)
            {
                mask.set_action(Action::PlaceRoad(EdgeId(e as u8)));
            }
        }
    }

    fn vertex_accessible_for_road(&self, player: PlayerId, v: usize) -> bool {
        if let Some(owner) = self.board.vertices[v].owner() {
            return owner == player;
        }
        self.board.vertex_has_friendly_road(v, player)
    }

    fn can_play_dev_card(&self, card: DevelopmentCard) -> bool {
        if card == DevelopmentCard::VictoryPoint || self.dev_card_played_this_turn {
            return false;
        }
        let held = self.players[self.current_player.idx()].dev_cards[card];
        let bought = self.dev_cards_bought_this_turn[card];
        held > bought
    }
}

// =========================================================================
// State machine transition
// =========================================================================

impl Game {
    pub fn apply_action(
        &mut self,
        action: Action,
        rng: &mut impl Rng,
    ) -> Result<(), InvalidAction> {
        let mask = self.action_mask();
        if !mask.get(action.to_index()) {
            return Err(InvalidAction {
                action,
                reason: "action not legal in current state",
            });
        }

        let topo = &*TOPO;
        let pid = self.acting_player();
        let pi = pid.idx();

        match action {
            Action::PlaceSettlement(v) => {
                let vi = v.idx();
                self.board.vertices[vi] = Vertex::Settlement(pid);
                self.players[pi].settlements_left -= 1;
                if let Some(pt) = self.board.port_at(vi) {
                    self.players[pi].update_ports(pt);
                }

                match &self.phase {
                    Phase::SetupSettlement { player, round } => {
                        let (player, round) = (*player, *round);
                        if round == 2 {
                            for &t in topo.vertex_tiles(vi) {
                                if let Some(r) = self.board.tiles[t as usize].terrain.resource()
                                    && self.board.bank[r] > 0
                                {
                                    self.players[pi].resources[r] += 1;
                                    self.board.bank[r] -= 1;
                                }
                            }
                        }
                        self.phase = Phase::SetupRoad {
                            player,
                            round,
                            settlement_vertex: v.0,
                        };
                    }
                    Phase::Main => {
                        self.pay(pid, SETTLEMENT_COST);
                        self.update_longest_road();
                        self.check_victory();
                    }
                    _ => unreachable!(),
                }
            }
            Action::PlaceRoad(e) => {
                self.board.edges[e.idx()] = Edge::Road(pid);
                self.players[pi].roads_left -= 1;

                match self.phase.clone() {
                    Phase::SetupRoad { player, round, .. } => self.advance_setup(player, round),
                    Phase::Main => {
                        self.pay(pid, ROAD_COST);
                        self.update_longest_road();
                        self.check_victory();
                    }
                    Phase::RoadBuilding { roads_left } => {
                        self.phase = if roads_left <= 1 || self.action_mask().count() == 0 {
                            Phase::Main
                        } else {
                            Phase::RoadBuilding {
                                roads_left: roads_left - 1,
                            }
                        };
                        self.update_longest_road();
                        self.check_victory();
                    }
                    _ => unreachable!(),
                }
            }
            Action::BuildCity(v) => {
                self.pay(pid, CITY_COST);
                self.board.vertices[v.idx()] = Vertex::City(pid);
                self.players[pi].cities_left -= 1;
                self.players[pi].settlements_left += 1;
                self.check_victory();
            }
            Action::BuyDevelopmentCard => {
                self.pay(pid, DEV_CARD_COST);
                let card = self.board.dev_card_deck.draw().expect("deck non-empty");
                self.players[pi].dev_cards += card;
                self.dev_cards_bought_this_turn += card;
                self.check_victory();
            }
            Action::PlayKnight => {
                self.players[pi]
                    .dev_cards
                    .checked_sub_assign(DevelopmentCard::Knight);
                self.players[pi].played_knights += 1;
                self.dev_card_played_this_turn = true;
                self.update_largest_army();
                self.phase = Phase::MoveRobber;
            }
            Action::PlayRoadBuilding => {
                self.players[pi]
                    .dev_cards
                    .checked_sub_assign(DevelopmentCard::RoadBuilding);
                self.dev_card_played_this_turn = true;
                let n = self.players[pi].roads_left.min(2);
                if n > 0 {
                    self.phase = Phase::RoadBuilding { roads_left: n };
                    if self.action_mask().count() == 0 {
                        self.phase = Phase::Main;
                    }
                }
            }
            Action::PlayYearOfPlenty(r1, r2) => {
                self.players[pi]
                    .dev_cards
                    .checked_sub_assign(DevelopmentCard::YearOfPlenty);
                self.dev_card_played_this_turn = true;
                let g1 = 1u8.min(self.board.bank[r1]);
                self.players[pi].resources[r1] += g1;
                self.board.bank[r1] -= g1;
                let g2 = 1u8.min(self.board.bank[r2]);
                self.players[pi].resources[r2] += g2;
                self.board.bank[r2] -= g2;
            }
            Action::PlayMonopoly(resource) => {
                self.players[pi]
                    .dev_cards
                    .checked_sub_assign(DevelopmentCard::Monopoly);
                self.dev_card_played_this_turn = true;
                let mut total = 0u8;
                for i in 0..4 {
                    if i != pi {
                        total += self.players[i].resources[resource];
                        self.players[i].resources[resource] = 0;
                    }
                }
                self.players[pi].resources[resource] += total;
            }
            Action::RollDice => {
                self.phase = Phase::ChanceRoll;
            }
            Action::MoveRobber(tile) => {
                self.board.robber = tile;
                let mut candidates = [false; 4];
                for &v in &topo.tile_vertices[tile.idx()] {
                    if let Some(owner) = self.board.vertices[v as usize].owner()
                        && owner != pid
                        && self.players[owner.idx()].resources.total() > 0
                    {
                        candidates[owner.idx()] = true;
                    }
                }
                self.phase = Phase::Steal { candidates };
            }
            Action::StealFrom(target) => {
                let total = self.players[target.idx()].resources.total();
                if total > 0 {
                    let pick = rng.gen_range(0..total);
                    let mut cum = 0u8;
                    for r in ALL_RESOURCES {
                        cum += self.players[target.idx()].resources[r];
                        if pick < cum {
                            self.players[target.idx()].resources[r] -= 1;
                            self.players[pi].resources[r] += 1;
                            break;
                        }
                    }
                }
                self.phase = if self.has_rolled_this_turn {
                    Phase::Main
                } else {
                    Phase::PreRoll
                };
            }
            Action::StealFromNone => {
                self.phase = if self.has_rolled_this_turn {
                    Phase::Main
                } else {
                    Phase::PreRoll
                };
            }
            Action::DiscardResource(resource) => {
                if let Phase::Discard {
                    mut remaining,
                    active,
                } = self.phase
                {
                    self.players[active.idx()].resources[resource] -= 1;
                    self.board.bank[resource] += 1;

                    if self.players[active.idx()].resources.total() <= 7 {
                        remaining[active.idx()] = false;
                        self.phase = match remaining.iter().position(|&b| b) {
                            None => Phase::MoveRobber,
                            Some(next) => Phase::Discard {
                                remaining,
                                active: PlayerId(next as u8),
                            },
                        };
                    } else {
                        self.phase = Phase::Discard { remaining, active };
                    }
                }
            }
            Action::BankTrade { give, receive } => {
                let rate = self.players[pi].trade_rate(give);
                self.players[pi].resources[give] -= rate;
                self.board.bank[give] += rate;
                self.players[pi].resources[receive] += 1;
                self.board.bank[receive] -= 1;
            }
            Action::EndTurn => {
                self.current_player = self.current_player.next();
                self.turn_number += 1;
                self.dev_card_played_this_turn = false;
                self.dev_cards_bought_this_turn = DevCardHand::EMPTY;
                self.has_rolled_this_turn = false;
                self.phase = Phase::PreRoll;
            }
        }

        Ok(())
    }

    fn advance_setup(&mut self, player: PlayerId, round: u8) {
        if round == 1 {
            self.phase = if player.0 < 3 {
                Phase::SetupSettlement {
                    player: player.next(),
                    round: 1,
                }
            } else {
                Phase::SetupSettlement { player, round: 2 }
            };
        } else if player.0 > 0 {
            self.phase = Phase::SetupSettlement {
                player: player.prev(),
                round: 2,
            };
        } else {
            self.current_player = PlayerId::P0;
            self.phase = Phase::PreRoll;
        }
    }

    fn pay(&mut self, player: PlayerId, cost: ResourceBank) {
        for i in 0..5 {
            self.players[player.idx()].resources.0[i] -= cost.0[i];
            self.board.bank.0[i] += cost.0[i];
        }
    }

    fn check_victory(&mut self) {
        if !matches!(self.phase, Phase::GameOver { .. })
            && self.victory_points(self.current_player) >= 10
        {
            self.phase = Phase::GameOver {
                winner: self.current_player,
            };
        }
    }

    fn update_longest_road(&mut self) {
        for i in 0..4 {
            let pid = PlayerId(i);
            let len = longest_road(&self.board, pid);
            if len >= 5 && len > self.longest_road_length {
                self.longest_road_length = len;
                self.longest_road_owner = Some(pid);
            }
        }
    }

    fn update_largest_army(&mut self) {
        for i in 0..4 {
            let k = self.players[i as usize].played_knights;
            if k >= 3 && k > self.largest_army_size {
                self.largest_army_size = k;
                self.largest_army_owner = Some(PlayerId(i));
            }
        }
        self.check_victory();
    }
}

// =========================================================================
// Chance node resolution
// =========================================================================

impl Game {
    pub const DICE_PROBS: [(u8, f64); 11] = [
        (2, 1.0 / 36.0),
        (3, 2.0 / 36.0),
        (4, 3.0 / 36.0),
        (5, 4.0 / 36.0),
        (6, 5.0 / 36.0),
        (7, 6.0 / 36.0),
        (8, 5.0 / 36.0),
        (9, 4.0 / 36.0),
        (10, 3.0 / 36.0),
        (11, 2.0 / 36.0),
        (12, 1.0 / 36.0),
    ];

    pub fn chance_outcomes(&self) -> &[(u8, f64); 11] {
        assert!(self.is_chance_node());
        &Self::DICE_PROBS
    }

    pub fn resolve_chance(&mut self, roll: u8) {
        assert!(self.is_chance_node());
        assert!((2..=12).contains(&roll));
        self.has_rolled_this_turn = true;

        if roll == 7 {
            let remaining = std::array::from_fn(|i| self.players[i].resources.total() > 7);
            self.phase = match remaining.iter().position(|&b| b) {
                None => Phase::MoveRobber,
                Some(first) => Phase::Discard {
                    remaining,
                    active: PlayerId(first as u8),
                },
            };
        } else {
            distribute_resources(&mut self.board, roll, &mut self.players);
            self.phase = Phase::Main;
        }
    }

    pub fn resolve_chance_random(&mut self, rng: &mut impl Rng) {
        let d1: u8 = rng.gen_range(1..=6);
        let d2: u8 = rng.gen_range(1..=6);
        self.resolve_chance(d1 + d2);
    }

    pub fn victory_points(&self, player: PlayerId) -> u8 {
        let mut vp = 0u8;
        for v in &self.board.vertices {
            match v {
                Vertex::Settlement(p) if *p == player => vp += 1,
                Vertex::City(p) if *p == player => vp += 2,
                _ => {}
            }
        }
        if self.longest_road_owner == Some(player) {
            vp += 2;
        }
        if self.largest_army_owner == Some(player) {
            vp += 2;
        }
        vp += self.players[player.idx()].dev_cards[DevelopmentCard::VictoryPoint];
        vp
    }
}

// =========================================================================
// Game rules
// =========================================================================

fn distribute_resources(board: &mut crate::board::Board, roll: u8, players: &mut [Player; 4]) {
    let topo = &*TOPO;
    for t in 0..NUM_TILES {
        let tile = &board.tiles[t];
        if tile.number != roll || board.robber == TileId(t as u8) {
            continue;
        }
        let resource = match tile.terrain.resource() {
            Some(r) => r,
            None => continue,
        };
        for &v in &topo.tile_vertices[t] {
            let (pid, amount) = match board.vertices[v as usize] {
                Vertex::Settlement(pid) => (pid, 1u8),
                Vertex::City(pid) => (pid, 2u8),
                _ => continue,
            };
            let give = amount.min(board.bank[resource]);
            if give > 0 {
                players[pid.idx()].resources[resource] += give;
                board.bank[resource] -= give;
            }
        }
    }
}

pub fn longest_road(board: &crate::board::Board, player: PlayerId) -> u8 {
    let topo = &*TOPO;
    let mut best = 0u8;
    let mut visited = [false; NUM_EDGES];
    for start in 0..NUM_EDGES {
        if board.edges[start].owner() != Some(player) {
            continue;
        }
        visited[start] = true;
        for &ep in &topo.edge_vertices[start] {
            dfs_road(board, player, ep as usize, 1, &mut visited, &mut best, topo);
        }
        visited[start] = false;
    }
    best
}

fn dfs_road(
    board: &crate::board::Board,
    player: PlayerId,
    vertex: usize,
    depth: u8,
    visited: &mut [bool; NUM_EDGES],
    best: &mut u8,
    topo: &Topology,
) {
    if depth > *best {
        *best = depth;
    }
    match board.vertices[vertex] {
        Vertex::Settlement(p) | Vertex::City(p) if p != player => return,
        _ => {}
    }
    for &edge in topo.vertex_edge_list(vertex) {
        let e = edge as usize;
        if visited[e] || board.edges[e].owner() != Some(player) {
            continue;
        }
        let [v1, v2] = topo.edge_vertices[e];
        let next = if v1 as usize == vertex {
            v2 as usize
        } else {
            v1 as usize
        };
        visited[e] = true;
        dfs_road(board, player, next, depth + 1, visited, best, topo);
        visited[e] = false;
    }
}

// =========================================================================
// Error
// =========================================================================

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InvalidAction {
    pub action: Action,
    pub reason: &'static str,
}

impl std::fmt::Display for InvalidAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid action {:?}: {}", self.action, self.reason)
    }
}

impl std::error::Error for InvalidAction {}
