use rand::Rng;

use crate::action::{Action, ActionMask};
use crate::board::{
    DevelopmentCard, Edge, NUM_EDGES, NUM_TILES, NUM_VERTICES, Resource, ResourceBank, TOPO,
    TileId, Topology, Vertex,
};
use crate::types::*;
use crate::{Game, Phase};

const ALL_RESOURCES: [Resource; 5] = [
    Resource::Brick,
    Resource::Lumber,
    Resource::Ore,
    Resource::Grain,
    Resource::Wool,
];

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
            Phase::Discard { active, .. } => {
                let p = &self.players[active.idx()];
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
                if candidates.is_empty() {
                    mask.set_action(Action::StealFromNone);
                } else {
                    for pid in candidates.iter() {
                        if self.players[pid.idx()].resources.total() > 0 {
                            mask.set_action(Action::StealFrom(pid));
                        }
                    }
                    if mask.is_empty() {
                        mask.set_action(Action::StealFromNone);
                    }
                }
            }
            Phase::Main => self.legal_main_phase(&mut mask),
            Phase::RoadBuilding { .. } => {
                let pid = self.current_player;
                if self.players[pid.idx()].roads_left > 0 {
                    self.add_legal_road_placements(pid, &mut mask);
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
                if let Vertex::Settlement(p) = self.board.vertices[v] {
                    if p == pid {
                        mask.set_action(Action::BuildCity(VertexId(v as u8)));
                    }
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
            let ok_v1 = self.vertex_accessible_for_road(player, v1 as usize);
            let ok_v2 = self.vertex_accessible_for_road(player, v2 as usize);
            if ok_v1 || ok_v2 {
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
        if card == DevelopmentCard::VictoryPoint {
            return false;
        }
        if self.dev_card_played_this_turn {
            return false;
        }
        let playable = self.players[self.current_player.idx()]
            .dev_cards
            .get(card)
            .saturating_sub(self.dev_cards_bought_this_turn.get(card));
        playable > 0
    }
}

// =========================================================================
// State machine transitions
// =========================================================================

impl Game {
    pub fn apply_action(
        &mut self,
        action: Action,
        rng: &mut impl Rng,
    ) -> Result<(), InvalidAction> {
        let mask = self.action_mask();
        if !mask.is_set(action.to_index()) {
            return Err(InvalidAction {
                action,
                reason: "action not legal in current state",
            });
        }

        match action {
            Action::PlaceSettlement(v) => self.do_place_settlement(v),
            Action::PlaceRoad(e) => self.do_place_road(e),
            Action::BuildCity(v) => self.do_build_city(v),
            Action::BuyDevelopmentCard => self.do_buy_dev_card(),
            Action::PlayKnight => self.do_play_knight(),
            Action::PlayRoadBuilding => self.do_play_road_building(),
            Action::PlayYearOfPlenty(r1, r2) => self.do_year_of_plenty(r1, r2),
            Action::PlayMonopoly(r) => self.do_monopoly(r),
            Action::RollDice => self.do_roll_dice(),
            Action::MoveRobber(t) => self.do_move_robber(t),
            Action::StealFrom(p) => self.do_steal(p, rng),
            Action::StealFromNone => self.after_steal(),
            Action::DiscardResource(r) => self.do_discard(r),
            Action::BankTrade { give, receive } => self.do_bank_trade(give, receive),
            Action::EndTurn => self.do_end_turn(),
        }
        Ok(())
    }

    // ----- Setup -----

    pub const SETTLEMENT_COST: ResourceBank = ResourceBank([1, 1, 0, 1, 1]);
    pub const CITY_COST: ResourceBank = ResourceBank([0, 0, 3, 2, 0]);
    pub const ROAD_COST: ResourceBank = ResourceBank([1, 1, 0, 0, 0]);
    pub const DEV_CARD_COST: ResourceBank = ResourceBank([0, 0, 1, 1, 1]);

    fn do_place_settlement(&mut self, v: VertexId) {
        let vi = v.idx();
        let topo = &*TOPO;

        match &self.phase {
            Phase::SetupSettlement { player, round } => {
                let player = *player;
                let round = *round;

                self.board.vertices[vi] = Vertex::Settlement(player);
                self.players[player.idx()].settlements_left -= 1;

                if let Some(pt) = self.board.port_at(vi) {
                    self.players[player.idx()].update_ports(pt);
                }

                if round == 2 {
                    for &t in topo.vertex_tiles(vi) {
                        if let Some(resource) = self.board.tiles[t as usize].terrain.resource() {
                            if self.board.bank[resource] > 0 {
                                self.players[player.idx()].resources[resource] += 1;
                                self.board.bank[resource] -= 1;
                            }
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
                let pid = self.current_player;
                self.pay(pid, SETTLEMENT_COST);
                self.board.vertices[vi] = Vertex::Settlement(pid);
                self.players[pid.idx()].settlements_left -= 1;

                if let Some(pt) = self.board.port_at(vi) {
                    self.players[pid.idx()].update_ports(pt);
                }

                self.update_longest_road();
                self.check_victory();
            }
            _ => unreachable!(),
        }
    }

    fn do_place_road(&mut self, e: EdgeId) {
        let ei = e.idx();
        match self.phase.clone() {
            Phase::SetupRoad { player, round, .. } => {
                self.board.edges[ei] = Edge::Road(player);
                self.players[player.idx()].roads_left -= 1;
                self.advance_setup(player, round);
            }
            Phase::Main => {
                let pid = self.current_player;
                self.pay(pid, ROAD_COST);
                self.board.edges[ei] = Edge::Road(pid);
                self.players[pid.idx()].roads_left -= 1;
                self.update_longest_road();
                self.check_victory();
            }
            Phase::RoadBuilding { roads_left } => {
                let pid = self.current_player;
                self.board.edges[ei] = Edge::Road(pid);
                self.players[pid.idx()].roads_left -= 1;
                if roads_left <= 1 {
                    self.phase = Phase::Main;
                } else {
                    self.phase = Phase::RoadBuilding {
                        roads_left: roads_left - 1,
                    };
                    if self.action_mask().is_empty() {
                        self.phase = Phase::Main;
                    }
                }
                self.update_longest_road();
                self.check_victory();
            }
            _ => unreachable!(),
        }
    }

    fn advance_setup(&mut self, player: PlayerId, round: u8) {
        if round == 1 {
            if player.0 < 3 {
                self.phase = Phase::SetupSettlement {
                    player: player.next(),
                    round: 1,
                };
            } else {
                self.phase = Phase::SetupSettlement { player, round: 2 };
            }
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

    // ----- Dice -----

    fn do_roll_dice(&mut self) {
        self.phase = Phase::ChanceRoll;
    }

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
            let mut discard_mask = PlayerMask::NONE;
            for i in 0..4 {
                if self.players[i].resources.total() > 7 {
                    discard_mask.insert(PlayerId(i as u8));
                }
            }
            if discard_mask.is_empty() {
                self.phase = Phase::MoveRobber;
            } else {
                let first = discard_mask.first().unwrap();
                self.phase = Phase::Discard {
                    remaining: discard_mask,
                    active: first,
                };
            }
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

    // ----- Building -----

    fn do_build_city(&mut self, v: VertexId) {
        let pid = self.current_player;
        self.pay(pid, CITY_COST);
        self.board.vertices[v.idx()] = Vertex::City(pid);
        self.players[pid.idx()].cities_left -= 1;
        self.players[pid.idx()].settlements_left += 1;
        self.check_victory();
    }

    fn do_buy_dev_card(&mut self) {
        let pid = self.current_player;
        self.pay(pid, DEV_CARD_COST);
        let card = self
            .board
            .dev_card_deck
            .draw()
            .expect("deck verified non-empty");
        self.players[pid.idx()].dev_cards.add(card);
        self.dev_cards_bought_this_turn.add(card);
        self.check_victory();
    }

    // ----- Dev cards -----

    fn do_play_knight(&mut self) {
        let pid = self.current_player;
        self.players[pid.idx()]
            .dev_cards
            .remove(DevelopmentCard::Knight);
        self.players[pid.idx()].played_knights += 1;
        self.dev_card_played_this_turn = true;
        self.update_largest_army();
        self.phase = Phase::MoveRobber;
    }

    fn do_play_road_building(&mut self) {
        let pid = self.current_player;
        self.players[pid.idx()]
            .dev_cards
            .remove(DevelopmentCard::RoadBuilding);
        self.dev_card_played_this_turn = true;
        let roads_available = self.players[pid.idx()].roads_left.min(2);
        if roads_available == 0 {
            return;
        }
        self.phase = Phase::RoadBuilding {
            roads_left: roads_available,
        };
        if self.action_mask().is_empty() {
            self.phase = Phase::Main;
        }
    }

    fn do_year_of_plenty(&mut self, r1: Resource, r2: Resource) {
        let pid = self.current_player;
        self.players[pid.idx()]
            .dev_cards
            .remove(DevelopmentCard::YearOfPlenty);
        self.dev_card_played_this_turn = true;

        let give1 = 1u8.min(self.board.bank[r1]);
        self.players[pid.idx()].resources[r1] += give1;
        self.board.bank[r1] -= give1;

        let give2 = 1u8.min(self.board.bank[r2]);
        self.players[pid.idx()].resources[r2] += give2;
        self.board.bank[r2] -= give2;
    }

    fn do_monopoly(&mut self, resource: Resource) {
        let pid = self.current_player;
        self.players[pid.idx()]
            .dev_cards
            .remove(DevelopmentCard::Monopoly);
        self.dev_card_played_this_turn = true;

        let mut total = 0u8;
        for i in 0..4 {
            if i != pid.idx() {
                let amount = self.players[i].resources[resource];
                total += amount;
                self.players[i].resources[resource] = 0;
            }
        }
        self.players[pid.idx()].resources[resource] += total;
    }

    // ----- Robber / steal -----

    fn do_move_robber(&mut self, tile: TileId) {
        self.board.robber = tile;
        let topo = &*TOPO;
        let mut candidates = PlayerMask::NONE;
        for &v in &topo.tile_vertices[tile.0 as usize] {
            if let Some(owner) = self.board.vertices[v as usize].owner() {
                if owner != self.current_player && self.players[owner.idx()].resources.total() > 0 {
                    candidates.insert(owner);
                }
            }
        }
        self.phase = Phase::Steal { candidates };
    }

    fn do_steal(&mut self, target: PlayerId, rng: &mut impl Rng) {
        let total = self.players[target.idx()].resources.total();
        if total > 0 {
            let pick = rng.gen_range(0..total);
            let mut cumulative = 0u8;
            for r in ALL_RESOURCES {
                cumulative += self.players[target.idx()].resources[r];
                if pick < cumulative {
                    self.players[target.idx()].resources[r] -= 1;
                    self.players[self.current_player.idx()].resources[r] += 1;
                    break;
                }
            }
        }
        self.after_steal();
    }

    fn after_steal(&mut self) {
        if self.has_rolled_this_turn {
            self.phase = Phase::Main;
        } else {
            self.phase = Phase::PreRoll;
        }
    }

    // ----- Discard -----

    fn do_discard(&mut self, resource: Resource) {
        if let Phase::Discard {
            mut remaining,
            active,
        } = self.phase
        {
            self.players[active.idx()].resources[resource] -= 1;
            self.board.bank[resource] += 1;

            if self.players[active.idx()].resources.total() <= 7 {
                remaining.remove(active);
                if remaining.is_empty() {
                    self.phase = Phase::MoveRobber;
                } else {
                    let next = remaining.first().unwrap();
                    self.phase = Phase::Discard {
                        remaining,
                        active: next,
                    };
                }
            } else {
                self.phase = Phase::Discard { remaining, active };
            }
        }
    }

    // ----- Trading -----

    fn do_bank_trade(&mut self, give: Resource, receive: Resource) {
        let pid = self.current_player;
        let rate = self.players[pid.idx()].trade_rate(give);
        self.players[pid.idx()].resources[give] -= rate;
        self.board.bank[give] += rate;
        self.players[pid.idx()].resources[receive] += 1;
        self.board.bank[receive] -= 1;
    }

    // ----- End turn -----

    fn do_end_turn(&mut self) {
        self.current_player = self.current_player.next();
        self.turn_number += 1;
        self.dev_card_played_this_turn = false;
        self.dev_cards_bought_this_turn = DevCardHand::EMPTY;
        self.has_rolled_this_turn = false;
        self.phase = Phase::PreRoll;
    }

    // ----- Helpers -----

    fn pay(&mut self, player: PlayerId, cost: ResourceBank) {
        for i in 0..5 {
            self.players[player.idx()].resources.0[i] -= cost.0[i];
            self.board.bank.0[i] += cost.0[i];
        }
    }

    pub fn victory_points(&self, player: PlayerId) -> u8 {
        let mut vp = 0u8;
        for v in &self.board.vertices {
            match v {
                Vertex::Settlement(pid) if *pid == player => vp += 1,
                Vertex::City(pid) if *pid == player => vp += 2,
                _ => {}
            }
        }
        if self.longest_road_owner == Some(player) {
            vp += 2;
        }
        if self.largest_army_owner == Some(player) {
            vp += 2;
        }
        vp += self.players[player.idx()]
            .dev_cards
            .get(DevelopmentCard::VictoryPoint);
        vp
    }

    fn check_victory(&mut self) {
        if matches!(self.phase, Phase::GameOver { .. }) {
            return;
        }
        if self.victory_points(self.current_player) >= 10 {
            self.phase = Phase::GameOver {
                winner: self.current_player,
            };
        }
    }

    fn update_longest_road(&mut self) {
        let mut best_len = self.longest_road_length;
        let mut best_owner = self.longest_road_owner;
        for i in 0..4 {
            let pid = PlayerId(i);
            let len = longest_road(&self.board, pid);
            if len >= 5 && len > best_len {
                best_len = len;
                best_owner = Some(pid);
            }
        }
        self.longest_road_length = best_len;
        self.longest_road_owner = best_owner;
    }

    fn update_largest_army(&mut self) {
        for i in 0..4 {
            let pid = PlayerId(i);
            let knights = self.players[i as usize].played_knights;
            if knights >= 3 && knights > self.largest_army_size {
                self.largest_army_size = knights;
                self.largest_army_owner = Some(pid);
            }
        }
        self.check_victory();
    }
}

// =========================================================================
// Game rules (pure functions on board state)
// =========================================================================

fn distribute_resources(
    board: &mut crate::board::Board,
    roll: u8,
    players: &mut [crate::player::Player; 4],
) {
    let topo = &*TOPO;
    for t in 0..19u8 {
        let tile = &board.tiles[t as usize];
        if tile.number != roll || board.robber == TileId(t) {
            continue;
        }
        let resource = match tile.terrain.resource() {
            Some(r) => r,
            None => continue,
        };
        for &v in &topo.tile_vertices[t as usize] {
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

    for start_edge in 0..NUM_EDGES {
        if board.edges[start_edge].owner() != Some(player) {
            continue;
        }
        visited[start_edge] = true;
        for &endpoint in &topo.edge_vertices[start_edge] {
            dfs_road(
                board,
                player,
                endpoint as usize,
                1,
                &mut visited,
                &mut best,
                topo,
            );
        }
        visited[start_edge] = false;
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
        let next_v = if v1 as usize == vertex {
            v2 as usize
        } else {
            v1 as usize
        };
        visited[e] = true;
        dfs_road(board, player, next_v, depth + 1, visited, best, topo);
        visited[e] = false;
    }
}
