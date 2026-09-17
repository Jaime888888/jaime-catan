use rand::{Rng, RngExt};

use crate::action::{Action, ActionMask};
use crate::board::{
    Board, DevelopmentCard, Edge, NUM_EDGES, NUM_TILES, NUM_VERTICES, Resource, ResourceBank, TOPO,
    Vertex,
};
use crate::{EdgeId, Game, Phase, PlayerId, TileId, TurnFlags, VertexId};

const SETTLEMENT_COST: ResourceBank = ResourceBank([1, 1, 0, 1, 1]);
const CITY_COST: ResourceBank = ResourceBank([0, 0, 3, 2, 0]);
const ROAD_COST: ResourceBank = ResourceBank([1, 1, 0, 0, 0]);
const DEV_CARD_COST: ResourceBank = ResourceBank([0, 0, 1, 1, 1]);

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

pub enum Turn<'a> {
    Player(PlayerTurn<'a>),
    Chance(ChanceTurn<'a>),
    Terminal,
}

pub struct PlayerTurn<'a> {
    game: &'a mut Game,
    pub mask: ActionMask,
}

pub struct ChanceTurn<'a> {
    game: &'a mut Game,
}

impl Game {
    pub fn action_mask(&self) -> ActionMask {
        let topo = &*TOPO;
        let mut mask = ActionMask::EMPTY;

        match &self.phase {
            Phase::GameOver { .. } | Phase::ChanceRoll => return mask,
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
            Phase::Discard {
                active: PlayerId(i),
                ..
            } => {
                let p = &self.players[*i as usize];
                for r in Resource::ALL {
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
            Phase::Main => {
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
                    self.legal_road_placements(pid, &mut mask);
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
                if player.resources.can_afford(DEV_CARD_COST)
                    && self.board.dev_card_deck.remaining() > 0
                {
                    mask.set_action(Action::BuyDevelopmentCard);
                }
                if self.can_play_dev_card(DevelopmentCard::Knight) {
                    mask.set_action(Action::PlayKnight);
                }
                if self.can_play_dev_card(DevelopmentCard::RoadBuilding) && player.roads_left > 0 {
                    mask.set_action(Action::PlayRoadBuilding);
                }
                if self.can_play_dev_card(DevelopmentCard::Monopoly) {
                    for r in Resource::ALL {
                        mask.set_action(Action::PlayMonopoly(r));
                    }
                }
                if self.can_play_dev_card(DevelopmentCard::YearOfPlenty) {
                    for (i, &r1) in Resource::ALL.iter().enumerate() {
                        for &r2 in &Resource::ALL[i..] {
                            let needed = if r1 == r2 { 2 } else { 1 };
                            if self.board.bank[r1] >= needed && self.board.bank[r2] >= 1 {
                                mask.set_action(Action::PlayYearOfPlenty(r1, r2));
                            }
                        }
                    }
                }
                for give in Resource::ALL {
                    let rate = player.trade_rate(give);
                    if player.resources[give] >= rate {
                        for recv in Resource::ALL {
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
            Phase::RoadBuilding { .. } => {
                if self.players[self.current_player.idx()].roads_left > 0 {
                    self.legal_road_placements(self.current_player, &mut mask);
                }
            }
        }

        mask
    }

    pub fn turn(&mut self) -> Turn<'_> {
        match self.phase {
            Phase::GameOver { .. } => Turn::Terminal,
            Phase::ChanceRoll => Turn::Chance(ChanceTurn { game: self }),
            _ => {
                let mask = self.action_mask();
                Turn::Player(PlayerTurn { game: self, mask })
            }
        }
    }
}

impl<'a> PlayerTurn<'a> {
    pub fn apply(self, action: Action, rng: &mut impl Rng) -> Result<(), InvalidAction> {
        if !self.mask.get(action.to_index()) {
            return Err(InvalidAction {
                action,
                reason: "action not legal in current state",
            });
        }

        let game = self.game;
        let topo = &*TOPO;
        let pid = game.acting_player();
        let pi = pid.idx();

        match action {
            Action::PlaceSettlement(v) => {
                let vi = v.idx();
                game.board.vertices[vi] = Vertex::Settlement(pid);
                game.players[pi].settlements_left -= 1;
                if let Some(pt) = game.board.port_at(vi) {
                    game.players[pi].update_ports(pt);
                }
                match &game.phase {
                    Phase::SetupSettlement { player, round } => {
                        let (player, round) = (*player, *round);
                        if round == 2 {
                            for &t in topo.vertex_tiles(vi) {
                                if let Some(r) = game.board.tiles[t as usize].terrain.resource()
                                    && game.board.bank[r] > 0
                                {
                                    game.players[pi].resources[r] += 1;
                                    game.board.bank[r] -= 1;
                                }
                            }
                        }
                        game.phase = Phase::SetupRoad {
                            player,
                            round,
                            settlement_vertex: v.0,
                        };
                    }
                    Phase::Main => {
                        game.pay(pid, SETTLEMENT_COST);
                        game.check_victory();
                    }
                    _ => unreachable!(),
                }
                game.compute_longest_road();
                game.check_victory();
            }
            Action::PlaceRoad(e) => {
                game.board.edges[e.idx()] = Edge::Road(pid);
                game.players[pi].roads_left -= 1;
                match game.phase {
                    Phase::SetupRoad { player, round, .. } => {
                        game.phase = if round == 1 {
                            if player.0 < 3 {
                                Phase::SetupSettlement {
                                    player: player.next(),
                                    round: 1,
                                }
                            } else {
                                Phase::SetupSettlement { player, round: 2 }
                            }
                        } else if player.0 > 0 {
                            Phase::SetupSettlement {
                                player: player.prev(),
                                round: 2,
                            }
                        } else {
                            game.current_player = PlayerId::P0;
                            Phase::PreRoll
                        };
                    }
                    Phase::Main => {
                        game.pay(pid, ROAD_COST);
                        game.check_victory();
                    }
                    Phase::RoadBuilding { roads_left } => {
                        game.phase = if roads_left <= 1 || !game.has_legal_road(pid) {
                            Phase::Main
                        } else {
                            Phase::RoadBuilding {
                                roads_left: roads_left - 1,
                            }
                        };
                        game.check_victory();
                    }
                    _ => unreachable!(),
                }
                game.compute_longest_road();
                game.check_victory();
            }
            Action::BuildCity(v) => {
                game.pay(pid, CITY_COST);
                game.board.vertices[v.idx()] = Vertex::City(pid);
                game.players[pi].cities_left -= 1;
                game.players[pi].settlements_left += 1;
                game.compute_longest_road();
                game.check_victory();
            }
            Action::BuyDevelopmentCard => {
                game.pay(pid, DEV_CARD_COST);
                let card = game.board.dev_card_deck.draw().expect("deck non-empty");
                game.players[pi].dev_cards += card;
                game.turn_flags.dev_cards_bought += card;
                game.check_victory();
            }
            Action::PlayKnight => {
                game.players[pi]
                    .dev_cards
                    .checked_sub_assign(DevelopmentCard::Knight);
                game.players[pi].played_knights += 1;
                game.turn_flags.dev_card_played = true;
                game.check_victory();
                game.phase = Phase::MoveRobber;
            }
            Action::PlayRoadBuilding => {
                game.players[pi]
                    .dev_cards
                    .checked_sub_assign(DevelopmentCard::RoadBuilding);
                game.turn_flags.dev_card_played = true;
                let n = game.players[pi].roads_left.min(2);
                if n > 0 && game.has_legal_road(pid) {
                    game.phase = Phase::RoadBuilding { roads_left: n };
                }
            }
            Action::PlayYearOfPlenty(r1, r2) => {
                game.players[pi]
                    .dev_cards
                    .checked_sub_assign(DevelopmentCard::YearOfPlenty);
                game.turn_flags.dev_card_played = true;
                let g1 = 1u8.min(game.board.bank[r1]);
                game.players[pi].resources[r1] += g1;
                game.board.bank[r1] -= g1;
                let g2 = 1u8.min(game.board.bank[r2]);
                game.players[pi].resources[r2] += g2;
                game.board.bank[r2] -= g2;
            }
            Action::PlayMonopoly(resource) => {
                game.players[pi]
                    .dev_cards
                    .checked_sub_assign(DevelopmentCard::Monopoly);
                game.turn_flags.dev_card_played = true;
                let mut total = 0u8;
                for i in 0..4 {
                    if i != pi {
                        total += game.players[i].resources[resource];
                        game.players[i].resources[resource] = 0;
                    }
                }
                game.players[pi].resources[resource] += total;
            }
            Action::RollDice => {
                game.phase = Phase::ChanceRoll;
            }
            Action::MoveRobber(tile) => {
                game.board.robber = tile;
                let mut candidates = [false; 4];
                for &v in &topo.tile_vertices[tile.idx()] {
                    if let Some(owner) = game.board.vertices[v as usize].owner()
                        && owner != pid
                        && game.players[owner.idx()].resources.total() > 0
                    {
                        candidates[owner.idx()] = true;
                    }
                }
                game.phase = Phase::Steal { candidates };
            }
            Action::StealFrom(target) => {
                let total = game.players[target.idx()].resources.total();
                if total > 0 {
                    let pick = rng.random_range(0..total);
                    let mut cum = 0u8;
                    for r in Resource::ALL {
                        cum += game.players[target.idx()].resources[r];
                        if pick < cum {
                            game.players[target.idx()].resources[r] -= 1;
                            game.players[pi].resources[r] += 1;
                            break;
                        }
                    }
                }
                game.phase = if game.turn_flags.has_rolled {
                    Phase::Main
                } else {
                    Phase::PreRoll
                };
            }
            Action::StealFromNone => {
                game.phase = if game.turn_flags.has_rolled {
                    Phase::Main
                } else {
                    Phase::PreRoll
                };
            }
            Action::DiscardResource(resource) => {
                if let Phase::Discard {
                    mut remaining,
                    targets,
                    active,
                } = game.phase
                {
                    game.players[active.idx()].resources[resource] -= 1;
                    game.board.bank[resource] += 1;
                    if game.players[active.idx()].resources.total() <= targets[active.idx()] {
                        remaining[active.idx()] = false;
                        game.phase = match remaining.iter().position(|&b| b) {
                            None => Phase::MoveRobber,
                            Some(next) => Phase::Discard {
                                remaining,
                                targets,
                                active: PlayerId(next as u8),
                            },
                        };
                    } else {
                        game.phase = Phase::Discard {
                            remaining,
                            targets,
                            active,
                        };
                    }
                }
            }
            Action::BankTrade { give, receive } => {
                let rate = game.players[pi].trade_rate(give);
                game.players[pi].resources[give] -= rate;
                game.board.bank[give] += rate;
                game.players[pi].resources[receive] += 1;
                game.board.bank[receive] -= 1;
            }
            Action::EndTurn => {
                game.current_player = game.current_player.next();
                game.turn_number += 1;
                game.turn_flags = TurnFlags::default();
                game.phase = Phase::PreRoll;
                game.check_victory();
            }
        }

        Ok(())
    }
}

impl<'a> ChanceTurn<'a> {
    pub fn resolve(self, roll: u8) {
        assert!((2..=12).contains(&roll));

        let topo = &*TOPO;
        self.game.turn_flags.has_rolled = true;

        if roll == 7 {
            let remaining = std::array::from_fn(|i| self.game.players[i].resources.total() > 7);
            let targets = std::array::from_fn(|i| {
                let total = self.game.players[i].resources.total();
                total.div_ceil(2)
            });

            self.game.phase = match remaining.iter().position(|&b| b) {
                None => Phase::MoveRobber,
                Some(first) => Phase::Discard {
                    remaining,
                    targets,
                    active: PlayerId(first as u8),
                },
            };
        } else {
            for t in 0..NUM_TILES {
                let tile = &self.game.board.tiles[t];
                if tile.number != roll || self.game.board.robber == TileId(t as u8) {
                    continue;
                }
                let Some(resource) = tile.terrain.resource() else {
                    continue;
                };

                for &v in &topo.tile_vertices[t] {
                    let (pid, amount) = match self.game.board.vertices[v as usize] {
                        Vertex::Settlement(pid) => (pid, 1u8),
                        Vertex::City(pid) => (pid, 2u8),
                        _ => continue,
                    };
                    let give = amount.min(self.game.board.bank[resource]);
                    if give > 0 {
                        self.game.players[pid.idx()].resources[resource] += give;
                        self.game.board.bank[resource] -= give;
                    }
                }
            }

            self.game.phase = Phase::Main;
        }
    }

    pub fn resolve_random(self, rng: &mut impl Rng) {
        self.resolve(rng.random_range(1..=6u8) + rng.random_range(1..=6u8));
    }
}

impl Game {
    fn legal_road_placements(&self, player: PlayerId, mask: &mut ActionMask) {
        let topo = &*TOPO;

        for e in 0..NUM_EDGES {
            if !self.board.edges[e].is_empty() {
                continue;
            }

            let [v1, v2] = topo.edge_vertices[e];
            if self.board.vertex_road_accessible(v1 as usize, player)
                || self.board.vertex_road_accessible(v2 as usize, player)
            {
                mask.set_action(Action::PlaceRoad(EdgeId(e as u8)));
            }
        }
    }

    fn has_legal_road(&self, player: PlayerId) -> bool {
        let topo = &*TOPO;

        (0..NUM_EDGES).any(|e| {
            self.board.edges[e].is_empty() && {
                let [v1, v2] = topo.edge_vertices[e];
                self.board.vertex_road_accessible(v1 as usize, player)
                    || self.board.vertex_road_accessible(v2 as usize, player)
            }
        })
    }

    fn can_play_dev_card(&self, card: DevelopmentCard) -> bool {
        if card == DevelopmentCard::VictoryPoint || self.turn_flags.dev_card_played {
            return false;
        }

        self.players[self.current_player.idx()].dev_cards[card]
            > self.turn_flags.dev_cards_bought[card]
    }

    fn pay(&mut self, player: PlayerId, cost: ResourceBank) {
        for i in 0..5 {
            self.players[player.idx()].resources.0[i] -= cost.0[i];
            self.board.bank.0[i] += cost.0[i];
        }
    }

    fn check_victory(&mut self) {
        if matches!(self.phase, Phase::GameOver { .. }) {
            return;
        }
        // Scan everyone: VP can depend on longest road (recomputed after builds), and
        // `current_player` alone is wrong after `EndTurn` if a win was missed earlier.
        for pid in PlayerId::ALL {
            if self.victory_points(pid) >= 10 {
                self.phase = Phase::GameOver { winner: pid };
                return;
            }
        }
    }

    fn compute_longest_road(&mut self) {
        let topo = &*TOPO;

        for i in 0..4 {
            let player = PlayerId(i as u8);
            let mut best = 0u8;
            let mut visited = [false; NUM_EDGES];

            for start in 0..NUM_EDGES {
                if self.board.edges[start].owner() != Some(player) {
                    continue;
                }

                visited[start] = true;
                for &ep in &topo.edge_vertices[start] {
                    dfs_road(&self.board, player, ep as usize, 1, &mut visited, &mut best);
                }
                visited[start] = false;
            }

            self.longest_road_len[i] = best;
        }
    }

    pub fn longest_road_owner(&self) -> Option<PlayerId> {
        let mut best_len = 0u8;
        let mut best_pid = None;
        for i in 0..4usize {
            let pid = PlayerId(i as u8);
            let len = self.longest_road_len[i];
            if len >= 5 && len > best_len {
                best_len = len;
                best_pid = Some(pid);
            }
        }

        best_pid
    }

    pub fn largest_army_owner(&self) -> Option<PlayerId> {
        let mut best_k = 2u8; // must exceed 2 (i.e. >= 3)
        let mut best_pid = None;
        for i in 0..4 {
            let k = self.players[i as usize].played_knights;

            if k > best_k {
                best_k = k;
                best_pid = Some(PlayerId(i));
            }
        }

        best_pid
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

        if self.longest_road_owner() == Some(player) {
            vp += 2;
        }

        if self.largest_army_owner() == Some(player) {
            vp += 2;
        }

        vp += self.players[player.idx()].dev_cards[DevelopmentCard::VictoryPoint];
        vp
    }
}

fn dfs_road(
    board: &Board,
    player: PlayerId,
    vertex: usize,
    depth: u8,
    visited: &mut [bool; NUM_EDGES],
    best: &mut u8,
) {
    if depth > *best {
        *best = depth;
    }

    let topo = &*TOPO;

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
        dfs_road(board, player, next, depth + 1, visited, best);
        visited[e] = false;
    }
}

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
