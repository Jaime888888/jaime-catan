use std::fmt;

use crate::board::{
    DevCardHand, DevelopmentCard, Edge, Port, Resource, ResourceBank, Terrain, Tile, Vertex,
};
use crate::{Action, Game, Phase, Player, PlayerId, TileId};

impl fmt::Display for Terrain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Terrain::Desert => "Desert",
            Terrain::Hills => "Hills",
            Terrain::Forest => "Forest",
            Terrain::Mountains => "Mtns",
            Terrain::Fields => "Fields",
            Terrain::Pasture => "Pasture",
        })
    }
}

impl fmt::Display for Resource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Resource::Brick => "Brick",
            Resource::Lumber => "Lumber",
            Resource::Ore => "Ore",
            Resource::Grain => "Grain",
            Resource::Wool => "Wool",
        })
    }
}

impl fmt::Display for Port {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Port::ThreeToOne => write!(f, "3:1"),
            Port::TwoToOne(r) => write!(f, "2:1 {r}"),
        }
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

impl fmt::Display for DevelopmentCard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            DevelopmentCard::Knight => "Knight",
            DevelopmentCard::VictoryPoint => "VP",
            DevelopmentCard::RoadBuilding => "RoadBuild",
            DevelopmentCard::YearOfPlenty => "YoP",
            DevelopmentCard::Monopoly => "Monopoly",
        })
    }
}

impl fmt::Display for DevCardHand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names = ["Kn", "VP", "RB", "YP", "Mo"];
        let mut first = true;
        for (i, &count) in self.0.iter().enumerate() {
            if count > 0 {
                if !first {
                    write!(f, " ")?;
                }
                write!(f, "{}x{}", count, names[i])?;
                first = false;
            }
        }
        if first {
            write!(f, "(none)")?;
        }
        Ok(())
    }
}

impl fmt::Display for Action {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Action::PlaceSettlement(v) => write!(f, "Settlement@v{}", v.0),
            Action::PlaceRoad(e) => write!(f, "Road@e{}", e.0),
            Action::BuildCity(v) => write!(f, "City@v{}", v.0),
            Action::BuyDevelopmentCard => write!(f, "BuyDevCard"),
            Action::PlayKnight => write!(f, "PlayKnight"),
            Action::PlayRoadBuilding => write!(f, "PlayRoadBuilding"),
            Action::PlayYearOfPlenty(r1, r2) => write!(f, "YoP({r1},{r2})"),
            Action::PlayMonopoly(r) => write!(f, "Monopoly({r})"),
            Action::RollDice => write!(f, "RollDice"),
            Action::MoveRobber(t) => write!(f, "Robber->t{}", t.0),
            Action::StealFrom(p) => write!(f, "Steal({p})"),
            Action::StealFromNone => write!(f, "StealNone"),
            Action::DiscardResource(r) => write!(f, "Discard({r})"),
            Action::BankTrade { give, receive } => write!(f, "Trade({give}->{receive})"),
            Action::EndTurn => write!(f, "EndTurn"),
        }
    }
}

impl fmt::Display for Phase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Phase::SetupSettlement { player, round } => {
                write!(f, "Setup R{round}: {player} place settlement")
            }
            Phase::SetupRoad { player, round, .. } => {
                write!(f, "Setup R{round}: {player} place road")
            }
            Phase::PreRoll => write!(f, "PreRoll"),
            Phase::ChanceRoll => write!(f, "Dice"),
            Phase::Discard { active, .. } => write!(f, "{active} must discard"),
            Phase::MoveRobber => write!(f, "Move robber"),
            Phase::Steal { .. } => write!(f, "Steal"),
            Phase::Main => write!(f, "Main"),
            Phase::RoadBuilding { roads_left } => write!(f, "RoadBuilding({roads_left} left)"),
            Phase::GameOver { winner } => write!(f, "Game Over: {winner} wins"),
        }
    }
}

impl fmt::Display for Player {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}  {}  ·  dev [{}]  ·  {} knights played",
            self.id, self.resources, self.dev_cards, self.played_knights,
        )
    }
}

fn board_terrain_word(t: Terrain) -> &'static str {
    match t {
        Terrain::Desert => "Desert",
        Terrain::Hills => "Hills",
        Terrain::Forest => "Forest",
        Terrain::Mountains => "Mtns",
        Terrain::Fields => "Fields",
        Terrain::Pasture => "Pasture",
    }
}

/// Text inside one bordered hex face (centered in `TILE_INNER_W` columns).
const TILE_INNER_W: usize = 12;
/// Horizontal gap (spaces) between adjacent tile boxes.
const TILE_GAP: usize = 2;

fn hex_tile_stride() -> usize {
    (2 + TILE_INNER_W) + TILE_GAP
}

/// Left offset of the first `┌` in each row (pointy-top honeycomb: rows of 4 nest between rows of 5).
fn hex_row_starts(stride: usize) -> [usize; 5] {
    let half = stride / 2;
    [stride, half, 0, half, stride]
}

/// Piece counts on one line, same spirit as `ResourceBank` / `DevCardHand` brackets.
fn player_piece_inline(game: &Game, pid: PlayerId) -> String {
    let mut settlements_on_board = 0u8;
    let mut cities_on_board = 0u8;
    for v in &game.board.vertices {
        match v {
            Vertex::Settlement(p) if *p == pid => settlements_on_board += 1,
            Vertex::City(p) if *p == pid => cities_on_board += 1,
            _ => {}
        }
    }
    let mut roads_on_board = 0u8;
    for e in &game.board.edges {
        if let Edge::Road(p) = e {
            if *p == pid {
                roads_on_board += 1;
            }
        }
    }
    let pl = &game.players[pid.idx()];
    format!(
        "on[S:{} C:{} R:{}] sup[S:{} C:{} R:{}]",
        settlements_on_board,
        cities_on_board,
        roads_on_board,
        pl.settlements_left,
        pl.cities_left,
        pl.roads_left,
    )
}

fn tile_face_text(tile: &Tile, robber_here: bool) -> String {
    if tile.terrain == Terrain::Desert {
        if robber_here {
            "Desert ●".to_string()
        } else {
            "Desert".to_string()
        }
    } else {
        let tail = if robber_here { "●" } else { "" };
        format!(
            "{} {:02}{}",
            board_terrain_word(tile.terrain),
            tile.number,
            tail
        )
    }
}

impl fmt::Display for Game {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let stride = hex_tile_stride();
        let row_start = hex_row_starts(stride);
        let board_graphic_w = 5 * stride - TILE_GAP;
        const BOARD_CONTENT_INDENT: usize = 4;
        let w = 72usize
            .max(board_graphic_w + BOARD_CONTENT_INDENT + 8)
            .max(board_graphic_w + 16);

        let rule = "─".repeat(w);
        let status = format!(
            "Turn {:>3}  ·  {}  ·  {}",
            self.turn_number,
            self.acting_player(),
            self.phase
        );
        writeln!(f, "╭{rule}╮")?;
        writeln!(f, "│{:^width$}│", status, width = w)?;
        writeln!(f, "╰{rule}╯")?;

        writeln!(f)?;
        writeln!(f, "  Board")?;
        writeln!(f, "  {}", "─".repeat(w.saturating_sub(2)))?;

        let center_slack = w.saturating_sub(BOARD_CONTENT_INDENT + board_graphic_w);
        let hex_origin_pad = BOARD_CONTENT_INDENT + center_slack / 2;

        const ROW_SIZES: [usize; 5] = [3, 4, 5, 4, 3];
        let mut t = 0;
        for (row, &size) in ROW_SIZES.iter().enumerate() {
            if row > 0 {
                writeln!(f)?;
            }
            let pad = format!(
                "{}{}",
                " ".repeat(hex_origin_pad),
                " ".repeat(row_start[row])
            );
            let gap = " ".repeat(TILE_GAP);
            let mut tops = String::new();
            let mut mids = String::new();
            let mut bots = String::new();
            for j in 0..size {
                if j > 0 {
                    tops.push_str(&gap);
                    mids.push_str(&gap);
                    bots.push_str(&gap);
                }
                let tile = &self.board.tiles[t + j];
                let robber_here = self.board.robber == TileId((t + j) as u8);
                let face = tile_face_text(tile, robber_here);
                tops.push('┌');
                tops.push_str(&"─".repeat(TILE_INNER_W));
                tops.push('┐');
                mids.push_str(&format!("│{:^inner$}│", face, inner = TILE_INNER_W));
                bots.push('└');
                bots.push_str(&"─".repeat(TILE_INNER_W));
                bots.push('┘');
            }
            writeln!(f, "{pad}{tops}")?;
            writeln!(f, "{pad}{mids}")?;
            writeln!(f, "{pad}{bots}")?;
            t += size;
        }

        writeln!(f)?;
        writeln!(f, "  Players")?;
        writeln!(f, "  {}", "─".repeat(w.saturating_sub(2)))?;
        for p in &self.players {
            let vp = self.victory_points(p.id);
            let mut awards = Vec::new();
            if self.longest_road_owner() == Some(p.id) {
                awards.push("longest road");
            }
            if self.largest_army_owner() == Some(p.id) {
                awards.push("largest army");
            }
            let award_str = if awards.is_empty() {
                "—".to_string()
            } else {
                awards.join(" · ")
            };
            writeln!(
                f,
                "    {:>3}   {:>2} vp   {:<22}  {}  ·  [{}]  ·  {}",
                p.id,
                vp,
                award_str,
                p.resources,
                p.dev_cards,
                player_piece_inline(self, p.id)
            )?;
        }

        writeln!(f)?;
        writeln!(
            f,
            "  Bank {}  ·  development deck: {} cards",
            self.board.bank,
            self.board.dev_card_deck.remaining()
        )?;
        Ok(())
    }
}
