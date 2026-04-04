use std::fmt;
use std::io::{self, IsTerminal};

use crate::board::{
    Board, DevCardHand, DevelopmentCard, Edge, Port, Resource, ResourceBank, Terrain, Tile, Vertex,
    NUM_EDGES, NUM_TILES, NUM_VERTICES, TOPO,
};
use crate::{Action, Game, Phase, Player, PlayerId, TileId};

const ANSI_RESET: &str = "\x1b[0m";
const ANSI_DIM: &str = "\x1b[90m";

fn color_enabled() -> bool {
    io::stdout().is_terminal()
}

/// Bright ANSI foreground for default four-player order (matches physical Catan).
fn player_fg(pid: PlayerId, color: bool) -> &'static str {
    if !color {
        return "";
    }
    match pid.0 {
        0 => "\x1b[91m", // bright red
        1 => "\x1b[92m", // bright green
        2 => "\x1b[93m", // bright yellow
        3 => "\x1b[94m", // bright blue
        _ => "\x1b[97m",
    }
}

// --- Pointy-top hex mesh from `TOPO.vertex_pos` (same plane as harbor math) ---

/// Horizontal spacing of the mesh (√3 factor still applied to layout `vx`).
const LAYOUT_SCALE_X: f64 = 2.65;
/// Vertical spacing (~0.57× X): shorter rows; edge chars still use true X/Y geometry.
const LAYOUT_SCALE_Y: f64 = 1.52;
/// √3 (layout x-scale matches `board` harbor angle math).
const SQRT_3: f64 = 1.732_050_807_568_877_2;

fn layout_to_cell(vx: i8, vy: i8) -> (i32, i32) {
    let px = vx as f64 * SQRT_3;
    let py = vy as f64;
    let x = (px * LAYOUT_SCALE_X).round() as i32;
    let y = (-py * LAYOUT_SCALE_Y).round() as i32;
    (x, y)
}

fn tile_center_cell(tile_idx: usize) -> (i32, i32) {
    let tv = TOPO.tile_vertices[tile_idx];
    let mut sx = 0i32;
    let mut sy = 0i32;
    for &vid in &tv {
        let (vx, vy) = TOPO.vertex_pos[vid as usize];
        let (cx, cy) = layout_to_cell(vx, vy);
        sx += cx;
        sy += cy;
    }
    (sx / 6, sy / 6)
}

fn canvas_bounds() -> (i32, i32, i32, i32) {
    let mut min_x = i32::MAX;
    let mut max_x = i32::MIN;
    let mut min_y = i32::MAX;
    let mut max_y = i32::MIN;
    for i in 0..NUM_VERTICES {
        let (vx, vy) = TOPO.vertex_pos[i];
        let (x, y) = layout_to_cell(vx, vy);
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }
    for t in 0..NUM_TILES {
        let (cx, cy) = tile_center_cell(t);
        min_x = min_x.min(cx - 2);
        max_x = max_x.max(cx + 2);
        min_y = min_y.min(cy - 1);
        max_y = max_y.max(cy + 1);
    }
    const M: i32 = 3;
    (min_x - M, max_x + M, min_y - M, max_y + M)
}

fn bresenham_line(x0: i32, y0: i32, x1: i32, y1: i32) -> Vec<(i32, i32)> {
    let mut pts = Vec::new();
    let mut x = x0;
    let mut y = y0;
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    loop {
        pts.push((x, y));
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
    pts
}

/// Single rune for an entire edge from layout endpoints (stable vs. Bresenham zig-zag).
fn world_edge_char(p1: (i8, i8), p2: (i8, i8)) -> char {
    let wx = (p2.0 - p1.0) as f64 * SQRT_3 * LAYOUT_SCALE_X;
    let wy = -(p2.1 - p1.1) as f64 * LAYOUT_SCALE_Y;
    if wx == 0.0 && wy == 0.0 {
        return '·';
    }
    let ang = wy.atan2(wx).to_degrees();
    if ang.abs() < 25.0 || ang.abs() > 155.0 {
        '─'
    } else if ang > 52.0 && ang < 128.0 {
        '│'
    } else if ang > 0.0 {
        '╲'
    } else {
        '╱'
    }
}

#[derive(Clone, Copy, Default)]
enum CellFg {
    #[default]
    None,
    Outline,
    Road(PlayerId),
    Building(PlayerId),
    Label,
}

#[derive(Clone)]
struct PlotCell {
    ch: char,
    fg: CellFg,
}

impl Default for PlotCell {
    fn default() -> Self {
        Self {
            ch: ' ',
            fg: CellFg::None,
        }
    }
}

fn fg_priority(fg: CellFg) -> u8 {
    match fg {
        CellFg::None => 0,
        CellFg::Outline => 1,
        CellFg::Label => 2,
        CellFg::Road(_) => 3,
        CellFg::Building(_) => 4,
    }
}

fn plot(buf: &mut [Vec<PlotCell>], ox: usize, oy: usize, ch: char, fg: CellFg) {
    let c = &mut buf[oy][ox];
    if c.ch == ' ' {
        c.ch = ch;
        c.fg = fg;
        return;
    }
    let p_new = fg_priority(fg);
    let p_old = fg_priority(c.fg);
    if p_new > p_old {
        c.ch = ch;
        c.fg = fg;
        return;
    }
    if p_new < p_old {
        return;
    }
    if c.ch == ch {
        if matches!(fg, CellFg::Road(_)) {
            c.fg = fg;
        }
        return;
    }
    c.ch = '+';
    if matches!(fg, CellFg::Road(_)) {
        c.fg = fg;
    }
}

fn terrain_abbr(t: Terrain) -> char {
    match t {
        Terrain::Desert => 'D',
        Terrain::Hills => 'H',
        Terrain::Forest => 'F',
        Terrain::Mountains => 'M',
        Terrain::Fields => 'G',
        Terrain::Pasture => 'P',
    }
}

fn tile_label_triple(tile: &Tile, robber_here: bool) -> [char; 3] {
    let a = terrain_abbr(tile.terrain);
    if tile.terrain == Terrain::Desert {
        let b = if robber_here { '●' } else { '·' };
        return [a, b, '·'];
    }
    let n = tile.number;
    let (b, c) = if n >= 10 {
        (
            char::from_digit((n / 10) as u32, 10).unwrap(),
            char::from_digit((n % 10) as u32, 10).unwrap(),
        )
    } else {
        (' ', char::from_digit(n as u32, 10).unwrap())
    };
    [a, b, c]
}

fn format_canvas_row(row: &[PlotCell], use_color: bool) -> String {
    let mut s = String::with_capacity(row.len() * 8);
    for cell in row {
        if cell.ch == ' ' {
            s.push(' ');
            continue;
        }
        match cell.fg {
            CellFg::None => s.push(cell.ch),
            CellFg::Outline => {
                if use_color {
                    s.push_str(ANSI_DIM);
                }
                s.push(cell.ch);
                if use_color {
                    s.push_str(ANSI_RESET);
                }
            }
            CellFg::Label => s.push(cell.ch),
            CellFg::Road(pid) => {
                if use_color {
                    s.push_str(player_fg(pid, true));
                }
                s.push(cell.ch);
                if use_color {
                    s.push_str(ANSI_RESET);
                }
            }
            CellFg::Building(pid) => {
                if use_color {
                    s.push_str(player_fg(pid, true));
                }
                s.push(cell.ch);
                if use_color {
                    s.push_str(ANSI_RESET);
                }
            }
        }
    }
    s
}

fn hex_board_lines(board: &Board, use_color: bool) -> (Vec<String>, usize) {
    let (min_x, max_x, min_y, max_y) = canvas_bounds();
    let w = (max_x - min_x + 1) as usize;
    let h = (max_y - min_y + 1) as usize;
    let ox = |x: i32| -> usize { (x - min_x) as usize };
    let oy = |y: i32| -> usize { (y - min_y) as usize };

    let mut buf = vec![vec![PlotCell::default(); w]; h];

    for eid in 0..NUM_EDGES {
        let [v1, v2] = TOPO.edge_vertices[eid];
        let p_a = TOPO.vertex_pos[v1 as usize];
        let p_b = TOPO.vertex_pos[v2 as usize];
        let (x0, y0) = layout_to_cell(p_a.0, p_a.1);
        let (x1, y1) = layout_to_cell(p_b.0, p_b.1);
        let pts = bresenham_line(x0, y0, x1, y1);
        let fg = match board.edges[eid] {
            Edge::Empty => CellFg::Outline,
            Edge::Road(p) => CellFg::Road(p),
        };
        let ch = world_edge_char(p_a, p_b);
        if pts.len() >= 3 {
            for &(px, py) in &pts[1..pts.len() - 1] {
                plot(&mut buf, ox(px), oy(py), ch, fg);
            }
        } else if pts.len() == 2 {
            let (px, py) = pts[0];
            plot(&mut buf, ox(px), oy(py), ch, fg);
        }
    }

    for vid in 0..NUM_VERTICES {
        match board.vertices[vid] {
            Vertex::Empty => {}
            Vertex::Settlement(p) => {
                let (vx, vy) = TOPO.vertex_pos[vid];
                let (cx, cy) = layout_to_cell(vx, vy);
                plot(&mut buf, ox(cx), oy(cy), 's', CellFg::Building(p));
            }
            Vertex::City(p) => {
                let (vx, vy) = TOPO.vertex_pos[vid];
                let (cx, cy) = layout_to_cell(vx, vy);
                plot(&mut buf, ox(cx), oy(cy), 'C', CellFg::Building(p));
            }
        }
    }

    for t in 0..NUM_TILES {
        let (cx, cy) = tile_center_cell(t);
        let robber = board.robber == TileId(t as u8);
        let trip = tile_label_triple(&board.tiles[t], robber);
        let base_x = cx - 1;
        for (k, &ch) in trip.iter().enumerate() {
            let xi = ox(base_x + k as i32);
            let yi = oy(cy);
            if xi < w && yi < h {
                let c = &mut buf[yi][xi];
                if c.ch == ' ' || matches!(c.fg, CellFg::Outline) {
                    c.ch = ch;
                    c.fg = CellFg::Label;
                }
            }
        }
    }

    let lines = (0..h)
        .map(|y| format_canvas_row(&buf[y], use_color))
        .collect::<Vec<_>>();
    (lines, w)
}

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
        if let Edge::Road(p) = e
            && *p == pid
        {
            roads_on_board += 1;
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

impl fmt::Display for Game {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let use_color = color_enabled();
        let (hex_lines, hex_w) = hex_board_lines(&self.board, use_color);
        const BOARD_CONTENT_INDENT: usize = 4;
        let w = 72usize
            .max(hex_w + BOARD_CONTENT_INDENT + 8)
            .max(hex_w + 16);

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

        let center_slack = w.saturating_sub(BOARD_CONTENT_INDENT + hex_w);
        let hex_origin_pad = BOARD_CONTENT_INDENT + center_slack / 2;
        let pad = " ".repeat(hex_origin_pad);
        for line in &hex_lines {
            writeln!(f, "{pad}{line}")?;
        }

        if use_color {
            writeln!(
                f,
                "  Dim ╱─│╲ = coast; colored = road by player; s/C = vertex; letters = tile (terrain + number)"
            )?;
            writeln!(f)?;
        }

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
            write!(f, "    ")?;
            if use_color {
                write!(f, "{}", player_fg(p.id, true))?;
            }
            write!(f, "{}", p.id)?;
            if use_color {
                write!(f, "{}", ANSI_RESET)?;
            }
            writeln!(
                f,
                "   {:>2} vp   {:<22}  {}  ·  [{}]  ·  {}",
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
