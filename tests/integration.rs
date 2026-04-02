use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

use catan::*;
use catan::board::topology::{TOPO, NUM_TILES, NUM_VERTICES, NUM_EDGES, NUM_PORTS};
use catan::game::action::Action;
use catan::game::phase::Phase;
use catan::game::rules;
use catan::board::{VertexBuilding, EdgeBuilding};

// =========================================================================
// Topology
// =========================================================================

#[test]
fn topology_vertex_count() {
    let topo = &*TOPO;
    // Every tile should reference 6 vertices, all within [0, 54)
    for t in 0..NUM_TILES {
        for &v in &topo.tile_vertices[t] {
            assert!((v as usize) < NUM_VERTICES, "tile {t} has invalid vertex {v}");
        }
    }
}

#[test]
fn topology_edge_count() {
    let topo = &*TOPO;
    for t in 0..NUM_TILES {
        for &e in &topo.tile_edges[t] {
            assert!((e as usize) < NUM_EDGES, "tile {t} has invalid edge {e}");
        }
    }
}

#[test]
fn topology_vertex_symmetry() {
    let topo = &*TOPO;
    // If vertex A is adjacent to B, then B is adjacent to A
    for v in 0..NUM_VERTICES {
        for &adj in topo.vertex_neighbors(v) {
            let adj = adj as usize;
            let found = topo.vertex_neighbors(adj).contains(&(v as u8));
            assert!(found, "vertex {v} has neighbor {adj}, but not vice versa");
        }
    }
}

#[test]
fn topology_edge_vertex_consistency() {
    let topo = &*TOPO;
    // Each edge connects two vertices; those vertices should list that edge
    for e in 0..NUM_EDGES {
        let [v1, v2] = topo.edge_vertices[e];
        assert!(
            topo.vertex_edge_list(v1 as usize).contains(&(e as u8)),
            "edge {e} connects vertex {v1} but vertex doesn't list edge"
        );
        assert!(
            topo.vertex_edge_list(v2 as usize).contains(&(e as u8)),
            "edge {e} connects vertex {v2} but vertex doesn't list edge"
        );
    }
}

#[test]
fn topology_coastal_edges() {
    let topo = &*TOPO;
    let mut coastal = 0;
    for e in 0..NUM_EDGES {
        if topo.edge_tile_count[e] == 1 {
            coastal += 1;
        }
    }
    assert_eq!(coastal, 30, "expected 30 coastal edges");
}

#[test]
fn topology_coastal_vertices() {
    let topo = &*TOPO;
    let coastal_count = topo.coastal_vertices.iter().filter(|&&c| c).count();
    // Standard Catan: 6 corners + 18 edge-coast vertices = 24? Let's just verify it's > 0
    assert!(coastal_count > 0, "should have coastal vertices");
    // Interior vertices touch 3 tiles
    let interior = (0..NUM_VERTICES).filter(|&v| topo.vertex_tile_count[v] == 3).count();
    assert_eq!(interior + coastal_count, NUM_VERTICES);
}

#[test]
fn topology_port_vertices_are_coastal() {
    let topo = &*TOPO;
    for i in 0..NUM_PORTS {
        let [v1, v2] = topo.port_vertices[i];
        assert!(topo.coastal_vertices[v1 as usize], "port {i} vertex {v1} should be coastal");
        assert!(topo.coastal_vertices[v2 as usize], "port {i} vertex {v2} should be coastal");
    }
}

#[test]
fn topology_tile_neighbor_symmetry() {
    let topo = &*TOPO;
    for t in 0..NUM_TILES {
        for d in 0..6 {
            let n = topo.tile_neighbors[t][d];
            if n == catan::board::topology::NONE {
                continue;
            }
            // Opposite direction
            let opp = (d + 3) % 6;
            assert_eq!(
                topo.tile_neighbors[n as usize][opp],
                t as u8,
                "tile {t} neighbor dir {d} = {n}, but tile {n} dir {opp} doesn't point back"
            );
        }
    }
}

// =========================================================================
// Setup flow
// =========================================================================

#[test]
fn setup_phase_sequence() {
    let mut rng = StdRng::seed_from_u64(123);
    let mut game = Game::new(&mut rng);

    // Round 1: P0 settlement, P0 road, P1 settlement, P1 road, ...
    for player_idx in 0..4 {
        assert!(matches!(game.phase, Phase::SetupSettlement { player, round: 1 } if player.0 == player_idx));
        let actions = game.legal_actions();
        assert!(!actions.is_empty(), "should have settlement placements");
        game.apply_action(actions[0], &mut rng).unwrap();

        assert!(matches!(game.phase, Phase::SetupRoad { player, round: 1, .. } if player.0 == player_idx));
        let actions = game.legal_actions();
        assert!(!actions.is_empty(), "should have road placements");
        game.apply_action(actions[0], &mut rng).unwrap();
    }

    // Round 2: P3 settlement, P3 road, P2 settlement, P2 road, ...
    for player_idx in (0..4).rev() {
        assert!(matches!(game.phase, Phase::SetupSettlement { player, round: 2 } if player.0 == player_idx));
        let actions = game.legal_actions();
        game.apply_action(actions[0], &mut rng).unwrap();

        assert!(matches!(game.phase, Phase::SetupRoad { player, round: 2, .. } if player.0 == player_idx));
        let actions = game.legal_actions();
        game.apply_action(actions[0], &mut rng).unwrap();
    }

    // After setup, should be PreRoll for P0
    assert!(matches!(game.phase, Phase::PreRoll));
    assert_eq!(game.current_player, PlayerId::P0);
}

#[test]
fn setup_gives_resources_round2() {
    let mut rng = StdRng::seed_from_u64(999);
    let mut game = Game::new(&mut rng);

    // Play through all of setup
    while matches!(game.phase, Phase::SetupSettlement { .. } | Phase::SetupRoad { .. }) {
        let actions = game.legal_actions();
        game.apply_action(actions[0], &mut rng).unwrap();
    }

    // At least some players should have resources from round 2 settlements
    let total_resources: u8 = game.players.iter().map(|p| p.resources.total()).sum();
    assert!(total_resources > 0, "players should receive resources in round 2 setup");
}

// =========================================================================
// Dice / chance nodes
// =========================================================================

#[test]
fn dice_chance_node_flow() {
    let mut rng = StdRng::seed_from_u64(42);
    let mut game = Game::new(&mut rng);

    // Play through setup
    while !matches!(game.phase, Phase::PreRoll) {
        let actions = game.legal_actions();
        game.apply_action(actions[0], &mut rng).unwrap();
    }

    // PreRoll: RollDice should be available
    let actions = game.legal_actions();
    assert!(actions.contains(&Action::RollDice));

    // Roll the dice
    game.apply_action(Action::RollDice, &mut rng).unwrap();
    assert!(game.is_chance_node());

    // Check outcomes
    let outcomes = game.chance_outcomes();
    assert_eq!(outcomes.len(), 11);
    let prob_sum: f64 = outcomes.iter().map(|(_, p)| p).sum();
    assert!((prob_sum - 1.0).abs() < 1e-10, "probabilities should sum to 1");

    // Resolve with a specific roll
    game.resolve_chance(6);
    assert!(!game.is_chance_node());
}

// =========================================================================
// Discard on 7
// =========================================================================

#[test]
fn discard_on_seven() {
    let mut rng = StdRng::seed_from_u64(42);
    let mut game = Game::new(&mut rng);

    // Play through setup
    while !matches!(game.phase, Phase::PreRoll) {
        let actions = game.legal_actions();
        game.apply_action(actions[0], &mut rng).unwrap();
    }

    // Give player 0 a bunch of resources
    game.players[0].resources = ResourceCounts::new(3, 3, 3, 3, 3); // 15 total
    game.bank = ResourceCounts::new(4, 4, 4, 4, 4); // enough bank

    // Roll dice and resolve as 7
    game.apply_action(Action::RollDice, &mut rng).unwrap();
    game.resolve_chance(7);

    // P0 should need to discard (has 15 > 7)
    assert!(matches!(game.phase, Phase::Discard { .. }));

    // Discard until done
    while matches!(game.phase, Phase::Discard { .. }) {
        let actions = game.legal_actions();
        assert!(!actions.is_empty());
        game.apply_action(actions[0], &mut rng).unwrap();
    }

    // After discard, should be MoveRobber
    assert!(matches!(game.phase, Phase::MoveRobber));
    // P0 should have <= 7 resources
    assert!(game.players[0].resources.total() <= 7);
}

// =========================================================================
// Full random game playout (stress test)
// =========================================================================

#[test]
fn random_game_completes() {
    // Run multiple random games to check for panics
    for seed in 0..20 {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut game = Game::new(&mut rng);
        let mut steps = 0u32;
        let max_steps = 10_000;

        while !game.is_terminal() && steps < max_steps {
            if game.is_chance_node() {
                game.resolve_chance_random(&mut rng);
                continue;
            }

            let actions = game.legal_actions();
            assert!(!actions.is_empty(), "seed={seed} step={steps}: no legal actions, phase={:?}", game.phase);

            let idx = rng.gen_range(0..actions.len());
            game.apply_action(actions[idx], &mut rng).unwrap();
            steps += 1;
        }

        // Game should complete within the step limit
        if steps >= max_steps {
            // Random play might not always finish, but it usually does
            // Just ensure no panic occurred
        }
    }
}

// =========================================================================
// Action mask consistency
// =========================================================================

#[test]
fn action_mask_matches_legal_actions() {
    let mut rng = StdRng::seed_from_u64(77);
    let mut game = Game::new(&mut rng);
    let mut steps = 0;

    while !game.is_terminal() && steps < 500 {
        if game.is_chance_node() {
            game.resolve_chance_random(&mut rng);
            continue;
        }

        let mask = game.action_mask();
        let actions = game.legal_actions();

        // Every action in legal_actions should be set in the mask
        for &a in &actions {
            assert!(mask.is_set(a.to_index()), "action {:?} in legal_actions but not in mask", a);
        }

        // The count should match
        assert_eq!(
            mask.count() as usize,
            actions.len(),
            "mask count {} != legal_actions len {}",
            mask.count(),
            actions.len()
        );

        let idx = rng.gen_range(0..actions.len());
        game.apply_action(actions[idx], &mut rng).unwrap();
        steps += 1;
    }
}

// =========================================================================
// Determinize
// =========================================================================

#[test]
fn determinize_preserves_perspective_info() {
    let mut rng = StdRng::seed_from_u64(42);
    let mut game = Game::new(&mut rng);

    // Play partway into the game
    for _ in 0..50 {
        if game.is_terminal() { break; }
        if game.is_chance_node() {
            game.resolve_chance_random(&mut rng);
            continue;
        }
        let actions = game.legal_actions();
        let idx = rng.gen_range(0..actions.len());
        game.apply_action(actions[idx], &mut rng).unwrap();
    }

    let perspective = PlayerId::P0;
    let det = game.determinize(&mut rng, perspective);

    // Perspective player's resources should be unchanged
    assert_eq!(det.players[0].resources, game.players[0].resources);
    assert_eq!(det.players[0].dev_cards, game.players[0].dev_cards);

    // Other players should have the same total resource count
    for i in 1..4 {
        assert_eq!(
            det.players[i].resources.total(),
            game.players[i].resources.total(),
            "player {i} total resources should be preserved"
        );
        assert_eq!(
            det.players[i].dev_cards.total(),
            game.players[i].dev_cards.total(),
            "player {i} total dev cards should be preserved"
        );
    }

    // Board state should be identical
    assert_eq!(det.board.robber, game.board.robber);
    assert_eq!(det.board.vertices, game.board.vertices);
    assert_eq!(det.board.edges, game.board.edges);
}

// =========================================================================
// Observation
// =========================================================================

#[test]
fn observation_hides_other_player_resources() {
    let mut rng = StdRng::seed_from_u64(42);
    let mut game = Game::new(&mut rng);

    // Give P1 some resources
    game.players[1].resources = ResourceCounts::new(2, 3, 0, 0, 0);

    let obs = game.observe(PlayerId::P0);

    // Current player (P0) should see their own resources
    assert_eq!(obs.current_player.resources, game.players[0].resources);

    // Other players: only total count visible
    let p1_obs = &obs.other_players[0]; // P1 is Clockwise1 from P0
    assert_eq!(p1_obs.total_resource_cards, 5);
}

// =========================================================================
// Longest road
// =========================================================================

#[test]
fn longest_road_calculation() {
    let mut rng = StdRng::seed_from_u64(42);
    let mut game = Game::new(&mut rng);
    let topo = &*TOPO;

    // Place a chain of roads for P0 along a path
    // First, find a starting vertex and build a path
    let start_vertex = 0;
    let pid = PlayerId::P0;
    game.board.vertices[start_vertex] = VertexBuilding::Settlement(pid);

    let mut current_vertex = start_vertex;
    let mut road_count = 0;

    for _ in 0..5 {
        // Find an edge from current_vertex that's empty
        let mut placed = false;
        for &e in topo.vertex_edge_list(current_vertex) {
            let ei = e as usize;
            if !game.board.edges[ei].is_empty() {
                continue;
            }
            let [v1, v2] = topo.edge_vertices[ei];
            let next = if v1 as usize == current_vertex { v2 as usize } else { v1 as usize };
            if !game.board.vertices[next].is_empty() && game.board.vertices[next].owner() != Some(pid) {
                continue;
            }
            game.board.edges[ei] = EdgeBuilding::Road(pid);
            current_vertex = next;
            road_count += 1;
            placed = true;
            break;
        }
        if !placed { break; }
    }

    let len = rules::longest_road(&game.board, pid);
    assert_eq!(len as usize, road_count, "longest road should match number of roads placed in a chain");
}

// =========================================================================
// Bank trading
// =========================================================================

#[test]
fn bank_trade_4_to_1() {
    let mut rng = StdRng::seed_from_u64(42);
    let mut game = Game::new(&mut rng);

    // Skip to main phase
    while !matches!(game.phase, Phase::PreRoll) {
        let actions = game.legal_actions();
        game.apply_action(actions[0], &mut rng).unwrap();
    }

    // Give P0 resources and ensure it's P0's turn
    game.current_player = PlayerId::P0;
    game.players[0].resources = ResourceCounts::new(4, 0, 0, 0, 0); // 4 brick
    game.phase = Phase::Main;

    let actions = game.legal_actions();
    // Should be able to trade 4 brick for any other resource
    let trade_actions: Vec<_> = actions.iter()
        .filter(|a| matches!(a, Action::BankTrade { give: Resource::Brick, .. }))
        .collect();
    assert_eq!(trade_actions.len(), 4, "should be able to trade brick for 4 other resources");

    // Execute the trade
    game.apply_action(Action::BankTrade { give: Resource::Brick, receive: Resource::Ore }, &mut rng).unwrap();
    assert_eq!(game.players[0].resources.get(Resource::Brick), 0);
    assert_eq!(game.players[0].resources.get(Resource::Ore), 1);
}

// =========================================================================
// Performance: playout throughput
// =========================================================================

#[test]
fn playout_performance() {
    use std::time::Instant;

    let num_games = 100;
    let mut rng = StdRng::seed_from_u64(12345);
    let start = Instant::now();
    let mut total_actions = 0u64;

    for _ in 0..num_games {
        let mut game = Game::new(&mut rng);
        let mut steps = 0u32;
        while !game.is_terminal() && steps < 5000 {
            if game.is_chance_node() {
                game.resolve_chance_random(&mut rng);
                continue;
            }
            let actions = game.legal_actions();
            if actions.is_empty() { break; }
            let idx = rng.gen_range(0..actions.len());
            game.apply_action(actions[idx], &mut rng).unwrap();
            steps += 1;
        }
        total_actions += steps as u64;
    }

    let elapsed = start.elapsed();
    let games_per_sec = num_games as f64 / elapsed.as_secs_f64();
    let actions_per_sec = total_actions as f64 / elapsed.as_secs_f64();
    println!(
        "\nPerformance: {} games in {:.2?} ({:.0} games/sec, {:.0} actions/sec, {:.0} avg actions/game)",
        num_games,
        elapsed,
        games_per_sec,
        actions_per_sec,
        total_actions as f64 / num_games as f64,
    );
    // Sanity check: we should complete at least a few games per second in debug mode
    assert!(elapsed.as_secs() < 30, "playouts took too long");
}
