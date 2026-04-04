use std::mem::{align_of, size_of};
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use catan_sim::board::{Edge, NUM_EDGES, NUM_PORTS, NUM_TILES, NUM_VERTICES, TOPO, Vertex};
use catan_sim::observation::{OBSERVATION_LEN, Observation};
use catan_sim::{ACTION_SPACE_SIZE, Action, ActionMask, Game, PlayerId, Turn};

// =========================================================================
// Action indexing
// =========================================================================

#[test]
fn action_index_roundtrip() {
    for i in 0..ACTION_SPACE_SIZE {
        let action = Action::from_index(i);
        assert_eq!(action.to_index(), i, "roundtrip failed at {i}: {action:?}");
    }
}

#[test]
fn action_mask_set_get_iter() {
    let mut m = ActionMask::EMPTY;
    assert_eq!(m.count(), 0);
    m.set(0);
    m.set(253);
    m.set(100);
    assert_eq!(m.count(), 3);
    assert!(m.get(0));
    assert!(m.get(100));
    assert!(m.get(253));
    assert!(!m.get(1));
    let items: Vec<_> = m.actions().collect();
    assert_eq!(
        items,
        vec![
            Action::from_index(0),
            Action::from_index(100),
            Action::from_index(253),
        ]
    );
}

// =========================================================================
// Observation layout
// =========================================================================

#[test]
fn observation_packed_layout() {
    assert_eq!(OBSERVATION_LEN, 386);
    assert_eq!(size_of::<Observation>(), OBSERVATION_LEN);
    assert_eq!(align_of::<Observation>(), 2);
}

#[test]
fn observe_as_bytes_all_perspectives() {
    let mut rng = StdRng::seed_from_u64(99);
    for _ in 0..32 {
        let game = Game::new(&mut rng);
        for &pid in &PlayerId::ALL {
            let obs = game.observe(pid);
            assert_eq!(obs.as_bytes().len(), OBSERVATION_LEN);
        }
    }
}

// =========================================================================
// Topology invariants
// =========================================================================

#[test]
fn topology_vertex_count() {
    let topo = &*TOPO;
    for t in 0..NUM_TILES {
        for &v in &topo.tile_vertices[t] {
            assert!((v as usize) < NUM_VERTICES);
        }
    }
}

#[test]
fn topology_edge_vertex_consistency() {
    let topo = &*TOPO;
    for e in 0..NUM_EDGES {
        let [v1, v2] = topo.edge_vertices[e];
        assert!((v1 as usize) < NUM_VERTICES);
        assert!((v2 as usize) < NUM_VERTICES);
        assert_ne!(v1, v2);
    }
}

#[test]
fn topology_vertex_neighbor_symmetry() {
    let topo = &*TOPO;
    for v in 0..NUM_VERTICES {
        for &adj in topo.vertex_neighbors(v) {
            assert!(
                topo.vertex_neighbors(adj as usize).contains(&(v as u8)),
                "v{v} -> v{adj} but not reverse"
            );
        }
    }
}

#[test]
fn topology_port_vertices_valid() {
    let topo = &*TOPO;
    for i in 0..NUM_PORTS {
        let [v1, v2] = topo.port_vertices[i];
        assert!((v1 as usize) < NUM_VERTICES);
        assert!((v2 as usize) < NUM_VERTICES);
        assert_eq!(
            topo.port_type(v1 as usize, &[catan_sim::board::Port::ThreeToOne; 9]),
            Some(catan_sim::board::Port::ThreeToOne)
        );
    }
}

// =========================================================================
// Setup flow
// =========================================================================

#[test]
fn setup_produces_8_settlements_8_roads() {
    let mut rng = StdRng::seed_from_u64(42);
    let mut game = Game::new(&mut rng);

    while !matches!(game.phase, catan_sim::Phase::PreRoll) {
        match game.turn() {
            Turn::Player(turn) => {
                let actions: Vec<_> = turn.mask.actions().collect();
                turn.apply(actions[0], &mut rng).unwrap();
            }
            _ => panic!("unexpected phase during setup"),
        }
    }

    let settlements = game
        .board
        .vertices
        .iter()
        .filter(|v| matches!(v, Vertex::Settlement(_)))
        .count();
    let roads = game
        .board
        .edges
        .iter()
        .filter(|e| matches!(e, Edge::Road(_)))
        .count();
    assert_eq!(settlements, 8, "each player places 2 settlements");
    assert_eq!(roads, 8, "each player places 2 roads");
}

#[test]
fn setup_round2_gives_resources() {
    let mut rng = StdRng::seed_from_u64(123);
    let mut game = Game::new(&mut rng);

    while !matches!(game.phase, catan_sim::Phase::PreRoll) {
        match game.turn() {
            Turn::Player(turn) => {
                let actions: Vec<_> = turn.mask.actions().collect();
                turn.apply(actions[0], &mut rng).unwrap();
            }
            _ => panic!("unexpected phase during setup"),
        }
    }

    let total: u8 = game.players.iter().map(|p| p.resources.total()).sum();
    assert!(
        total > 0,
        "players should receive resources from round 2 settlements"
    );
}

// =========================================================================
// Discard rule
// =========================================================================

#[test]
fn discard_halves_hand() {
    let mut rng = StdRng::seed_from_u64(42);
    let mut game = Game::new(&mut rng);

    // Play through setup
    while !matches!(game.phase, catan_sim::Phase::PreRoll) {
        match game.turn() {
            Turn::Player(turn) => {
                let actions: Vec<_> = turn.mask.actions().collect();
                turn.apply(actions[0], &mut rng).unwrap();
            }
            _ => panic!(),
        }
    }

    // Give P0 10 resources, roll a 7
    game.players[0].resources = catan_sim::board::ResourceBank([2, 2, 2, 2, 2]);

    // Enter dice phase, resolve as 7
    match game.turn() {
        Turn::Player(turn) => turn.apply(Action::RollDice, &mut rng).unwrap(),
        _ => panic!(),
    };
    match game.turn() {
        Turn::Chance(chance) => chance.resolve(7),
        _ => panic!(),
    };

    // P0 should need to discard: floor(10/2) = 5, keeping ceil(10/2) = 5
    assert!(matches!(game.phase, catan_sim::Phase::Discard { .. }));

    // Discard until done
    while matches!(game.phase, catan_sim::Phase::Discard { .. }) {
        match game.turn() {
            Turn::Player(turn) => {
                let actions: Vec<_> = turn.mask.actions().collect();
                turn.apply(actions[0], &mut rng).unwrap();
            }
            _ => panic!(),
        }
    }

    assert_eq!(
        game.players[0].resources.total(),
        5,
        "should keep ceil(10/2)=5 cards"
    );
}

// =========================================================================
// End-to-end random playouts
// =========================================================================

#[test]
fn random_games_complete_without_panic() {
    for seed in 0..50 {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut game = Game::new(&mut rng);
        let mut steps = 0u32;
        loop {
            match game.turn() {
                Turn::Terminal => break,
                Turn::Chance(c) => c.resolve_random(&mut rng),
                Turn::Player(turn) => {
                    let actions: Vec<_> = turn.mask.actions().collect();
                    assert!(
                        !actions.is_empty(),
                        "seed={seed} step={steps}: no legal actions"
                    );
                    let idx = rng.gen_range(0..actions.len());
                    turn.apply(actions[idx], &mut rng).unwrap();
                    steps += 1;
                    if steps > 10_000 {
                        break;
                    }
                }
            }
        }
    }
}

#[test]
fn action_mask_matches_legal_actions() {
    let mut rng = StdRng::seed_from_u64(77);
    let mut game = Game::new(&mut rng);
    let mut steps = 0;
    loop {
        match game.turn() {
            Turn::Terminal => break,
            Turn::Chance(c) => c.resolve_random(&mut rng),
            Turn::Player(turn) => {
                let mask = turn.mask;
                let actions: Vec<_> = mask.actions().collect();
                for &a in &actions {
                    assert!(mask.get(a.to_index()), "{a:?} in actions but not mask");
                }
                assert_eq!(mask.count() as usize, actions.len());
                let idx = rng.gen_range(0..actions.len());
                turn.apply(actions[idx], &mut rng).unwrap();
                steps += 1;
                if steps > 2000 {
                    break;
                }
            }
        }
    }
}

#[test]
fn observation_stable_across_perspectives() {
    let mut rng = StdRng::seed_from_u64(42);
    let mut game = Game::new(&mut rng);

    // Play 100 steps
    for _ in 0..100 {
        match game.turn() {
            Turn::Terminal => break,
            Turn::Chance(c) => c.resolve_random(&mut rng),
            Turn::Player(turn) => {
                let actions: Vec<_> = turn.mask.actions().collect();
                turn.apply(actions[rng.gen_range(0..actions.len())], &mut rng)
                    .unwrap();
            }
        }
    }

    // Observations from all perspectives should have same tile data
    let obs0 = game.observe(PlayerId::P0);
    let obs1 = game.observe(PlayerId::P1);
    for t in 0..NUM_TILES {
        assert_eq!(obs0.tiles[t].terrain, obs1.tiles[t].terrain);
        assert_eq!(obs0.tiles[t].number, obs1.tiles[t].number);
        assert_eq!(obs0.tiles[t].has_robber, obs1.tiles[t].has_robber);
    }
    assert_eq!(obs0.meta.resource_bank, obs1.meta.resource_bank);
}

#[test]
fn victory_points_consistent() {
    let mut rng = StdRng::seed_from_u64(42);
    let mut game = Game::new(&mut rng);
    loop {
        match game.turn() {
            Turn::Terminal => break,
            Turn::Chance(c) => c.resolve_random(&mut rng),
            Turn::Player(turn) => {
                let actions: Vec<_> = turn.mask.actions().collect();
                turn.apply(actions[rng.gen_range(0..actions.len())], &mut rng)
                    .unwrap();
            }
        }
    }

    if let Some(winner) = game.winner() {
        assert!(
            game.victory_points(winner) >= 10,
            "winner should have >= 10 VP"
        );
    }
}

// =========================================================================
// Stress / performance
// =========================================================================

#[test]
fn playout_throughput() {
    let num_games = 200;
    let mut rng = StdRng::seed_from_u64(12345);
    let start = Instant::now();
    let mut total_actions = 0u64;

    for _ in 0..num_games {
        let mut game = Game::new(&mut rng);
        let mut steps = 0u32;
        loop {
            match game.turn() {
                Turn::Terminal => break,
                Turn::Chance(c) => c.resolve_random(&mut rng),
                Turn::Player(turn) => {
                    let actions: Vec<_> = turn.mask.actions().collect();
                    if actions.is_empty() {
                        break;
                    }
                    turn.apply(actions[rng.gen_range(0..actions.len())], &mut rng)
                        .unwrap();
                    steps += 1;
                    if steps > 5000 {
                        break;
                    }
                }
            }
        }
        total_actions += steps as u64;
    }

    let elapsed = start.elapsed();
    let games_per_sec = num_games as f64 / elapsed.as_secs_f64();
    let actions_per_sec = total_actions as f64 / elapsed.as_secs_f64();
    eprintln!(
        "{num_games} games in {elapsed:.2?} ({games_per_sec:.0} games/sec, {actions_per_sec:.0} actions/sec, {:.0} avg actions/game)",
        total_actions as f64 / num_games as f64,
    );
    assert!(elapsed.as_secs() < 60, "took too long");
}

#[test]
fn clone_and_diverge() {
    let mut rng = StdRng::seed_from_u64(42);
    let mut game = Game::new(&mut rng);

    // Play into the main game
    for _ in 0..100 {
        match game.turn() {
            Turn::Terminal => return,
            Turn::Chance(c) => c.resolve_random(&mut rng),
            Turn::Player(turn) => {
                let actions: Vec<_> = turn.mask.actions().collect();
                turn.apply(actions[rng.gen_range(0..actions.len())], &mut rng)
                    .unwrap();
            }
        }
    }

    let mut game_a = game.clone();
    let mut game_b = game.clone();
    let mut rng_a = StdRng::seed_from_u64(100);
    let mut rng_b = StdRng::seed_from_u64(200);

    for _ in 0..500 {
        for (g, r) in [(&mut game_a, &mut rng_a), (&mut game_b, &mut rng_b)] {
            match g.turn() {
                Turn::Terminal => continue,
                Turn::Chance(c) => c.resolve_random(r),
                Turn::Player(turn) => {
                    let actions: Vec<_> = turn.mask.actions().collect();
                    turn.apply(actions[r.gen_range(0..actions.len())], r)
                        .unwrap();
                }
            }
        }
    }

    assert_ne!(
        game_a.board.bank.0, game_b.board.bank.0,
        "cloned games with different seeds should diverge"
    );
}
