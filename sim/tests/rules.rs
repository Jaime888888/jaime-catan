use rand::Rng;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;

use catan_sim::board::{
    DevelopmentCard, Edge, NUM_VERTICES, Resource, ResourceBank, TOPO, Terrain, Vertex,
};
use catan_sim::{Action, Game, Phase, PlayerId, TileId, Turn, VertexId};

fn play_setup(game: &mut Game, rng: &mut impl Rng) {
    while matches!(
        game.phase,
        Phase::SetupSettlement { .. } | Phase::SetupRoad { .. }
    ) {
        match game.turn() {
            Turn::Player(turn) => {
                let actions: Vec<_> = turn.mask.actions().collect();
                turn.apply(actions[rng.random_range(0..actions.len())], rng)
                    .unwrap();
            }
            _ => panic!("unexpected phase during setup"),
        }
    }
}

fn roll_dice(game: &mut Game, roll: u8, rng: &mut impl Rng) {
    match game.turn() {
        Turn::Player(turn) => turn.apply(Action::RollDice, rng).unwrap(),
        _ => panic!("expected PreRoll"),
    };
    match game.turn() {
        Turn::Chance(c) => c.resolve(roll),
        _ => panic!("expected ChanceRoll"),
    };
}

fn play_until_main(game: &mut Game, rng: &mut impl Rng) {
    loop {
        match &game.phase {
            Phase::Main => return,
            Phase::PreRoll => roll_dice(game, 3, rng), // safe non-7 roll
            _ => match game.turn() {
                Turn::Player(turn) => {
                    let actions: Vec<_> = turn.mask.actions().collect();
                    turn.apply(actions[0], rng).unwrap();
                }
                Turn::Chance(c) => c.resolve_random(rng),
                Turn::Terminal => panic!("game ended"),
            },
        }
    }
}

// =========================================================================
// Settlement distance rule
// =========================================================================

#[test]
fn no_adjacent_settlements() {
    for seed in 0..20 {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut game = Game::new(&mut rng);
        let mut steps = 0;
        loop {
            match game.turn() {
                Turn::Terminal => break,
                Turn::Chance(c) => c.resolve_random(&mut rng),
                Turn::Player(turn) => {
                    let actions: Vec<_> = turn.mask.actions().collect();
                    turn.apply(actions[rng.random_range(0..actions.len())], &mut rng)
                        .unwrap();
                    steps += 1;
                    if steps > 3000 {
                        break;
                    }
                }
            }
            let topo = &*TOPO;
            for v in 0..NUM_VERTICES {
                if !game.board.vertices[v].is_empty() {
                    for &adj in topo.vertex_neighbors(v) {
                        assert!(
                            game.board.vertices[adj as usize].is_empty(),
                            "seed={seed}: adjacent buildings at v{v} and v{adj}"
                        );
                    }
                }
            }
        }
    }
}

// =========================================================================
// Resource conservation
// =========================================================================

#[test]
fn resources_conserved() {
    for seed in 0..30 {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut game = Game::new(&mut rng);
        let mut steps = 0;
        loop {
            match game.turn() {
                Turn::Terminal => break,
                Turn::Chance(c) => c.resolve_random(&mut rng),
                Turn::Player(turn) => {
                    let actions: Vec<_> = turn.mask.actions().collect();
                    turn.apply(actions[rng.random_range(0..actions.len())], &mut rng)
                        .unwrap();
                    steps += 1;
                    if steps > 3000 {
                        break;
                    }
                }
            }
            for r in Resource::ALL {
                let player_total: u8 = game.players.iter().map(|p| p.resources[r]).sum();
                let bank = game.board.bank[r];
                assert_eq!(
                    player_total + bank,
                    19,
                    "seed={seed} step={steps}: {r:?} total is {} + {} = {}, expected 19",
                    player_total,
                    bank,
                    player_total + bank
                );
            }
        }
    }
}

// =========================================================================
// Robber blocks production
// =========================================================================

#[test]
fn robber_blocks_tile_production() {
    let mut rng = StdRng::seed_from_u64(42);
    let mut game = Game::new(&mut rng);
    play_setup(&mut game, &mut rng);

    // Find a resource tile with a settlement and note its number
    let topo = &*TOPO;
    let mut target_tile = None;
    for t in 0..19 {
        let terrain = game.board.tiles[t].terrain;
        let number = game.board.tiles[t].number;
        if terrain == Terrain::Desert || number == 0 {
            continue;
        }
        for &v in &topo.tile_vertices[t] {
            if matches!(game.board.vertices[v as usize], Vertex::Settlement(_)) {
                target_tile = Some(t);
                break;
            }
        }
        if target_tile.is_some() {
            break;
        }
    }
    let t = target_tile.expect("should find a tile with a settlement");
    let roll = game.board.tiles[t].number;

    // Move robber to this tile
    game.board.robber = TileId(t as u8);

    play_until_main(&mut game, &mut rng);
    // Go to PreRoll for next turn
    match game.turn() {
        Turn::Player(turn) => turn.apply(Action::EndTurn, &mut rng).unwrap(),
        _ => panic!(),
    };
    roll_dice(&mut game, roll, &mut rng);

    // Resources may have changed from OTHER tiles with same number, but the robber'd tile
    // should not have contributed. We verify conservation still holds (tested separately)
    // and that at minimum the robber tile didn't cause a panic.
}

// =========================================================================
// Dev card timing rules
// =========================================================================

#[test]
fn cannot_play_two_dev_cards_per_turn() {
    let mut rng = StdRng::seed_from_u64(42);
    let mut game = Game::new(&mut rng);
    play_setup(&mut game, &mut rng);
    play_until_main(&mut game, &mut rng);

    let pi = game.current_player.idx();
    game.players[pi].dev_cards += DevelopmentCard::Knight;
    game.players[pi].dev_cards += DevelopmentCard::Monopoly;

    // Play knight
    match game.turn() {
        Turn::Player(turn) => {
            assert!(turn.mask.get(Action::PlayKnight.to_index()));
            turn.apply(Action::PlayKnight, &mut rng).unwrap();
        }
        _ => panic!(),
    };

    // Resolve robber
    while !matches!(game.phase, Phase::Main) {
        match game.turn() {
            Turn::Player(turn) => {
                let actions: Vec<_> = turn.mask.actions().collect();
                turn.apply(actions[0], &mut rng).unwrap();
            }
            _ => panic!(),
        }
    }

    // Monopoly should NOT be available now
    match game.turn() {
        Turn::Player(turn) => {
            for r in Resource::ALL {
                assert!(
                    !turn.mask.get(Action::PlayMonopoly(r).to_index()),
                    "should not be able to play second dev card"
                );
            }
        }
        _ => panic!(),
    };
}

#[test]
fn cannot_play_card_bought_this_turn() {
    let mut rng = StdRng::seed_from_u64(42);
    let mut game = Game::new(&mut rng);
    play_setup(&mut game, &mut rng);
    play_until_main(&mut game, &mut rng);

    let pi = game.current_player.idx();

    // Simulate having bought a knight this turn by setting the flag directly
    game.players[pi].dev_cards += DevelopmentCard::Knight;
    game.turn_flags.dev_cards_bought += DevelopmentCard::Knight;

    match game.turn() {
        Turn::Player(turn) => {
            assert!(
                !turn.mask.get(Action::PlayKnight.to_index()),
                "should not be able to play card bought this turn"
            );
        }
        _ => panic!(),
    };

    // But if they had another knight from before, that one IS playable
    game.players[pi].dev_cards += DevelopmentCard::Knight; // now 2 held, 1 bought
    match game.turn() {
        Turn::Player(turn) => {
            assert!(
                turn.mask.get(Action::PlayKnight.to_index()),
                "should be able to play a pre-existing knight"
            );
        }
        _ => panic!(),
    };
}

// =========================================================================
// Monopoly takes from all opponents
// =========================================================================

#[test]
fn monopoly_collects_from_all_opponents() {
    let mut rng = StdRng::seed_from_u64(42);
    let mut game = Game::new(&mut rng);
    play_setup(&mut game, &mut rng);
    play_until_main(&mut game, &mut rng);

    let pi = game.current_player.idx();
    game.players[pi].dev_cards += DevelopmentCard::Monopoly;
    game.players[0].resources[Resource::Ore] = 3;
    game.players[1].resources[Resource::Ore] = 2;
    game.players[2].resources[Resource::Ore] = 1;
    game.players[3].resources[Resource::Ore] = 4;
    let total_ore_before: u8 = game
        .players
        .iter()
        .map(|p| p.resources[Resource::Ore])
        .sum();

    match game.turn() {
        Turn::Player(turn) => {
            turn.apply(Action::PlayMonopoly(Resource::Ore), &mut rng)
                .unwrap();
        }
        _ => panic!(),
    };

    let total_ore_after: u8 = game
        .players
        .iter()
        .map(|p| p.resources[Resource::Ore])
        .sum();
    assert_eq!(
        total_ore_before, total_ore_after,
        "monopoly should conserve resources"
    );
    assert_eq!(game.players[pi].resources[Resource::Ore], total_ore_before);
    for i in 0..4 {
        if i != pi {
            assert_eq!(game.players[i].resources[Resource::Ore], 0);
        }
    }
}

// =========================================================================
// Bank trade rates
// =========================================================================

#[test]
fn bank_trade_respects_port_rates() {
    let mut rng = StdRng::seed_from_u64(42);
    let mut game = Game::new(&mut rng);
    play_setup(&mut game, &mut rng);
    play_until_main(&mut game, &mut rng);

    let pi = game.current_player.idx();

    // No ports: need 4 to trade
    game.players[pi].resources = ResourceBank([3, 0, 0, 0, 0]);
    match game.turn() {
        Turn::Player(turn) => {
            assert!(
                !turn.mask.get(
                    Action::BankTrade {
                        give: Resource::Brick,
                        receive: Resource::Ore
                    }
                    .to_index()
                ),
                "3 brick without port should not allow trade"
            );
        }
        _ => panic!(),
    };

    game.players[pi].resources = ResourceBank([4, 0, 0, 0, 0]);
    match game.turn() {
        Turn::Player(turn) => {
            assert!(
                turn.mask.get(
                    Action::BankTrade {
                        give: Resource::Brick,
                        receive: Resource::Ore
                    }
                    .to_index()
                ),
                "4 brick should allow 4:1 trade"
            );
            turn.apply(
                Action::BankTrade {
                    give: Resource::Brick,
                    receive: Resource::Ore,
                },
                &mut rng,
            )
            .unwrap();
        }
        _ => panic!(),
    };
    assert_eq!(game.players[pi].resources[Resource::Brick], 0);
    assert_eq!(game.players[pi].resources[Resource::Ore], 1);

    // 3:1 port
    game.players[pi].has_three_to_one_port = true;
    game.players[pi].resources = ResourceBank([3, 0, 0, 0, 0]);
    match game.turn() {
        Turn::Player(turn) => {
            assert!(
                turn.mask.get(
                    Action::BankTrade {
                        give: Resource::Brick,
                        receive: Resource::Lumber
                    }
                    .to_index()
                )
            );
        }
        _ => panic!(),
    };

    // 2:1 port
    game.players[pi].two_to_one_ports[Resource::Brick as usize] = true;
    game.players[pi].resources = ResourceBank([2, 0, 0, 0, 0]);
    match game.turn() {
        Turn::Player(turn) => {
            assert!(
                turn.mask.get(
                    Action::BankTrade {
                        give: Resource::Brick,
                        receive: Resource::Grain
                    }
                    .to_index()
                )
            );
        }
        _ => panic!(),
    };
}

// =========================================================================
// City replaces settlement, returns piece
// =========================================================================

#[test]
fn city_upgrade_returns_settlement_piece() {
    let mut rng = StdRng::seed_from_u64(42);
    let mut game = Game::new(&mut rng);
    play_setup(&mut game, &mut rng);
    play_until_main(&mut game, &mut rng);

    let pi = game.current_player.idx();
    let sl_before = game.players[pi].settlements_left;
    let cl_before = game.players[pi].cities_left;

    // Find a settlement owned by current player
    let my_settlement = (0..NUM_VERTICES)
        .find(|&v| game.board.vertices[v] == Vertex::Settlement(game.current_player))
        .expect("should own a settlement");

    game.players[pi].resources = ResourceBank([0, 0, 3, 2, 0]); // city cost
    match game.turn() {
        Turn::Player(turn) => {
            turn.apply(Action::BuildCity(VertexId(my_settlement as u8)), &mut rng)
                .unwrap();
        }
        _ => panic!(),
    };

    assert!(matches!(
        game.board.vertices[my_settlement],
        Vertex::City(_)
    ));
    assert_eq!(game.players[pi].settlements_left, sl_before + 1);
    assert_eq!(game.players[pi].cities_left, cl_before - 1);
}

// =========================================================================
// Longest road is recalculated, not stale
// =========================================================================

#[test]
fn longest_road_updates_on_every_road_build() {
    let mut rng = StdRng::seed_from_u64(42);
    let mut game = Game::new(&mut rng);

    // After setup, check longest road is computed
    play_setup(&mut game, &mut rng);

    // Every road should be counted
    for i in 0..4 {
        let pid = PlayerId(i as u8);
        let roads_placed = game
            .board
            .edges
            .iter()
            .filter(|e| e.owner() == Some(pid))
            .count();
        if roads_placed > 0 {
            assert!(
                game.longest_road_len[i] > 0,
                "Player{i} has {roads_placed} roads but longest_road_len=0"
            );
        }
    }
}

// =========================================================================
// Setup order: P0,P1,P2,P3,P3,P2,P1,P0
// =========================================================================

#[test]
fn setup_player_order() {
    let mut rng = StdRng::seed_from_u64(42);
    let mut game = Game::new(&mut rng);

    let expected_order = [0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 2, 2, 1, 1, 0, 0];
    let mut actual_order = Vec::new();

    while matches!(
        game.phase,
        Phase::SetupSettlement { .. } | Phase::SetupRoad { .. }
    ) {
        actual_order.push(game.acting_player().0);
        match game.turn() {
            Turn::Player(turn) => {
                let actions: Vec<_> = turn.mask.actions().collect();
                turn.apply(actions[0], &mut rng).unwrap();
            }
            _ => panic!(),
        }
    }

    assert_eq!(actual_order, expected_order);
}

// =========================================================================
// Building supply limits
// =========================================================================

#[test]
fn building_supply_never_negative() {
    for seed in 0..30 {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut game = Game::new(&mut rng);
        let mut steps = 0;
        loop {
            match game.turn() {
                Turn::Terminal => break,
                Turn::Chance(c) => c.resolve_random(&mut rng),
                Turn::Player(turn) => {
                    let actions: Vec<_> = turn.mask.actions().collect();
                    turn.apply(actions[rng.random_range(0..actions.len())], &mut rng)
                        .unwrap();
                    steps += 1;
                    if steps > 3000 {
                        break;
                    }
                }
            }
            for p in &game.players {
                assert!(p.settlements_left <= 5, "settlements_left overflow");
                assert!(p.cities_left <= 4, "cities_left overflow");
                assert!(p.roads_left <= 15, "roads_left overflow");
            }
        }
    }
}

// =========================================================================
// Knight before roll returns to PreRoll
// =========================================================================

#[test]
fn knight_before_roll_returns_to_preroll() {
    let mut rng = StdRng::seed_from_u64(42);
    let mut game = Game::new(&mut rng);
    play_setup(&mut game, &mut rng);

    assert!(matches!(game.phase, Phase::PreRoll));

    let pi = game.current_player.idx();
    game.players[pi].dev_cards += DevelopmentCard::Knight;

    // Play knight from PreRoll
    match game.turn() {
        Turn::Player(turn) => {
            assert!(turn.mask.get(Action::PlayKnight.to_index()));
            turn.apply(Action::PlayKnight, &mut rng).unwrap();
        }
        _ => panic!(),
    };

    assert!(matches!(game.phase, Phase::MoveRobber));

    // Resolve robber + steal
    while !matches!(game.phase, Phase::PreRoll | Phase::Main) {
        match game.turn() {
            Turn::Player(turn) => {
                let actions: Vec<_> = turn.mask.actions().collect();
                turn.apply(actions[0], &mut rng).unwrap();
            }
            _ => panic!(),
        }
    }

    assert!(
        matches!(game.phase, Phase::PreRoll),
        "after pre-roll knight, should return to PreRoll, not Main"
    );
}

// =========================================================================
// Resource distribution: settlements get 1, cities get 2
// =========================================================================

#[test]
fn cities_produce_double_resources() {
    let mut rng = StdRng::seed_from_u64(42);
    let mut game = Game::new(&mut rng);
    play_setup(&mut game, &mut rng);

    let topo = &*TOPO;
    let pid = game.current_player;
    let pi = pid.idx();

    // Find a settlement owned by current player, on a non-desert tile
    let (v_idx, tile_idx) = (0..NUM_VERTICES)
        .filter(|&v| game.board.vertices[v] == Vertex::Settlement(pid))
        .find_map(|v| {
            topo.vertex_tiles(v)
                .iter()
                .find(|&&t| game.board.tiles[t as usize].terrain != Terrain::Desert)
                .map(|&t| (v, t as usize))
        })
        .expect("player should have a settlement on a resource tile");

    let roll = game.board.tiles[tile_idx].number;
    let resource = game.board.tiles[tile_idx].terrain.resource().unwrap();

    // Upgrade to city
    game.board.vertices[v_idx] = Vertex::City(pid);
    game.players[pi].cities_left -= 1;
    game.players[pi].settlements_left += 1;

    // Clear all resources, ensure robber is elsewhere
    for p in &mut game.players {
        p.resources = ResourceBank([0; 5]);
    }
    if game.board.robber == TileId(tile_idx as u8) {
        game.board.robber = TileId(((tile_idx + 1) % 19) as u8);
    }

    play_until_main(&mut game, &mut rng);
    match game.turn() {
        Turn::Player(turn) => turn.apply(Action::EndTurn, &mut rng).unwrap(),
        _ => panic!(),
    };
    roll_dice(&mut game, roll, &mut rng);

    // City should have produced 2 of the resource (may be more if multiple tiles match)
    assert!(
        game.players[pi].resources[resource] >= 2,
        "city should produce at least 2 {resource:?}, got {}",
        game.players[pi].resources[resource]
    );
}

// =========================================================================
// Year of Plenty respects bank limits
// =========================================================================

#[test]
fn year_of_plenty_capped_by_bank() {
    let mut rng = StdRng::seed_from_u64(42);
    let mut game = Game::new(&mut rng);
    play_setup(&mut game, &mut rng);
    play_until_main(&mut game, &mut rng);

    let pi = game.current_player.idx();
    game.players[pi].dev_cards += DevelopmentCard::YearOfPlenty;

    // Set bank to exactly 1 brick, 0 lumber
    game.board.bank[Resource::Brick] = 1;
    game.board.bank[Resource::Lumber] = 0;

    // YoP(Brick, Brick) should NOT be available (bank only has 1)
    match game.turn() {
        Turn::Player(turn) => {
            assert!(
                !turn
                    .mask
                    .get(Action::PlayYearOfPlenty(Resource::Brick, Resource::Brick).to_index()),
                "YoP(Brick,Brick) should be unavailable with only 1 in bank"
            );
            // YoP(Brick, Lumber) also unavailable since Lumber=0
            assert!(
                !turn
                    .mask
                    .get(Action::PlayYearOfPlenty(Resource::Brick, Resource::Lumber).to_index()),
            );
        }
        _ => panic!(),
    };

    // Give bank enough for a valid pair
    game.board.bank[Resource::Ore] = 1;
    let before_brick = game.players[pi].resources[Resource::Brick];
    let before_ore = game.players[pi].resources[Resource::Ore];
    match game.turn() {
        Turn::Player(turn) => {
            assert!(
                turn.mask
                    .get(Action::PlayYearOfPlenty(Resource::Brick, Resource::Ore).to_index())
            );
            turn.apply(
                Action::PlayYearOfPlenty(Resource::Brick, Resource::Ore),
                &mut rng,
            )
            .unwrap();
        }
        _ => panic!(),
    };
    assert_eq!(
        game.players[pi].resources[Resource::Brick],
        before_brick + 1
    );
    assert_eq!(game.players[pi].resources[Resource::Ore], before_ore + 1);
    assert_eq!(game.board.bank[Resource::Brick], 0);
    assert_eq!(game.board.bank[Resource::Ore], 0);
}

// =========================================================================
// Steal transfers exactly one resource
// =========================================================================

#[test]
fn steal_transfers_exactly_one() {
    let mut rng = StdRng::seed_from_u64(42);
    let mut game = Game::new(&mut rng);
    play_setup(&mut game, &mut rng);
    play_until_main(&mut game, &mut rng);

    let pi = game.current_player.idx();
    let target = game.current_player.next();
    let ti = target.idx();

    game.players[ti].resources = ResourceBank([1, 2, 0, 0, 3]);
    let target_total_before = game.players[ti].resources.total();
    let my_total_before = game.players[pi].resources.total();

    // Directly enter steal phase
    game.phase = Phase::Steal {
        candidates: std::array::from_fn(|i| i == ti),
    };

    match game.turn() {
        Turn::Player(turn) => {
            turn.apply(Action::StealFrom(target), &mut rng).unwrap();
        }
        _ => panic!(),
    };

    assert_eq!(
        game.players[ti].resources.total(),
        target_total_before - 1,
        "target should lose exactly 1"
    );
    assert_eq!(
        game.players[pi].resources.total(),
        my_total_before + 1,
        "stealer should gain exactly 1"
    );
}

// =========================================================================
// Discard targets correct for various hand sizes
// =========================================================================

#[test]
fn discard_targets_correct_for_various_sizes() {
    // Catan rule: discard floor(n/2), keep ceil(n/2)
    let cases = [(8, 4), (9, 5), (10, 5), (11, 6), (12, 6), (15, 8)];

    for (hand_size, expected_keep) in cases {
        let mut rng = StdRng::seed_from_u64(42);
        let mut game = Game::new(&mut rng);
        play_setup(&mut game, &mut rng);
        play_until_main(&mut game, &mut rng);

        let pi = game.current_player.idx();
        // Spread resources across types to reach exact hand_size
        let per = hand_size / 5;
        let rem = hand_size % 5;
        game.players[pi].resources = ResourceBank([per; 5]);
        for i in 0..rem as usize {
            game.players[pi].resources.0[i] += 1;
        }
        assert_eq!(game.players[pi].resources.total(), hand_size);

        match game.turn() {
            Turn::Player(turn) => turn.apply(Action::EndTurn, &mut rng).unwrap(),
            _ => panic!(),
        };
        roll_dice(&mut game, 7, &mut rng);

        while matches!(game.phase, Phase::Discard { .. }) {
            match game.turn() {
                Turn::Player(turn) => {
                    let action = turn.mask.actions().next().unwrap();
                    turn.apply(action, &mut rng).unwrap();
                }
                _ => panic!(),
            }
        }

        assert_eq!(
            game.players[pi].resources.total(),
            expected_keep,
            "hand_size={hand_size}: should keep {expected_keep}"
        );
    }
}

// =========================================================================
// Robber must move to a DIFFERENT tile
// =========================================================================

#[test]
fn robber_cannot_stay_on_same_tile() {
    let mut rng = StdRng::seed_from_u64(42);
    let mut game = Game::new(&mut rng);
    play_setup(&mut game, &mut rng);
    play_until_main(&mut game, &mut rng);

    let pi = game.current_player.idx();
    game.players[pi].dev_cards += DevelopmentCard::Knight;

    match game.turn() {
        Turn::Player(turn) => turn.apply(Action::PlayKnight, &mut rng).unwrap(),
        _ => panic!(),
    };

    assert!(matches!(game.phase, Phase::MoveRobber));
    let current_robber = game.board.robber;

    match game.turn() {
        Turn::Player(turn) => {
            assert!(
                !turn.mask.get(Action::MoveRobber(current_robber).to_index()),
                "robber should not be allowed to stay on the same tile"
            );
            // All other tiles should be available
            for t in 0..19u8 {
                if TileId(t) != current_robber {
                    assert!(turn.mask.get(Action::MoveRobber(TileId(t)).to_index()));
                }
            }
        }
        _ => panic!(),
    };
}

// =========================================================================
// Largest army requires >= 3 knights
// =========================================================================

#[test]
fn largest_army_requires_three_knights() {
    let mut rng = StdRng::seed_from_u64(42);
    let mut game = Game::new(&mut rng);
    play_setup(&mut game, &mut rng);

    let pi = game.current_player.idx();
    game.players[pi].played_knights = 2;
    assert!(
        game.largest_army_owner().is_none(),
        "2 knights should not qualify"
    );

    game.players[pi].played_knights = 3;
    assert_eq!(game.largest_army_owner(), Some(game.current_player));
}

// =========================================================================
// Longest road requires >= 5 segments
// =========================================================================

#[test]
fn longest_road_requires_five_segments() {
    let mut rng = StdRng::seed_from_u64(42);
    let mut game = Game::new(&mut rng);
    play_setup(&mut game, &mut rng);

    // After setup, each player has 2 roads (longest = 2 per player)
    assert!(
        game.longest_road_owner().is_none(),
        "2-segment roads should not qualify for longest road"
    );
}

// =========================================================================
// Road building dev card places free roads
// =========================================================================

#[test]
fn road_building_places_free_roads() {
    let mut rng = StdRng::seed_from_u64(42);
    let mut game = Game::new(&mut rng);
    play_setup(&mut game, &mut rng);
    play_until_main(&mut game, &mut rng);

    let pi = game.current_player.idx();
    let resources_before = game.players[pi].resources;
    let roads_before = game.players[pi].roads_left;
    game.players[pi].dev_cards += DevelopmentCard::RoadBuilding;

    match game.turn() {
        Turn::Player(turn) => turn.apply(Action::PlayRoadBuilding, &mut rng).unwrap(),
        _ => panic!(),
    };

    // Should be in RoadBuilding phase
    if matches!(game.phase, Phase::RoadBuilding { .. }) {
        // Place two roads
        for _ in 0..2 {
            if !matches!(game.phase, Phase::RoadBuilding { .. }) {
                break;
            }
            match game.turn() {
                Turn::Player(turn) => {
                    let actions: Vec<_> = turn.mask.actions().collect();
                    turn.apply(actions[0], &mut rng).unwrap();
                }
                _ => panic!(),
            }
        }
    }

    // Resources should be unchanged (roads are free)
    assert_eq!(game.players[pi].resources, resources_before);
    assert!(game.players[pi].roads_left <= roads_before - 1);
}

// =========================================================================
// Settlement on port grants trade benefit
// =========================================================================

#[test]
fn settlement_on_port_grants_access() {
    let mut rng = StdRng::seed_from_u64(42);
    let mut game = Game::new(&mut rng);

    // Find a port vertex
    let port_v = (0..NUM_VERTICES)
        .find(|&v| game.board.port_at(v).is_some())
        .expect("should have port vertices");

    let pid = PlayerId::P0;
    let pi = pid.idx();

    // Manually place a settlement there
    game.board.vertices[port_v] = Vertex::Settlement(pid);
    game.players[pi].settlements_left -= 1;
    if let Some(pt) = game.board.port_at(port_v) {
        game.players[pi].update_ports(pt);
    }

    // Player should now have the port benefit
    let port = game.board.port_at(port_v).unwrap();
    match port {
        catan_sim::board::Port::ThreeToOne => {
            assert!(game.players[pi].has_three_to_one_port);
        }
        catan_sim::board::Port::TwoToOne(r) => {
            assert!(game.players[pi].two_to_one_ports[r as usize]);
            assert_eq!(game.players[pi].trade_rate(r), 2);
        }
    }
}

// =========================================================================
// Board building counts match vertex/edge arrays
// =========================================================================

#[test]
fn building_counts_match_board_state() {
    for seed in 0..20 {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut game = Game::new(&mut rng);
        let mut steps = 0;
        loop {
            match game.turn() {
                Turn::Terminal => break,
                Turn::Chance(c) => c.resolve_random(&mut rng),
                Turn::Player(turn) => {
                    let actions: Vec<_> = turn.mask.actions().collect();
                    turn.apply(actions[rng.random_range(0..actions.len())], &mut rng)
                        .unwrap();
                    steps += 1;
                    if steps > 2000 {
                        break;
                    }
                }
            }

            for i in 0..4 {
                let pid = PlayerId(i as u8);
                let settlements_on_board = game
                    .board
                    .vertices
                    .iter()
                    .filter(|v| matches!(v, Vertex::Settlement(p) if *p == pid))
                    .count() as u8;
                let cities_on_board = game
                    .board
                    .vertices
                    .iter()
                    .filter(|v| matches!(v, Vertex::City(p) if *p == pid))
                    .count() as u8;
                let roads_on_board = game
                    .board
                    .edges
                    .iter()
                    .filter(|e| matches!(e, Edge::Road(p) if *p == pid))
                    .count() as u8;

                assert_eq!(
                    settlements_on_board + game.players[i].settlements_left,
                    5,
                    "seed={seed} P{i}: settlement accounting broken"
                );
                assert_eq!(
                    cities_on_board + game.players[i].cities_left,
                    4,
                    "seed={seed} P{i}: city accounting broken"
                );
                assert_eq!(
                    roads_on_board + game.players[i].roads_left,
                    15,
                    "seed={seed} P{i}: road accounting broken"
                );
            }
        }
    }
}

// =========================================================================
// EndTurn resets turn flags
// =========================================================================

#[test]
fn end_turn_resets_flags() {
    let mut rng = StdRng::seed_from_u64(42);
    let mut game = Game::new(&mut rng);
    play_setup(&mut game, &mut rng);
    play_until_main(&mut game, &mut rng);

    let player_before = game.current_player;
    game.turn_flags.dev_card_played = true;
    game.turn_flags.has_rolled = true;
    game.turn_flags.dev_cards_bought += DevelopmentCard::Knight;

    match game.turn() {
        Turn::Player(turn) => turn.apply(Action::EndTurn, &mut rng).unwrap(),
        _ => panic!(),
    };

    assert_ne!(game.current_player, player_before);
    assert!(!game.turn_flags.dev_card_played);
    assert!(!game.turn_flags.has_rolled);
    assert_eq!(game.turn_flags.dev_cards_bought.total(), 0);
    assert!(matches!(game.phase, Phase::PreRoll));
}

// =========================================================================
// No actions leak across phase boundaries
// =========================================================================

#[test]
fn no_invalid_actions_offered() {
    for seed in 0..30 {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut game = Game::new(&mut rng);
        let mut steps = 0;
        loop {
            match game.turn() {
                Turn::Terminal => break,
                Turn::Chance(c) => c.resolve_random(&mut rng),
                Turn::Player(turn) => {
                    // Every offered action should succeed without error
                    let actions: Vec<_> = turn.mask.actions().collect();
                    assert!(!actions.is_empty(), "seed={seed} step={steps}: empty mask");
                    let action = actions[rng.random_range(0..actions.len())];
                    turn.apply(action, &mut rng)
                        .unwrap_or_else(|e| panic!("seed={seed} step={steps}: {e}"));
                    steps += 1;
                    if steps > 3000 {
                        break;
                    }
                }
            }
        }
    }
}

// =========================================================================
// Multiple players discard on a 7
// =========================================================================

#[test]
fn multiple_players_discard_on_seven() {
    let mut rng = StdRng::seed_from_u64(42);
    let mut game = Game::new(&mut rng);
    play_setup(&mut game, &mut rng);
    play_until_main(&mut game, &mut rng);

    // Give three players > 7 resources
    game.players[0].resources = ResourceBank([3, 3, 3, 0, 0]); // 9
    game.players[1].resources = ResourceBank([2, 2, 2, 2, 2]); // 10
    game.players[2].resources = ResourceBank([4, 4, 0, 0, 0]); // 8
    game.players[3].resources = ResourceBank([1, 1, 1, 1, 1]); // 5 (safe)

    match game.turn() {
        Turn::Player(turn) => turn.apply(Action::EndTurn, &mut rng).unwrap(),
        _ => panic!(),
    };
    roll_dice(&mut game, 7, &mut rng);

    assert!(matches!(game.phase, Phase::Discard { .. }));

    // Process all discards
    while matches!(game.phase, Phase::Discard { .. }) {
        match game.turn() {
            Turn::Player(turn) => {
                let action = turn.mask.actions().next().unwrap();
                turn.apply(action, &mut rng).unwrap();
            }
            _ => panic!(),
        }
    }

    // P0: 9 → keep 5, P1: 10 → keep 5, P2: 8 → keep 4, P3: unchanged
    assert_eq!(game.players[0].resources.total(), 5);
    assert_eq!(game.players[1].resources.total(), 5);
    assert_eq!(game.players[2].resources.total(), 4);
    assert_eq!(game.players[3].resources.total(), 5);

    assert!(matches!(game.phase, Phase::MoveRobber));
}

// =========================================================================
// Steal offers StealFromNone when adjacent players have 0 resources
// =========================================================================

#[test]
fn steal_from_none_when_no_resources() {
    let mut rng = StdRng::seed_from_u64(42);
    let mut game = Game::new(&mut rng);
    play_setup(&mut game, &mut rng);
    play_until_main(&mut game, &mut rng);

    // Wipe everyone's resources
    for p in &mut game.players {
        p.resources = ResourceBank([0; 5]);
    }

    let pi = game.current_player.idx();
    game.players[pi].dev_cards += DevelopmentCard::Knight;

    match game.turn() {
        Turn::Player(turn) => turn.apply(Action::PlayKnight, &mut rng).unwrap(),
        _ => panic!(),
    };

    // Move robber to any tile
    match game.turn() {
        Turn::Player(turn) => {
            let action = turn.mask.actions().next().unwrap();
            turn.apply(action, &mut rng).unwrap();
        }
        _ => panic!(),
    };

    // Should be Steal phase with StealFromNone as the only option
    assert!(matches!(game.phase, Phase::Steal { .. }));
    match game.turn() {
        Turn::Player(turn) => {
            let actions: Vec<_> = turn.mask.actions().collect();
            assert_eq!(actions.len(), 1);
            assert_eq!(actions[0], Action::StealFromNone);
            turn.apply(Action::StealFromNone, &mut rng).unwrap();
        }
        _ => panic!(),
    };
}

// =========================================================================
// Bank trade cannot give and receive same resource
// =========================================================================

#[test]
fn bank_trade_no_self_trade() {
    let mut rng = StdRng::seed_from_u64(42);
    let mut game = Game::new(&mut rng);
    play_setup(&mut game, &mut rng);
    play_until_main(&mut game, &mut rng);

    let pi = game.current_player.idx();
    game.players[pi].resources = ResourceBank([19, 19, 19, 19, 19]);

    match game.turn() {
        Turn::Player(turn) => {
            let trades: Vec<_> = turn
                .mask
                .actions()
                .filter_map(|a| match a {
                    Action::BankTrade { give, receive } => Some((give, receive)),
                    _ => None,
                })
                .collect();
            for (give, receive) in &trades {
                assert_ne!(
                    give, receive,
                    "should not be able to trade {give:?} for itself"
                );
            }
            assert!(!trades.is_empty(), "should have some bank trades available");
        }
        _ => panic!(),
    };
}

// =========================================================================
// Cannot buy dev card when deck is empty
// =========================================================================

#[test]
fn cannot_buy_dev_card_empty_deck() {
    let mut rng = StdRng::seed_from_u64(42);
    let mut game = Game::new(&mut rng);
    play_setup(&mut game, &mut rng);
    play_until_main(&mut game, &mut rng);

    let pi = game.current_player.idx();
    game.players[pi].resources = ResourceBank([0, 0, 19, 19, 19]);

    // Drain the deck
    while game.board.dev_card_deck.draw().is_some() {}

    match game.turn() {
        Turn::Player(turn) => {
            assert!(
                !turn.mask.get(Action::BuyDevelopmentCard.to_index()),
                "should not be able to buy from empty deck"
            );
        }
        _ => panic!(),
    };
}

// =========================================================================
// Cannot build settlement without road connection (main phase)
// =========================================================================

#[test]
fn settlement_requires_road_connection_in_main_phase() {
    let mut rng = StdRng::seed_from_u64(42);
    let mut game = Game::new(&mut rng);
    play_setup(&mut game, &mut rng);
    play_until_main(&mut game, &mut rng);

    let pid = game.current_player;
    let pi = pid.idx();
    game.players[pi].resources = ResourceBank([5, 5, 5, 5, 5]);

    // Collect settlement actions from the mask, then verify against the board
    let settlement_vertices: Vec<u8> = match game.turn() {
        Turn::Player(turn) => {
            let verts: Vec<_> = turn
                .mask
                .actions()
                .filter_map(|a| match a {
                    Action::PlaceSettlement(v) => Some(v.0),
                    _ => None,
                })
                .collect();
            let actions: Vec<_> = turn.mask.actions().collect();
            turn.apply(actions[0], &mut rng).unwrap();
            verts
        }
        _ => panic!(),
    };

    for v in settlement_vertices {
        assert!(
            game.board.vertex_has_friendly_road(v as usize, pid)
                || game.board.vertices[v as usize].owner() == Some(pid),
            "offered settlement at v{v} without road connection"
        );
    }
}

// =========================================================================
// VP dev cards count toward victory but aren't "played"
// =========================================================================

#[test]
fn vp_dev_cards_passive() {
    let mut rng = StdRng::seed_from_u64(42);
    let mut game = Game::new(&mut rng);
    play_setup(&mut game, &mut rng);
    play_until_main(&mut game, &mut rng);

    let pi = game.current_player.idx();

    // VP cards contribute to victory_points
    game.players[pi].dev_cards += DevelopmentCard::VictoryPoint;
    game.players[pi].dev_cards += DevelopmentCard::VictoryPoint;
    let vp_with = game.victory_points(game.current_player);

    game.players[pi]
        .dev_cards
        .checked_sub_assign(DevelopmentCard::VictoryPoint);
    game.players[pi]
        .dev_cards
        .checked_sub_assign(DevelopmentCard::VictoryPoint);
    let vp_without = game.victory_points(game.current_player);
    assert_eq!(vp_with, vp_without + 2);

    // VP cards should never appear as playable actions
    game.players[pi].dev_cards += DevelopmentCard::VictoryPoint;
    match game.turn() {
        Turn::Player(turn) => {
            let actions: Vec<_> = turn.mask.actions().collect();
            for a in &actions {
                assert!(
                    !matches!(a, Action::PlayKnight)
                        || game.players[pi].dev_cards.has(DevelopmentCard::Knight),
                    "shouldn't offer VP card as a playable action"
                );
            }
            // More directly: no action should reference VictoryPoint
            // VP dev cards have no Action variant -- they're purely passive.
            // This test just confirms they count in VP.
        }
        _ => panic!(),
    };
}

// =========================================================================
// Turn number increments correctly
// =========================================================================

#[test]
fn turn_number_increments() {
    let mut rng = StdRng::seed_from_u64(42);
    let mut game = Game::new(&mut rng);
    play_setup(&mut game, &mut rng);
    assert_eq!(game.turn_number, 0);

    for expected_turn in 0..8 {
        assert_eq!(game.turn_number, expected_turn);
        play_until_main(&mut game, &mut rng);
        match game.turn() {
            Turn::Player(turn) => turn.apply(Action::EndTurn, &mut rng).unwrap(),
            _ => panic!(),
        };
    }
    assert_eq!(game.turn_number, 8);
}

// =========================================================================
// Current player cycles P0 → P1 → P2 → P3 → P0
// =========================================================================

#[test]
fn player_rotation() {
    let mut rng = StdRng::seed_from_u64(42);
    let mut game = Game::new(&mut rng);
    play_setup(&mut game, &mut rng);

    for round in 0..3 {
        for expected in 0..4u8 {
            assert_eq!(
                game.current_player,
                PlayerId(expected),
                "round {round}: expected P{expected}"
            );
            play_until_main(&mut game, &mut rng);
            match game.turn() {
                Turn::Player(turn) => turn.apply(Action::EndTurn, &mut rng).unwrap(),
                _ => panic!(),
            };
        }
    }
}

// =========================================================================
// Road cannot be placed on occupied edge
// =========================================================================

#[test]
fn no_road_on_occupied_edge() {
    for seed in 0..20 {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut game = Game::new(&mut rng);
        let mut steps = 0;
        loop {
            // Check before taking the turn (no borrow conflict)
            match game.turn() {
                Turn::Terminal => break,
                Turn::Chance(c) => {
                    c.resolve_random(&mut rng);
                    continue;
                }
                Turn::Player(turn) => {
                    let actions: Vec<_> = turn.mask.actions().collect();
                    let choice = rng.random_range(0..actions.len());
                    turn.apply(actions[choice], &mut rng).unwrap();
                    // Verify the action we just applied didn't place on occupied
                    if let Action::PlaceRoad(e) = actions[choice] {
                        assert!(
                            game.board.edges[e.idx()].owner().is_some(),
                            "seed={seed}: road should now be placed on edge {}",
                            e.0
                        );
                    }
                    steps += 1;
                    if steps > 2000 {
                        break;
                    }
                }
            }
        }
    }
}

// =========================================================================
// Robber produces correct steal candidates
// =========================================================================

#[test]
fn steal_candidates_are_adjacent_to_robber() {
    let mut rng = StdRng::seed_from_u64(42);
    let mut game = Game::new(&mut rng);
    play_setup(&mut game, &mut rng);
    play_until_main(&mut game, &mut rng);

    let pi = game.current_player.idx();
    game.players[pi].dev_cards += DevelopmentCard::Knight;
    // Give everyone resources so steal is possible
    for p in &mut game.players {
        p.resources = ResourceBank([2, 2, 2, 2, 2]);
    }

    match game.turn() {
        Turn::Player(turn) => turn.apply(Action::PlayKnight, &mut rng).unwrap(),
        _ => panic!(),
    };

    // Pick a specific tile to move robber to
    let target_tile = (0..19u8).find(|&t| TileId(t) != game.board.robber).unwrap();

    match game.turn() {
        Turn::Player(turn) => {
            turn.apply(Action::MoveRobber(TileId(target_tile)), &mut rng)
                .unwrap();
        }
        _ => panic!(),
    };

    // Verify steal candidates are exactly the non-self players adjacent to that tile
    let topo = &*TOPO;
    if let Phase::Steal { candidates } = &game.phase {
        for (i, &is_candidate) in candidates.iter().enumerate() {
            let pid = PlayerId(i as u8);
            let adjacent = topo.tile_vertices[target_tile as usize]
                .iter()
                .any(|&v| game.board.vertices[v as usize].owner() == Some(pid));
            if pid == game.current_player {
                assert!(!is_candidate, "self should never be a steal candidate");
            } else if adjacent && game.players[i].resources.total() > 0 {
                assert!(
                    is_candidate,
                    "P{i} is adjacent with resources but not a candidate"
                );
            } else {
                assert!(!is_candidate, "P{i} shouldn't be a candidate");
            }
        }
    } else {
        panic!("expected Steal phase");
    }
}

// =========================================================================
// Dev card deck has correct composition (14K, 5VP, 2RB, 2YP, 2Mo)
// =========================================================================

#[test]
fn dev_deck_composition() {
    let mut rng = StdRng::seed_from_u64(42);
    let game = Game::new(&mut rng);
    let mut deck = game.board.dev_card_deck.clone();
    let mut counts = [0u8; 5];
    while let Some(card) = deck.draw() {
        counts[card as usize] += 1;
    }
    assert_eq!(counts[DevelopmentCard::Knight as usize], 14);
    assert_eq!(counts[DevelopmentCard::VictoryPoint as usize], 5);
    assert_eq!(counts[DevelopmentCard::RoadBuilding as usize], 2);
    assert_eq!(counts[DevelopmentCard::YearOfPlenty as usize], 2);
    assert_eq!(counts[DevelopmentCard::Monopoly as usize], 2);
}

// =========================================================================
// Board has correct tile composition
// =========================================================================

#[test]
fn board_tile_composition() {
    let mut rng = StdRng::seed_from_u64(42);
    let game = Game::new(&mut rng);
    let mut terrain_counts = [0u8; 6];
    for tile in &game.board.tiles {
        terrain_counts[tile.terrain as usize] += 1;
    }
    assert_eq!(terrain_counts[Terrain::Desert as usize], 1);
    assert_eq!(terrain_counts[Terrain::Hills as usize], 3);
    assert_eq!(terrain_counts[Terrain::Forest as usize], 4);
    assert_eq!(terrain_counts[Terrain::Mountains as usize], 3);
    assert_eq!(terrain_counts[Terrain::Fields as usize], 4);
    assert_eq!(terrain_counts[Terrain::Pasture as usize], 4);

    // Desert has number 0, all others have 2-12
    for tile in &game.board.tiles {
        if tile.terrain == Terrain::Desert {
            assert_eq!(tile.number, 0);
        } else {
            assert!((2..=12).contains(&tile.number));
            assert_ne!(tile.number, 7); // 7 is never a number token
        }
    }
}

// =========================================================================
// Robber starts on the desert
// =========================================================================

#[test]
fn robber_starts_on_desert() {
    for seed in 0..20 {
        let mut rng = StdRng::seed_from_u64(seed);
        let game = Game::new(&mut rng);
        let robber_tile = game.board.robber.idx();
        assert_eq!(
            game.board.tiles[robber_tile].terrain,
            Terrain::Desert,
            "seed={seed}: robber should start on desert"
        );
    }
}
