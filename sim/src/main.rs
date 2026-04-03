use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

use catan_sim::{Game, PlayerId};

fn main() {
    let mut rng = StdRng::seed_from_u64(42);
    let mut game = Game::new(&mut rng);

    println!("=== Catan Simulator ===");
    println!("Playing a random game...\n");

    let mut action_count = 0u32;
    loop {
        if game.is_terminal() {
            break;
        }
        if game.is_chance_node() {
            game.resolve_chance_random(&mut rng);
            continue;
        }

        let actions = game.legal_actions();
        if actions.is_empty() {
            println!("No legal actions – phase: {:?}", game.phase);
            break;
        }

        let idx = rng.gen_range(0..actions.len());
        game.apply_action(actions[idx], &mut rng).unwrap();
        action_count += 1;

        if action_count.is_multiple_of(100) {
            println!(
                "  action #{action_count}: turn={}, player={}, phase={:?}",
                game.turn_number,
                game.acting_player(),
                game.phase
            );
        }
    }

    println!(
        "\nGame over after {action_count} actions and {} turns.",
        game.turn_number
    );
    if let Some(winner) = game.winner() {
        println!("Winner: {winner} with {} VP", game.victory_points(winner));
    }
    for i in 0..4 {
        let pid = PlayerId(i);
        println!(
            "  {}: {} VP, resources={}",
            pid,
            game.victory_points(pid),
            game.players[i as usize].resources
        );
    }
}
