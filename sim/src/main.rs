use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

use catan_sim::{Game, Turn};

fn main() {
    // let mut rng = StdRng::seed_from_u64(42);
    let mut rng = StdRng::from_entropy();
    let mut game = Game::new(&mut rng);

    println!("{game}\n");

    let mut action_count = 0u32;
    loop {
        match game.turn() {
            Turn::Terminal => break,
            Turn::Chance(chance) => chance.resolve_random(&mut rng),
            Turn::Player(turn) => {
                let actions = turn.mask.actions().collect::<Vec<_>>();
                let idx = rng.gen_range(0..actions.len());
                let action = actions[idx];
                turn.apply(action, &mut rng).unwrap();
                action_count += 1;

                if action_count.is_multiple_of(200) {
                    println!("--- action #{action_count}: {action} ---");
                    println!("{game}\n");
                }
            }
        }
    }

    println!("=== FINAL ({action_count} actions) ===");
    println!("{game}");
}
