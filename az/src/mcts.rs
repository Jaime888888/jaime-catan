use crate::Game;
use rand::RngExt;

pub struct ActionDistribution<const ACT: usize> {
    pub visits: [u32; ACT],
}

impl<const ACT: usize> ActionDistribution<ACT> {
    pub fn policy(&self, temperature: f32) -> [f32; ACT] {
        let mut out = [0.0f32; ACT];

        if temperature < 1e-6 {
            let best = self
                .visits
                .iter()
                .enumerate()
                .filter(|&(_, &n)| n > 0)
                .max_by_key(|&(_, &n)| n)
                .map(|(i, _)| i);
            if let Some(i) = best {
                out[i] = 1.0;
            }
            return out;
        }

        let inv_t = 1.0 / temperature;
        let sum: f32 = self.visits.iter().map(|&n| (n as f32).powf(inv_t)).sum();

        if sum < 1e-8 {
            return out;
        }

        for (i, &n) in self.visits.iter().enumerate() {
            out[i] = (n as f32).powf(inv_t) / sum;
        }

        out
    }

    pub fn pick(&self, temperature: f32, rng: &mut impl rand::Rng) -> usize {
        let policy = self.policy(temperature);
        let r: f32 = rng.random_range(0.0..1.0);
        let mut cum = 0.0;
        let mut last_nonzero = 0;

        for (i, &p) in policy.iter().enumerate() {
            if p > 0.0 {
                last_nonzero = i;
            }
            cum += p;
            if r < cum {
                return i;
            }
        }

        last_nonzero
    }
}

enum NodeState<const PLAYERS: usize> {
    Leaf,
    Interior,
    Terminal([f32; PLAYERS]),
}

struct Node<const PLAYERS: usize> {
    visit_count: u32,
    value_sum: f32,
    prior: f32,
    acting_player: usize,
    children: Vec<(u16, u32)>,
    state: NodeState<PLAYERS>,
}

impl<const PLAYERS: usize> Node<PLAYERS> {
    fn new(prior: f32, acting_player: usize) -> Self {
        Self {
            visit_count: 0,
            value_sum: 0.0,
            prior,
            acting_player,
            children: Vec::new(),
            state: NodeState::Leaf,
        }
    }

    fn puct_score(&self, c_puct: f32, sqrt_parent: f32) -> f32 {
        let q = if self.visit_count == 0 {
            0.0
        } else {
            self.value_sum / self.visit_count as f32
        };
        q + c_puct * self.prior * sqrt_parent / (1.0 + self.visit_count as f32)
    }
}

pub struct Tree<const ACT: usize, const PLAYERS: usize> {
    nodes: Vec<Node<PLAYERS>>,
}

pub struct Leaf<G> {
    path: Vec<u32>,
    pub state: G,
    node_id: u32,
}

impl<const ACT: usize, const PLAYERS: usize> Default for Tree<ACT, PLAYERS> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const ACT: usize, const PLAYERS: usize> Tree<ACT, PLAYERS> {
    pub fn new() -> Self {
        let mut nodes = Vec::with_capacity(ACT * 1024);
        nodes.push(Node::new(1.0, 0));
        Self { nodes }
    }

    /// Walk from root to a leaf via PUCT selection. Known-terminal
    /// nodes are backpropped directly without cloning the game state.
    /// Non-terminal leaves are returned for network evaluation +
    /// [`expand`](Self::expand).
    pub fn select<G, const OBS: usize>(&mut self, root_state: &G, c_puct: f32) -> Option<Leaf<G>>
    where
        G: Game<ACT, OBS, PLAYERS>,
    {
        self.nodes[0].acting_player = G::player_to_index(root_state.current_player());

        let mut path = Vec::with_capacity(64);
        path.push(0u32);
        let mut state = root_state.clone();
        let mut current = 0u32;

        loop {
            let current_idx = current as usize;
            match &self.nodes[current_idx].state {
                NodeState::Terminal(values) => {
                    let values = *values;
                    self.backprop(&path, &values);
                    return None;
                }
                NodeState::Leaf => {
                    if self.nodes[current_idx].visit_count > 0 {
                        return None;
                    }
                    break;
                }
                NodeState::Interior => {
                    let node = &self.nodes[current_idx];
                    if node.children.is_empty() {
                        break;
                    }
                    let sqrt_parent = (node.visit_count as f32).sqrt();
                    let &(action_idx, child_id) = node
                        .children
                        .iter()
                        .max_by(|&&(_, a), &&(_, b)| {
                            let sa = self.nodes[a as usize].puct_score(c_puct, sqrt_parent);
                            let sb = self.nodes[b as usize].puct_score(c_puct, sqrt_parent);
                            sa.total_cmp(&sb)
                        })
                        .unwrap();

                    let action = G::index_to_action(action_idx as usize);
                    state.apply(action);
                    path.push(child_id);
                    current = child_id;
                }
            }
        }

        if state.is_terminal() {
            let values: [f32; PLAYERS] =
                std::array::from_fn(|p| state.result(G::index_to_player(p)));
            self.nodes[current as usize].state = NodeState::Terminal(values);
            self.backprop(&path, &values);
            return None;
        }

        for &node_id in &path {
            self.nodes[node_id as usize].visit_count += 1;
        }
        Some(Leaf {
            path,
            state,
            node_id: current,
        })
    }

    /// Expand a non-terminal leaf: create children from legal actions
    /// with priors derived from `policy` logits, then backprop `values`.
    pub fn expand<G, const OBS: usize>(
        &mut self,
        leaf: &Leaf<G>,
        policy: &[f32; ACT],
        values: [f32; PLAYERS],
    ) where
        G: Game<ACT, OBS, PLAYERS>,
    {
        let nid = leaf.node_id as usize;
        self.nodes[nid].state = NodeState::Interior;
        self.nodes[nid].acting_player = G::player_to_index(leaf.state.current_player());

        let legal = leaf.state.legal_actions();
        debug_assert!(!legal.is_empty(), "expand called with no legal actions");
        let max_logit = legal
            .iter()
            .map(|&a| policy[G::action_to_index(a)])
            .fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = legal
            .iter()
            .map(|&a| (policy[G::action_to_index(a)] - max_logit).exp())
            .sum();

        let mut children = Vec::with_capacity(legal.len());
        for &action in &legal {
            let idx = G::action_to_index(action);
            let prior = (policy[idx] - max_logit).exp() / exp_sum;
            let child_id = self.nodes.len() as u32;
            self.nodes.push(Node::new(prior, 0));
            children.push((idx as u16, child_id));
        }
        self.nodes[nid].children = children;

        for &node_id in leaf.path.iter().rev() {
            let node = &mut self.nodes[node_id as usize];
            node.value_sum += values[node.acting_player];
        }
    }

    /// Full backprop: increment visit count and add value.
    /// Used for paths that did NOT receive virtual loss (e.g. terminals).
    fn backprop(&mut self, path: &[u32], values: &[f32; PLAYERS]) {
        for &node_id in path.iter().rev() {
            let node = &mut self.nodes[node_id as usize];
            node.visit_count += 1;
            node.value_sum += values[node.acting_player];
        }
    }

    pub fn action_distribution(&self) -> ActionDistribution<ACT> {
        let root = &self.nodes[0];
        let mut visits = [0u32; ACT];
        for &(action_idx, child_id) in &root.children {
            visits[action_idx as usize] = self.nodes[child_id as usize].visit_count;
        }

        ActionDistribution { visits }
    }
}
