use crate::decay::Decay;

use super::Choice;

#[derive(Debug)]
pub struct EpsilonGreedy<D: Decay> {
    epsilon: D,
}

impl<D: Decay> EpsilonGreedy<D> {
    pub fn new(epsilon: D) -> Self {
        EpsilonGreedy { epsilon }
    }

    // with probability epsilon, explore
    // with probability 1 - epsilon, exploit
    pub fn choose(&self, step: u32) -> Choice {
        if self.epsilon.decay(step) > rand::random() {
            Choice::Explore
        } else {
            Choice::Exploit
        }
    }
}
