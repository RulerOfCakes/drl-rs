use crate::memory::transition::Transition;
use burn::prelude::Backend;
use rand::seq::SliceRandom;

pub struct ReplayMemory<B: Backend, const D: usize> {
    capacity: usize,
    memory: Vec<Transition<B, D>>,
}

impl<B: Backend, const D: usize> ReplayMemory<B, D> {
    pub fn new(capacity: usize) -> Self {
        ReplayMemory {
            capacity,
            memory: Vec::with_capacity(capacity),
        }
    }

    pub fn push(&mut self, transition: Transition<B, D>) {
        if self.memory.len() >= self.capacity {
            self.memory.remove(0);
        }

        self.memory.push(transition);
    }

    pub fn sample(&self, batch_size: usize) -> Vec<Transition<B, D>> {
        let mut rng = rand::thread_rng();

        self.memory
            .choose_multiple(&mut rng, batch_size)
            .cloned()
            .collect()
    }
}
