use crate::memory::transition::Transition;
use gym::environment::Environment;
use rand::seq::SliceRandom;

#[derive(Debug)]
pub struct ReplayMemory<E: Environment> {
    capacity: usize,
    memory: Vec<Transition<E>>,
}

impl<E: Environment> ReplayMemory<E> {
    pub fn new(capacity: usize) -> Self {
        ReplayMemory {
            capacity,
            memory: Vec::with_capacity(capacity),
        }
    }

    pub fn push(&mut self, transition: Transition<E>) {
        if self.memory.len() >= self.capacity {
            self.memory.remove(0);
        }

        self.memory.push(transition);
    }

    pub fn sample(&self, batch_size: usize) -> Option<Vec<Transition<E>>> {
        if batch_size > self.memory.len() {
            return None;
        }
        let mut rng = rand::thread_rng();

        Some(
            self.memory
                .choose_multiple(&mut rng, batch_size)
                .cloned()
                .collect(),
        )
    }
}
