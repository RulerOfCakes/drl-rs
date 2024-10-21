use anyhow::Result;
use std::fmt::Debug;
pub trait Environment {
    type State: Clone + Debug;
    type Reward: Clone + Debug;
    type Action: Clone + Debug;
    fn reset(&mut self) -> Result<Self::State>;
    fn step(&mut self, action: Self::Action) -> Option<(Self::State, Self::Reward)>;
    fn is_active(&self) -> bool {
        true
    }
    fn zero_reward(&self) -> Self::Reward;
}
