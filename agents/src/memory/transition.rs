use gym::environment::Environment;

#[derive(Debug)]
pub struct Transition<E: Environment> {
    pub state: E::State,
    pub action: E::Action,
    pub reward: E::Reward,
    pub next_state: Option<E::State>,
}

impl<E: Environment> Transition<E> {
    pub fn new(
        state: E::State,
        action: E::Action,
        reward: E::Reward,
        next_state: Option<E::State>,
    ) -> Self {
        Transition {
            state,
            action,
            reward,
            next_state,
        }
    }
}

// Due to how derive macros work, we need to specifically implement Clone
// to keep the compiler happy about Transition<E> always being Clone.
impl<E: Environment> Clone for Transition<E> {
    fn clone(&self) -> Self {
        Transition {
            state: self.state.clone(),
            action: self.action.clone(),
            reward: self.reward.clone(),
            next_state: self.next_state.clone(),
        }
    }
}
