use burn::{prelude::Backend, tensor::Tensor};

#[derive(Debug, Clone)]
pub struct Transition<B: Backend, const D: usize> {
    state: Tensor<B, D>,
    action: Tensor<B, D>,
    next_state: Tensor<B, D>,
    reward: f32,
    done: bool,
}
