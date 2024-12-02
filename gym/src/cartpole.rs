use anyhow::Result;
use burn::{
    prelude::Backend,
    tensor::{Float, Int, Tensor, TensorData},
};
use gym_rs::{core::Env, envs::classical_control::cartpole::CartPoleEnv as GymEnv};

pub use gym_rs::utils::renderer::RenderMode;

use crate::{environment::Environment, to_tensor::ToTensor};

pub struct CartPoleEnv {
    gym: GymEnv,
}

impl CartPoleEnv {
    pub fn new(render_mode: RenderMode) -> Self {
        Self {
            gym: GymEnv::new(render_mode),
        }
    }
}

impl Default for CartPoleEnv {
    fn default() -> Self {
        Self::new(RenderMode::Human)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CartPoleAction(i32);

impl From<usize> for CartPoleAction {
    fn from(action: usize) -> Self {
        Self(action as i32)
    }
}

impl<B: Backend> ToTensor<B, 2, Int> for Vec<CartPoleAction> {
    fn to_tensor(self, device: &B::Device) -> Tensor<B, 2, Int> {
        let len = self.len();
        let data = TensorData::new(self.into_iter().map(|x| x.0).collect::<Vec<_>>(), [len]);
        Tensor::<B, 1, Int>::from_data(data, device).unsqueeze_dim(1)
    }
}

impl<B: Backend> ToTensor<B, 2, Float> for Vec<f32> {
    fn to_tensor(self, device: &B::Device) -> Tensor<B, 2, Float> {
        let len = self.len();
        let data = TensorData::new(self, [len]);
        Tensor::<B, 1, Float>::from_data(data, device).unsqueeze_dim(1)
    }
}

impl<B: Backend> ToTensor<B, 2, Float> for Vec<Vec<f32>> {
    fn to_tensor(self, device: &B::Device) -> Tensor<B, 2, Float> {
        let len = self.len();
        let inner = self[0].len();
        let data = TensorData::new(
            self.into_iter().flatten().collect::<Vec<_>>(),
            [len * inner],
        );
        Tensor::<B, 1, Float>::from_data(data, device).reshape([-1, inner as i32])
    }
}

impl Environment for CartPoleEnv {
    // force cast state and reward to lower precision for faster training
    type State = Vec<f32>;
    type Action = CartPoleAction;
    type Reward = f32;

    fn reset(&mut self) -> Result<Self::State> {
        let (observation_info, _) = self.gym.reset(None, false, None);
        let observation_info: Vec<f64> = observation_info.into();

        Ok(observation_info.into_iter().map(|x| x as f32).collect())
    }

    fn step(&mut self, action: Self::Action) -> Option<(Self::State, Self::Reward)> {
        let action_reward = self.gym.step(action.0 as usize);
        if action_reward.done {
            return None;
        }
        let observation_info: Vec<f64> = action_reward.observation.into();
        let reward = action_reward.reward.0 as f32;
        Some((
            observation_info.into_iter().map(|x| x as f32).collect(),
            reward,
        ))
    }

    fn zero_reward(&self) -> Self::Reward {
        0.
    }

    fn is_active(&self) -> bool {
        true
    }
}
