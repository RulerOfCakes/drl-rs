use burn::backend::Wgpu;
use gym_rs::{envs::classical_control::cartpole::CartPoleEnv, utils::renderer::RenderMode};

use crate::model::DQNModelConfig;

mod model;

fn main() {
    type MyBackend = Wgpu<f32, i32>;

    let device = Default::default();
    let model = DQNModelConfig::new().init::<MyBackend>(&device);

    let mut env = CartPoleEnv::new(RenderMode::Human);
    println!("{:?}", model);
}
