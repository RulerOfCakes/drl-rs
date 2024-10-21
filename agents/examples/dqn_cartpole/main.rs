use anyhow::Result;
use burn::backend::{wgpu::WgpuDevice, Autodiff, Wgpu};
use gym::{
    cartpole::{CartPoleEnv, RenderMode},
    environment::Environment,
};

use crate::model::CartpoleModelConfig;
use agents::algorithms::dqn::agent::{
    DQNAgent, DQNAgentHyperParameters, DQNAgentHyperParametersBuilder,
};

mod model;

use std::sync::LazyLock;

static DEVICE: LazyLock<WgpuDevice> = LazyLock::new(WgpuDevice::default);

static NUM_EPISODES: u32 = 1024;

fn main() -> Result<()> {
    type MyBackend = Autodiff<Wgpu>;

    let model = CartpoleModelConfig::new().init::<MyBackend>(&*DEVICE);
    let hyperparams: DQNAgentHyperParameters = DQNAgentHyperParametersBuilder::default()
        .build()
        .expect("Failed to build hyperparameters");

    let mut agent = DQNAgent::new(model.clone(), model, &*DEVICE, hyperparams);

    let mut env = CartPoleEnv::new(RenderMode::Human);

    for ep in 0..NUM_EPISODES {
        dbg!(ep);
        agent = agent.episode(&mut env)?;
        let _ = env.reset();
    }

    Ok(())
}
