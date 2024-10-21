use burn::{
    grad_clipping::GradientClippingConfig,
    module::AutodiffModule,
    optim::{AdamWConfig, GradientsParams, Optimizer},
    prelude::*,
    tensor::{backend::AutodiffBackend, ElementConversion, Tensor},
    // train::{ClassificationOutput, TrainStep},
};
use derive_builder::Builder;
use nn::loss::{MseLoss, Reduction};

use crate::{
    decay::{Decay, ExponentialDecay},
    exploration::{epsilon_greedy::EpsilonGreedy, Choice},
    memory::{replay::ReplayMemory, transition::Transition},
};

use gym::{environment::Environment, to_tensor::ToTensor};

use anyhow::Result;

pub trait DQNModel<B: AutodiffBackend, E: Environment, const INPUT_DIM: usize>:
    AutodiffModule<B>
{
    // input: [batch_size, D-1], output: [batch_size, 1]
    fn forward(&self, observations: Tensor<B, INPUT_DIM>) -> Tensor<B, 2>;

    // periodically update the target model with the policy model
    // avoid hard updates to prevent abrupt changes in the target model
    fn soft_update(self, policy_model: &Self, learning_rate: f32) -> Self;

    fn output_size(&self) -> usize;
}

#[derive(Builder, Debug)]
pub struct DQNAgentHyperParameters {
    #[builder(default = "16384")]
    memory_capacity: usize,
    #[builder(default = "128")]
    batch_size: usize,
    #[builder(default = "0.999")]
    gamma: f32, // discount factor
    // for exponential epsilon greedy
    #[builder(default = "0.9")]
    eps_start: f32,
    #[builder(default = "0.05")]
    eps_end: f32,
    #[builder(default = "0.05")]
    eps_decay: f32,
    // learning rates
    #[builder(default = "0.005")]
    target_learning_rate: f32,
    #[builder(default = "1")]
    target_update_interval: u32,
    #[builder(default = "0.001")]
    optimizer_learning_rate: f32,
}

impl DQNAgentHyperParameters {
    pub fn get_decay(&self) -> ExponentialDecay {
        ExponentialDecay::new(self.eps_start, self.eps_end, self.eps_decay)
    }
}

#[derive(Debug)]
pub struct DQNAgent<B: AutodiffBackend, M, E: Environment, D: Decay, const INPUT_DIM: usize> {
    policy_model: M,
    target_model: M,
    device: &'static B::Device,
    memory: ReplayMemory<E>,
    exploration: EpsilonGreedy<D>,
    batch_size: usize,
    gamma: f32,
    target_learning_rate: f32,
    target_update_interval: u32,
    optimizer_learning_rate: f32,
    // self-initialized state
    episodes_elapsed: u32,
    steps: u32,
}

impl<
        B: AutodiffBackend<FloatElem = f32, IntElem = i32>,
        M,
        E: Environment,
        const INPUT_DIM: usize,
    > DQNAgent<B, M, E, ExponentialDecay, INPUT_DIM>
where
    M: DQNModel<B, E, INPUT_DIM>,
    Vec<E::State>: ToTensor<B, INPUT_DIM, Float>, // the state vector is directly used as the input batch
    Vec<E::Action>: ToTensor<B, 2, Int>,
    Vec<E::Reward>: ToTensor<B, 2, Float>,
    E::Action: From<usize>,
{
    pub fn new(
        policy_model: M,
        target_model: M,
        device: &'static B::Device,
        hyperparams: DQNAgentHyperParameters,
    ) -> Self {
        DQNAgent {
            policy_model,
            target_model,
            device,
            memory: ReplayMemory::new(hyperparams.memory_capacity),
            exploration: EpsilonGreedy::new(hyperparams.get_decay()),
            batch_size: hyperparams.batch_size,
            gamma: hyperparams.gamma,
            target_learning_rate: hyperparams.target_learning_rate,
            target_update_interval: hyperparams.target_update_interval,
            optimizer_learning_rate: hyperparams.optimizer_learning_rate,
            episodes_elapsed: 0,
            steps: 0,
        }
    }

    // Choose an action based on the current state with epsilon greedy strategy
    pub fn act(&self, state: E::State) -> usize {
        match self.exploration.choose(self.steps) {
            Choice::Explore => rand::random::<usize>() % self.policy_model.output_size(),
            Choice::Exploit => {
                let state_tensor = vec![state].to_tensor(self.device);

                self.policy_model
                    .forward(state_tensor)
                    .argmax(1) // this will panic if the index tensor is multidimensional
                    .into_scalar()
                    .elem::<u32>() as usize
            }
        }
    }

    pub fn step(mut self, optimizer: &mut impl Optimizer<M, B>) -> Self {
        let Some(transitions) = self.memory.sample(self.batch_size) else {
            return self;
        };

        // boolean mask of terminality on states
        let non_terminal_mask: Tensor<B, 2, Bool> = transitions
            .iter()
            .map(|transition| transition.next_state.is_some())
            .collect::<Vec<_>>()
            .to_tensor(self.device)
            .unsqueeze_dim(1);

        // Convert to tensors
        let (states, actions, rewards, next_states) = transitions.into_iter().fold(
            (Vec::new(), Vec::new(), Vec::new(), Vec::new()),
            |(mut states, mut actions, mut rewards, mut next_states), transition| {
                states.push(transition.state);
                actions.push(transition.action);
                rewards.push(transition.reward);
                if let Some(next_state) = transition.next_state {
                    next_states.push(next_state);
                }
                (states, actions, rewards, next_states)
            },
        );

        // collect tensor states, actions, rewards, and next_states(note that next_states is clipped)
        let states: Tensor<B, INPUT_DIM> = states.to_tensor(self.device);
        let actions: Tensor<B, 2, Int> = actions.to_tensor(self.device);
        let rewards: Tensor<B, 2, Float> = rewards.to_tensor(self.device);
        let next_states: Tensor<B, INPUT_DIM> = next_states.to_tensor(self.device);

        // compute q values - Q(s_t) of the chosen actions in each state
        let q_values = self.policy_model.forward(states).gather(1, actions);

        // Compute the maximum Q values obtainable from each next state for the backwards pass
        // max_a Q(s_{t+1})
        let expected_q_values = Tensor::zeros([self.batch_size, 1], self.device).mask_where(
            non_terminal_mask,
            self.target_model.forward(next_states).max_dim(1).detach(),
        );

        // Thanks to the mask, we correctly get y_i = r_i + γ max_a Q(s_{t+1}) or discard the latter term for terminal states
        let discounted_expected_return = rewards + (expected_q_values * self.gamma);

        let loss = MseLoss::new().forward(q_values, discounted_expected_return, Reduction::Mean);

        // Perform backpropagation on policy model
        let grads = GradientsParams::from_grads(loss.backward(), &self.policy_model);
        self.policy_model = optimizer.step(
            self.optimizer_learning_rate.into(),
            self.policy_model,
            grads,
        );

        // Perform a periodic soft update on the parameters of the target network for stable convergence
        self.target_model = if self.episodes_elapsed % self.target_update_interval == 0 {
            self.target_model
                .soft_update(&self.policy_model, self.target_learning_rate)
        } else {
            self.target_model
        };
        self
    }

    // Train the agent for one episode
    pub fn episode(mut self, env: &mut E) -> Result<Self> {
        let mut optimizer = AdamWConfig::new()
            .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
            .init::<B, M>();

        // initial state must be valid
        let mut state: E::State = env.reset()?;

        for step in 0.. {
            self.steps = step;
            let mut terminate = false;
            let action = self.act(state.clone());
            match env.step(action.into()) {
                Some((next_state, reward)) => {
                    self.memory.push(Transition::new(
                        state,
                        action.into(),
                        reward,
                        Some(next_state.clone()),
                    ));
                    state = next_state;
                }
                None => {
                    self.memory.push(Transition::new(
                        state.clone(),
                        action.into(),
                        env.zero_reward(),
                        None,
                    ));
                    terminate = true;
                }
            }
            self = self.step(&mut optimizer);
            if terminate {
                break;
            }
        }
        Ok(self)
    }
}
