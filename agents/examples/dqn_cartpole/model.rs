use agents::algorithms::dqn::agent::DQNModel;
use burn::{
    config::Config,
    module::{Module, Param},
    nn::{Linear, LinearConfig, Relu},
    prelude::Backend,
    tensor::{backend::AutodiffBackend, Tensor},
};
use gym::environment::Environment;

#[derive(Module, Debug)]
pub struct CartpoleModel<B: Backend> {
    l1: Linear<B>,
    l2: Linear<B>,
    l3: Linear<B>,
    activation: Relu,
}

// the Config trait helps define default values for the struct fields
#[derive(Config, Debug)]
pub struct CartpoleModelConfig {
    #[config(default = 4)]
    observations: usize,
    #[config(default = 2)]
    actions: usize,
}

impl CartpoleModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CartpoleModel<B> {
        CartpoleModel {
            l1: LinearConfig::new(self.observations, 128).init(device),
            l2: LinearConfig::new(128, 128).init(device),
            l3: LinearConfig::new(128, self.actions).init(device),
            activation: Relu::new(),
        }
    }
}

impl<B: AutodiffBackend, E: Environment> DQNModel<B, E, 2> for CartpoleModel<B> {
    /// # Shape
    /// - Input: [batch_size, 4] - an expansion of the CartPole observation
    /// - Output: [batch_size, actions] - the Q-values for each action
    fn forward(&self, observations: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.l1.forward(observations);
        let x = self.activation.forward(x);
        let x = self.l2.forward(x);
        let x = self.activation.forward(x);

        self.l3.forward(x)
    }

    fn soft_update(self, policy_model: &Self, learning_rate: f32) -> Self {
        Self {
            l1: update_linear(self.l1, &policy_model.l1, learning_rate),
            l2: update_linear(self.l2, &policy_model.l2, learning_rate),
            l3: update_linear(self.l3, &policy_model.l3, learning_rate),
            activation: self.activation,
        }
    }

    fn output_size(&self) -> usize {
        2
    }
}

// soft update the weights of the target model
fn update_tensor<B: Backend, const D: usize>(
    t1: Param<Tensor<B, D>>,
    t2: &Param<Tensor<B, D>>,
    learning_rate: f32,
) -> Param<Tensor<B, D>> {
    t1.map(|t| t * (1. - learning_rate) + t2.val() * learning_rate)
}

fn update_linear<B: Backend>(mut l1: Linear<B>, l2: &Linear<B>, learning_rate: f32) -> Linear<B> {
    l1.weight = update_tensor(l1.weight, &l2.weight, learning_rate);
    l1.bias = match (l1.bias, &l2.bias) {
        (Some(b1), Some(b2)) => Some(update_tensor(b1, b2, learning_rate)),
        _ => None,
    };
    l1
}
