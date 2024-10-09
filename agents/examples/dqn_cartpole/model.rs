use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig, Relu},
    prelude::Backend,
    tensor::Tensor,
};

#[derive(Module, Debug)]
pub struct DQNModel<B: Backend> {
    l1: Linear<B>,
    l2: Linear<B>,
    l3: Linear<B>,
    activation: Relu,
}

// the Config trait helps define default values for the struct fields
#[derive(Config, Debug)]
pub struct DQNModelConfig {
    #[config(default = 4)]
    observations: usize,
    #[config(default = 1)]
    actions: usize,
}

impl DQNModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DQNModel<B> {
        DQNModel {
            l1: LinearConfig::new(self.observations, 128).init(device),
            l2: LinearConfig::new(128, 128).init(device),
            l3: LinearConfig::new(128, self.actions).init(device),
            activation: Relu::new(),
        }
    }
}

impl<B: Backend> DQNModel<B> {
    /// # Shape
    /// - Input: [batch_size, 4] - an expansion of the CartPole observation
    /// - Output: [batch_size, actions] - the Q-values for each action
    pub fn forward(&self, observations: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.l1.forward(observations);
        let x = self.activation.forward(x);
        let x = self.l2.forward(x);
        let x = self.activation.forward(x);

        self.l3.forward(x)
    }
}
