use std::fmt::Debug;

pub trait Decay: Debug {
    fn decay(&self, step: u32) -> f32;
}

#[derive(Debug)]
pub struct ExponentialDecay {
    initial: f32,
    min: f32,
    decay: f32,
}

impl ExponentialDecay {
    pub fn new(initial: f32, min: f32, decay: f32) -> Self {
        ExponentialDecay {
            initial,
            min,
            decay,
        }
    }
}

impl Decay for ExponentialDecay {
    fn decay(&self, step: u32) -> f32 {
        self.min + (self.initial - self.min) * (-self.decay * step as f32).exp()
    }
}
