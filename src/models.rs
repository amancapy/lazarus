use std::iter::zip;

use burn::nn::Linear;
use burn::{backend, config, prelude::*};
use nn::LinearConfig;

use burn::module::Module;
use burn::nn::Relu;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, T};

#[derive(Module, Clone, Debug, Default)]
pub struct Tanh {}

impl Tanh {
    pub fn new() -> Self {
        Self {}
    }
    pub fn forward<B: Backend, const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        burn::tensor::activation::tanh(input)
    }
}

#[derive(Module, Clone, Debug, Default)]
pub struct Sigmoid {}

impl Sigmoid {
    pub fn new() -> Self {
        Self {}
    }
    pub fn forward<B: Backend, const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        burn::tensor::activation::sigmoid(input)
    }
}

#[derive(Debug, Clone)]
pub enum Activation {
    Relu(Relu),
    Tanh(Tanh),
    Sigmoid(Sigmoid),
    Identity,
}

impl Forward for Activation {
    fn forward<B: Backend, const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        match self {
            Activation::Relu(r) => r.forward(input),
            Activation::Tanh(t) => t.forward(input),
            Activation::Sigmoid(s) => s.forward(input),
            Activation::Identity => input,
        }
    }
}

trait Forward {
    fn forward<B: Backend, const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D>;
}

#[derive(Debug, Clone)]
pub struct FF<B: Backend> {
    lins: Vec<Linear<B>>,
    acts: Vec<Activation>,
}

pub fn create_ff<B: Backend>(
    layer_sizes: Vec<usize>,
    activations: Vec<Activation>,
    device: &Device<B>,
) -> FF<B> {
    assert!(
        layer_sizes.len() == activations.len(),
        "layer-sizes Vec and activations Vec must be equal in length. use Identity if needed."
    );
    FF {
        lins: (0..layer_sizes.len() - 1)
            .into_iter()
            .map(|i| LinearConfig::new(layer_sizes[i], layer_sizes[i + 1]).init(device))
            .collect(),
        acts: activations,
    }
}

impl<B: Backend> FF<B> {
    pub fn forward(&self, mut x: Tensor<B, 2>) -> Tensor<B, 2> {
        for (lin, act) in zip(&self.lins, &self.acts) {
            x = lin.forward(x);
            x = act.forward(x);
        }

        return x;
    }
}

#[derive(Clone)]
pub struct SumFxModel<B: Backend> {
    being_model: FF<B>,
    fo_model: FF<B>,
    speechlet_model: FF<B>,

    final_model: FF<B>,
}

impl<B: Backend> SumFxModel<B> {
    pub fn new(
        being_config: (Vec<usize>, Vec<Activation>),
        fo_config: (Vec<usize>, Vec<Activation>),
        speechlet_config: (Vec<usize>, Vec<Activation>),
        final_config: (Vec<usize>, Vec<Activation>),

        device: &Device<B>,
    ) -> Self {
        assert!(
            being_config.0.last() == fo_config.0.last()
                && being_config.0.last() == speechlet_config.0.last(),
            "all sensory models must output the same shape"
        );
        assert!(
            final_config.0.first() == being_config.0.last(),
            "sensory model output and final model input must be the same size"
        );

        SumFxModel {
            being_model: create_ff::<B>(being_config.0, being_config.1, device),
            fo_model: create_ff::<B>(fo_config.0, fo_config.1, device),
            speechlet_model: create_ff::<B>(speechlet_config.0, speechlet_config.1, device),

            final_model: create_ff(final_config.0, final_config.1, device),
        }
    }

    pub fn forward(
        &self,
        being_tensor: Tensor<B, 2>,
        fo_tensor: Tensor<B, 2>,
        speechlet_tensor: Tensor<B, 2>,
    ) -> Tensor<B, 1> {
        let beings_output = self.being_model.forward(being_tensor).mean_dim(0);
        // println!("{}", 1);
        let fo_output = self.fo_model.forward(fo_tensor).mean_dim(0);
        // println!("{}", 2);
        let speechlet_output = self.speechlet_model.forward(speechlet_tensor).mean_dim(0);
        // println!("{}", 3);

        let intermediate = (beings_output + fo_output + speechlet_output) / 3.;
        
        let final_output = self.final_model.forward(intermediate).squeeze(0);
        // println!("{}", 4);

        final_output
    }
}

/* baseline model forward:

    let being_model_output = being_model(being_inputs).mean(axis=0).squeeze(0);
    let fo_model_output = fo_model(fo_inputs).mean(axis=0).squeeze(0);
    let speechlet_model_output = speechlet_model(speechlet_inputs).mean(axis=0).squeeze(0);

    let intermediate = (being_model_output + fo_model_output + speechlet_output) / 3.;

    let model_output = final_output_model(intermediate);

    return model_output
*/

/* Ideally:
    set-transformer implementation for each input type, then final_output_model(intermediate) similarly.
    I remember reading something along the lines that their model subsumes sum({f(x) for all x})
*/
