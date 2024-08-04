use std::borrow::{Borrow, BorrowMut};
use std::iter::zip;

use burn::nn::Linear;
use burn::{backend, config, prelude::*};
use nn::LinearConfig;

use burn::module::{Module, Param};
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

trait Forward {
    fn forward<B: Backend, const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D>;
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

#[derive(Debug, Clone)]
pub struct FF<B: Backend> {
    pub lins: Vec<Linear<B>>,
    pub acts: Vec<Activation>,
}

pub fn create_ff<B: Backend>(
    layer_sizes: Vec<usize>,
    activations: Vec<Activation>,
    device: &Device<B>,
) -> FF<B> {
    assert!(
        !layer_sizes.is_empty(),
        "layer_sizes vec or activations vec can not be empty"
    );
    assert!(
        layer_sizes.len() == activations.len(),
        "layer-sizes Vec and activations Vec must be equal in length. use Identity if needed."
    );
    FF {
        lins: (0..layer_sizes.len() - 1)
            .into_iter()
            .map(|i| {
                LinearConfig::new(layer_sizes[i], layer_sizes[i + 1])
                    .init(device)
                    .no_grad()
            })
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
    pub being_model: FF<B>,
    pub fo_model: FF<B>,
    pub speechlet_model: FF<B>,
    pub self_model: FF<B>,

    pub final_model: FF<B>,
    pub concat_before_final: bool,
}

// I'm letting this live on as a testament to tunnel-vision
// trait CustomInit {
//     fn custom_init<B: Backend>(
//         &self,
//         weight: Tensor<B, 2>,
//         bias_vec: Option<Tensor<B, 1>>,
//         device: &B::Device,
//     ) -> Linear<B>;
// }

// impl CustomInit for LinearConfig {
//     fn custom_init<B: Backend>(
//         &self,
//         weight: Tensor<B, 2>,
//         bias: Option<Tensor<B, 1>>,
//         device: &B::Device,
//     ) -> Linear<B> {

//         let weight = Param::from_data(weight.to_data(), device);
//         let bias = if self.bias {
//             Some(Param::from_data(bias.unwrap().to_data(), device))
//         } else {
//             None
//         };

//         Linear { weight, bias }
//     }
// }

impl<B: Backend> SumFxModel<B> {
    pub fn new(
        being_config: (Vec<usize>, Vec<Activation>),
        fo_config: (Vec<usize>, Vec<Activation>),
        speechlet_config: (Vec<usize>, Vec<Activation>),
        self_config: (Vec<usize>, Vec<Activation>),
        final_config: (Vec<usize>, Vec<Activation>),

        concat_before_final: bool,
        device: &Device<B>,
    ) -> Self {
        if !concat_before_final {
            assert!(
                being_config.0.last() == fo_config.0.last()
                    && being_config.0.last() == speechlet_config.0.last()
                    && being_config.0.last() == self_config.0.last(),
                "all sensory models must output the same shape, since you chose add mode"
            );
            assert!(
                final_config.0.first() == being_config.0.last(),
                "sensory model output and final model input must be the same size, since you chose mean mode"
            );
        } else {
            assert!(
                &(being_config.0.last().unwrap() + fo_config.0.last().unwrap() + speechlet_config.0.last().unwrap() + self_config.0.last().unwrap()) == final_config.0.first().unwrap(),
                "sensory model output sizes must add up to final model input size, since you chose concat mode"
            );
        }

        SumFxModel {
            being_model: create_ff::<B>(being_config.0, being_config.1, device),
            fo_model: create_ff::<B>(fo_config.0, fo_config.1, device),
            speechlet_model: create_ff::<B>(speechlet_config.0, speechlet_config.1, device),
            self_model: create_ff::<B>(self_config.0, self_config.1, device),

            final_model: create_ff(final_config.0, final_config.1, device),
            concat_before_final: concat_before_final,
        }
    }

    pub fn forward(
        &self,
        being_tensor: Tensor<B, 2>,
        fo_tensor: Tensor<B, 2>,
        speechlet_tensor: Tensor<B, 2>,
        self_tensor: Tensor<B, 2>,
    ) -> Tensor<B, 1> {
        let beings_output = self.being_model.forward(being_tensor).mean_dim(0);
        let fo_output = self.fo_model.forward(fo_tensor).mean_dim(0);
        let speechlet_output = self.speechlet_model.forward(speechlet_tensor).mean_dim(0);
        let self_output = self.self_model.forward(self_tensor);

        let intermediate: Tensor<B, 2>;
        if self.concat_before_final {
            intermediate = Tensor::cat(
                vec![beings_output, fo_output, speechlet_output, self_output],
                1,
            );
        } else {
            intermediate = (beings_output + fo_output + speechlet_output + self_output) / 4.;
        };

        let final_output = self.final_model.forward(intermediate).squeeze(0);

        final_output
    }

    pub fn mutate(self, recomb_weight: f32, device: &Device<B>) {
        let models: Vec<SumFxModel<B>> = vec![];
        for mut model in [
            self.being_model,
            self.fo_model,
            self.speechlet_model,
            self.self_model,
            self.final_model,
        ] {
            let mut newlins: Vec<Linear<B>> = vec![];

            for lin in model.lins {
                let [inp_size, outp_size] = lin.weight.shape().dims;
                let mutation_lin: Linear<B> = LinearConfig::new(inp_size, outp_size).init(device);

                let weight = Param::from_tensor(
                    mutation_lin.weight.val().mul_scalar(recomb_weight)
                        + lin.weight.val().mul_scalar(1. - recomb_weight),
                );

                let bias = {
                    if let None = lin.bias {
                        None
                    } else {
                        Some(Param::from_tensor(
                            mutation_lin.bias.unwrap().val().mul_scalar(recomb_weight)
                                + lin.bias.unwrap().val().mul_scalar(1. - recomb_weight),
                        ))
                    }
                };
                
                let newlin = Linear {
                    weight: weight,
                    bias: bias,
                };
                newlins.push(newlin);
            }
        }
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
