use std::iter::zip;

use burn::nn::Linear;
use burn::prelude::*;
use nn::attention::MultiHeadAttention;
use nn::{LinearConfig, Lstm};

use burn::module::{Module, Param};
use burn::nn::Relu;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

pub fn tensorize_2dvec<B: Backend>(
    vec: &Vec<Vec<f32>>,
    shape: [usize; 2],
    device: &Device<B>,
) -> Tensor<B, 2> {
    Tensor::<B, 1>::from_floats(
        vec.clone()
            .into_iter()
            .flatten()
            .collect::<Vec<f32>>()
            .as_slice(),
        device,
    )
    .reshape(shape)
}

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

// bias is decided by lin1
pub fn combine_linears<B: Backend>(
    lin1: Linear<B>,
    lin2: Linear<B>,
    left_weight: f32,
    right_weight: f32,
) -> Linear<B> {
    assert!(
        lin1.weight.shape() == lin1.weight.shape(),
        "linear constructs do not match"
    );

    let weight =
        lin1.weight.val().mul_scalar(left_weight) + lin2.weight.val().mul_scalar(right_weight);
    let bias = {
        if lin1.bias.is_none() {
            None
        } else {
            Some(
                lin1.bias.unwrap().val().mul_scalar(left_weight)
                    + lin2.bias.unwrap().val().mul_scalar(right_weight),
            )
        }
    };
    let (weight, bias) = (
        Param::from_tensor(weight),
        Some(Param::from_tensor(bias.unwrap())),
    );

    Linear { weight, bias }
}

#[derive(Debug, Clone)]
pub struct FF<B: Backend> {
    pub lins: Vec<Linear<B>>,
    pub acts: Vec<Activation>,

    pub config: (Vec<usize>, Vec<Activation>),
}

impl<B: Backend> FF<B> {
    pub fn new(layer_sizes: Vec<usize>, activations: Vec<Activation>, device: &Device<B>) -> FF<B> {
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
            acts: activations.clone(),

            config: (layer_sizes, activations),
        }
    }

    pub fn forward(&self, mut x: Tensor<B, 2>) -> Tensor<B, 2> {
        for (lin, act) in zip(&self.lins, &self.acts) {
            x = lin.forward(x);
            x = act.forward(x);
        }

        return x;
    }
}

pub fn splice_ffs<B: Backend>(
    mut ff1: FF<B>,
    ff2: FF<B>,
    left_weight: f32,
) -> FF<B> {
    for (ff1_lin, ff2_lin) in zip(&mut ff1.lins, ff2.lins) {
        let weight = ff1_lin.weight.clone().val();
        let mask: Tensor<B, 2> = weight.ones_like().mul_scalar(left_weight);
        
        let ff1_mask: Tensor<B, 2, Bool> = weight.random_like(burn::tensor::Distribution::Uniform(0., 1.)).greater_equal(mask);
        let ff2_mask: Tensor<B, 2, Bool> = ff1_mask.clone().bool_not();

        let weight = ff1_lin.weight.val().mask_fill(ff1_mask, 0.) + ff2_lin.weight.val().mask_fill(ff2_mask, 0.);
        ff1_lin.weight = Param::from_tensor(weight);

        if !ff1_lin.bias.is_none() {
            let bias = ff1_lin.bias.clone().unwrap().val();
            let mask: Tensor<B, 1> = bias.ones_like().mul_scalar(left_weight);

            let ff1_mask: Tensor<B, 1, Bool> = bias.random_like(burn::tensor::Distribution::Uniform(0., 1.)).greater_equal(mask);
            let ff2_mask: Tensor<B, 1, Bool> = ff1_mask.clone().bool_not();

            let bias = ff1_lin.bias.clone().unwrap().val().mask_fill(ff1_mask, 0.) + ff2_lin.bias.unwrap().val().mask_fill(ff2_mask, 0.);
            ff1_lin.bias = Some(Param::from_tensor(bias));
        }
    }

    ff1
}

pub fn combine_lstms<B: Backend>(
    lstm_1: Lstm<B>,
    lstm_2: Lstm<B>,
    left_weight: f32,
    right_weight: f32,
) -> Lstm<B> {
    let mut record_1 = lstm_1.clone().into_record();
    let record_2 = lstm_2.into_record();

    for (gate_1, gate_2) in zip(
        [
            &mut record_1.input_gate,
            &mut record_1.forget_gate,
            &mut record_1.output_gate,
            &mut record_1.cell_gate,
        ],
        [
            record_2.input_gate,
            record_2.forget_gate,
            record_2.output_gate,
            record_2.cell_gate,
        ],
    ) {
        let (i_1, h_1) = (&gate_1.input_transform, &gate_1.hidden_transform);
        let (i_1, h_1) = (
            Linear {
                weight: i_1.weight.clone(),
                bias: i_1.bias.clone(),
            },
            Linear {
                weight: h_1.weight.clone(),
                bias: h_1.bias.clone(),
            },
        );

        let (i_2, h_2) = (gate_2.input_transform, gate_2.hidden_transform);
        let (i_2, h_2) = (
            Linear {
                weight: i_2.weight.clone(),
                bias: i_2.bias.clone(),
            },
            Linear {
                weight: h_2.weight.clone(),
                bias: h_2.bias.clone(),
            },
        );

        gate_1.input_transform = combine_linears(i_1, i_2, left_weight, right_weight).into_record();
        gate_1.hidden_transform =
            combine_linears(h_1, h_2, left_weight, right_weight).into_record();
    }

    lstm_1.load_record(record_1).no_grad()
}

pub fn combine_mhas<B: Backend>(
    mha1: MultiHeadAttention<B>,
    mha2: MultiHeadAttention<B>,
    left_weight: f32,
    right_weight: f32,
) -> MultiHeadAttention<B> {
    let mut record_1 = mha1.clone().into_record();
    let record_2 = mha2.into_record();

    for (lin1, lin2) in zip(
        [
            &mut record_1.query,
            &mut record_1.key,
            &mut record_1.value,
            &mut record_1.output,
        ],
        [
            record_2.query,
            record_2.key,
            record_2.value,
            record_2.output,
        ],
    ) {
        let l1 = Linear {
            weight: lin1.weight.clone(),
            bias: lin1.bias.clone(),
        };
        let l2 = Linear {
            weight: lin2.weight.clone(),
            bias: lin2.bias.clone(),
        };

        let comb = combine_linears(l1, l2, left_weight, right_weight);

        lin1.weight = comb.weight;
        lin1.bias = comb.bias;
    }

    mha1.load_record(record_1)
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
