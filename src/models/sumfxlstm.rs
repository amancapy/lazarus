use std::iter::zip;

use burn::nn::Linear;
use burn::prelude::*;
use nn::attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig};
use nn::{LinearConfig, Lstm, LstmConfig};

use burn::module::{ConstantRecord, Module, Param};
use burn::nn::Relu;
use burn::tensor::backend::Backend;
use burn::tensor::{activation, Tensor};

use crate::being_nn::{combine_linears, combine_lstms, Activation, Tanh, FF};
use crate::{combine_ffs, B_OUTPUT_LEN, GENOME_LEN, SPEECHLET_LEN};

#[derive(Clone)]
pub struct SumFxLstmModel<B: Backend> {
    pub being_model: FF<B>,
    pub fo_model: FF<B>,
    pub speechlet_model: FF<B>,
    pub self_model: FF<B>,

    pub lstm: Lstm<B>,
    pub final_model: FF<B>,

    pub concat_before_final: bool,
    pub intermediate_dim: usize,
    pub lstm_inp_size: usize,

    state: (Tensor<B, 2>, Tensor<B, 2>),
}

impl<B: Backend> SumFxLstmModel<B> {
    pub fn new(
        being_config: (Vec<usize>, Vec<Activation>),
        fo_config: (Vec<usize>, Vec<Activation>),
        speechlet_config: (Vec<usize>, Vec<Activation>),
        self_config: (Vec<usize>, Vec<Activation>),
        final_config: (Vec<usize>, Vec<Activation>),

        concat_before_final: bool,

        device: &Device<B>,
    ) -> Self {
        let lstm_inp_size = {
            if !concat_before_final {
                being_config.0.last().unwrap().clone()
            } else {
                being_config.0.last().unwrap()
                    + fo_config.0.last().unwrap()
                    + speechlet_config.0.last().unwrap()
                    + self_config.0.last().unwrap()
            }
        };

        let intermediate_dim: usize;

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
            intermediate_dim = being_config.0.last().unwrap().clone();
        } else {
            assert!(
                &(being_config.0.last().unwrap() + fo_config.0.last().unwrap() + speechlet_config.0.last().unwrap() + self_config.0.last().unwrap()) == final_config.0.first().unwrap(),
                "sensory model output sizes must add up to final model input size, since you chose concat mode"
            );
            intermediate_dim = being_config.0.last().unwrap()
                + fo_config.0.last().unwrap()
                + speechlet_config.0.last().unwrap()
                + self_config.0.last().unwrap();
        }

        SumFxLstmModel {
            being_model: FF::new(being_config.0, being_config.1, device),
            fo_model: FF::new(fo_config.0, fo_config.1, device),
            speechlet_model: FF::new(speechlet_config.0, speechlet_config.1, device),
            self_model: FF::new(self_config.0, self_config.1, device),
            lstm: LstmConfig::new(lstm_inp_size, lstm_inp_size, true)
                .init(device)
                .no_grad(),
            final_model: FF::new(final_config.0, final_config.1, device),

            concat_before_final: concat_before_final,
            intermediate_dim: intermediate_dim,
            lstm_inp_size: lstm_inp_size,
            state: (
                Tensor::<B, 2>::zeros([1, intermediate_dim], device).no_grad(),
                Tensor::<B, 2>::zeros([1, intermediate_dim], device).no_grad(),
            ),
        }
    }

    pub fn standard_model(device: &Device<B>) -> Self {
        let being_config = (
            vec![3 + GENOME_LEN, 8],
            vec![Activation::Tanh(Tanh {}), Activation::Tanh(Tanh {})],
        );
        let fo_config = (
            vec![5, 8],
            vec![Activation::Tanh(Tanh {}), Activation::Tanh(Tanh {})],
        );
        let speechlet_config = (
            vec![SPEECHLET_LEN, 8],
            vec![Activation::Tanh(Tanh {}), Activation::Tanh(Tanh {})],
        );
        let self_config = (
            vec![5, 8],
            vec![Activation::Tanh(Tanh {}), Activation::Tanh(Tanh {})],
        );
        let final_config = (
            vec![32, B_OUTPUT_LEN],
            vec![Activation::Tanh(Tanh {}), Activation::Tanh(Tanh {})],
        );
        return SumFxLstmModel::new(
            being_config,
            fo_config,
            speechlet_config,
            self_config,
            final_config,
            true,
            device,
        );
    }

    pub fn forward(
        &mut self,
        being_tensor: Tensor<B, 2>,
        fo_tensor: Tensor<B, 2>,
        speechlet_tensor: Tensor<B, 2>,
        self_tensor: Tensor<B, 2>,
    ) -> Tensor<B, 1> {
        let beings_output = self.being_model.forward(being_tensor).mean_dim(0);
        let fo_output = self.fo_model.forward(fo_tensor).mean_dim(0);
        let speechlet_output = self.speechlet_model.forward(speechlet_tensor).mean_dim(0);
        let self_output = self.self_model.forward(self_tensor);

        let intermediate: Tensor<B, 2> = {
            if self.concat_before_final {
                Tensor::cat(
                    vec![beings_output, fo_output, speechlet_output, self_output],
                    1,
                )
            } else {
                (beings_output + fo_output + speechlet_output + self_output) / 4.
            }
        };

        let (c, h) = self
            .lstm
            .forward(intermediate.clone().unsqueeze(), Some(self.state.clone()));

        let (c, h): (Tensor<B, 2>, Tensor<B, 2>) = (c.squeeze(0).no_grad(), h.squeeze(0).no_grad());
        self.state = (c.clone(), h.clone());

        let final_output = self.final_model.forward(h).squeeze(0);
        let final_output = activation::tanh(final_output);

        final_output
    }

    pub fn crossover(
        self,
        other: SumFxLstmModel<B>,
        crossover_weight: f32,
        device: &Device<B>,
    ) -> Self {
        let being_model = combine_ffs(
            self.being_model,
            other.being_model,
            crossover_weight,
            1. - crossover_weight,
        );
        let fo_model = combine_ffs(
            self.fo_model,
            other.fo_model,
            crossover_weight,
            1. - crossover_weight,
        );
        let speechlet_model = combine_ffs(
            self.speechlet_model,
            other.speechlet_model,
            crossover_weight,
            1. - crossover_weight,
        );
        let self_model = combine_ffs(
            self.self_model,
            other.self_model,
            crossover_weight,
            1. - crossover_weight,
        );
        let final_model = combine_ffs(
            self.final_model,
            other.final_model,
            crossover_weight,
            1. - crossover_weight,
        );

        return SumFxLstmModel {
            being_model: being_model,
            fo_model: fo_model,
            speechlet_model: speechlet_model,
            self_model: self_model,
            final_model: final_model,

            lstm: combine_lstms(
                self.lstm,
                other.lstm,
                crossover_weight,
                1. - crossover_weight,
            ),

            concat_before_final: self.concat_before_final,
            intermediate_dim: self.intermediate_dim,
            lstm_inp_size: self.lstm_inp_size,
            state: (
                Tensor::<B, 2>::zeros([1, self.intermediate_dim as usize], device),
                Tensor::<B, 2>::zeros([1, self.intermediate_dim as usize], device),
            ),
        };
    }
    pub fn mutate(self, mutation_rate: f32, device: &Device<B>) -> Self {
        let mut new_models: Vec<FF<B>> = vec![];

        for model in [
            self.being_model,
            self.fo_model,
            self.speechlet_model,
            self.self_model,
            self.final_model,
        ] {
            let config = model.config.clone();
            let mutation_model = FF::new(config.0, config.1, device);
            let new_model = combine_ffs(model, mutation_model, 1., mutation_rate);
            new_models.push(new_model);
        }

        let mutation_lstm =
            LstmConfig::new(self.lstm_inp_size, self.lstm_inp_size, true).init(device);

        return SumFxLstmModel {
            being_model: new_models[0].to_owned(),
            fo_model: new_models[1].to_owned(),
            speechlet_model: new_models[2].to_owned(),
            self_model: new_models[3].to_owned(),
            lstm: combine_lstms(self.lstm, mutation_lstm, 1., mutation_rate),
            final_model: new_models[4].to_owned(),

            concat_before_final: self.concat_before_final,
            intermediate_dim: self.intermediate_dim,
            lstm_inp_size: self.lstm_inp_size,
            state: (
                Tensor::<B, 2>::zeros([1, self.intermediate_dim as usize], device),
                Tensor::<B, 2>::zeros([1, self.intermediate_dim as usize], device),
            ),
        };
    }
}
