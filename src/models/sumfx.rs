use std::iter::zip;
use std::process::exit;

use burn::nn::Linear;
use burn::prelude::*;
use nn::LinearConfig;

use burn::module::{Module, Param};
use burn::tensor::backend::Backend;
use burn::tensor::{activation, BasicOps, Numeric, Tensor};

use crate::being_nn::{combine_linears, Activation, Tanh, FF};
use crate::{splice_ffs, B_OUTPUT_LEN, GENOME_LEN, SPEECHLET_LEN};

#[derive(Clone)]
pub struct SumFxModel<B: Backend> {
    pub being_model: FF<B>,
    pub fo_model: FF<B>,
    pub speechlet_model: FF<B>,
    pub self_model: FF<B>,

    pub final_model: FF<B>,

    pub concat_before_final: bool,
    pub intermediate_dim: usize,
}

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

        SumFxModel {
            being_model: FF::new(being_config.0, being_config.1, device),
            fo_model: FF::new(fo_config.0, fo_config.1, device),
            speechlet_model: FF::new(speechlet_config.0, speechlet_config.1, device),
            self_model: FF::new(self_config.0, self_config.1, device),
            final_model: FF::new(final_config.0, final_config.1, device),

            concat_before_final: concat_before_final,
            intermediate_dim: intermediate_dim,
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
        return SumFxModel::new(
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

        let final_output = self.final_model.forward(intermediate).squeeze(0);
        let final_output = activation::tanh(final_output);

        final_output
    }

    pub fn crossover(
        self,
        other: SumFxModel<B>,
        crossover_weight: f32,
        device: &Device<B>,
    ) -> Self {
        let being_model = splice_ffs(
            self.being_model,
            other.being_model,
            crossover_weight,
        );
        let fo_model = splice_ffs(
            self.fo_model,
            other.fo_model,
            crossover_weight,
        );
        let speechlet_model = splice_ffs(
            self.speechlet_model,
            other.speechlet_model,
            crossover_weight,
        );
        let self_model = splice_ffs(
            self.self_model,
            other.self_model,
            crossover_weight,
        );
        let final_model = splice_ffs(
            self.final_model,
            other.final_model,
            crossover_weight,
        );

        return SumFxModel {
            being_model: being_model,
            fo_model: fo_model,
            speechlet_model: speechlet_model,
            self_model: self_model,
            final_model: final_model,

            concat_before_final: self.concat_before_final,
            intermediate_dim: self.intermediate_dim,
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

            let model = splice_ffs(model, mutation_model, 1. - mutation_rate);
            new_models.push(model.clone());
        }

        return SumFxModel {
            being_model: new_models[0].to_owned(),
            fo_model: new_models[1].to_owned(),
            speechlet_model: new_models[2].to_owned(),
            self_model: new_models[3].to_owned(),
            final_model: new_models[4].to_owned(),

            concat_before_final: self.concat_before_final,
            intermediate_dim: self.intermediate_dim,
        };
    }
}
