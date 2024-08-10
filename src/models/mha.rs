use std::iter::zip;

use burn::{
    module::Module,
    nn::{
        attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig},
        Linear, LinearConfig, Lstm, LstmConfig,
    },
    prelude::Backend,
    tensor::{activation, Device, Tensor},
};

use crate::{
    being_nn::{combine_linears, combine_lstms, combine_mhas, Activation, Tanh, FF}, combine_ffs, B_OUTPUT_LEN, GENOME_LEN, SPEECHLET_LEN
};

#[derive(Clone)]
pub struct MhaModel<B: Backend> {
    pub being_mha: MultiHeadAttention<B>,
    pub fo_mha: MultiHeadAttention<B>,
    pub speechlet_mha: MultiHeadAttention<B>,

    pub being_model: FF<B>,
    pub fo_model: FF<B>,
    pub speechlet_model: FF<B>,
    pub self_model: FF<B>,

    pub final_model: FF<B>,

    pub concat_before_final: bool,
    pub intermediate_dim: usize,
    pub num_heads: usize,
    pub inp_sizes: (usize, usize, usize),
}

impl<B: Backend> MhaModel<B> {
    pub fn new(
        being_config: (usize, usize, Activation),
        fo_config: (usize, usize, Activation),
        speechlet_config: (usize, usize, Activation),
        self_config: (Vec<usize>, Vec<Activation>),
        final_config: (Vec<usize>, Vec<Activation>),

        concat_before_final: bool,
        num_heads: usize,

        device: &Device<B>,
    ) -> Self {
        let (being_inp_size, being_out_size, being_act) = being_config;
        let (fo_inp_size, fo_out_size, fo_act) = fo_config;
        let (speechlet_inp_size, speechlet_out_size, speechlet_act) = speechlet_config;

        let lstm_inp_size = {
            if !concat_before_final {
                being_out_size
            } else {
                being_out_size + fo_out_size + speechlet_out_size + self_config.0.last().unwrap()
            }
        };

        let intermediate_dim: usize;

        if !concat_before_final {
            assert!(
                being_out_size == fo_out_size
                    && being_out_size == speechlet_out_size
                    && &being_out_size == self_config.0.last().unwrap(),
                "all sensory models must output the same shape, since you chose add mode"
            );
            assert!(
                final_config.0.first().unwrap() == &being_out_size,
                "sensory model output and final model input must be the same size, since you chose mean mode"
            );
            intermediate_dim = being_out_size;
        } else {
            assert!(
                &(being_out_size + fo_out_size + speechlet_out_size + self_config.0.last().unwrap()) == final_config.0.first().unwrap(),
                "sensory model output sizes must add up to final model input size, since you chose concat mode"
            );
            intermediate_dim =
                being_out_size + fo_out_size + speechlet_out_size + self_config.0.last().unwrap();
        }

        MhaModel {
            being_mha: MultiHeadAttentionConfig::new(being_inp_size, num_heads)
                .init(device)
                .no_grad(),
            fo_mha: MultiHeadAttentionConfig::new(fo_inp_size, num_heads)
                .init(device)
                .no_grad(),
            speechlet_mha: MultiHeadAttentionConfig::new(speechlet_inp_size, num_heads)
                .init(device)
                .no_grad(),

            being_model: FF::new(
                vec![being_inp_size, being_out_size],
                vec![being_act.clone(), being_act],
                device,
            ),
            fo_model: FF::new(
                vec![fo_inp_size, fo_out_size],
                vec![fo_act.clone(), fo_act],
                device,
            ),
            speechlet_model: FF::new(
                vec![speechlet_inp_size, speechlet_out_size],
                vec![speechlet_act.clone(), speechlet_act],
                device,
            ),
            self_model: FF::new(self_config.0, self_config.1, device),
            final_model: FF::new(final_config.0, final_config.1, device),

            concat_before_final: concat_before_final,
            intermediate_dim: intermediate_dim,
            num_heads: num_heads,
            inp_sizes: (being_inp_size, fo_inp_size, speechlet_inp_size),
        }
    }

    pub fn standard_model(device: &Device<B>) -> Self {
        let being_config = (3 + GENOME_LEN, 8, Activation::Tanh(Tanh {}));
        let fo_config = (5, 8, Activation::Tanh(Tanh {}));
        let speechlet_config = (SPEECHLET_LEN, 8, Activation::Tanh(Tanh {}));
        let self_config = (
            vec![5, 8],
            vec![Activation::Tanh(Tanh {}), Activation::Tanh(Tanh {})],
        );
        let final_config = (
            vec![32, B_OUTPUT_LEN],
            vec![Activation::Tanh(Tanh {}), Activation::Tanh(Tanh {})],
        );
        return MhaModel::new(
            being_config,
            fo_config,
            speechlet_config,
            self_config,
            final_config,
            true,
            1,
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
        let (being_tensor, fo_tensor, speechlet_tensor) = (
            being_tensor.unsqueeze(),
            fo_tensor.unsqueeze(),
            speechlet_tensor.unsqueeze(),
        );

        let beings_output = self
            .being_mha
            .forward(MhaInput::new(
                being_tensor.clone(),
                being_tensor.clone(),
                being_tensor,
            ))
            .context
            .squeeze(0);
        let beings_output = self.being_model.forward(beings_output).mean_dim(0);

        let fo_output = self
            .fo_mha
            .forward(MhaInput::new(
                fo_tensor.clone(),
                fo_tensor.clone(),
                fo_tensor.clone(),
            ))
            .context
            .squeeze(0);
        let fo_output = self.fo_model.forward(fo_output).mean_dim(0);

        let speechlet_output = self
            .speechlet_mha
            .forward(MhaInput::new(
                speechlet_tensor.clone(),
                speechlet_tensor.clone(),
                speechlet_tensor.clone(),
            ))
            .context
            .squeeze(0);
        let speechlet_output = self.speechlet_model.forward(speechlet_output).mean_dim(0);

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

    pub fn crossover(self, other: Self, crossover_weight: f32, device: &Device<B>) -> Self {
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

        return MhaModel {
            being_mha: combine_mhas(
                self.being_mha,
                other.being_mha,
                crossover_weight,
                1. - crossover_weight,
            ),
            fo_mha: combine_mhas(
                self.fo_mha,
                other.fo_mha,
                crossover_weight,
                1. - crossover_weight,
            ),
            speechlet_mha: combine_mhas(
                self.speechlet_mha,
                other.speechlet_mha,
                crossover_weight,
                1. - crossover_weight,
            ),

            being_model: being_model,
            fo_model: fo_model,
            speechlet_model: speechlet_model,
            self_model: self_model,
            final_model: final_model,

            concat_before_final: self.concat_before_final,
            intermediate_dim: self.intermediate_dim,
            num_heads: self.num_heads,
            inp_sizes: self.inp_sizes,
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

        let being_mutation =
            MultiHeadAttentionConfig::new(self.inp_sizes.0, self.num_heads).init(device);
        let fo_mutation =
            MultiHeadAttentionConfig::new(self.inp_sizes.1, self.num_heads).init(device);
        let speechlet_mutation =
            MultiHeadAttentionConfig::new(self.inp_sizes.2, self.num_heads).init(device);

        return MhaModel {
            self_model: new_models[3].to_owned(),
            final_model: new_models[4].to_owned(),

            concat_before_final: self.concat_before_final,
            intermediate_dim: self.intermediate_dim,

            being_mha: combine_mhas(self.being_mha, being_mutation, 1., mutation_rate),
            fo_mha: combine_mhas(self.fo_mha, fo_mutation, 1., mutation_rate),
            speechlet_mha: combine_mhas(self.speechlet_mha, speechlet_mutation, 1., mutation_rate),

            being_model: new_models[0].to_owned(),
            fo_model: new_models[1].to_owned(),
            speechlet_model: new_models[2].to_owned(),

            num_heads: self.num_heads,
            inp_sizes: self.inp_sizes,
        };
    }
}
