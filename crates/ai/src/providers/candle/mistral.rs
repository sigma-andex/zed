// Adapted from https://github.com/huggingface/candle/blob/cdc4c172c42b5c31b3063afd20cc7055d60f9af8/candle-examples/examples/mistral/main.rs
use anyhow::{Error as E, Result};

use derive_new::new;

use super::logits_processor::LogitsProcessor;
use super::token_output_stream::TokenOutputStream;
use crate::providers::open_ai::{OpenAiRequest, RequestMessage};
use candle_core::utils::metal_is_available;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

use candle_transformers::models::mistral::{Config, Model as Mistral};
use candle_transformers::models::quantized_mistral::Model as QMistral;

#[derive(Clone)]
pub enum Model {
    Mistral(Mistral),
    Quantized(QMistral),
}

#[derive(Clone)]
pub struct TextGeneration {
    model: Model,
    device: Device,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: Model,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
        }
    }

    fn apply_chat_template(request: OpenAiRequest) -> String {
        let prompt = request
            .messages
            .into_iter()
            .map(|RequestMessage { content, role }| match role {
                crate::providers::open_ai::Role::User => {
                    format!("[INST]{}[/INST] ", content)
                }
                crate::providers::open_ai::Role::Assistant => {
                    format!("{}", content)
                }
                crate::providers::open_ai::Role::System => {
                    format!("<s>{}</s>", content)
                }
            })
            .collect();
        prompt
    }

    pub fn run(
        &mut self,
        sender: futures::channel::mpsc::UnboundedSender<anyhow::Result<String>>,
        prompt: OpenAiRequest,
        sample_len: usize,
    ) -> Result<()> {
        let prompt = TextGeneration::apply_chat_template(prompt);
        println!("Running model with prompt {:?}", &prompt);
        use std::io::Write;
        self.tokenizer.clear();
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        for &t in tokens.iter() {
            if let Some(t) = self.tokenizer.next_token(t)? {
                print!("{t}")
            }
        }
        std::io::stdout().flush()?;

        let mut generated_tokens = 0usize;
        let eos_token = match self.tokenizer.get_token("</s>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find the </s> token"),
        };
        let start_gen = std::time::Instant::now();
        println!("Starting to generate {} tokens...", sample_len);
        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);

            let ctxt: &[u32] = &tokens.get(start_pos..).unwrap_or_else(|| {
                println!("Index out of bounds");
                panic!("Index out of bounds")
            });

            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = match &mut self.model {
                Model::Mistral(m) => m.forward(&input, start_pos)?,
                Model::Quantized(m) => m.forward(&input, start_pos)?,
            };

            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
            };

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;
            if next_token == eos_token {
                println!("Finished generation.");
                break;
            }
            if let Some(t) = self.tokenizer.next_token(next_token)? {
                print!("{t}");
                sender.unbounded_send(Ok(t))?;
                std::io::stdout().flush()?;
            }
        }
        sender.close_channel();
        let dt = start_gen.elapsed();
        if let Some(rest) = self.tokenizer.decode_rest().map_err(E::msg)? {
            print!("{rest}");
        }
        std::io::stdout().flush()?;
        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );

        // [TODO] Fix KV cache shape
        print!("Clearing kv cache...");
        match &mut self.model {
            Model::Mistral(m) => {
                m.clear_kv_cache();
                ()
            }
            Model::Quantized(m) => {
                m.clear_kv_cache();
                ()
            }
        };
        println!("done!");
        Ok(())
    }
}

#[derive(Debug, new)]
pub struct ModelArgs {
    /// Run on CPU rather than on GPU.
    cpu: bool,

    /// The temperature used to generate samples.
    temperature: Option<f64>,

    /// Nucleus sampling probability cutoff.
    top_p: Option<f64>,

    /// The seed to use when generating random samples.
    seed: u64,

    model_id: Option<String>,

    revision: String,

    tokenizer_file: Option<String>,

    weight_files: Option<String>,

    // quantized: bool,
    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    repeat_last_n: usize,
}

pub fn load_model(args: ModelArgs) -> anyhow::Result<TextGeneration> {
    println!(
        "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
        args.temperature.unwrap_or(0.),
        args.repeat_penalty,
        args.repeat_last_n
    );

    let start = std::time::Instant::now();
    let api = Api::new()?;
    let model_id = match args.model_id {
        Some(model_id) => model_id,
        None => {
            // if args.quantized {
            "lmz/candle-mistral".to_string()
            // } else {
            //     "mistralai/Mistral-7B-v0.1".to_string()
            // }
        }
    };
    let repo = api.repo(Repo::with_revision(
        model_id,
        RepoType::Model,
        args.revision,
    ));
    let tokenizer_filename = match args.tokenizer_file {
        Some(file) => std::path::PathBuf::from(file),
        None => repo.get("tokenizer.json")?,
    };
    let filenames = match args.weight_files {
        Some(files) => files
            .split(',')
            .map(std::path::PathBuf::from)
            .collect::<Vec<_>>(),
        None => {
            // if args.quantized {
            vec![repo.get("model-q4k.gguf")?]
            // } else {
            //     candle_examples::hub_load_safetensors(&repo, "model.safetensors.index.json")?
            // }
        }
    };
    println!("retrieved the files in {:?}", start.elapsed());
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let start = std::time::Instant::now();
    let config = Config::config_7b_v0_1(false);
    let device = if args.cpu || !metal_is_available() {
        println!("Using CPU 🐌");
        Ok::<Device, anyhow::Error>(Device::Cpu)
    } else {
        println!("Using Metal 🦾");
        Ok(Device::new_metal(0)?)
    }?;
    // let (model, device) = if args.quantized {
    let filename = &filenames[0];
    let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(filename, &device)?;
    let model = Model::Quantized(QMistral::new(&config, vb)?);
    //     (Model::Quantized(model), device)
    // } else {
    //     let dtype = if device.is_cuda() {
    //         DType::BF16
    //     } else {
    //         DType::F32
    //     };
    //     let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
    //     let model = Mistral::new(&config, vb)?;
    //     (Model::Mistral(model), device)
    // };

    println!("loaded the model in {:?}", start.elapsed());

    let pipeline = TextGeneration::new(
        model,
        tokenizer,
        args.seed,
        args.temperature,
        args.top_p,
        args.repeat_penalty,
        args.repeat_last_n,
        &device,
    );
    // pipeline.run(&args.prompt, args.sample_len)?;
    Ok(pipeline)
}
