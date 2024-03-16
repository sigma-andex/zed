// Adapted from https://github.com/huggingface/candle/blob/cdc4c172c42b5c31b3063afd20cc7055d60f9af8/candle-examples/examples/starcoder2/main.rs
use anyhow::{Error as E, Result};
use candle_transformers::models::starcoder2::Model;
use derive_new::new;

use super::logits_processor::LogitsProcessor;
use super::token_output_stream::TokenOutputStream;
use crate::providers::open_ai::{OpenAiRequest, RequestMessage};
use candle_core::utils::metal_is_available;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

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
            .map(|RequestMessage { content, role }| {
                println!("{:?} {}", role, content);
                match role {
                    crate::providers::open_ai::Role::User => format!("<USER>{}</USER>", content),
                    crate::providers::open_ai::Role::Assistant => {
                        format!("<ASSISTANT>{}</ASSISTANT>", content)
                    }
                    crate::providers::open_ai::Role::System => {
                        format!("<SYSTEM>{}</SYSTEM>", content)
                    }
                }
            })
            .collect();
        // let prompt = r#"// Fix this bug
        //     fn main() {
        //     println!("Hello, 🗺️");
        //     let x = 5;
        //     x = 10;
        //     println!("{}", x);
        // }"#
        // .to_string();
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
        let eos_token = match self.tokenizer.get_token("<|endoftext|>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find the <|endoftext|> token"),
        };
        let start_gen = std::time::Instant::now();
        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, start_pos)?;
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
                break;
            }
            if let Some(t) = self.tokenizer.next_token(next_token)? {
                print!("{t}\n");
                sender.unbounded_send(Ok(t))?;
                std::io::stdout().flush()?;
            }
        }
        let dt = start_gen.elapsed();
        if let Some(rest) = self.tokenizer.decode_rest().map_err(E::msg)? {
            print!("{rest}");
        }
        std::io::stdout().flush()?;
        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );
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

    config_file: Option<String>,

    tokenizer_file: Option<String>,

    weight_files: Option<String>,

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
        None => "bigcode/starcoder2-3b".to_string(),
    };
    let repo = api.repo(Repo::with_revision(
        model_id,
        RepoType::Model,
        args.revision,
    ));
    let config_file = match args.config_file {
        Some(file) => std::path::PathBuf::from(file),
        None => repo.get("config.json")?,
    };
    let tokenizer_file = match args.tokenizer_file {
        Some(file) => std::path::PathBuf::from(file),
        None => repo.get("tokenizer.json")?,
    };
    let filenames = match args.weight_files {
        Some(files) => files
            .split(',')
            .map(std::path::PathBuf::from)
            .collect::<Vec<_>>(),
        None => vec![repo.get("model.safetensors")?],
    };
    println!("retrieved the files in {:?}", start.elapsed());
    let tokenizer = Tokenizer::from_file(tokenizer_file).map_err(E::msg)?;

    let start = std::time::Instant::now();
    let config = serde_json::from_reader(std::fs::File::open(config_file)?)?;
    let device = if args.cpu || !metal_is_available() {
        println!("Using CPU 🐌");
        Ok::<Device, anyhow::Error>(Device::Cpu)
    } else {
        println!("Using Metal 🦾");
        Ok(Device::new_metal(0)?)
    }?;
    let dtype = if device.is_cuda() {
        DType::BF16
    } else {
        DType::F32
    };
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
    let model = Model::new(&config, vb)?;

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
