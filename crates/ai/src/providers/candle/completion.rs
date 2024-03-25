use std::{
    panic,
    sync::{Arc, Mutex},
};

use super::mistral;
use super::mistral::TextGeneration;
use crate::{
    auth::{CredentialProvider, ProviderCredential},
    completion::CompletionProvider,
    models::TruncationDirection,
};
use anyhow::Result;
use futures::{future::BoxFuture, stream, FutureExt, Stream, StreamExt};
use gpui::BackgroundExecutor;
use util::ResultExt;

#[derive(Clone)]
pub struct CandleAiLanguageModel {
    pipeline: Arc<Mutex<TextGeneration>>,
}
impl CandleAiLanguageModel {
    pub fn load(model_name: &str) -> anyhow::Result<Self> {
        let seed = rand::random();
        let model_id = "lmz/candle-mistral";
        let args = mistral::ModelArgs::new(
            false,
            Some(0.5),
            None,
            seed,
            Some(model_id.to_string()),
            "main".to_string(),
            None,
            None,
            1.1,
            64,
        );
        let pipeline = Arc::new(Mutex::new(mistral::load_model(args)?));
        Ok(Self { pipeline })
    }
}

#[derive(Clone)]
pub struct CandleCompletionProvider {
    model: CandleAiLanguageModel,
    executor: BackgroundExecutor,
}

impl CandleCompletionProvider {
    pub async fn new(model_name: String, executor: BackgroundExecutor) -> anyhow::Result<Self> {
        let model = executor
            .spawn(async move { CandleAiLanguageModel::load(&model_name) })
            .await?;
        Ok(Self { model, executor })
    }
}
impl CredentialProvider for CandleCompletionProvider {
    fn has_credentials(&self) -> bool {
        true
    }

    fn retrieve_credentials(
        &self,
        cx: &mut gpui::AppContext,
    ) -> futures::prelude::future::BoxFuture<crate::auth::ProviderCredential> {
        async move { ProviderCredential::NotNeeded }.boxed()
    }

    fn save_credentials(
        &self,
        cx: &mut gpui::AppContext,
        credential: crate::auth::ProviderCredential,
    ) -> futures::prelude::future::BoxFuture<()> {
        async move { () }.boxed()
    }

    fn delete_credentials(
        &self,
        cx: &mut gpui::AppContext,
    ) -> futures::prelude::future::BoxFuture<()> {
        async move { () }.boxed()
    }
}
#[derive(Clone)]
pub struct FakeLanguageModel {
    pub capacity: usize,
}

impl crate::models::LanguageModel for FakeLanguageModel {
    fn name(&self) -> String {
        "dummy".to_string()
    }
    fn count_tokens(&self, content: &str) -> anyhow::Result<usize> {
        anyhow::Ok(content.chars().collect::<Vec<char>>().len())
    }
    fn truncate(
        &self,
        content: &str,
        length: usize,
        direction: TruncationDirection,
    ) -> anyhow::Result<String> {
        println!("TRYING TO TRUNCATE: {:?}", length.clone());

        if length > self.count_tokens(content)? {
            println!("NOT TRUNCATING");
            return anyhow::Ok(content.to_string());
        }

        anyhow::Ok(match direction {
            TruncationDirection::End => content.chars().collect::<Vec<char>>()[..length]
                .into_iter()
                .collect::<String>(),
            TruncationDirection::Start => content.chars().collect::<Vec<char>>()[length..]
                .into_iter()
                .collect::<String>(),
        })
    }
    fn capacity(&self) -> anyhow::Result<usize> {
        anyhow::Ok(self.capacity)
    }
}
async fn stream_completion(
    executor: BackgroundExecutor,
    pipeline: Arc<Mutex<TextGeneration>>,
    request: Box<dyn crate::completion::CompletionRequest>,
) -> Result<impl Stream<Item = Result<String>>> {
    let prompt = request.as_openai_request();

    let (tx, rx) = futures::channel::mpsc::unbounded::<Result<String>>();
    let _text_generation = executor
        .spawn(async move {
            let mut pl = pipeline.lock().unwrap();

            let result = pl.run(tx, prompt, 128);
            match result {
                Ok(_) => {}
                Err(err) => {
                    println!("{:?}", err);
                }
            }
            drop(pl);
            anyhow::Ok(())
        })
        .detach();

    Ok(rx)
}
impl CompletionProvider for CandleCompletionProvider {
    fn base_model(&self) -> Box<dyn crate::models::LanguageModel> {
        Box::new(FakeLanguageModel { capacity: 100 })
    }

    fn complete(
        &self,
        prompt: Box<dyn crate::completion::CompletionRequest>,
    ) -> futures::prelude::future::BoxFuture<
        'static,
        gpui::Result<futures::prelude::stream::BoxStream<'static, gpui::Result<String>>>,
    > {
        let pipeline = self.model.pipeline.clone();
        let stream = stream_completion(self.executor.clone(), pipeline, prompt);

        async move {
            let stream = stream.await?;
            let stream = stream.boxed();
            Ok(stream)
        }
        .boxed()
    }

    fn box_clone(&self) -> Box<dyn CompletionProvider> {
        Box::new((*self).clone())
    }
}
