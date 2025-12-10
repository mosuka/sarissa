//! VectorEngine 埋め込み関連の型定義
//!
//! このモジュールは埋め込みレジストリ、埋め込みインスタンス、埋め込み実行器を提供する。

use std::collections::HashMap;
use std::future::Future;
use std::sync::{Arc, mpsc};

use parking_lot::RwLock;
use tokio::runtime::Builder as TokioRuntimeBuilder;

use crate::embedding::image_embedder::ImageEmbedder;
use crate::embedding::text_embedder::TextEmbedder;
use crate::error::{PlatypusError, Result};
use crate::vector::engine::config::{VectorEmbedderConfig, VectorEmbedderProvider};

#[cfg(feature = "embeddings-multimodal")]
use crate::embedding::candle_multimodal_embedder::CandleMultimodalEmbedder;
#[cfg(feature = "embeddings-candle")]
use crate::embedding::candle_text_embedder::CandleTextEmbedder;
#[cfg(feature = "embeddings-openai")]
use crate::embedding::openai_text_embedder::OpenAITextEmbedder;

#[derive(Clone)]
pub(crate) struct EmbedderInstance {
    pub(crate) text: Option<Arc<dyn TextEmbedder>>,
    pub(crate) image: Option<Arc<dyn ImageEmbedder>>,
}

impl EmbedderInstance {
    pub(crate) fn text_only(embedder: Arc<dyn TextEmbedder>) -> Self {
        Self {
            text: Some(embedder),
            image: None,
        }
    }

    pub(crate) fn text_and_image(text: Arc<dyn TextEmbedder>, image: Arc<dyn ImageEmbedder>) -> Self {
        Self {
            text: Some(text),
            image: Some(image),
        }
    }
}

pub(crate) struct VectorEmbedderRegistry {
    configs: HashMap<String, VectorEmbedderConfig>,
    instances: RwLock<HashMap<String, EmbedderInstance>>,
}

impl VectorEmbedderRegistry {
    pub(crate) fn new(configs: HashMap<String, VectorEmbedderConfig>) -> Self {
        Self {
            configs,
            instances: RwLock::new(HashMap::new()),
        }
    }

    pub(crate) fn resolve_text(&self, embedder_id: &str) -> Result<Arc<dyn TextEmbedder>> {
        let instance = self.ensure_instance(embedder_id)?;
        instance.text.clone().ok_or_else(|| {
            PlatypusError::invalid_config(format!(
                "embedder '{embedder_id}' does not expose text embedding capabilities"
            ))
        })
    }

    pub(crate) fn resolve_image(&self, embedder_id: &str) -> Result<Arc<dyn ImageEmbedder>> {
        let instance = self.ensure_instance(embedder_id)?;
        instance.image.clone().ok_or_else(|| {
            PlatypusError::invalid_config(format!(
                "embedder '{embedder_id}' does not expose image embedding capabilities"
            ))
        })
    }

    fn ensure_instance(&self, embedder_id: &str) -> Result<EmbedderInstance> {
        if let Some(instance) = self.instances.read().get(embedder_id) {
            return Ok(instance.clone());
        }

        let config = self.configs.get(embedder_id).ok_or_else(|| {
            PlatypusError::invalid_config(format!(
                "embedder '{embedder_id}' is not defined in VectorEngineConfig.embedders"
            ))
        })?;

        let instance = self.instantiate(embedder_id, config)?;
        self.instances
            .write()
            .insert(embedder_id.to_string(), instance.clone());
        Ok(instance)
    }

    fn instantiate(
        &self,
        embedder_id: &str,
        config: &VectorEmbedderConfig,
    ) -> Result<EmbedderInstance> {
        match config.provider {
            VectorEmbedderProvider::CandleText => {
                Self::instantiate_candle_text(embedder_id, config)
            }
            VectorEmbedderProvider::CandleMultimodal => {
                Self::instantiate_candle_multimodal(embedder_id, config)
            }
            VectorEmbedderProvider::OpenAiText => {
                Self::instantiate_openai_text(embedder_id, config)
            }
            VectorEmbedderProvider::External => Err(PlatypusError::invalid_config(format!(
                "embedder '{embedder_id}' uses provider 'external' and requires a runtime instance via register_embedder_instance"
            ))),
        }
    }

    pub(crate) fn register_external(
        &self,
        embedder_id: String,
        embedder: Arc<dyn TextEmbedder>,
    ) -> Result<()> {
        self.register_external_with_image(embedder_id, embedder, None)
    }

    pub(crate) fn register_external_with_image(
        &self,
        embedder_id: String,
        text_embedder: Arc<dyn TextEmbedder>,
        image_embedder: Option<Arc<dyn ImageEmbedder>>,
    ) -> Result<()> {
        let config = self.configs.get(&embedder_id).ok_or_else(|| {
            PlatypusError::invalid_config(format!(
                "cannot register embedder '{embedder_id}' that is not declared in config"
            ))
        })?;

        if !matches!(config.provider, VectorEmbedderProvider::External) {
            return Err(PlatypusError::invalid_config(format!(
                "embedder '{embedder_id}' does not use provider 'external'"
            )));
        }

        let instance = match image_embedder {
            Some(image) => EmbedderInstance::text_and_image(text_embedder, image),
            None => EmbedderInstance::text_only(text_embedder),
        };
        self.instances.write().insert(embedder_id, instance);
        Ok(())
    }

    #[cfg(feature = "embeddings-candle")]
    fn instantiate_candle_text(
        _embedder_id: &str,
        config: &VectorEmbedderConfig,
    ) -> Result<EmbedderInstance> {
        let embedder = Arc::new(CandleTextEmbedder::new(config.model.as_str())?);
        Ok(EmbedderInstance::text_only(embedder))
    }

    #[cfg(not(feature = "embeddings-candle"))]
    fn instantiate_candle_text(
        embedder_id: &str,
        _: &VectorEmbedderConfig,
    ) -> Result<EmbedderInstance> {
        Err(Self::missing_feature("embeddings-candle", embedder_id))
    }

    #[cfg(feature = "embeddings-multimodal")]
    fn instantiate_candle_multimodal(
        _embedder_id: &str,
        config: &VectorEmbedderConfig,
    ) -> Result<EmbedderInstance> {
        let embedder = Arc::new(CandleMultimodalEmbedder::new(config.model.as_str())?);
        let text: Arc<dyn TextEmbedder> = embedder.clone();
        let image: Arc<dyn ImageEmbedder> = embedder;
        Ok(EmbedderInstance::text_and_image(text, image))
    }

    #[cfg(not(feature = "embeddings-multimodal"))]
    fn instantiate_candle_multimodal(
        embedder_id: &str,
        _: &VectorEmbedderConfig,
    ) -> Result<EmbedderInstance> {
        Err(Self::missing_feature("embeddings-multimodal", embedder_id))
    }

    #[cfg(feature = "embeddings-openai")]
    fn instantiate_openai_text(
        _embedder_id: &str,
        config: &VectorEmbedderConfig,
    ) -> Result<EmbedderInstance> {
        let api_key = config
            .options
            .get("api_key")
            .and_then(|value| value.as_str())
            .ok_or_else(|| {
                PlatypusError::invalid_config("OpenAI embedder requires 'api_key' option")
            })?
            .to_string();

        let dimension = config
            .options
            .get("dimension")
            .and_then(|value| value.as_u64())
            .map(|value| value as usize);

        let embedder = if let Some(custom_dim) = dimension {
            OpenAITextEmbedder::with_dimension(api_key, config.model.clone(), custom_dim)?
        } else {
            OpenAITextEmbedder::new(api_key, config.model.clone())?
        };

        Ok(EmbedderInstance::text_only(Arc::new(embedder)))
    }

    #[cfg(not(feature = "embeddings-openai"))]
    fn instantiate_openai_text(
        embedder_id: &str,
        _: &VectorEmbedderConfig,
    ) -> Result<EmbedderInstance> {
        Err(Self::missing_feature("embeddings-openai", embedder_id))
    }

    #[allow(dead_code)]
    fn missing_feature(feature: &str, embedder_id: &str) -> PlatypusError {
        PlatypusError::invalid_config(format!(
            "embedder '{embedder_id}' requires feature '{feature}'"
        ))
    }
}

#[derive(Clone)]
pub(crate) struct EmbedderExecutor {
    runtime: Arc<tokio::runtime::Runtime>,
}

impl EmbedderExecutor {
    pub(crate) fn new() -> Result<Self> {
        let runtime = TokioRuntimeBuilder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .map_err(|err| {
                PlatypusError::internal(format!("failed to initialize embedder runtime: {err}"))
            })?;
        Ok(Self {
            runtime: Arc::new(runtime),
        })
    }

    pub(crate) fn run<F, T>(&self, future: F) -> Result<T>
    where
        F: Future<Output = Result<T>> + Send + 'static,
        T: Send + 'static,
    {
        let (tx, rx) = mpsc::channel();
        let handle = self.runtime.handle().clone();
        handle.spawn(async move {
            let _ = tx.send(future.await);
        });
        rx.recv().map_err(|err| {
            PlatypusError::internal(format!("embedder task channel closed: {err}"))
        })?
    }
}
