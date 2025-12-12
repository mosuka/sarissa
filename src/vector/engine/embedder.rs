//! VectorEngine embedding related type definitions.
//!
//! This module provides embedding registry, embedding instance, and embedding executor.

use std::collections::HashMap;
use std::future::Future;
use std::sync::{Arc, mpsc};

use parking_lot::RwLock;
use tokio::runtime::Builder as TokioRuntimeBuilder;

use crate::embedding::image_embedder::ImageEmbedder;
use crate::embedding::text_embedder::TextEmbedder;
use crate::error::{PlatypusError, Result};

/// An instance holding text and/or image embedder references.
#[derive(Clone)]
pub(crate) struct EmbedderInstance {
    pub(crate) text: Option<Arc<dyn TextEmbedder>>,
    pub(crate) image: Option<Arc<dyn ImageEmbedder>>,
}

/// Registry for managing embedder instances.
///
/// This registry stores embedder instances that can be resolved by their ID.
/// Embedders are registered via `VectorIndexConfig.embedder` field using
/// `PerFieldEmbedder` or similar implementations.
pub(crate) struct VectorEmbedderRegistry {
    instances: RwLock<HashMap<String, EmbedderInstance>>,
}

impl VectorEmbedderRegistry {
    /// Create a new empty embedder registry.
    pub(crate) fn new() -> Self {
        Self {
            instances: RwLock::new(HashMap::new()),
        }
    }

    /// Resolve a text embedder by its ID.
    ///
    /// Returns an error if the embedder is not registered or does not support text embedding.
    pub(crate) fn resolve_text(&self, embedder_id: &str) -> Result<Arc<dyn TextEmbedder>> {
        let instances = self.instances.read();
        let instance = instances.get(embedder_id).ok_or_else(|| {
            PlatypusError::invalid_config(format!(
                "embedder '{embedder_id}' is not registered. Use VectorIndexConfig.embedder field with PerFieldEmbedder to configure embedders."
            ))
        })?;
        instance.text.clone().ok_or_else(|| {
            PlatypusError::invalid_config(format!(
                "embedder '{embedder_id}' does not expose text embedding capabilities"
            ))
        })
    }

    /// Resolve an image embedder by its ID.
    ///
    /// Returns an error if the embedder is not registered or does not support image embedding.
    pub(crate) fn resolve_image(&self, embedder_id: &str) -> Result<Arc<dyn ImageEmbedder>> {
        let instances = self.instances.read();
        let instance = instances.get(embedder_id).ok_or_else(|| {
            PlatypusError::invalid_config(format!(
                "embedder '{embedder_id}' is not registered. Use VectorIndexConfig.embedder field with PerFieldEmbedder to configure embedders."
            ))
        })?;
        instance.image.clone().ok_or_else(|| {
            PlatypusError::invalid_config(format!(
                "embedder '{embedder_id}' does not expose image embedding capabilities"
            ))
        })
    }

    /// Register an embedder instance from the Embedder trait object.
    ///
    /// This method is used by the `VectorIndexConfig.embedder` field API.
    pub(crate) fn register_from_embedder_trait(
        &self,
        embedder_id: String,
        text_embedder: Option<Arc<dyn TextEmbedder>>,
        image_embedder: Option<Arc<dyn ImageEmbedder>>,
    ) {
        // Only register if at least one embedder is provided
        if text_embedder.is_some() || image_embedder.is_some() {
            let instance = EmbedderInstance {
                text: text_embedder,
                image: image_embedder,
            };
            self.instances.write().insert(embedder_id, instance);
        }
    }
}

/// Executor for running async embedding operations.
#[derive(Clone)]
pub(crate) struct EmbedderExecutor {
    runtime: Arc<tokio::runtime::Runtime>,
}

impl EmbedderExecutor {
    /// Create a new embedder executor with a tokio runtime.
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

    /// Run an async future and wait for its result.
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
