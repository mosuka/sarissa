//! VectorEngine embedding related type definitions.
//!
//! This module provides embedding registry, embedding instance, and embedding executor.

use std::collections::HashMap;
use std::future::Future;
use std::sync::{Arc, mpsc};

use parking_lot::RwLock;
use tokio::runtime::Builder as TokioRuntimeBuilder;

use crate::embedding::embedder::Embedder;
use crate::error::{Result, SarissaError};

/// Registry for managing embedder instances keyed byフィールド名.
///
/// Embedders are registered via `VectorIndexConfig.embedder` (e.g., `PerFieldEmbedder`).
pub(crate) struct VectorEmbedderRegistry {
    instances: RwLock<HashMap<String, Arc<dyn Embedder>>>,
}

impl VectorEmbedderRegistry {
    /// Create a new empty embedder registry.
    pub(crate) fn new() -> Self {
        Self {
            instances: RwLock::new(HashMap::new()),
        }
    }

    /// Register an embedder instance by its key（フィールド名）。
    pub(crate) fn register(&self, field_name: impl Into<String>, embedder: Arc<dyn Embedder>) {
        self.instances.write().insert(field_name.into(), embedder);
    }

    /// Resolve an embedder by its ID.
    ///
    /// Returns an error if the embedder is not registered.
    pub(crate) fn resolve(&self, field_name: &str) -> Result<Arc<dyn Embedder>> {
        let instances = self.instances.read();
        instances.get(field_name).cloned().ok_or_else(|| {
            SarissaError::invalid_config(format!(
                "embedder '{field_name}' is not registered. Use VectorIndexConfig.embedder field with PerFieldEmbedder to configure embedders."
            ))
        })
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
                SarissaError::internal(format!("failed to initialize embedder runtime: {err}"))
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
        rx.recv()
            .map_err(|err| SarissaError::internal(format!("embedder task channel closed: {err}")))?
    }
}
