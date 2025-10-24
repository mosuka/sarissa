//! Vector indexing module for building and maintaining vector indexes.
//!
//! This module handles all vector index construction, maintenance, and optimization:
//! - Building HNSW, Flat, and IVF indexes
//! - Text embedding generation
//! - Vector quantization and compression
//! - Index optimization and maintenance

pub mod optimization;
pub mod quantization;
pub mod reader;
pub mod reader_factory;
pub mod writer;

use serde::{Deserialize, Serialize};
use std::sync::{Arc, RwLock};

use crate::error::{Result, SageError};
use crate::storage::traits::Storage;
use crate::vector::writer::VectorIndexWriter;
use crate::vector::{DistanceMetric, Vector};

/// Configuration for vector index construction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorIndexWriterConfig {
    /// Vector dimension.
    pub dimension: usize,
    /// Index type to build.
    pub index_type: VectorIndexType,
    /// Distance metric to use.
    pub distance_metric: DistanceMetric,
    /// Whether to normalize vectors.
    pub normalize_vectors: bool,
    /// Whether to use quantization.
    pub use_quantization: bool,
    /// Quantization method.
    pub quantization_method: quantization::QuantizationMethod,
    /// Build in parallel.
    pub parallel_build: bool,
    /// Memory limit for construction (in bytes).
    pub memory_limit: Option<usize>,
}

impl Default for VectorIndexWriterConfig {
    fn default() -> Self {
        Self {
            dimension: 128,
            index_type: VectorIndexType::HNSW,
            distance_metric: DistanceMetric::Cosine,
            normalize_vectors: true,
            use_quantization: false,
            quantization_method: quantization::QuantizationMethod::None,
            parallel_build: true,
            memory_limit: None,
        }
    }
}

/// Types of vector indexes that can be built.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VectorIndexType {
    /// Flat index for exact search.
    Flat,
    /// HNSW index for approximate search.
    HNSW,
    /// IVF index for memory-efficient search.
    IVF,
}

/// Factory for creating vector index writers.
pub struct VectorIndexWriterFactory;

impl VectorIndexWriterFactory {
    /// Create a new vector index builder based on configuration.
    pub fn create_builder(config: VectorIndexWriterConfig) -> Result<Box<dyn VectorIndexWriter>> {
        match config.index_type {
            VectorIndexType::Flat => Ok(Box::new(writer::flat::FlatIndexWriter::new(config)?)),
            VectorIndexType::HNSW => Ok(Box::new(writer::hnsw::HnswIndexWriter::new(config)?)),
            VectorIndexType::IVF => Ok(Box::new(writer::ivf::IvfIndexWriter::new(config)?)),
        }
    }

    /// Create a new vector index builder with storage support.
    pub fn create_builder_with_storage(
        config: VectorIndexWriterConfig,
        storage: Arc<dyn Storage>,
    ) -> Result<Box<dyn VectorIndexWriter>> {
        match config.index_type {
            VectorIndexType::Flat => Ok(Box::new(writer::flat::FlatIndexWriter::with_storage(
                config, storage,
            )?)),
            VectorIndexType::HNSW => Ok(Box::new(writer::hnsw::HnswIndexWriter::with_storage(
                config, storage,
            )?)),
            VectorIndexType::IVF => Ok(Box::new(writer::ivf::IvfIndexWriter::with_storage(
                config, storage,
            )?)),
        }
    }

    /// Load an existing vector index from storage.
    pub fn load_builder(
        config: VectorIndexWriterConfig,
        storage: Arc<dyn Storage>,
        path: &str,
    ) -> Result<Box<dyn VectorIndexWriter>> {
        match config.index_type {
            VectorIndexType::Flat => Ok(Box::new(writer::flat::FlatIndexWriter::load(
                config, storage, path,
            )?)),
            VectorIndexType::HNSW => Ok(Box::new(writer::hnsw::HnswIndexWriter::load(
                config, storage, path,
            )?)),
            VectorIndexType::IVF => Ok(Box::new(writer::ivf::IvfIndexWriter::load(
                config, storage, path,
            )?)),
        }
    }
}

/// In-memory vector index that manages the lifecycle of builders and readers.
/// This is similar to FileIndex in the lexical module.
pub struct VectorIndex {
    config: VectorIndexWriterConfig,
    builder: Arc<RwLock<Box<dyn VectorIndexWriter>>>,
    is_finalized: Arc<RwLock<bool>>,
    storage: Option<Arc<dyn Storage>>,
}

impl VectorIndex {
    /// Create a new in-memory vector index.
    pub fn create(config: VectorIndexWriterConfig) -> Result<Self> {
        let builder = VectorIndexWriterFactory::create_builder(config.clone())?;
        Ok(Self {
            config,
            builder: Arc::new(RwLock::new(builder)),
            is_finalized: Arc::new(RwLock::new(false)),
            storage: None,
        })
    }

    /// Create a new vector index with storage support.
    pub fn create_with_storage(
        config: VectorIndexWriterConfig,
        storage: Arc<dyn Storage>,
    ) -> Result<Self> {
        let builder =
            VectorIndexWriterFactory::create_builder_with_storage(config.clone(), storage.clone())?;
        Ok(Self {
            config,
            builder: Arc::new(RwLock::new(builder)),
            is_finalized: Arc::new(RwLock::new(false)),
            storage: Some(storage),
        })
    }

    /// Load an existing vector index from storage.
    pub fn load(
        config: VectorIndexWriterConfig,
        storage: Arc<dyn Storage>,
        path: &str,
    ) -> Result<Self> {
        let builder =
            VectorIndexWriterFactory::load_builder(config.clone(), storage.clone(), path)?;
        Ok(Self {
            config,
            builder: Arc::new(RwLock::new(builder)),
            is_finalized: Arc::new(RwLock::new(true)),
            storage: Some(storage),
        })
    }

    /// Add vectors to the index.
    pub fn add_vectors(&mut self, vectors: Vec<(u64, Vector)>) -> Result<()> {
        let finalized = *self.is_finalized.read().unwrap();
        if finalized {
            return Err(SageError::InvalidOperation(
                "Cannot add vectors to finalized index".to_string(),
            ));
        }

        let mut builder = self.builder.write().unwrap();
        builder.add_vectors(vectors)?;
        Ok(())
    }

    /// Finalize the index construction.
    pub fn finalize(&mut self) -> Result<()> {
        let mut builder = self.builder.write().unwrap();
        builder.finalize()?;
        *self.is_finalized.write().unwrap() = true;
        Ok(())
    }

    /// Optimize the index.
    pub fn optimize(&mut self) -> Result<()> {
        let mut builder = self.builder.write().unwrap();
        builder.optimize()?;
        Ok(())
    }

    /// Get the configuration.
    pub fn config(&self) -> &VectorIndexWriterConfig {
        &self.config
    }

    /// Get build progress (0.0 to 1.0).
    pub fn progress(&self) -> f32 {
        let builder = self.builder.read().unwrap();
        builder.progress()
    }

    /// Get estimated memory usage.
    pub fn estimated_memory_usage(&self) -> usize {
        let builder = self.builder.read().unwrap();
        builder.estimated_memory_usage()
    }

    /// Check if the index is finalized.
    pub fn is_finalized(&self) -> bool {
        *self.is_finalized.read().unwrap()
    }

    /// Get vectors from this index.
    /// Returns a copy of all vectors stored in the index.
    pub fn vectors(&self) -> Result<Vec<(u64, Vector)>> {
        let finalized = *self.is_finalized.read().unwrap();
        if !finalized {
            return Err(SageError::InvalidOperation(
                "Index must be finalized before accessing vectors".to_string(),
            ));
        }

        let builder = self.builder.read().unwrap();
        Ok(builder.vectors().to_vec())
    }

    /// Write the index to storage.
    /// The index must be finalized before calling this method.
    pub fn write(&self, path: &str) -> Result<()> {
        let finalized = *self.is_finalized.read().unwrap();
        if !finalized {
            return Err(SageError::InvalidOperation(
                "Index must be finalized before writing".to_string(),
            ));
        }

        let builder = self.builder.read().unwrap();
        if !builder.has_storage() {
            return Err(SageError::InvalidOperation(
                "Index was not created with storage support".to_string(),
            ));
        }

        builder.write(path)
    }

    /// Check if this index has storage configured.
    pub fn has_storage(&self) -> bool {
        self.storage.is_some()
    }

    /// Create a reader for this index.
    /// Returns a boxed VectorIndexReader that can be used for searching.
    pub fn reader(&self) -> Result<Arc<dyn crate::vector::reader::VectorIndexReader>> {
        let finalized = *self.is_finalized.read().unwrap();
        if !finalized {
            return Err(SageError::InvalidOperation(
                "Index must be finalized before creating a reader".to_string(),
            ));
        }

        // If storage is available, load from storage
        if self.storage.is_some() {
            // We need a path to load from - for now, we'll use a default
            // In a real implementation, this would be stored in the VectorIndex
            return Err(SageError::InvalidOperation(
                "Reader creation from storage not yet implemented. Use load() instead.".to_string(),
            ));
        }

        // Otherwise, create from in-memory vectors
        let vectors = self.vectors()?;
        let reader = crate::vector::reader::SimpleVectorReader::new(
            vectors,
            self.config.dimension,
            self.config.distance_metric,
        )?;
        Ok(Arc::new(reader))
    }
}
