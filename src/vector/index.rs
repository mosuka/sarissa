//! Vector indexing module for building and maintaining vector indexes.
//!
//! This module handles all vector index construction, maintenance, and optimization:
//! - Building HNSW, Flat, and IVF indexes
//! - Text embedding generation
//! - Vector quantization and compression
//! - Index optimization and maintenance

pub mod flat_builder;
pub mod hnsw_builder;
pub mod ivf_builder;
pub mod optimization;
pub mod quantization;
pub mod writer;

use serde::{Deserialize, Serialize};
use std::sync::{Arc, RwLock};

use crate::error::{Result, SageError};
use crate::vector::{DistanceMetric, Vector};

/// Configuration for vector index construction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorIndexBuildConfig {
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

impl Default for VectorIndexBuildConfig {
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

/// Trait for vector index builders.
pub trait VectorIndexBuilder: Send + Sync {
    /// Build an index from a collection of vectors.
    fn build(&mut self, vectors: Vec<(u64, Vector)>) -> Result<()>;

    /// Add vectors incrementally during construction.
    fn add_vectors(&mut self, vectors: Vec<(u64, Vector)>) -> Result<()>;

    /// Finalize the index construction.
    fn finalize(&mut self) -> Result<()>;

    /// Get build progress (0.0 to 1.0).
    fn progress(&self) -> f32;

    /// Get estimated memory usage.
    fn estimated_memory_usage(&self) -> usize;

    /// Optimize the built index.
    fn optimize(&mut self) -> Result<()>;

    /// Get access to the stored vectors.
    /// Returns a reference to the vectors stored in the builder.
    fn vectors(&self) -> &[(u64, Vector)];
}

/// Factory for creating vector index builders.
pub struct VectorIndexBuilderFactory;

impl VectorIndexBuilderFactory {
    /// Create a new vector index builder based on configuration.
    pub fn create_builder(config: VectorIndexBuildConfig) -> Result<Box<dyn VectorIndexBuilder>> {
        match config.index_type {
            VectorIndexType::Flat => {
                Ok(Box::new(flat_builder::FlatVectorIndexBuilder::new(config)?))
            }
            VectorIndexType::HNSW => Ok(Box::new(hnsw_builder::HnswIndexBuilder::new(config)?)),
            VectorIndexType::IVF => Ok(Box::new(ivf_builder::IvfIndexBuilder::new(config)?)),
        }
    }
}

/// In-memory vector index that manages the lifecycle of builders and readers.
/// This is similar to FileIndex in the lexical module.
pub struct VectorIndex {
    config: VectorIndexBuildConfig,
    builder: Arc<RwLock<Box<dyn VectorIndexBuilder>>>,
    is_finalized: Arc<RwLock<bool>>,
}

impl VectorIndex {
    /// Create a new in-memory vector index.
    pub fn create(config: VectorIndexBuildConfig) -> Result<Self> {
        let builder = VectorIndexBuilderFactory::create_builder(config.clone())?;
        Ok(Self {
            config,
            builder: Arc::new(RwLock::new(builder)),
            is_finalized: Arc::new(RwLock::new(false)),
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
    pub fn config(&self) -> &VectorIndexBuildConfig {
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

    /// Get a reader for this index.
    /// This creates an in-memory reader that can access the built index data.
    pub fn reader(&self) -> Result<crate::vector::reader::InMemoryVectorIndexReader> {
        let finalized = *self.is_finalized.read().unwrap();
        if !finalized {
            return Err(SageError::InvalidOperation(
                "Index must be finalized before creating a reader".to_string(),
            ));
        }

        // Access the builder to get vector data
        let builder = self.builder.read().unwrap();

        // Extract vectors from the builder
        let vectors = self.extract_vectors_from_builder(&**builder)?;

        crate::vector::reader::InMemoryVectorIndexReader::new(
            vectors,
            self.config.dimension,
            self.config.distance_metric,
        )
    }

    /// Extract vectors from the builder (helper method).
    fn extract_vectors_from_builder(
        &self,
        builder: &dyn VectorIndexBuilder,
    ) -> Result<Vec<(u64, Vector)>> {
        // Use the vectors() method from the trait
        Ok(builder.vectors().to_vec())
    }
}
