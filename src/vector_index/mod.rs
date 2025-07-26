//! Vector indexing module for building and maintaining vector indexes.
//!
//! This module handles all vector index construction, maintenance, and optimization:
//! - Building HNSW, Flat, and IVF indexes
//! - Text embedding generation
//! - Vector quantization and compression
//! - Index optimization and maintenance

pub mod embeddings;
pub mod flat_builder;
pub mod hnsw_builder;
pub mod ivf_builder;
pub mod optimization;
pub mod quantization;
pub mod writer;

pub use embeddings::{EmbeddingConfig, EmbeddingEngine, EmbeddingMethod};
pub use flat_builder::FlatVectorIndexBuilder;
pub use hnsw_builder::HnswIndexBuilder;
pub use ivf_builder::IvfIndexBuilder;
pub use optimization::VectorIndexOptimizer;
pub use quantization::{QuantizationMethod, VectorQuantizer};
pub use writer::{VectorIndexWriter, VectorWriterConfig};

use crate::error::Result;
use crate::vector::{DistanceMetric, Vector};
use serde::{Deserialize, Serialize};

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
    pub quantization_method: QuantizationMethod,
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
            quantization_method: QuantizationMethod::None,
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
}

/// Factory for creating vector index builders.
pub struct VectorIndexBuilderFactory;

impl VectorIndexBuilderFactory {
    /// Create a new vector index builder based on configuration.
    pub fn create_builder(config: VectorIndexBuildConfig) -> Result<Box<dyn VectorIndexBuilder>> {
        match config.index_type {
            VectorIndexType::Flat => Ok(Box::new(FlatVectorIndexBuilder::new(config)?)),
            VectorIndexType::HNSW => Ok(Box::new(HnswIndexBuilder::new(config)?)),
            VectorIndexType::IVF => Ok(Box::new(IvfIndexBuilder::new(config)?)),
        }
    }
}
