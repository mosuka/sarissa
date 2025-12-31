//! Vector indexing module for building and maintaining vector indexes.
//!
//! This module handles all vector index construction, maintenance, and optimization:
//! - Building HNSW, Flat, and IVF indexes
//! - Text embedding generation
//! - Vector quantization and compression
//! - Index optimization and maintenance

pub mod config;
pub mod factory;
pub mod field;
pub mod flat;
pub mod hnsw;
pub mod io;
pub mod ivf;
pub mod storage;

use std::sync::{Arc, RwLock};

use crate::error::{Result, SarissaError};
use crate::storage::Storage;
use crate::vector::core::vector::Vector;
use crate::vector::index::config::{
    FlatIndexConfig, HnswIndexConfig, IvfIndexConfig, VectorIndexTypeConfig,
};
use crate::vector::reader::VectorIndexReader;
use crate::vector::writer::VectorIndexWriter;

/// Trait for vector index implementations.
///
/// This trait defines the low-level interface for individual vector indexes
/// (Flat, HNSW, IVF, etc.). Each index type implements this trait to provide
/// reader/writer access and basic lifecycle management.
///
/// This is analogous to [`crate::lexical::index::LexicalIndex`] in the lexical module.
/// For high-level, document-centric operations, see
/// [`crate::vector::collection::VectorCollection`] which manages multiple
/// vector fields and is used by [`crate::vector::engine::VectorEngine`].
pub trait VectorIndex: Send + Sync + std::fmt::Debug {
    /// Get a reader for this index.
    ///
    /// Returns a reader that can be used to query the index.
    fn reader(&self) -> Result<Arc<dyn VectorIndexReader>>;

    /// Get a writer for this index.
    ///
    /// Returns a writer that can be used to add or update vectors.
    fn writer(&self) -> Result<Box<dyn VectorIndexWriter>>;

    /// Get the storage backend for this index.
    ///
    /// Returns a reference to the underlying storage.
    fn storage(&self) -> &Arc<dyn Storage>;

    /// Close the index and release resources.
    ///
    /// This should flush any pending writes and release all resources.
    /// Uses interior mutability for thread-safe access.
    fn close(&self) -> Result<()>;

    /// Check if the index is closed.
    ///
    /// Returns true if the index has been closed.
    fn is_closed(&self) -> bool;

    /// Get index statistics.
    ///
    /// Returns statistics about the index such as vector count, dimension, etc.
    fn stats(&self) -> Result<VectorIndexStats>;

    /// Optimize the index.
    ///
    /// Performs index optimization to improve query performance.
    /// Uses interior mutability for thread-safe access.
    fn optimize(&self) -> Result<()>;
}

/// Statistics about a vector index.
#[derive(Debug, Clone)]
pub struct VectorIndexStats {
    /// Number of vectors in the index.
    pub vector_count: u64,

    /// Dimension of vectors.
    pub dimension: usize,

    /// Total size of the index in bytes.
    pub total_size: u64,

    /// Number of deleted vectors.
    pub deleted_count: u64,

    /// Last modified time (seconds since epoch).
    pub last_modified: u64,
}

/// Internal implementation for managing vector index lifecycle.
///
/// This structure wraps a vector index writer and manages its state.
/// For most use cases, prefer using `VectorEngine` which provides a higher-level interface.
///
/// # Note
///
/// This is an internal implementation detail. The public API for vector indexes
/// is defined by the `VectorIndex` trait and `VectorIndexFactory`.
pub struct ManagedVectorIndex {
    config: VectorIndexTypeConfig,
    builder: Arc<RwLock<Box<dyn VectorIndexWriter>>>,
    is_finalized: Arc<RwLock<bool>>,
    storage: Option<Arc<dyn Storage>>,
}

impl ManagedVectorIndex {
    /// Create a new vector index with the given configuration and storage.
    ///
    /// # Arguments
    ///
    /// * `config` - Vector index configuration including index type
    /// * `storage` - Storage backend (MemoryStorage, FileStorage, etc.)
    pub fn new(config: VectorIndexTypeConfig, storage: Arc<dyn Storage>) -> Result<Self> {
        // Create builder based on config type
        let builder: Box<dyn VectorIndexWriter> = match &config {
            VectorIndexTypeConfig::Flat(flat_config) => {
                let writer_config = Self::default_writer_config();
                Box::new(flat::writer::FlatIndexWriter::with_storage(
                    flat_config.clone(),
                    writer_config,
                    storage.clone(),
                )?)
            }
            VectorIndexTypeConfig::HNSW(hnsw_config) => {
                let writer_config = Self::default_writer_config();
                Box::new(hnsw::writer::HnswIndexWriter::with_storage(
                    hnsw_config.clone(),
                    writer_config,
                    storage.clone(),
                )?)
            }
            VectorIndexTypeConfig::IVF(ivf_config) => {
                let writer_config = Self::default_writer_config();
                Box::new(ivf::writer::IvfIndexWriter::with_storage(
                    ivf_config.clone(),
                    writer_config,
                    storage.clone(),
                )?)
            }
        };

        Ok(Self {
            config,
            builder: Arc::new(RwLock::new(builder)),
            is_finalized: Arc::new(RwLock::new(false)),
            storage: Some(storage),
        })
    }

    /// Helper to create a default writer config.
    fn default_writer_config() -> crate::vector::writer::VectorIndexWriterConfig {
        crate::vector::writer::VectorIndexWriterConfig::default()
    }

    /// Add vectors to the index.
    pub fn add_vectors(&mut self, vectors: Vec<(u64, String, Vector)>) -> Result<()> {
        let finalized = *self.is_finalized.read().unwrap();
        if finalized {
            return Err(SarissaError::InvalidOperation(
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
    pub fn config(&self) -> &VectorIndexTypeConfig {
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
    pub fn vectors(&self) -> Result<Vec<(u64, String, Vector)>> {
        let finalized = *self.is_finalized.read().unwrap();
        if !finalized {
            return Err(SarissaError::InvalidOperation(
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
            return Err(SarissaError::InvalidOperation(
                "Index must be finalized before writing".to_string(),
            ));
        }

        let builder = self.builder.read().unwrap();
        if !builder.has_storage() {
            return Err(SarissaError::InvalidOperation(
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
            return Err(SarissaError::InvalidOperation(
                "Index must be finalized before creating a reader".to_string(),
            ));
        }

        // If storage is available, load from storage
        if self.storage.is_some() {
            // We need a path to load from - for now, we'll use a default
            // In a real implementation, this would be stored in the VectorIndex
            return Err(SarissaError::InvalidOperation(
                "Reader creation from storage not yet implemented. Use load() instead.".to_string(),
            ));
        }

        // Otherwise, create from in-memory vectors
        let vectors = self.vectors()?;
        let reader = crate::vector::reader::SimpleVectorReader::new(
            vectors,
            self.config.dimension(),
            self.config.distance_metric(),
        )?;
        Ok(Arc::new(reader))
    }
}
