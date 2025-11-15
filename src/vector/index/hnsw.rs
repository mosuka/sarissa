//! HNSW vector index implementation.

pub mod maintenance;
pub mod reader;
pub mod searcher;
pub mod segment;
pub mod writer;

use std::path::Path;
use std::sync::Arc;

use crate::error::{Result, YatagarasuError};
use crate::storage::Storage;
use crate::vector::index::config::HnswIndexConfig;
use crate::vector::index::hnsw::writer::HnswIndexWriter;
use crate::vector::index::{VectorIndex, VectorIndexStats};
use crate::vector::reader::VectorIndexReader;
use crate::vector::writer::{VectorIndexWriter, VectorIndexWriterConfig};

/// Metadata for the HNSW index.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct IndexMetadata {
    /// Number of vectors in the index.
    vector_count: u64,
    /// Vector dimension.
    dimension: usize,
    /// Creation timestamp.
    created: u64,
    /// Last modification timestamp.
    modified: u64,
}

impl Default for IndexMetadata {
    fn default() -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        Self {
            vector_count: 0,
            dimension: 0,
            created: now,
            modified: now,
        }
    }
}

/// A concrete HNSW vector index implementation.
#[derive(Debug)]
pub struct HnswIndex {
    /// The storage backend.
    storage: Arc<dyn Storage>,

    /// HNSW index specific configuration.
    config: HnswIndexConfig,

    /// Whether the index is closed.
    closed: bool,

    /// Index metadata.
    metadata: IndexMetadata,
}

impl HnswIndex {
    /// Create a new HNSW index in the given storage.
    pub fn create(storage: Arc<dyn Storage>, config: HnswIndexConfig) -> Result<Self> {
        let metadata = IndexMetadata {
            dimension: config.dimension,
            ..Default::default()
        };

        let index = HnswIndex {
            storage,
            config,
            closed: false,
            metadata,
        };

        index.write_metadata()?;
        Ok(index)
    }

    /// Open an existing HNSW index from storage.
    pub fn open(storage: Arc<dyn Storage>, config: HnswIndexConfig) -> Result<Self> {
        if !storage.file_exists("metadata.json") {
            return Err(YatagarasuError::index("Index does not exist"));
        }

        let metadata = Self::read_metadata(storage.as_ref())?;

        Ok(HnswIndex {
            storage,
            config,
            closed: false,
            metadata,
        })
    }

    /// Create an index in a directory.
    pub fn create_in_dir<P: AsRef<Path>>(dir: P, config: HnswIndexConfig) -> Result<Self> {
        use crate::storage::file::{FileStorage, FileStorageConfig};

        let storage_config = FileStorageConfig::new(&dir);
        let storage = Arc::new(FileStorage::new(&dir, storage_config)?);
        Self::create(storage, config)
    }

    /// Open an index from a directory.
    pub fn open_dir<P: AsRef<Path>>(dir: P, config: HnswIndexConfig) -> Result<Self> {
        use crate::storage::file::{FileStorage, FileStorageConfig};

        let storage_config = FileStorageConfig::new(&dir);
        let storage = Arc::new(FileStorage::new(&dir, storage_config)?);
        Self::open(storage, config)
    }

    /// Write metadata to storage.
    fn write_metadata(&self) -> Result<()> {
        let metadata_json = serde_json::to_string_pretty(&self.metadata)
            .map_err(|e| YatagarasuError::index(format!("Failed to serialize metadata: {e}")))?;

        let mut output = self.storage.create_output("metadata.json")?;
        std::io::Write::write_all(&mut output, metadata_json.as_bytes())?;
        output.close()?;

        Ok(())
    }

    /// Read metadata from storage.
    fn read_metadata(storage: &dyn Storage) -> Result<IndexMetadata> {
        let input = storage.open_input("metadata.json")?;
        let metadata: IndexMetadata = serde_json::from_reader(input)
            .map_err(|e| YatagarasuError::index(format!("Failed to deserialize metadata: {e}")))?;
        Ok(metadata)
    }

    /// Update metadata.
    fn update_metadata(&mut self) -> Result<()> {
        self.metadata.modified = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        self.write_metadata()
    }

    /// Check if the index is closed.
    fn check_closed(&self) -> Result<()> {
        if self.closed {
            return Err(YatagarasuError::InvalidOperation(
                "Index is closed".to_string(),
            ));
        }
        Ok(())
    }
}

impl VectorIndex for HnswIndex {
    fn reader(&self) -> Result<Arc<dyn VectorIndexReader>> {
        self.check_closed()?;

        use crate::vector::index::hnsw::reader::HnswIndexReader;

        let reader = HnswIndexReader::load(
            self.storage.clone(),
            "default_index",
            self.config.distance_metric,
        )?;
        Ok(Arc::new(reader))
    }

    fn writer(&self) -> Result<Box<dyn VectorIndexWriter>> {
        self.check_closed()?;

        let writer = HnswIndexWriter::with_storage(
            self.config.clone(),
            VectorIndexWriterConfig::default(),
            self.storage.clone(),
        )?;
        Ok(Box::new(writer))
    }

    fn storage(&self) -> &Arc<dyn Storage> {
        &self.storage
    }

    fn close(&mut self) -> Result<()> {
        if !self.closed {
            self.closed = true;
        }
        Ok(())
    }

    fn is_closed(&self) -> bool {
        self.closed
    }

    fn stats(&self) -> Result<VectorIndexStats> {
        self.check_closed()?;

        Ok(VectorIndexStats {
            vector_count: self.metadata.vector_count,
            dimension: self.metadata.dimension,
            total_size: 0,
            deleted_count: 0,
            last_modified: self.metadata.modified,
        })
    }

    fn optimize(&mut self) -> Result<()> {
        self.check_closed()?;
        self.update_metadata()?;
        Ok(())
    }
}
