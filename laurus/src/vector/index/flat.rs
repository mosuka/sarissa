//! Flat vector index implementation.

pub mod reader;
pub mod searcher;
pub mod segment;
pub mod segmented;
pub mod writer;

#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use parking_lot::RwLock;

use crate::embedding::embedder::Embedder;
use crate::error::{LaurusError, Result};
use crate::storage::Storage;
use crate::storage::manifest as manifest_io;
use crate::vector::index::config::FlatIndexConfig;
use crate::vector::index::flat::searcher::FlatVectorSearcher;
use crate::vector::index::flat::writer::FlatIndexWriter;
use crate::vector::index::{VectorIndex, VectorIndexStats};
use crate::vector::reader::VectorIndexReader;
use crate::vector::search::searcher::VectorIndexSearcher;
use crate::vector::store::embedding_writer::EmbeddingVectorIndexWriter;
use crate::vector::writer::{VectorIndexWriter, VectorIndexWriterConfig};

/// Metadata for the flat index.
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
        let now = crate::util::time::now_secs();
        Self {
            vector_count: 0,
            dimension: 0,
            created: now,
            modified: now,
        }
    }
}

/// A concrete flat vector index implementation.
pub struct FlatIndex {
    /// The name of the index.
    name: String,

    /// The storage backend.
    storage: Arc<dyn Storage>,

    /// Flat index specific configuration.
    config: FlatIndexConfig,

    /// Whether the index is closed (thread-safe).
    closed: AtomicBool,

    /// Index metadata (thread-safe).
    metadata: RwLock<IndexMetadata>,
}

impl std::fmt::Debug for FlatIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FlatIndex")
            .field("name", &self.name)
            .field("storage", &self.storage)
            .field("config", &self.config)
            .field("closed", &self.closed.load(Ordering::SeqCst))
            .field("metadata", &*self.metadata.read())
            .finish()
    }
}

impl FlatIndex {
    /// Create a new flat index in the given storage.
    pub fn create(storage: Arc<dyn Storage>, name: &str, config: FlatIndexConfig) -> Result<Self> {
        let metadata = IndexMetadata {
            dimension: config.dimension,
            ..Default::default()
        };

        let index = FlatIndex {
            name: name.to_string(),
            storage,
            config,
            closed: AtomicBool::new(false),
            metadata: RwLock::new(metadata),
        };

        index.write_metadata()?;
        Ok(index)
    }

    /// Open an existing flat index from storage.
    pub fn open(storage: Arc<dyn Storage>, name: &str, config: FlatIndexConfig) -> Result<Self> {
        if !storage.file_exists("metadata.json") {
            return Err(LaurusError::index("Index does not exist"));
        }

        let metadata = Self::read_metadata(storage.as_ref(), name)?;

        // Validate dimension consistency between stored metadata and config.
        if metadata.dimension != 0 && metadata.dimension != config.dimension {
            return Err(LaurusError::index(format!(
                "Dimension mismatch: stored {}, config {}",
                metadata.dimension, config.dimension
            )));
        }

        Ok(FlatIndex {
            name: name.to_string(),
            storage,
            config,
            closed: AtomicBool::new(false),
            metadata: RwLock::new(metadata),
        })
    }

    /// Create an index in a directory.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn create_in_dir<P: AsRef<Path>>(
        dir: P,
        name: &str,
        config: FlatIndexConfig,
    ) -> Result<Self> {
        use crate::storage::file::{FileStorage, FileStorageConfig};

        let storage_config = FileStorageConfig::new(&dir);
        let storage = Arc::new(FileStorage::new(&dir, storage_config)?);
        Self::create(storage, name, config)
    }

    /// Open an index from a directory.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn open_dir<P: AsRef<Path>>(dir: P, name: &str, config: FlatIndexConfig) -> Result<Self> {
        use crate::storage::file::{FileStorage, FileStorageConfig};

        let storage_config = FileStorageConfig::new(&dir);
        let storage = Arc::new(FileStorage::new(&dir, storage_config)?);
        Self::open(storage, name, config)
    }

    /// Write metadata to storage.
    fn write_metadata(&self) -> Result<()> {
        // Atomic and checksummed, matching the HNSW path (#1023). Serialize
        // under the lock, write outside it — holding a `parking_lot::RwLock`
        // across I/O deadlocks as soon as an error path formats `self`.
        let metadata_json = {
            let metadata = self.metadata.read();
            serde_json::to_vec(&*metadata)
                .map_err(|e| LaurusError::index(format!("Failed to serialize metadata: {e}")))?
        };

        manifest_io::save_checksummed(self.storage.as_ref(), "metadata.json", None, &metadata_json)
    }

    /// Read metadata from storage.
    fn read_metadata(storage: &dyn Storage, _: &str) -> Result<IndexMetadata> {
        // Verifies the checksum; pre-#1023 raw-JSON files still load.
        match manifest_io::load_checksummed_json::<IndexMetadata>(storage, "metadata.json", None)? {
            Some((metadata, _format)) => Ok(metadata),
            None => Err(LaurusError::index("metadata.json is missing or empty")),
        }
    }

    /// Update metadata.
    fn update_metadata(&self) -> Result<()> {
        {
            let mut metadata = self.metadata.write();
            metadata.modified = crate::util::time::now_secs();
        }
        self.write_metadata()
    }

    /// Check if the index is closed.
    fn check_closed(&self) -> Result<()> {
        if self.closed.load(Ordering::SeqCst) {
            return Err(LaurusError::InvalidOperation("Index is closed".to_string()));
        }
        Ok(())
    }
}

impl VectorIndex for FlatIndex {
    fn reader(&self) -> Result<Arc<dyn VectorIndexReader>> {
        self.check_closed()?;

        use crate::vector::index::flat::reader::FlatVectorIndexReader;

        // Load the index data from storage using the index name
        let reader = FlatVectorIndexReader::load(
            self.storage.clone(),
            &self.name,
            self.config.distance_metric,
        )?;
        Ok(Arc::new(reader))
    }

    fn writer(&self) -> Result<Box<dyn VectorIndexWriter>> {
        self.check_closed()?;

        let inner_writer = FlatIndexWriter::with_storage(
            self.config.clone(),
            VectorIndexWriterConfig::default(),
            self.name.clone(),
            self.storage.clone(),
        )?;

        // Wrap with EmbeddingVectorIndexWriter for automatic text/image embedding
        let embedder = self.embedder();
        let writer = EmbeddingVectorIndexWriter::new(Box::new(inner_writer), embedder);
        Ok(Box::new(writer))
    }

    fn storage(&self) -> &Arc<dyn Storage> {
        &self.storage
    }

    /// Mark the index as closed.
    ///
    /// Callers must call `commit()` before `close()` to persist pending data.
    fn close(&self) -> Result<()> {
        self.closed.store(true, Ordering::SeqCst);
        Ok(())
    }

    fn is_closed(&self) -> bool {
        self.closed.load(Ordering::SeqCst)
    }

    fn stats(&self) -> Result<VectorIndexStats> {
        self.check_closed()?;

        let metadata = self.metadata.read();
        Ok(VectorIndexStats {
            vector_count: metadata.vector_count,
            dimension: metadata.dimension,
            total_size: 0,
            deleted_count: 0,
            last_modified: metadata.modified,
        })
    }

    fn optimize(&self) -> Result<()> {
        self.check_closed()?;
        self.update_metadata()?;
        Ok(())
    }

    fn searcher(&self) -> Result<Box<dyn VectorIndexSearcher>> {
        self.check_closed()?;
        let reader = self.reader()?;
        Ok(Box::new(FlatVectorSearcher::new(reader)?))
    }

    fn embedder(&self) -> Arc<dyn Embedder> {
        Arc::clone(&self.config.embedder)
    }
}
