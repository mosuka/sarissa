//! IVF vector index implementation.

pub mod field_reader;
pub mod maintenance;
pub mod reader;
pub mod searcher;
pub mod segment;
pub mod tests;
pub mod writer;

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};

use crate::error::{Result, SarissaError};
use crate::storage::Storage;
use crate::vector::index::config::IvfIndexConfig;
use crate::vector::index::ivf::writer::IvfIndexWriter;
use crate::vector::index::{VectorIndex, VectorIndexStats};
use crate::vector::reader::VectorIndexReader;
use crate::vector::writer::{VectorIndexWriter, VectorIndexWriterConfig};

/// Metadata for the IVF index.
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

/// A concrete IVF vector index implementation.
pub struct IvfIndex {
    /// The name of the index.
    name: String,

    /// The storage backend.
    storage: Arc<dyn Storage>,

    /// IVF index specific configuration.
    config: IvfIndexConfig,

    /// Whether the index is closed (thread-safe).
    closed: AtomicBool,

    /// Index metadata (thread-safe).
    metadata: RwLock<IndexMetadata>,
}

impl std::fmt::Debug for IvfIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IvfIndex")
            .field("name", &self.name)
            .field("storage", &self.storage)
            .field("config", &self.config)
            .field("closed", &self.closed.load(Ordering::SeqCst))
            .field("metadata", &*self.metadata.read().unwrap())
            .finish()
    }
}

impl IvfIndex {
    /// Create a new IVF index in the given storage.
    pub fn create(storage: Arc<dyn Storage>, name: &str, config: IvfIndexConfig) -> Result<Self> {
        let metadata = IndexMetadata {
            dimension: config.dimension,
            ..Default::default()
        };

        let index = IvfIndex {
            name: name.to_string(),
            storage,
            config,
            closed: AtomicBool::new(false),
            metadata: RwLock::new(metadata),
        };

        index.write_metadata()?;
        Ok(index)
    }

    /// Open an existing IVF index from storage.
    pub fn open(storage: Arc<dyn Storage>, name: &str, config: IvfIndexConfig) -> Result<Self> {
        let metadata_file = format!("{}.json", name);
        if !storage.file_exists(&metadata_file) {
            return Err(SarissaError::index("Index does not exist"));
        }

        let metadata = Self::read_metadata(storage.as_ref(), name)?;

        Ok(IvfIndex {
            name: name.to_string(),
            storage,
            config,
            closed: AtomicBool::new(false),
            metadata: RwLock::new(metadata),
        })
    }

    /// Create an index in a directory.
    pub fn create_in_dir<P: AsRef<Path>>(
        dir: P,
        name: &str,
        config: IvfIndexConfig,
    ) -> Result<Self> {
        use crate::storage::file::{FileStorage, FileStorageConfig};

        let storage_config = FileStorageConfig::new(&dir);
        let storage = Arc::new(FileStorage::new(&dir, storage_config)?);
        Self::create(storage, name, config)
    }

    /// Open an index from a directory.
    pub fn open_dir<P: AsRef<Path>>(dir: P, name: &str, config: IvfIndexConfig) -> Result<Self> {
        use crate::storage::file::{FileStorage, FileStorageConfig};

        let storage_config = FileStorageConfig::new(&dir);
        let storage = Arc::new(FileStorage::new(&dir, storage_config)?);
        Self::open(storage, name, config)
    }

    /// Write metadata to storage.
    fn write_metadata(&self) -> Result<()> {
        let metadata = self
            .metadata
            .read()
            .map_err(|_| SarissaError::index("Failed to acquire metadata read lock"))?;
        let metadata_json = serde_json::to_string_pretty(&*metadata)
            .map_err(|e| SarissaError::index(format!("Failed to serialize metadata: {e}")))?;
        drop(metadata);

        let metadata_file = format!("{}.json", self.name);
        let mut output = self.storage.create_output(&metadata_file)?;
        std::io::Write::write_all(&mut output, metadata_json.as_bytes())?;
        output.close()?;

        Ok(())
    }

    /// Read metadata from storage.
    fn read_metadata(storage: &dyn Storage, name: &str) -> Result<IndexMetadata> {
        let metadata_file = format!("{}.json", name);
        let input = storage.open_input(&metadata_file)?;
        let metadata: IndexMetadata = serde_json::from_reader(input)
            .map_err(|e| SarissaError::index(format!("Failed to deserialize metadata: {e}")))?;
        Ok(metadata)
    }

    /// Update metadata.
    fn update_metadata(&self) -> Result<()> {
        {
            let mut metadata = self
                .metadata
                .write()
                .map_err(|_| SarissaError::index("Failed to acquire metadata write lock"))?;
            metadata.modified = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
        }
        self.write_metadata()
    }

    /// Check if the index is closed.
    fn check_closed(&self) -> Result<()> {
        if self.closed.load(Ordering::SeqCst) {
            return Err(SarissaError::InvalidOperation(
                "Index is closed".to_string(),
            ));
        }
        Ok(())
    }
}

impl VectorIndex for IvfIndex {
    fn reader(&self) -> Result<Arc<dyn VectorIndexReader>> {
        self.check_closed()?;

        use crate::vector::index::ivf::reader::IvfIndexReader;

        let reader = IvfIndexReader::load(&*self.storage, &self.name, self.config.distance_metric)?;
        Ok(Arc::new(reader))
    }

    fn writer(&self) -> Result<Box<dyn VectorIndexWriter>> {
        self.check_closed()?;

        let writer = IvfIndexWriter::with_storage(
            self.config.clone(),
            VectorIndexWriterConfig::default(),
            self.name.clone(),
            self.storage.clone(),
        )?;
        Ok(Box::new(writer))
    }

    fn storage(&self) -> &Arc<dyn Storage> {
        &self.storage
    }

    fn close(&self) -> Result<()> {
        self.closed.store(true, Ordering::SeqCst);
        Ok(())
    }

    fn is_closed(&self) -> bool {
        self.closed.load(Ordering::SeqCst)
    }

    fn stats(&self) -> Result<VectorIndexStats> {
        self.check_closed()?;

        let metadata = self
            .metadata
            .read()
            .map_err(|_| SarissaError::index("Failed to acquire metadata read lock"))?;
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
}
