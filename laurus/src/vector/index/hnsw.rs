//! HNSW vector index implementation.

pub mod graph;
pub mod maintenance;
pub mod reader;
pub mod searcher;
pub mod segment;
pub mod writer;

#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use parking_lot::RwLock;

use crate::embedding::embedder::Embedder;
use crate::error::{LaurusError, Result};
use crate::maintenance::deletion::DeletionBitmap;
use crate::storage::Storage;
use crate::storage::structured::{StructReader, StructWriter};
use crate::vector::index::config::HnswIndexConfig;
use crate::vector::index::hnsw::searcher::HnswSearcher;
use crate::vector::index::hnsw::writer::HnswIndexWriter;
use crate::vector::index::{VectorIndex, VectorIndexStats};
use crate::vector::reader::VectorIndexReader;
use crate::vector::search::searcher::VectorIndexSearcher;
use crate::vector::store::embedding_writer::EmbeddingVectorIndexWriter;
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
        let now = crate::util::time::now_secs();
        Self {
            vector_count: 0,
            dimension: 0,
            created: now,
            modified: now,
        }
    }
}

/// A concrete HNSW vector index implementation.
pub struct HnswIndex {
    /// The name of the index.
    name: String,

    /// The storage backend.
    storage: Arc<dyn Storage>,

    /// HNSW index specific configuration.
    config: HnswIndexConfig,

    /// Whether the index is closed (thread-safe).
    closed: AtomicBool,

    /// Index metadata (thread-safe).
    metadata: RwLock<IndexMetadata>,

    /// Logical deletion bitmap for this single-segment index (Issue #624).
    ///
    /// Lazily loaded from `<name>.delmap` (or created on the first soft
    /// delete). When present and non-empty it is attached to readers so the
    /// deletion-aware HNSW traversal (Issue #665) excludes deleted documents
    /// without rebuilding the graph; [`Self::optimize`] physically reclaims
    /// them. `None` means "not yet loaded / no deletions".
    deletion: RwLock<Option<Arc<DeletionBitmap>>>,
}

impl std::fmt::Debug for HnswIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HnswIndex")
            .field("name", &self.name)
            .field("storage", &self.storage)
            .field("config", &self.config)
            .field("closed", &self.closed.load(Ordering::SeqCst))
            .field("metadata", &*self.metadata.read())
            .finish()
    }
}

impl HnswIndex {
    /// Create a new HNSW index in the given storage.
    pub fn create(storage: Arc<dyn Storage>, name: &str, config: HnswIndexConfig) -> Result<Self> {
        let metadata = IndexMetadata {
            dimension: config.dimension,
            ..Default::default()
        };

        let index = HnswIndex {
            name: name.to_string(),
            storage,
            config,
            closed: AtomicBool::new(false),
            metadata: RwLock::new(metadata),
            deletion: RwLock::new(None),
        };

        index.write_metadata()?;
        Ok(index)
    }

    /// Open an existing HNSW index from storage.
    pub fn open(storage: Arc<dyn Storage>, name: &str, config: HnswIndexConfig) -> Result<Self> {
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

        Ok(HnswIndex {
            name: name.to_string(),
            storage,
            config,
            closed: AtomicBool::new(false),
            metadata: RwLock::new(metadata),
            deletion: RwLock::new(None),
        })
    }

    /// Create an index in a directory.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn create_in_dir<P: AsRef<Path>>(
        dir: P,
        name: &str,
        config: HnswIndexConfig,
    ) -> Result<Self> {
        use crate::storage::file::{FileStorage, FileStorageConfig};

        let storage_config = FileStorageConfig::new(&dir);
        let storage = Arc::new(FileStorage::new(&dir, storage_config)?);
        Self::create(storage, name, config)
    }

    /// Open an index from a directory.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn open_dir<P: AsRef<Path>>(dir: P, name: &str, config: HnswIndexConfig) -> Result<Self> {
        use crate::storage::file::{FileStorage, FileStorageConfig};

        let storage_config = FileStorageConfig::new(&dir);
        let storage = Arc::new(FileStorage::new(&dir, storage_config)?);
        Self::open(storage, name, config)
    }

    /// Write metadata to storage.
    fn write_metadata(&self) -> Result<()> {
        let metadata = self.metadata.read();
        let metadata_json = serde_json::to_string_pretty(&*metadata)
            .map_err(|e| LaurusError::index(format!("Failed to serialize metadata: {e}")))?;
        drop(metadata);

        let mut output = self.storage.create_output("metadata.json")?;
        std::io::Write::write_all(&mut output, metadata_json.as_bytes())?;
        output.close()?;

        Ok(())
    }

    /// Read metadata from storage.
    fn read_metadata(storage: &dyn Storage, _: &str) -> Result<IndexMetadata> {
        let input = storage.open_input("metadata.json")?;
        let metadata: IndexMetadata = serde_json::from_reader(input)
            .map_err(|e| LaurusError::index(format!("Failed to deserialize metadata: {e}")))?;
        Ok(metadata)
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

    /// File name holding this index's deletion bitmap (`<name>.delmap`).
    fn delmap_file_name(&self) -> String {
        format!("{}.delmap", self.name)
    }

    /// Resolve this index's deletion bitmap (Issue #624).
    ///
    /// Returns the cached bitmap if present; otherwise loads it from
    /// `<name>.delmap` when that file exists. When `create_if_missing` is
    /// `true` and no bitmap exists yet, a fresh empty bitmap is created and
    /// cached. The returned `Arc` shares interior mutability with the cached
    /// instance, so callers see subsequent marks.
    ///
    /// The bitmap is created with an unbounded id range (`[0, u64::MAX - 1]`)
    /// because the single-segment index has no fixed doc-id span; the range
    /// check is therefore a no-op and `total_docs` is not used (compaction is
    /// driven explicitly by [`Self::optimize`], not by deletion ratio).
    ///
    /// # Arguments
    ///
    /// * `create_if_missing` - Create and cache an empty bitmap when none exists.
    ///
    /// # Errors
    ///
    /// Returns an error if reading an existing `.delmap` file fails.
    fn load_or_get_bitmap(&self, create_if_missing: bool) -> Result<Option<Arc<DeletionBitmap>>> {
        if let Some(bitmap) = self.deletion.read().as_ref() {
            return Ok(Some(bitmap.clone()));
        }

        let mut guard = self.deletion.write();
        // Re-check under the write lock in case another thread populated it.
        if let Some(bitmap) = guard.as_ref() {
            return Ok(Some(bitmap.clone()));
        }

        let file = self.delmap_file_name();
        if self.storage.file_exists(&file) {
            let input = self.storage.open_input(&file)?;
            let mut reader = StructReader::new(input)?;
            let bitmap = Arc::new(DeletionBitmap::read_from_storage(&mut reader)?);
            *guard = Some(bitmap.clone());
            return Ok(Some(bitmap));
        }

        if create_if_missing {
            let bitmap = Arc::new(DeletionBitmap::new(self.name.clone(), 0, u64::MAX - 1));
            *guard = Some(bitmap.clone());
            return Ok(Some(bitmap));
        }

        Ok(None)
    }

    /// Drop all logical deletions and remove the `.delmap` file (Issue #624).
    ///
    /// Called after [`Self::optimize`] has physically rebuilt the graph
    /// without the deleted documents.
    ///
    /// # Errors
    ///
    /// Returns an error if deleting the `.delmap` file fails.
    fn clear_deletions(&self) -> Result<()> {
        *self.deletion.write() = None;
        let file = self.delmap_file_name();
        if self.storage.file_exists(&file) {
            self.storage.delete_file(&file)?;
        }
        Ok(())
    }
}

impl VectorIndex for HnswIndex {
    fn reader(&self) -> Result<Arc<dyn VectorIndexReader>> {
        self.check_closed()?;

        use crate::vector::index::hnsw::reader::HnswIndexReader;

        let mut reader = HnswIndexReader::load(
            self.storage.clone(),
            &self.name,
            self.config.distance_metric,
        )?;
        // Attach the logical-deletion bitmap so the deletion-aware traversal
        // (Issue #665) excludes soft-deleted documents (Issue #624).
        if let Some(bitmap) = self.load_or_get_bitmap(false)?
            && bitmap.deleted_count.load(Ordering::Relaxed) > 0
        {
            reader.set_deletion_bitmap(bitmap);
        }
        Ok(Arc::new(reader))
    }

    fn writer(&self) -> Result<Box<dyn VectorIndexWriter>> {
        self.check_closed()?;

        let inner_writer = HnswIndexWriter::with_storage(
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
    /// This method only sets the closed flag and releases resources.
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

        // Physically reclaim soft-deleted documents (Issue #624). Reuse the
        // writer's existing full-rebuild path: load the current graph +
        // vectors, drop the deleted ids (this removes them from the buffer and
        // invalidates the graph), then commit to rebuild a clean graph without
        // them. The expensive rebuild happens here (intentional compaction),
        // not on every delete.
        let deleted: Vec<u64> = match self.load_or_get_bitmap(false)? {
            Some(bitmap) if bitmap.deleted_count.load(Ordering::Relaxed) > 0 => {
                bitmap.get_deleted_docs()
            }
            _ => {
                self.update_metadata()?;
                return Ok(());
            }
        };

        let mut writer = HnswIndexWriter::with_storage(
            self.config.clone(),
            VectorIndexWriterConfig::default(),
            self.name.clone(),
            self.storage.clone(),
        )?;
        for doc_id in &deleted {
            writer.delete_document(*doc_id)?;
        }
        writer.commit()?;

        self.clear_deletions()?;
        self.update_metadata()?;
        Ok(())
    }

    fn searcher(&self) -> Result<Box<dyn VectorIndexSearcher>> {
        self.check_closed()?;
        let reader = self.reader()?;
        // Pull the schema-level default `ef_search` from the config so the
        // monolithic search path honours `HnswOption.default_ef_search`
        // (Issue #644). Per-query overrides via
        // `VectorIndexQueryParams.ef_search` still take precedence.
        Ok(Box::new(HnswSearcher::with_default_ef_search(
            reader,
            self.config.default_ef_search,
        )?))
    }

    fn embedder(&self) -> Arc<dyn Embedder> {
        Arc::clone(&self.config.embedder)
    }

    fn supports_soft_delete(&self) -> bool {
        true
    }

    fn soft_delete_document(&self, doc_id: u64) -> Result<()> {
        self.check_closed()?;
        let bitmap = self
            .load_or_get_bitmap(true)?
            .ok_or_else(|| LaurusError::internal("deletion bitmap unexpectedly missing"))?;
        bitmap.delete_document(doc_id)?;
        Ok(())
    }

    fn persist_deletions(&self) -> Result<()> {
        let guard = self.deletion.read();
        if let Some(bitmap) = guard.as_ref()
            && bitmap.deleted_count.load(Ordering::Relaxed) > 0
        {
            let file = self.delmap_file_name();
            let output = self.storage.create_output(&file)?;
            let mut writer = StructWriter::new(output);
            bitmap.write_to_storage(&mut writer)?;
            writer.close()?;
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests;
