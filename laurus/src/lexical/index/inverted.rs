//! Inverted index implementation for full-text search.
//!
//! This module provides the core inverted index implementation:
//! - Core data structures (posting lists, term enumeration)
//! - Index creation and management
//! - Writer for building the index
//! - Reader for querying the index
//! - Searcher for executing searches
//! - Segment management and merging
//! - Index maintenance operations
//! - Query types for searching

use std::collections::HashMap;
use std::io::Read;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use parking_lot::RwLock;

use serde::{Deserialize, Serialize};

use crate::error::{LaurusError, Result};
use crate::lexical::core::field::FieldOption;
use crate::lexical::index::LexicalIndex;
use crate::lexical::index::config::InvertedIndexConfig;
use crate::lexical::reader::LexicalIndexReader;
use crate::lexical::search::searcher::LexicalSearcher;
use crate::lexical::writer::LexicalIndexWriter;
use crate::storage::Storage;
#[cfg(not(target_arch = "wasm32"))]
use crate::storage::file::{FileStorage, FileStorageConfig};

pub(crate) mod bmw;
pub mod core;
pub mod maintenance;
pub(crate) mod per_segment_view;
pub mod query_cache;
pub mod reader;
pub mod searcher;
pub mod segment;
pub mod writer;

use self::reader::{InvertedIndexReader, InvertedIndexReaderConfig};
use self::searcher::InvertedIndexSearcher;
use self::segment::SegmentInfo;
use self::writer::{InvertedIndexWriter, InvertedIndexWriterConfig};

/// Metadata about an inverted index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexMetadata {
    /// Version of the index format.
    pub version: u32,

    /// Creation time (seconds since epoch).
    pub created: u64,

    /// Last modified time (seconds since epoch).
    pub modified: u64,

    /// Number of documents indexed.
    pub doc_count: u64,

    /// Generation number for updates.
    pub generation: u64,

    /// Number of deleted documents.
    #[serde(default)]
    pub deleted_count: u64,

    /// Last processed WAL sequence number.
    #[serde(default)]
    pub last_wal_seq: u64,
}

/// Statistics about an inverted index.
#[derive(Debug, Clone)]
pub struct InvertedIndexStats {
    /// Number of documents in the index.
    pub doc_count: u64,

    /// Number of unique terms in the index.
    pub term_count: u64,

    /// Number of segments in the index.
    pub segment_count: u32,

    /// Total size of the index in bytes.
    pub total_size: u64,

    /// Number of deleted documents.
    pub deleted_count: u64,

    /// Last modified time (seconds since epoch).
    pub last_modified: u64,
}

impl Default for IndexMetadata {
    fn default() -> Self {
        let now = crate::util::time::now_secs();

        IndexMetadata {
            version: 1,
            created: now,
            modified: now,
            doc_count: 0,
            generation: 0,
            deleted_count: 0,
            last_wal_seq: 0,
        }
    }
}

/// A concrete inverted index implementation for schema-less lexical indexing.
pub struct InvertedIndex {
    /// The storage backend.
    storage: Arc<dyn Storage>,

    /// Inverted index specific configuration.
    config: InvertedIndexConfig,

    /// Fields added dynamically at runtime via [`add_field()`](Self::add_field).
    /// These are merged with `config.fields` when creating a new writer.
    extra_fields: RwLock<HashMap<String, FieldOption>>,

    /// Whether the index is closed (thread-safe).
    closed: AtomicBool,

    /// Index metadata (thread-safe).
    metadata: RwLock<IndexMetadata>,
}

impl std::fmt::Debug for InvertedIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InvertedIndex")
            .field("storage", &self.storage)
            .field("config", &self.config)
            .field("closed", &self.closed.load(Ordering::SeqCst))
            .field("metadata", &*self.metadata.read())
            .finish()
    }
}

impl InvertedIndex {
    /// Create a new index in the given storage.
    pub fn create(storage: Arc<dyn Storage>, config: InvertedIndexConfig) -> Result<Self> {
        let metadata = IndexMetadata::default();

        let index = InvertedIndex {
            storage,
            config,
            extra_fields: RwLock::new(HashMap::new()),
            closed: AtomicBool::new(false),
            metadata: RwLock::new(metadata),
        };

        index.write_metadata()?;
        Ok(index)
    }

    /// Open an existing index from storage.
    pub fn open(storage: Arc<dyn Storage>, config: InvertedIndexConfig) -> Result<Self> {
        if !storage.file_exists("metadata.json") {
            return Err(LaurusError::index("Index does not exist"));
        }

        let metadata = Self::read_metadata(storage.as_ref())?;

        Ok(InvertedIndex {
            storage,
            config,
            extra_fields: RwLock::new(HashMap::new()),
            closed: AtomicBool::new(false),
            metadata: RwLock::new(metadata),
        })
    }

    /// Create an index in a directory.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn create_in_dir<P: AsRef<Path>>(dir: P, config: InvertedIndexConfig) -> Result<Self> {
        let storage_config = FileStorageConfig::new(&dir);
        let storage = Arc::new(FileStorage::new(&dir, storage_config)?);
        Self::create(storage, config)
    }

    /// Open an index from a directory.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn open_dir<P: AsRef<Path>>(dir: P, config: InvertedIndexConfig) -> Result<Self> {
        let storage_config = FileStorageConfig::new(&dir);
        let storage = Arc::new(FileStorage::new(&dir, storage_config)?);
        Self::open(storage, config)
    }

    /// Open or create an index.
    pub fn open_or_create(storage: Arc<dyn Storage>, config: InvertedIndexConfig) -> Result<Self> {
        if storage.file_exists("metadata.json") {
            Self::open(storage, config)
        } else {
            Self::create(storage, config)
        }
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
    pub(crate) fn read_metadata(storage: &dyn Storage) -> Result<IndexMetadata> {
        let mut input = storage.open_input("metadata.json")?;
        let mut metadata_json = String::new();
        Read::read_to_string(&mut input, &mut metadata_json)?;

        let metadata: IndexMetadata = serde_json::from_str(&metadata_json)
            .map_err(|e| LaurusError::index(format!("Failed to deserialize metadata: {e}")))?;

        Ok(metadata)
    }

    /// Update metadata and write to storage.
    fn update_metadata(&self) -> Result<()> {
        {
            let mut metadata = self.metadata.write();
            metadata.modified = crate::util::time::now_secs();
        }

        self.write_metadata()
    }

    /// Update the document count in the index metadata.
    pub fn update_doc_count(&self, additional_docs: u64) -> Result<()> {
        self.check_closed()?;
        {
            let mut metadata = self.metadata.write();
            metadata.doc_count += additional_docs;
        }
        self.update_metadata()
    }

    /// Check if the index is closed.
    fn check_closed(&self) -> Result<()> {
        if self.closed.load(Ordering::SeqCst) {
            Err(LaurusError::index("Index is closed"))
        } else {
            Ok(())
        }
    }

    /// Load segment information from storage.
    fn load_segments(&self) -> Result<Vec<SegmentInfo>> {
        let files = self.storage.list_files()?;
        let mut segments = Vec::new();

        for file in &files {
            // Both freshly flushed segments (`segment_*`) and segments produced
            // by a merge (`merged_*`, Issue #754) are discovered here.
            if (file.starts_with("segment_") || file.starts_with("merged_"))
                && file.ends_with(".meta")
            {
                let mut input = self.storage.open_input(file)?;
                let mut data = Vec::new();
                Read::read_to_end(&mut input, &mut data)?;

                let segment_info: SegmentInfo = serde_json::from_slice(&data).map_err(|e| {
                    LaurusError::index(format!("Failed to parse segment metadata: {e}"))
                })?;

                segments.push(segment_info);
            }
        }

        segments.sort_by_key(|s| s.generation);
        Ok(segments)
    }

    /// Force-merge every current segment into a single new segment (Issue
    /// #754), the classic `optimize()` / force-merge semantics.
    ///
    /// Discovers the current segments, merges them with the (correct, typed)
    /// [`MergeEngine`](self::segment::merge_engine::MergeEngine), rewrites the
    /// merged segment's metadata generation so it sorts as the newest segment,
    /// and deletes the now-merged source segments' files so segment discovery
    /// ([`Self::load_segments`]) sees only the merged result. A no-op when
    /// fewer than two segments exist.
    fn force_merge_all(&self) -> Result<()> {
        let segments = self.load_segments()?;
        if segments.len() < 2 {
            // Zero or one segment: nothing to compact.
            return Ok(());
        }
        // The merged segment must sort as the newest, so its generation is one
        // past the highest source generation.
        let next_generation = segments.iter().map(|s| s.generation).max().unwrap_or(0) + 1;
        self.merge_segment_set(&segments, next_generation)
    }

    /// Auto-merge implementation behind the [`LexicalIndex::maybe_merge`] hook
    /// run after each commit (Issue #755).
    ///
    /// Keeps the segment count bounded without a manual
    /// [`optimize()`](LexicalIndex::optimize): when the number of segments
    /// exceeds [`InvertedIndexConfig::max_segments`], the smallest
    /// [`merge_factor`](InvertedIndexConfig::merge_factor) segments are merged
    /// into one (Lucene-style "merge small segments first"). A single merge is
    /// performed per call; repeated commits converge the count. Cheap when no
    /// merge is needed (a segment count check). Disable by raising
    /// `max_segments`.
    fn auto_merge(&self) -> Result<()> {
        let segments = self.load_segments()?;
        if segments.len() <= self.config.max_segments as usize {
            // Under the threshold: nothing to do.
            return Ok(());
        }

        // Merge the smallest `merge_factor` segments (small-first keeps merge
        // cost low and bounds per-commit latency).
        let mut by_size: Vec<(SegmentInfo, u64)> = segments
            .iter()
            .map(|s| (s.clone(), self.segment_size_bytes(&s.segment_id)))
            .collect();
        by_size.sort_by_key(|(_, size)| *size);

        let take = (self.config.merge_factor as usize).clamp(2, segments.len());
        let subset: Vec<SegmentInfo> = by_size.into_iter().take(take).map(|(s, _)| s).collect();

        let next_generation = segments.iter().map(|s| s.generation).max().unwrap_or(0) + 1;
        self.merge_segment_set(&subset, next_generation)
    }

    /// Merge a set of source segments into a single new segment.
    ///
    /// Shared by [`Self::force_merge_all`] (all segments) and
    /// [`Self::maybe_merge`] (a policy-selected subset). Runs the (correct,
    /// typed) [`MergeEngine`](self::segment::merge_engine::MergeEngine), rewrites
    /// the merged segment's generation to `next_generation` so it sorts as the
    /// newest, and deletes the source segments (their `.meta` first, so they
    /// drop out of `.meta` file-scan discovery before their now-orphaned data
    /// files are removed — minimizing any window in which a document could be
    /// seen in both a source and the merged segment). A no-op for fewer than
    /// two sources.
    fn merge_segment_set(&self, sources: &[SegmentInfo], next_generation: u64) -> Result<()> {
        use self::segment::manager::{ManagedSegmentInfo, MergeCandidate, MergeStrategy};
        use self::segment::merge_engine::{MergeConfig, MergeEngine};

        if sources.len() < 2 {
            return Ok(());
        }

        let managed: Vec<ManagedSegmentInfo> = sources
            .iter()
            .map(|info| {
                let mut mi = ManagedSegmentInfo::new(info.clone());
                mi.size_bytes = self.segment_size_bytes(&info.segment_id);
                mi
            })
            .collect();
        let candidate = MergeCandidate {
            segments: sources.iter().map(|s| s.segment_id.clone()).collect(),
            priority: 1.0,
            estimated_size: 0,
            strategy: MergeStrategy::SizeBased,
        };

        let engine = MergeEngine::new(MergeConfig::default(), self.storage.clone());
        let result = engine.merge_segments(&candidate, &managed, next_generation)?;
        self.set_segment_generation(&result.new_segment.segment_info.segment_id, next_generation)?;

        // Delete each source's `.meta` first (drops it from discovery), then the
        // remaining data files.
        for info in sources {
            let meta = format!("{}.meta", info.segment_id);
            if self.storage.file_exists(&meta) {
                self.storage.delete_file(&meta)?;
            }
        }
        for info in sources {
            self.delete_segment_files(&info.segment_id)?;
        }

        Ok(())
    }

    /// Sum the on-disk size of every file belonging to `segment_id`.
    fn segment_size_bytes(&self, segment_id: &str) -> u64 {
        let prefix = format!("{segment_id}.");
        self.storage
            .list_files()
            .map(|files| {
                files
                    .iter()
                    .filter(|f| f.starts_with(&prefix))
                    .map(|f| self.storage.metadata(f).map(|m| m.size).unwrap_or(0))
                    .sum()
            })
            .unwrap_or(0)
    }

    /// Rewrite a segment's `.meta` with an updated generation.
    fn set_segment_generation(&self, segment_id: &str, generation: u64) -> Result<()> {
        let meta_file = format!("{segment_id}.meta");
        let mut input = self.storage.open_input(&meta_file)?;
        let mut data = Vec::new();
        Read::read_to_end(&mut input, &mut data)?;
        let mut info: SegmentInfo = serde_json::from_slice(&data)
            .map_err(|e| LaurusError::index(format!("Failed to parse segment metadata: {e}")))?;
        info.generation = generation;
        let json = serde_json::to_string_pretty(&info).map_err(|e| {
            LaurusError::index(format!("Failed to serialize segment metadata: {e}"))
        })?;
        let mut output = self.storage.create_output(&meta_file)?;
        std::io::Write::write_all(&mut output, json.as_bytes())?;
        output.close()?;
        Ok(())
    }

    /// Delete every file belonging to `segment_id`.
    fn delete_segment_files(&self, segment_id: &str) -> Result<()> {
        let prefix = format!("{segment_id}.");
        let files = self.storage.list_files()?;
        for file in files.iter().filter(|f| f.starts_with(&prefix)) {
            self.storage.delete_file(file)?;
        }
        Ok(())
    }

    /// Check if an index exists in the given directory.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn exists_in_dir<P: AsRef<Path>>(dir: P) -> bool {
        let metadata_path = dir.as_ref().join("metadata.json");
        metadata_path.exists()
    }

    /// Delete an index from the given directory.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn delete_in_dir<P: AsRef<Path>>(dir: P) -> Result<()> {
        let storage_config = FileStorageConfig::new(&dir);
        let storage = FileStorage::new(&dir, storage_config)?;

        for file in storage.list_files()? {
            storage.delete_file(&file)?;
        }

        Ok(())
    }

    /// List all files in the index.
    pub fn list_files(&self) -> Result<Vec<String>> {
        self.check_closed()?;
        self.storage.list_files()
    }

    /// Returns the last WAL (Write-Ahead Log) sequence number recorded in the index metadata.
    ///
    /// # Returns
    ///
    /// The last WAL sequence number as a `u64`.
    pub fn last_wal_seq(&self) -> u64 {
        self.metadata.read().last_wal_seq
    }

    /// Sets the last WAL (Write-Ahead Log) sequence number in the index metadata
    /// and persists the updated metadata to storage.
    ///
    /// # Arguments
    ///
    /// * `seq` - The new WAL sequence number to record.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or an error if the index is closed or the metadata write fails.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError`] if the index has been closed
    /// or if persisting the metadata fails.
    pub fn set_last_wal_seq(&self, seq: u64) -> Result<()> {
        self.check_closed()?;
        {
            let mut metadata = self.metadata.write();
            metadata.last_wal_seq = seq;
        }
        self.update_metadata()
    }
}

impl LexicalIndex for InvertedIndex {
    fn reader(&self) -> Result<Arc<dyn LexicalIndexReader>> {
        self.check_closed()?;

        let segments = self.load_segments()?;

        // Use analyzer from index config. The query/filter cache capacity must
        // be set explicitly here: `InvertedIndexReaderConfig::default()` would
        // otherwise mask the value configured on the index (Issue #578).
        let reader_config = InvertedIndexReaderConfig {
            analyzer: self.config.analyzer.clone(),
            query_filter_cache_capacity: self.config.query_filter_cache_capacity,
            ..InvertedIndexReaderConfig::default()
        };

        let reader = InvertedIndexReader::new(segments, self.storage.clone(), reader_config)?;
        Ok(Arc::new(reader))
    }

    fn writer(&self) -> Result<Box<dyn LexicalIndexWriter>> {
        self.check_closed()?;

        // Merge base config fields with dynamically added fields.
        let mut fields = self.config.fields.clone();
        fields.extend(
            self.extra_fields
                .read()
                .iter()
                .map(|(k, v)| (k.clone(), v.clone())),
        );

        // Use analyzer and shard_id from index config
        let writer_config = InvertedIndexWriterConfig {
            analyzer: self.config.analyzer.clone(),
            shard_id: self.config.shard_id,
            fields,
            ..Default::default()
        };
        let writer = InvertedIndexWriter::new(self.storage.clone(), writer_config)?;
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

    fn stats(&self) -> Result<InvertedIndexStats> {
        self.check_closed()?;

        let metadata = self.metadata.read();
        Ok(InvertedIndexStats {
            doc_count: metadata.doc_count,
            term_count: 0,
            segment_count: 0,
            total_size: 0,
            deleted_count: metadata.deleted_count,
            last_modified: metadata.modified,
        })
    }

    fn optimize(&self) -> Result<()> {
        self.check_closed()?;
        self.force_merge_all()?;
        self.update_metadata()?;
        Ok(())
    }

    fn maybe_merge(&self) -> Result<()> {
        self.check_closed()?;
        self.auto_merge()
    }

    fn refresh(&self) -> Result<()> {
        self.check_closed()?;
        let metadata = Self::read_metadata(self.storage.as_ref())?;
        *self.metadata.write() = metadata;
        Ok(())
    }

    fn searcher(&self) -> Result<Box<dyn LexicalSearcher>> {
        self.check_closed()?;
        let reader = self.reader()?;
        let searcher = InvertedIndexSearcher::from_arc(reader)
            .with_default_fields(self.config.default_fields.clone());
        Ok(Box::new(searcher))
    }

    fn default_fields(&self) -> Result<Vec<String>> {
        Ok(self.config.default_fields.clone())
    }

    fn add_field(&self, name: &str, option: FieldOption) -> Result<()> {
        // Check for duplicates in both base config and extra fields.
        if self.config.fields.contains_key(name) || self.extra_fields.read().contains_key(name) {
            return Err(LaurusError::invalid_argument(format!(
                "Field '{name}' already exists in the lexical index"
            )));
        }
        self.extra_fields.write().insert(name.to_string(), option);
        Ok(())
    }

    fn delete_field(&self, name: &str) -> Result<()> {
        // Only dynamically added fields (in extra_fields) can be removed at
        // the index level. Fields from the initial config remain in the
        // underlying index data but will be hidden from the engine-level schema.
        self.extra_fields.write().remove(name);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexical::core::document::Document;
    use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};
    use std::sync::Arc;

    #[allow(dead_code)]
    fn create_test_document(title: &str, body: &str) -> Document {
        Document::builder()
            .add_text("title", title)
            .add_text("body", body)
            .build()
    }

    #[test]
    fn test_inverted_index_writer_creation() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let config = InvertedIndexWriterConfig::default();

        let writer = InvertedIndexWriter::new(storage, config).unwrap();

        assert_eq!(writer.pending_docs(), 0);
        assert_eq!(writer.stats().docs_added, 0);
    }

    #[test]
    fn test_add_document() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let config = InvertedIndexWriterConfig::default();

        let mut writer = InvertedIndexWriter::new(storage, config).unwrap();
        let doc = create_test_document("Test Title", "This is test content");

        writer.add_document(doc).unwrap();

        assert_eq!(writer.pending_docs(), 1);
        assert_eq!(writer.stats().docs_added, 1);
        assert!(writer.stats().unique_terms > 0);
    }

    #[test]
    fn test_auto_flush() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let config = InvertedIndexWriterConfig {
            max_buffered_docs: 2,
            ..Default::default()
        };

        let mut writer = InvertedIndexWriter::new(storage.clone(), config).unwrap();

        // Add first document
        writer
            .add_document(create_test_document("Doc 1", "Content 1"))
            .unwrap();
        assert_eq!(writer.pending_docs(), 1);

        // Add second document - should trigger flush
        writer
            .add_document(create_test_document("Doc 2", "Content 2"))
            .unwrap();
        assert_eq!(writer.pending_docs(), 0); // Flushed
        assert_eq!(writer.stats().segments_created, 1);

        // Check that files were created
        let files = storage.list_files().unwrap();
        assert!(files.iter().any(|f| f.contains("segment_000000")));
    }

    #[test]
    fn test_commit() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let config = InvertedIndexWriterConfig::default();

        let mut writer = InvertedIndexWriter::new(storage.clone(), config).unwrap();

        writer
            .add_document(create_test_document("Test", "Content"))
            .unwrap();
        writer.commit().unwrap();

        assert_eq!(writer.pending_docs(), 0);

        // Check that files were created
        let files = storage.list_files().unwrap();
        assert!(files.contains(&"index.meta".to_string()));
        assert!(files.iter().any(|f| f.starts_with("segment_")));
    }

    #[test]
    fn test_rollback() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let config = InvertedIndexWriterConfig::default();

        let mut writer = InvertedIndexWriter::new(storage, config).unwrap();

        writer
            .add_document(create_test_document("Test", "Content"))
            .unwrap();
        assert_eq!(writer.pending_docs(), 1);

        writer.rollback().unwrap();
        assert_eq!(writer.pending_docs(), 0);
        assert_eq!(writer.stats().docs_added, 1); // Stats don't rollback
    }

    #[test]
    fn test_multiple_field_types() {
        // Schema-less mode: fields are inferred from document
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let config = InvertedIndexWriterConfig::default();

        let mut writer = InvertedIndexWriter::new(storage, config).unwrap();

        let doc = Document::builder()
            .add_text("title", "Test Document")
            .add_text("id", "doc1")
            .add_float("count", 42.0)
            .build();

        writer.add_document(doc).unwrap();
        writer.commit().unwrap();

        assert_eq!(writer.stats().docs_added, 1);
        assert!(writer.stats().unique_terms >= 3); // At least title, id, count fields
    }
}
