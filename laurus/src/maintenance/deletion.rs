//! Document deletion and compaction system.
//!
//! This module provides efficient document deletion using set-based
//! logical deletion and periodic compaction for space reclamation.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use ahash::AHashMap;
use roaring::RoaringTreemap;
use serde::{Deserialize, Serialize};

use crate::error::{LaurusError, Result};
use crate::storage::structured::{StructReader, StructWriter};
use crate::storage::{Storage, StorageInput, StorageOutput};

/// Configuration for deletion management.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeletionConfig {
    /// Compaction threshold (deletion ratio 0.0-1.0).
    pub compaction_threshold: f64,

    /// Enable automatic compaction.
    pub auto_compaction: bool,

    /// Compaction check interval in seconds.
    pub compaction_interval_secs: u64,

    /// Maximum memory for deletion bitmaps (in MB).
    pub max_bitmap_memory_mb: u64,

    /// Batch size for deletion operations.
    pub deletion_batch_size: usize,

    /// Enable deletion log for recovery.
    pub enable_deletion_log: bool,
}

impl Default for DeletionConfig {
    fn default() -> Self {
        DeletionConfig {
            compaction_threshold: 0.3,
            auto_compaction: true,
            compaction_interval_secs: 300, // 5 minutes
            max_bitmap_memory_mb: 64,
            deletion_batch_size: 1000,
            enable_deletion_log: true,
        }
    }
}

/// A hash set-based deletion tracker for a segment.
#[derive(Debug)]
pub struct DeletionBitmap {
    /// Segment ID this bitmap belongs to.
    pub segment_id: String,

    /// Set of deleted document IDs, stored as a Roaring bitmap (Issue #684).
    ///
    /// `RoaringTreemap` (u64) is far more compact than the previous
    /// `AHashSet<u64>` for the dense deletion sets that accumulate over a
    /// segment's life, and `contains` is a branch-light bit test rather than a
    /// hashed probe — measurable on the per-doc / per-neighbour `is_deleted`
    /// hot paths.
    pub deleted_docs: RwLock<RoaringTreemap>,

    /// Total number of documents in the segment.
    pub total_docs: AtomicU64,

    /// Minimum document ID in this segment.
    pub min_doc_id: u64,

    /// Maximum document ID in this segment.
    pub max_doc_id: u64,

    /// Number of deleted documents.
    pub deleted_count: AtomicU64,

    /// Timestamp of last modification.
    pub last_modified: AtomicU64,

    /// Version number for consistency.
    pub version: AtomicU64,
}

impl DeletionBitmap {
    /// Create a new deletion bitmap for a segment.
    pub fn new(segment_id: String, min_doc_id: u64, max_doc_id: u64) -> Self {
        let total_docs = if max_doc_id >= min_doc_id {
            max_doc_id - min_doc_id + 1
        } else {
            0
        };
        DeletionBitmap {
            segment_id,
            deleted_docs: RwLock::new(RoaringTreemap::new()),
            total_docs: AtomicU64::new(total_docs),
            min_doc_id,
            max_doc_id,
            deleted_count: AtomicU64::new(0),
            last_modified: AtomicU64::new(crate::util::time::now_secs()),
            version: AtomicU64::new(1),
        }
    }

    /// Mark a document as deleted.
    ///
    /// # Arguments
    ///
    /// * `doc_id` - The document ID to mark as deleted. Must be within the
    ///   `[min_doc_id, max_doc_id]` range of this segment.
    ///
    /// # Returns
    ///
    /// Returns `Ok(true)` if the document was newly deleted, or `Ok(false)` if
    /// it was already marked as deleted (idempotent). Returns `Err` if the
    /// document ID is outside this segment's range.
    pub fn delete_document(&self, doc_id: u64) -> Result<bool> {
        // Range check
        if doc_id < self.min_doc_id || doc_id > self.max_doc_id {
            return Err(LaurusError::index(format!(
                "Document ID {doc_id} is out of range [{}, {}] for segment {}",
                self.min_doc_id, self.max_doc_id, self.segment_id
            )));
        }

        let mut docs = self.deleted_docs.write().unwrap();
        // `RoaringTreemap::insert` returns `true` when the id was newly added.
        let newly_deleted = docs.insert(doc_id);
        if newly_deleted {
            self.deleted_count.fetch_add(1, Ordering::SeqCst);
            self.last_modified
                .store(crate::util::time::now_secs(), Ordering::SeqCst);
            self.version.fetch_add(1, Ordering::SeqCst);
        }

        Ok(newly_deleted)
    }

    /// Resize the bitmap to accommodate more documents.
    pub fn resize(&self, new_size: u64) {
        self.total_docs.store(new_size, Ordering::SeqCst);
    }

    /// Check if a document is deleted.
    pub fn is_deleted(&self, doc_id: u64) -> bool {
        self.deleted_docs.read().unwrap().contains(doc_id)
    }

    /// Get deletion ratio (0.0 to 1.0).
    pub fn deletion_ratio(&self) -> f64 {
        if self.total_docs.load(Ordering::SeqCst) == 0 {
            0.0
        } else {
            self.deleted_count.load(Ordering::SeqCst) as f64
                / self.total_docs.load(Ordering::SeqCst) as f64
        }
    }

    /// Get number of live (non-deleted) documents.
    pub fn live_count(&self) -> u64 {
        self.total_docs.load(Ordering::SeqCst) - self.deleted_count.load(Ordering::SeqCst)
    }

    /// Check if compaction is needed.
    pub fn needs_compaction(&self, threshold: f64) -> bool {
        self.deletion_ratio() > threshold
    }

    /// Get all deleted document IDs, in ascending order.
    pub fn get_deleted_docs(&self) -> Vec<u64> {
        let docs = self.deleted_docs.read().unwrap();
        docs.iter().collect()
    }

    /// Get an approximate memory usage of this deletion tracker in bytes.
    ///
    /// The estimate includes the struct itself, the segment ID string buffer,
    /// and the Roaring bitmap's serialized size (Issue #684) as a proxy for its
    /// in-memory footprint — far more accurate, and far smaller for dense
    /// deletion sets, than the previous `AHashSet::capacity()` heuristic.
    pub fn memory_usage(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.deleted_docs.read().unwrap().serialized_size()
            + self.segment_id.capacity()
    }

    /// Write bitmap to storage.
    ///
    /// Writes the v4 format (Issue #684): the metadata header is unchanged from
    /// v3, but the deleted-id set is a `RoaringTreemap::serialize_into` payload
    /// rather than a raw `u64` list — orders of magnitude smaller for dense
    /// deletion sets. v1/v2/v3 remain readable (see [`Self::read_from_storage`]).
    pub fn write_to_storage<W: StorageOutput>(&self, writer: &mut StructWriter<W>) -> Result<()> {
        // Write header
        writer.write_u32(0x44454C42)?; // "DELB" - Deletion Bitmap
        writer.write_u32(4)?; // Version 4 (Roaring bitmap with min/max doc_id)

        // Write metadata
        writer.write_string(&self.segment_id)?;
        writer.write_u64(self.total_docs.load(Ordering::SeqCst))?;
        writer.write_u64(self.deleted_count.load(Ordering::SeqCst))?;
        writer.write_u64(self.last_modified.load(Ordering::SeqCst))?;
        writer.write_u64(self.version.load(Ordering::SeqCst))?;
        writer.write_u64(self.min_doc_id)?;
        writer.write_u64(self.max_doc_id)?;

        // Write the deleted-id set as a Roaring payload (length-prefixed bytes).
        let docs = self.deleted_docs.read().unwrap();
        let mut payload = Vec::with_capacity(docs.serialized_size());
        docs.serialize_into(&mut payload)
            .map_err(|e| LaurusError::index(format!("Failed to serialize deletion bitmap: {e}")))?;
        writer.write_bytes(&payload)?;

        Ok(())
    }

    /// Read bitmap from storage.
    pub fn read_from_storage<R: StorageInput>(reader: &mut StructReader<R>) -> Result<Self> {
        // Read header
        let magic = reader.read_u32()?;
        if magic != 0x44454C42 {
            return Err(LaurusError::index("Invalid deletion bitmap format"));
        }

        let version = reader.read_u32()?;
        if version == 1 {
            // Legacy BitVec format
            let segment_id = reader.read_string()?;
            let total_docs = reader.read_u64()?;
            let deleted_count = reader.read_u64()?;
            let last_modified = reader.read_u64()?;
            let bitmap_version = reader.read_u64()?;

            let _bitmap_size = reader.read_varint()? as usize;
            let bitmap_bytes = reader.read_bytes()?;
            let bitvec = bit_vec::BitVec::from_bytes(&bitmap_bytes);

            let mut deleted_docs = RoaringTreemap::new();
            let mut min_doc_id = u64::MAX;
            let mut max_doc_id = 0;
            for (idx, bit) in bitvec.iter().enumerate() {
                if bit {
                    let doc_id = idx as u64;
                    deleted_docs.insert(doc_id);
                    min_doc_id = min_doc_id.min(doc_id);
                    max_doc_id = max_doc_id.max(doc_id);
                }
            }
            // If no docs were deleted, min/max might be default values,
            // but total_docs should give a hint for the range.
            // For version 1, we don't have explicit min/max, so we infer.
            // If total_docs is 0, min/max can be 0. Otherwise, assume 0 to total_docs-1.
            if total_docs > 0 && deleted_docs.is_empty() {
                min_doc_id = 0;
                max_doc_id = total_docs - 1;
            } else if total_docs == 0 {
                min_doc_id = 0;
                max_doc_id = 0;
            }

            Ok(DeletionBitmap {
                segment_id,
                deleted_docs: RwLock::new(deleted_docs),
                total_docs: AtomicU64::new(total_docs),
                min_doc_id,
                max_doc_id,
                deleted_count: AtomicU64::new(deleted_count),
                last_modified: AtomicU64::new(last_modified),
                version: AtomicU64::new(bitmap_version),
            })
        } else if version == 2 {
            // New HashSet format
            let segment_id = reader.read_string()?;
            let total_docs = reader.read_u64()?;
            let deleted_count = reader.read_u64()?;
            let last_modified = reader.read_u64()?;
            let bitmap_version = reader.read_u64()?;

            let deleted_id_count = reader.read_varint()? as usize;
            let _ = deleted_id_count; // count is informational; Roaring grows as needed
            let mut deleted_docs = RoaringTreemap::new();
            let mut min_doc_id = u64::MAX;
            let mut max_doc_id = 0;
            for _ in 0..deleted_id_count {
                let doc_id = reader.read_u64()?;
                deleted_docs.insert(doc_id);
                min_doc_id = min_doc_id.min(doc_id);
                max_doc_id = max_doc_id.max(doc_id);
            }
            // For version 2, we don't have explicit min/max, so we infer.
            // If total_docs is 0, min/max can be 0. Otherwise, assume 0 to total_docs-1.
            if total_docs > 0 && deleted_docs.is_empty() {
                min_doc_id = 0;
                max_doc_id = total_docs - 1;
            } else if total_docs == 0 {
                min_doc_id = 0;
                max_doc_id = 0;
            }

            Ok(DeletionBitmap {
                segment_id,
                deleted_docs: RwLock::new(deleted_docs),
                total_docs: AtomicU64::new(total_docs),
                min_doc_id,
                max_doc_id,
                deleted_count: AtomicU64::new(deleted_count),
                last_modified: AtomicU64::new(last_modified),
                version: AtomicU64::new(bitmap_version),
            })
        } else if version == 3 {
            // Version 3 (HashSet based with min/max doc_id)
            let segment_id = reader.read_string()?;
            let total_docs = reader.read_u64()?;
            let deleted_count = reader.read_u64()?;
            let last_modified = reader.read_u64()?;
            let bitmap_version = reader.read_u64()?;
            let min_doc_id = reader.read_u64()?;
            let max_doc_id = reader.read_u64()?;

            let deleted_id_count = reader.read_varint()? as usize;
            let _ = deleted_id_count; // count is informational; Roaring grows as needed
            let mut deleted_docs = RoaringTreemap::new();
            for _ in 0..deleted_id_count {
                deleted_docs.insert(reader.read_u64()?);
            }

            Ok(DeletionBitmap {
                segment_id,
                deleted_docs: RwLock::new(deleted_docs),
                total_docs: AtomicU64::new(total_docs),
                min_doc_id,
                max_doc_id,
                deleted_count: AtomicU64::new(deleted_count),
                last_modified: AtomicU64::new(last_modified),
                version: AtomicU64::new(bitmap_version),
            })
        } else if version == 4 {
            // Version 4 (Roaring bitmap with min/max doc_id) — Issue #684.
            let segment_id = reader.read_string()?;
            let total_docs = reader.read_u64()?;
            let deleted_count = reader.read_u64()?;
            let last_modified = reader.read_u64()?;
            let bitmap_version = reader.read_u64()?;
            let min_doc_id = reader.read_u64()?;
            let max_doc_id = reader.read_u64()?;

            let payload = reader.read_bytes()?;
            let deleted_docs = RoaringTreemap::deserialize_from(&payload[..]).map_err(|e| {
                LaurusError::index(format!("Failed to deserialize deletion bitmap: {e}"))
            })?;

            Ok(DeletionBitmap {
                segment_id,
                deleted_docs: RwLock::new(deleted_docs),
                total_docs: AtomicU64::new(total_docs),
                min_doc_id,
                max_doc_id,
                deleted_count: AtomicU64::new(deleted_count),
                last_modified: AtomicU64::new(last_modified),
                version: AtomicU64::new(bitmap_version),
            })
        } else {
            Err(LaurusError::index(format!(
                "Unsupported bitmap version: {version}"
            )))
        }
    }
}

/// Entry in the deletion log for recovery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeletionLogEntry {
    /// Timestamp of the deletion.
    pub timestamp: u64,

    /// Segment ID.
    pub segment_id: String,

    /// Document ID that was deleted.
    pub doc_id: u64,

    /// Reason for deletion.
    pub reason: String,

    /// Log sequence number.
    pub sequence: u64,
}

/// Log for tracking deletion operations.
#[derive(Debug)]
pub struct DeletionLog {
    /// Storage backend.
    storage: Arc<dyn Storage>,

    /// Current sequence number.
    sequence: std::sync::atomic::AtomicU64,

    /// Log file path.
    log_path: String,
}

impl DeletionLog {
    /// Create a new deletion log.
    pub fn new(storage: Arc<dyn Storage>, log_path: String) -> Result<Self> {
        let log = DeletionLog {
            storage,
            sequence: std::sync::atomic::AtomicU64::new(0),
            log_path,
        };

        // Load existing sequence number
        log.load_sequence()?;

        Ok(log)
    }

    /// Log a deletion operation.
    pub fn log_deletion(&self, segment_id: &str, doc_id: u64, reason: &str) -> Result<()> {
        let entry = DeletionLogEntry {
            timestamp: crate::util::time::now_secs(),
            segment_id: segment_id.to_string(),
            doc_id,
            reason: reason.to_string(),
            sequence: self
                .sequence
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst),
        };

        // Append to log file
        let output = self.storage.create_output_append(&self.log_path)?;
        let mut writer = StructWriter::new(output);

        // Write entry
        let json = serde_json::to_string(&entry)?;
        writer.write_string(&json)?;
        writer.write_u8(b'\n')?; // Newline separator
        writer.close()?;

        Ok(())
    }

    /// Load sequence number from existing log.
    fn load_sequence(&self) -> Result<()> {
        if let Ok(input) = self.storage.open_input(&self.log_path) {
            let mut reader = StructReader::new(input)?;
            let mut max_sequence = 0;

            // Read all entries to find max sequence
            while !reader.is_eof() {
                if let Ok(json) = reader.read_string() {
                    if let Ok(entry) = serde_json::from_str::<DeletionLogEntry>(&json) {
                        max_sequence = max_sequence.max(entry.sequence);
                    }
                    // Skip newline
                    if reader.read_u8().is_err() {
                        // EOF or error after string
                        break;
                    }
                } else {
                    // Failed to read string (EOF or corruption)
                    break;
                }
            }

            self.sequence
                .store(max_sequence + 1, std::sync::atomic::Ordering::SeqCst);
        }

        Ok(())
    }
}

/// Statistics about deletion operations.
#[derive(Debug, Clone, Default)]
pub struct DeletionStats {
    /// Total number of segments tracked.
    pub segments_tracked: usize,

    /// Total documents across all segments.
    pub total_docs: u64,

    /// Total deleted documents.
    pub total_deleted: u64,

    /// Overall deletion ratio.
    pub overall_deletion_ratio: f64,

    /// Number of segments needing compaction.
    pub segments_needing_compaction: usize,

    /// Total memory used by bitmaps (bytes).
    pub bitmap_memory_usage: usize,

    /// Number of deletion operations performed.
    pub deletion_operations: u64,

    /// Number of compaction operations performed.
    pub compaction_operations: u64,
}

/// Global deletion state across all segments.
#[derive(Debug, Clone)]
pub struct GlobalDeletionState {
    /// Total documents across all segments.
    pub total_documents: u64,

    /// Total deleted documents across all segments.
    pub total_deleted: u64,

    /// Global deletion ratio.
    pub global_deletion_ratio: f64,

    /// Segments that need compaction.
    pub compaction_candidates: Vec<String>,

    /// Last compaction timestamp.
    pub last_compaction: u64,

    /// Total space that can be reclaimed (bytes).
    pub reclaimable_space: u64,
}

impl Default for GlobalDeletionState {
    fn default() -> Self {
        Self::new()
    }
}

impl GlobalDeletionState {
    /// Create a new global deletion state.
    pub fn new() -> Self {
        GlobalDeletionState {
            total_documents: 0,
            total_deleted: 0,
            global_deletion_ratio: 0.0,
            compaction_candidates: Vec::new(),
            last_compaction: 0,
            reclaimable_space: 0,
        }
    }

    /// Check if global compaction is needed.
    pub fn needs_global_compaction(&self, threshold: f64) -> bool {
        self.global_deletion_ratio > threshold
    }

    /// Get efficiency metrics.
    pub fn efficiency_metrics(&self) -> (f64, u64, usize) {
        (
            self.global_deletion_ratio,
            self.reclaimable_space,
            self.compaction_candidates.len(),
        )
    }
}

/// Core deletion manager.
#[derive(Debug)]
pub struct DeletionManager {
    /// Configuration.
    config: DeletionConfig,

    /// Storage backend.
    storage: Arc<dyn Storage>,

    /// Deletion bitmaps per segment.
    bitmaps: RwLock<AHashMap<String, Arc<DeletionBitmap>>>,

    /// Deletion log for recovery.
    deletion_log: Option<DeletionLog>,

    /// Statistics.
    stats: RwLock<DeletionStats>,

    /// Global deletion state.
    global_state: RwLock<GlobalDeletionState>,
}

impl DeletionManager {
    /// Create a new deletion manager.
    pub fn new(config: DeletionConfig, storage: Arc<dyn Storage>) -> Result<Self> {
        let deletion_log = if config.enable_deletion_log {
            Some(DeletionLog::new(
                storage.clone(),
                "deletions.log".to_string(),
            )?)
        } else {
            None
        };

        let manager = DeletionManager {
            config,
            storage,
            bitmaps: RwLock::new(AHashMap::new()),
            deletion_log,
            stats: RwLock::new(DeletionStats::default()),
            global_state: RwLock::new(GlobalDeletionState::new()),
        };

        // Load existing bitmaps
        manager.load_bitmaps()?;

        // Initialize global state
        manager.update_global_state()?;

        Ok(manager)
    }

    /// Get deletion bitmap for a segment.
    pub fn get_bitmap(&self, segment_id: &str) -> Option<Arc<DeletionBitmap>> {
        self.bitmaps.read().unwrap().get(segment_id).cloned()
    }

    /// Initialize deletion tracking for a segment.
    pub fn initialize_segment(
        &self,
        segment_id: &str,
        min_doc_id: u64,
        max_doc_id: u64,
    ) -> Result<()> {
        let bitmaps = self.bitmaps.read().unwrap();
        if bitmaps.contains_key(segment_id) {
            // If segment already exists, update its min/max if necessary, or just return.
            // For now, we assume it's already correctly initialized.
            // A more robust system might check if min/max changed and update.
            return Ok(());
        }
        drop(bitmaps);

        let bitmap = Arc::new(DeletionBitmap::new(
            segment_id.to_string(),
            min_doc_id,
            max_doc_id,
        ));

        {
            let mut bitmaps = self.bitmaps.write().unwrap();
            bitmaps.insert(segment_id.to_string(), bitmap);
        }

        self.save_bitmap(segment_id)?;
        self.update_stats();
        let _ = self.update_global_state();

        Ok(())
    }

    /// Resize a segment's bitmap.
    pub fn resize_segment(&self, segment_id: &str, new_size: u64) -> Result<()> {
        let bitmaps = self.bitmaps.read().unwrap();
        if let Some(bitmap) = bitmaps.get(segment_id) {
            bitmap.resize(new_size);
            self.save_bitmap(segment_id)?;
        }
        Ok(())
    }

    /// Mark a document as deleted.
    pub fn delete_document(&self, segment_id: &str, doc_id: u64, reason: &str) -> Result<bool> {
        let was_deleted = {
            let bitmaps = self.bitmaps.read().unwrap();

            if let Some(bitmap) = bitmaps.get(segment_id) {
                bitmap.delete_document(doc_id)?
            } else {
                return Err(LaurusError::index(format!(
                    "Segment {segment_id} not found in deletion manager"
                )));
            }
        };

        // Log the deletion
        if let Some(ref log) = self.deletion_log {
            log.log_deletion(segment_id, doc_id, reason)?;
        }

        // Save updated bitmap
        if was_deleted {
            self.save_bitmap(segment_id)?;
            self.update_stats();
            let _ = self.update_global_state();
        }

        Ok(was_deleted)
    }

    /// Delete multiple documents in batch.
    pub fn delete_documents(&self, segment_id: &str, doc_ids: &[u64], reason: &str) -> Result<u64> {
        let mut deleted_count = 0;

        // Process in batches (read lock is sufficient with atomic bitmap)
        for chunk in doc_ids.chunks(self.config.deletion_batch_size) {
            {
                let bitmaps = self.bitmaps.read().unwrap();

                if let Some(bitmap) = bitmaps.get(segment_id) {
                    for &doc_id in chunk {
                        if bitmap.delete_document(doc_id)? {
                            deleted_count += 1;
                        }
                    }
                } else {
                    return Err(LaurusError::index(format!(
                        "Segment {segment_id} not found in deletion manager"
                    )));
                }
            }

            // Log deletions
            if let Some(ref log) = self.deletion_log {
                for &doc_id in chunk {
                    log.log_deletion(segment_id, doc_id, reason)?;
                }
            }
        }

        if deleted_count > 0 {
            self.save_bitmap(segment_id)?;
            self.update_stats();
            let _ = self.update_global_state();
        }

        Ok(deleted_count)
    }

    /// Check if a document is deleted.
    pub fn is_deleted(&self, segment_id: &str, doc_id: u64) -> bool {
        let bitmaps = self.bitmaps.read().unwrap();

        if let Some(bitmap) = bitmaps.get(segment_id) {
            bitmap.is_deleted(doc_id)
        } else {
            false
        }
    }

    /// Get deletion ratio for a segment.
    pub fn get_deletion_ratio(&self, segment_id: &str) -> f64 {
        let bitmaps = self.bitmaps.read().unwrap();

        if let Some(bitmap) = bitmaps.get(segment_id) {
            bitmap.deletion_ratio()
        } else {
            0.0
        }
    }

    /// Get segments that need compaction.
    pub fn get_compaction_candidates(&self) -> Vec<String> {
        let bitmaps = self.bitmaps.read().unwrap();

        bitmaps
            .values()
            .filter(|bitmap| bitmap.needs_compaction(self.config.compaction_threshold))
            .map(|bitmap| bitmap.segment_id.clone())
            .collect()
    }

    /// Get deleted document IDs for a segment.
    pub fn get_deleted_docs(&self, segment_id: &str) -> Vec<u64> {
        let bitmaps = self.bitmaps.read().unwrap();

        if let Some(bitmap) = bitmaps.get(segment_id) {
            bitmap.get_deleted_docs()
        } else {
            Vec::new()
        }
    }

    /// Remove tracking for a segment (after merge/deletion).
    pub fn remove_segment(&self, segment_id: &str) -> Result<()> {
        {
            let mut bitmaps = self.bitmaps.write().unwrap();
            bitmaps.remove(segment_id);
        }

        // Delete bitmap file
        let bitmap_file = format!("{segment_id}.delmap");
        let _ = self.storage.delete_file(&bitmap_file);

        self.update_stats();
        let _ = self.update_global_state();
        Ok(())
    }

    /// Save bitmap to storage.
    fn save_bitmap(&self, segment_id: &str) -> Result<()> {
        let bitmaps = self.bitmaps.read().unwrap();

        if let Some(bitmap) = bitmaps.get(segment_id) {
            let bitmap_file = format!("{segment_id}.delmap");
            let output = self.storage.create_output(&bitmap_file)?;
            let mut writer = StructWriter::new(output);
            bitmap.write_to_storage(&mut writer)?;
            writer.close()?;
        }

        Ok(())
    }

    /// Load existing bitmaps from storage.
    fn load_bitmaps(&self) -> Result<()> {
        let files = self.storage.list_files()?;

        for file in files {
            if file.ends_with(".delmap") {
                let input = self.storage.open_input(&file)?;
                let mut reader = StructReader::new(input)?;

                if let Ok(bitmap) = DeletionBitmap::read_from_storage(&mut reader) {
                    let mut bitmaps = self.bitmaps.write().unwrap();
                    bitmaps.insert(bitmap.segment_id.clone(), Arc::new(bitmap));
                }
            }
        }

        self.update_stats();
        let _ = self.update_global_state();
        Ok(())
    }

    /// Update internal statistics.
    fn update_stats(&self) {
        let bitmaps = self.bitmaps.read().unwrap();
        let mut stats = self.stats.write().unwrap();

        stats.segments_tracked = bitmaps.len();
        stats.total_docs = bitmaps
            .values()
            .map(|b| b.total_docs.load(Ordering::SeqCst))
            .sum();
        stats.total_deleted = bitmaps
            .values()
            .map(|b| b.deleted_count.load(Ordering::SeqCst))
            .sum();

        if stats.total_docs > 0 {
            stats.overall_deletion_ratio = stats.total_deleted as f64 / stats.total_docs as f64;
        }

        stats.segments_needing_compaction = bitmaps
            .values()
            .filter(|b| b.needs_compaction(self.config.compaction_threshold))
            .count();

        stats.bitmap_memory_usage = bitmaps.values().map(|b| b.memory_usage()).sum();
    }

    /// Get current statistics.
    pub fn get_stats(&self) -> DeletionStats {
        self.stats.read().unwrap().clone()
    }

    /// Get configuration.
    pub fn get_config(&self) -> &DeletionConfig {
        &self.config
    }

    /// Get global deletion state.
    pub fn get_global_state(&self) -> GlobalDeletionState {
        self.global_state.read().unwrap().clone()
    }

    /// Restore global deletion state (used for transaction rollback).
    pub fn restore_global_state(&self, state: GlobalDeletionState) -> Result<()> {
        let mut global_state = self.global_state.write().unwrap();
        *global_state = state;
        Ok(())
    }

    /// Update global deletion state based on current segment states.
    pub fn update_global_state(&self) -> Result<()> {
        let bitmaps = self.bitmaps.read().unwrap();
        let mut global_state = self.global_state.write().unwrap();

        // Calculate totals
        global_state.total_documents = bitmaps
            .values()
            .map(|b| b.total_docs.load(Ordering::SeqCst))
            .sum();
        global_state.total_deleted = bitmaps
            .values()
            .map(|b| b.deleted_count.load(Ordering::SeqCst))
            .sum();

        // Calculate global deletion ratio
        if global_state.total_documents > 0 {
            global_state.global_deletion_ratio =
                global_state.total_deleted as f64 / global_state.total_documents as f64;
        } else {
            global_state.global_deletion_ratio = 0.0;
        }

        // Find compaction candidates
        global_state.compaction_candidates = bitmaps
            .values()
            .filter(|b| b.needs_compaction(self.config.compaction_threshold))
            .map(|b| b.segment_id.clone())
            .collect();

        // Estimate reclaimable space (approximate)
        global_state.reclaimable_space = bitmaps
            .values()
            .map(|b| {
                if b.needs_compaction(self.config.compaction_threshold) {
                    // Rough estimate: deleted_ratio * segment_size
                    (b.deletion_ratio() * b.total_docs.load(Ordering::SeqCst) as f64 * 100.0) as u64 // 100 bytes per doc estimate
                } else {
                    0
                }
            })
            .sum();

        Ok(())
    }

    /// Check if automatic compaction should be triggered.
    pub fn should_trigger_auto_compaction(&self) -> bool {
        if !self.config.auto_compaction {
            return false;
        }

        let global_state = self.global_state.read().unwrap();
        let current_time = crate::util::time::now_secs();

        // Check if enough time has passed since last compaction
        let time_threshold = global_state.last_compaction + self.config.compaction_interval_secs;
        let time_to_compact = current_time >= time_threshold;

        // Check if deletion ratio exceeds threshold
        let ratio_threshold =
            global_state.needs_global_compaction(self.config.compaction_threshold);

        // Check if we have candidates
        let has_candidates = !global_state.compaction_candidates.is_empty();

        time_to_compact && ratio_threshold && has_candidates
    }

    /// Mark compaction as completed.
    pub fn mark_compaction_completed(&self, segments_compacted: &[String]) -> Result<()> {
        // Remove compacted segments from tracking
        for segment_id in segments_compacted {
            self.remove_segment(segment_id)?;
        }

        // Update global state
        let mut global_state = self.global_state.write().unwrap();
        global_state.last_compaction = crate::util::time::now_secs();

        // Update statistics
        let mut stats = self.stats.write().unwrap();
        stats.compaction_operations += 1;

        // Refresh global state
        drop(global_state);
        self.update_global_state()?;

        Ok(())
    }

    /// Get comprehensive deletion report.
    pub fn get_deletion_report(&self) -> DeletionReport {
        let stats = self.stats.read().unwrap();
        let global_state = self.global_state.read().unwrap();
        let bitmaps = self.bitmaps.read().unwrap();

        let segment_reports: Vec<SegmentDeletionReport> = bitmaps
            .values()
            .map(|bitmap| SegmentDeletionReport {
                segment_id: bitmap.segment_id.clone(),
                total_docs: bitmap.total_docs.load(Ordering::SeqCst),
                deleted_docs: bitmap.deleted_count.load(Ordering::SeqCst),
                deletion_ratio: bitmap.deletion_ratio(),
                needs_compaction: bitmap.needs_compaction(self.config.compaction_threshold),
                memory_usage: bitmap.memory_usage(),
                last_modified: bitmap.last_modified.load(Ordering::SeqCst),
            })
            .collect();

        DeletionReport {
            global_state: global_state.clone(),
            deletion_stats: stats.clone(),
            segment_reports,
            auto_compaction_enabled: self.config.auto_compaction,
            next_compaction_due: global_state.last_compaction
                + self.config.compaction_interval_secs,
        }
    }
}

/// Report about segment deletion status.
#[derive(Debug, Clone)]
pub struct SegmentDeletionReport {
    /// Segment identifier.
    pub segment_id: String,

    /// Total documents in segment.
    pub total_docs: u64,

    /// Deleted documents in segment.
    pub deleted_docs: u64,

    /// Deletion ratio.
    pub deletion_ratio: f64,

    /// Whether this segment needs compaction.
    pub needs_compaction: bool,

    /// Memory usage of deletion bitmap.
    pub memory_usage: usize,

    /// Last modification timestamp.
    pub last_modified: u64,
}

/// Comprehensive deletion report.
#[derive(Debug, Clone)]
pub struct DeletionReport {
    /// Global deletion state.
    pub global_state: GlobalDeletionState,

    /// Overall deletion statistics.
    pub deletion_stats: DeletionStats,

    /// Per-segment deletion reports.
    pub segment_reports: Vec<SegmentDeletionReport>,

    /// Whether auto-compaction is enabled.
    pub auto_compaction_enabled: bool,

    /// Timestamp when next compaction is due.
    pub next_compaction_due: u64,
}

impl DeletionReport {
    /// Get summary metrics.
    pub fn summary(&self) -> (f64, usize, u64, bool) {
        (
            self.global_state.global_deletion_ratio,
            self.global_state.compaction_candidates.len(),
            self.global_state.reclaimable_space,
            self.global_state.needs_global_compaction(0.3), // Default threshold
        )
    }

    /// Get segments by compaction urgency.
    pub fn segments_by_urgency(&self) -> (Vec<String>, Vec<String>, Vec<String>) {
        let mut urgent = Vec::new();
        let mut moderate = Vec::new();
        let mut low = Vec::new();

        for report in &self.segment_reports {
            if report.deletion_ratio > 0.5 {
                urgent.push(report.segment_id.clone());
            } else if report.deletion_ratio > 0.3 {
                moderate.push(report.segment_id.clone());
            } else if report.deletion_ratio > 0.1 {
                low.push(report.segment_id.clone());
            }
        }

        (urgent, moderate, low)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::storage::memory::MemoryStorage;
    use crate::storage::memory::MemoryStorageConfig;

    #[test]
    fn test_deletion_bitmap_creation() {
        let bitmap = DeletionBitmap::new("seg001".to_string(), 0, 999);

        assert_eq!(bitmap.segment_id, "seg001");
        assert_eq!(bitmap.total_docs.load(Ordering::SeqCst), 1000);
        assert_eq!(bitmap.deleted_count.load(Ordering::SeqCst), 0);
        assert_eq!(bitmap.deletion_ratio(), 0.0);
        assert_eq!(bitmap.live_count(), 1000);
    }

    #[test]
    fn test_deletion_bitmap_operations() {
        let bitmap = DeletionBitmap::new("seg001".to_string(), 0, 99);

        // Delete some documents
        assert!(bitmap.delete_document(5).unwrap());
        assert!(bitmap.delete_document(10).unwrap());
        assert!(bitmap.delete_document(15).unwrap());

        // Check deletion status
        assert!(bitmap.is_deleted(5));
        assert!(bitmap.is_deleted(10));
        assert!(bitmap.is_deleted(15));
        assert!(!bitmap.is_deleted(20));

        // Check counts
        assert_eq!(bitmap.deleted_count.load(Ordering::SeqCst), 3);
        assert_eq!(bitmap.live_count(), 97);
        assert_eq!(bitmap.deletion_ratio(), 0.03);

        // Try to delete same document again
        assert!(!bitmap.delete_document(5).unwrap());
        assert_eq!(bitmap.deleted_count.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn test_deletion_bitmap_out_of_range() {
        let bitmap = DeletionBitmap::new("seg001".to_string(), 0, 99);

        let result = bitmap.delete_document(150);
        assert!(result.is_err());

        assert!(!bitmap.is_deleted(150));
    }

    /// v4 (Roaring) `.delmap` round-trips: write then read yields the same
    /// deleted set, ordering, and metadata (Issue #684).
    #[test]
    fn test_deletion_bitmap_v4_round_trip() {
        let storage = MemoryStorage::new(MemoryStorageConfig::default());

        let bitmap = DeletionBitmap::new("seg-v4".to_string(), 0, 999);
        for id in [3u64, 7, 42, 900, 999] {
            bitmap.delete_document(id).unwrap();
        }

        {
            let output = storage.create_output("seg-v4.delmap").unwrap();
            let mut writer = StructWriter::new(output);
            bitmap.write_to_storage(&mut writer).unwrap();
            writer.close().unwrap();
        }

        let loaded = {
            let input = storage.open_input("seg-v4.delmap").unwrap();
            let mut reader = StructReader::new(input).unwrap();
            DeletionBitmap::read_from_storage(&mut reader).unwrap()
        };

        assert_eq!(loaded.segment_id, "seg-v4");
        assert_eq!(loaded.min_doc_id, 0);
        assert_eq!(loaded.max_doc_id, 999);
        assert_eq!(loaded.deleted_count.load(Ordering::SeqCst), 5);
        // `get_deleted_docs` is ascending for a Roaring bitmap.
        assert_eq!(loaded.get_deleted_docs(), vec![3, 7, 42, 900, 999]);
        assert!(loaded.is_deleted(42));
        assert!(!loaded.is_deleted(43));
    }

    /// Legacy v3 (raw-`u64`-list) `.delmap` payloads must still be readable
    /// after the Roaring migration (Issue #684 back-compat).
    #[test]
    fn test_deletion_bitmap_reads_v3_format() {
        let storage = MemoryStorage::new(MemoryStorageConfig::default());

        // Hand-write a v3 payload (the format prior to this change).
        {
            let output = storage.create_output("seg-v3.delmap").unwrap();
            let mut writer = StructWriter::new(output);
            writer.write_u32(0x44454C42).unwrap(); // magic "DELB"
            writer.write_u32(3).unwrap(); // version 3
            writer.write_string("seg-v3").unwrap();
            writer.write_u64(1000).unwrap(); // total_docs
            writer.write_u64(3).unwrap(); // deleted_count
            writer.write_u64(12345).unwrap(); // last_modified
            writer.write_u64(7).unwrap(); // bitmap version
            writer.write_u64(0).unwrap(); // min_doc_id
            writer.write_u64(999).unwrap(); // max_doc_id
            writer.write_varint(3).unwrap(); // deleted id count
            for id in [11u64, 222, 888] {
                writer.write_u64(id).unwrap();
            }
            writer.close().unwrap();
        }

        let loaded = {
            let input = storage.open_input("seg-v3.delmap").unwrap();
            let mut reader = StructReader::new(input).unwrap();
            DeletionBitmap::read_from_storage(&mut reader).unwrap()
        };

        assert_eq!(loaded.segment_id, "seg-v3");
        assert_eq!(loaded.min_doc_id, 0);
        assert_eq!(loaded.max_doc_id, 999);
        assert_eq!(loaded.get_deleted_docs(), vec![11, 222, 888]);
        assert!(loaded.is_deleted(222));
        assert!(!loaded.is_deleted(223));
    }

    #[test]
    fn test_deletion_manager_creation() {
        let config = DeletionConfig::default();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        let manager = DeletionManager::new(config, storage).unwrap();

        let stats = manager.get_stats();
        assert_eq!(stats.segments_tracked, 0);
        assert_eq!(stats.total_docs, 0);
    }

    #[test]
    fn test_deletion_manager_operations() {
        let config = DeletionConfig::default();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let manager = DeletionManager::new(config, storage).unwrap();

        // Initialize segment
        manager.initialize_segment("seg001", 0, 999).unwrap();

        // Delete documents
        assert!(
            manager
                .delete_document("seg001", 100, "test deletion")
                .unwrap()
        );
        assert!(
            manager
                .delete_document("seg001", 200, "test deletion")
                .unwrap()
        );

        // Check deletion status
        assert!(manager.is_deleted("seg001", 100));
        assert!(manager.is_deleted("seg001", 200));
        assert!(!manager.is_deleted("seg001", 300));

        // Check deletion ratio
        let ratio = manager.get_deletion_ratio("seg001");
        assert_eq!(ratio, 0.002); // 2/1000

        let stats = manager.get_stats();
        assert_eq!(stats.segments_tracked, 1);
        assert_eq!(stats.total_deleted, 2);
    }

    #[test]
    fn test_batch_deletion() {
        let config = DeletionConfig::default();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let manager = DeletionManager::new(config, storage).unwrap();

        manager.initialize_segment("seg001", 0, 19).unwrap(); // Reduced from 1000 to 20

        let doc_ids = vec![1, 2, 3, 4, 5]; // 5 docs out of 20
        let deleted_count = manager
            .delete_documents("seg001", &doc_ids, "batch deletion")
            .unwrap();

        assert_eq!(deleted_count, 5);

        for &doc_id in &doc_ids {
            assert!(manager.is_deleted("seg001", doc_id));
        }

        let ratio = manager.get_deletion_ratio("seg001");
        assert_eq!(ratio, 0.25); // 5/20 = 0.25
    }

    #[test]
    fn test_compaction_candidates() {
        let config = DeletionConfig {
            compaction_threshold: 0.1, // 10%
            ..Default::default()
        };

        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let manager = DeletionManager::new(config, storage).unwrap();

        // Initialize segments (reduced sizes)
        manager.initialize_segment("seg001", 0, 9).unwrap(); // Reduced from 1000 to 10
        manager.initialize_segment("seg002", 0, 9).unwrap(); // Reduced from 1000 to 10

        // Delete enough docs in seg001 to trigger compaction
        let doc_ids: Vec<u64> = vec![0, 1]; // 20% deletion (2/10)
        manager
            .delete_documents("seg001", &doc_ids, "test")
            .unwrap();

        // Delete fewer docs in seg002
        manager.delete_documents("seg002", &[0], "test").unwrap(); // 10% deletion (1/10)

        let candidates = manager.get_compaction_candidates();
        assert_eq!(candidates.len(), 1); // Only seg001 should be candidate (20% > 10% threshold)
        assert!(candidates.contains(&"seg001".to_string()));
    }

    #[test]
    fn test_global_deletion_state() {
        let config = DeletionConfig::default();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let manager = DeletionManager::new(config, storage).unwrap();

        // Initialize multiple segments (reduced sizes)
        manager.initialize_segment("seg001", 0, 9).unwrap(); // Reduced from 1000 to 10
        manager.initialize_segment("seg002", 0, 19).unwrap(); // Reduced from 2000 to 20

        // Delete documents in different segments
        let doc_ids1: Vec<u64> = (0..4).collect(); // 40% deletion (4/10)
        manager
            .delete_documents("seg001", &doc_ids1, "test")
            .unwrap();

        let doc_ids2: Vec<u64> = (0..2).collect(); // 10% deletion (2/20)
        manager
            .delete_documents("seg002", &doc_ids2, "test")
            .unwrap();

        // Get global state
        let global_state = manager.get_global_state();

        assert_eq!(global_state.total_documents, 30); // 10 + 20
        assert_eq!(global_state.total_deleted, 6); // 4 + 2
        assert!((global_state.global_deletion_ratio - 0.2).abs() < 0.001); // 6/30 = 0.2
        assert!(!global_state.compaction_candidates.is_empty());
        assert!(
            global_state
                .compaction_candidates
                .contains(&"seg001".to_string())
        );
    }

    #[test]
    fn test_deletion_report() {
        let config = DeletionConfig::default();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let manager = DeletionManager::new(config, storage).unwrap();

        // Initialize segments and add deletions (reduced sizes)
        manager.initialize_segment("seg001", 0, 9).unwrap(); // Reduced from 1000 to 10
        manager.initialize_segment("seg002", 0, 19).unwrap(); // Reduced from 2000 to 20

        let doc_ids: Vec<u64> = (0..4).collect(); // 40% deletion in seg001 (4/10)
        manager
            .delete_documents("seg001", &doc_ids, "test")
            .unwrap();

        // Get comprehensive deletion report
        let report = manager.get_deletion_report();

        assert_eq!(report.segment_reports.len(), 2);
        assert_eq!(report.global_state.total_documents, 30); // 10 + 20
        assert_eq!(report.global_state.total_deleted, 4);
        assert!(report.auto_compaction_enabled);

        // Test urgency classification
        let (urgent, moderate, low) = report.segments_by_urgency();
        assert_eq!(urgent.len(), 0); // seg001 has 40% deletion (moderate)
        assert_eq!(moderate.len(), 1); // seg001 falls into moderate category (30-50%)
        assert_eq!(low.len(), 0);

        // Test summary
        let (ratio, candidates, _space, needs_compaction) = report.summary();
        assert!((ratio - 0.133333).abs() < 0.001); // 4/30 ≈ 0.133
        assert_eq!(candidates, 1);
        // Note: space may be 0 if deletion ratio is below compaction threshold
        // space is u64, so >= 0 check is redundant
        assert!(!needs_compaction); // Below 30% threshold
    }

    #[test]
    fn test_auto_compaction_trigger() {
        let config = DeletionConfig {
            auto_compaction: true,
            compaction_threshold: 0.2,   // 20% threshold
            compaction_interval_secs: 1, // 1 second interval
            ..Default::default()
        };

        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let manager = DeletionManager::new(config, storage).unwrap();

        // Initialize segment and delete enough to trigger compaction
        manager.initialize_segment("seg001", 0, 99).unwrap();
        let doc_ids: Vec<u64> = (0..25).collect(); // 25% deletion
        manager
            .delete_documents("seg001", &doc_ids, "test")
            .unwrap();

        // Should trigger auto compaction after interval
        std::thread::sleep(std::time::Duration::from_secs(2));
        assert!(manager.should_trigger_auto_compaction());
    }

    #[test]
    fn test_mark_compaction_completed() {
        let config = DeletionConfig::default();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let manager = DeletionManager::new(config, storage).unwrap();

        // Initialize segments
        manager.initialize_segment("seg001", 0, 999).unwrap();
        manager.initialize_segment("seg002", 0, 1999).unwrap();

        // Mark compaction as completed for seg001
        let compacted_segments = vec!["seg001".to_string()];
        manager
            .mark_compaction_completed(&compacted_segments)
            .unwrap();

        // seg001 should be removed from tracking
        let global_state = manager.get_global_state();
        assert_eq!(global_state.total_documents, 2000); // Only seg002 remains

        // Compaction stats should be updated
        let stats = manager.get_stats();
        assert_eq!(stats.compaction_operations, 1);
    }
}
