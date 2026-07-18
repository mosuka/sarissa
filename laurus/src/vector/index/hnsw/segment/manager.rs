//! Segment manager for vector indexes.
//!
//! This module manages vector index segments, including segment metadata,
//! merging strategies, and segment lifecycle.
//!
//! The segment registry is persisted in a `segments.json` manifest. Since
//! #634 (PR-1 / #879) the manifest is **versioned, checksummed, and written
//! atomically** (temp file + CRC-32 trailer + fsync + rename — the same
//! #784/#786 standard the `.hnsw` segment files follow), so it can serve as
//! the commit pivot for the segment-per-commit pipeline: a crash mid-save
//! leaves the previous manifest intact, and a corrupted manifest fails the
//! open loudly instead of silently starting empty.

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::RwLock;

use crate::error::{LaurusError, Result};
use crate::storage::Storage;
use crate::storage::structured::{StructReader, StructWriter};

use super::merge_policy::MergePolicy;

/// Manifest file name for the segment registry.
const MANIFEST_FILE: &str = "segments.json";

/// Temporary file the manifest is staged in before the atomic rename.
const MANIFEST_TMP_FILE: &str = "segments.json.tmp";

/// Current manifest format version (see [`SegmentManifest`]).
const MANIFEST_VERSION: u32 = 1;

/// Versioned payload of the `segments.json` manifest (#634 PR-1 / #879).
///
/// Serialized as JSON and framed with a length prefix plus a CRC-32 trailer
/// (via `StructWriter`), so corruption is detected at load time. The legacy
/// format — a bare JSON array of [`ManagedSegmentInfo`] written in place —
/// is still readable for backward compatibility and is upgraded on the next
/// save.
#[derive(Debug, Serialize, Deserialize)]
struct SegmentManifest {
    /// Format version (currently [`MANIFEST_VERSION`]).
    version: u32,

    /// Next segment ordinal handed out by
    /// [`SegmentManager::generate_segment_id`], persisted so a reopen never
    /// re-issues an ID even if the highest-numbered segment was merged away.
    next_segment_id: u64,

    /// Last WAL sequence number applied to the segments in this manifest.
    ///
    /// Reserved for the segment-per-commit recovery pivot (#634 PR-4): the
    /// vector side persists no WAL checkpoint today, so this stays 0 until
    /// the wiring lands. Persisting it here keeps the manifest format stable
    /// across the campaign.
    last_wal_seq: u64,

    /// The registered segments.
    segments: Vec<ManagedSegmentInfo>,
}

/// Configuration for segment manager.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentManagerConfig {
    /// Maximum number of vectors per segment.
    pub max_vectors_per_segment: u64,

    /// Minimum number of vectors per segment before merging.
    pub min_vectors_per_segment: u64,

    /// Maximum number of segments before triggering merge.
    pub max_segments: u32,

    /// Merge factor (how many segments to merge at once).
    pub merge_factor: u32,
}

impl Default for SegmentManagerConfig {
    fn default() -> Self {
        Self {
            max_vectors_per_segment: 1000000,
            min_vectors_per_segment: 10000,
            max_segments: 100,
            merge_factor: 10,
        }
    }
}

/// Information about a managed segment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManagedSegmentInfo {
    /// Segment identifier.
    pub segment_id: String,

    /// Number of vectors in this segment.
    pub vector_count: u64,

    /// Vector offset for this segment.
    pub vector_offset: u64,

    /// Generation number of this segment.
    pub generation: u64,

    /// Whether this segment has deletions.
    pub has_deletions: bool,

    /// Size of the segment in bytes.
    pub size_bytes: u64,
}

impl ManagedSegmentInfo {
    /// Create a new managed segment info.
    pub fn new(segment_id: String, vector_count: u64, vector_offset: u64, generation: u64) -> Self {
        Self {
            segment_id,
            vector_count,
            vector_offset,
            generation,
            has_deletions: false,
            size_bytes: 0,
        }
    }

    /// Check if this segment should be merged based on config.
    pub fn should_merge(&self, config: &SegmentManagerConfig) -> bool {
        self.vector_count < config.min_vectors_per_segment
    }
}

/// Candidate segments for merging.
#[derive(Debug, Clone)]
pub struct MergeCandidate {
    /// Segments to merge.
    pub segments: Vec<ManagedSegmentInfo>,

    /// Total vector count.
    pub total_vectors: u64,

    /// Total size in bytes.
    pub total_size: u64,
}

/// Strategy for selecting segments to merge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MergeStrategy {
    /// Merge smallest segments first.
    Smallest,

    /// Merge segments with most deletions first.
    MostDeletions,

    /// Merge adjacent segments.
    Adjacent,
}

/// Urgency level for merge operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MergeUrgency {
    /// No urgent need to merge.
    Low,

    /// Should merge soon.
    Medium,

    /// Should merge immediately.
    High,
}

/// Plan for merging segments.
#[derive(Debug, Clone)]
pub struct MergePlan {
    /// Merge candidates.
    pub candidates: Vec<MergeCandidate>,

    /// Strategy used.
    pub strategy: MergeStrategy,

    /// Urgency level.
    pub urgency: MergeUrgency,
}

/// Statistics about segment manager.
#[derive(Debug, Clone)]
pub struct SegmentManagerStats {
    /// Total number of segments.
    pub segment_count: u32,

    /// Total number of vectors across all segments.
    pub total_vectors: u64,

    /// Total size of all segments in bytes.
    pub total_size: u64,

    /// Number of segments with deletions.
    pub segments_with_deletions: u32,

    /// Average vectors per segment.
    pub avg_vectors_per_segment: f64,
}

/// Manages segments for vector indexes.
#[derive(Debug)]
pub struct SegmentManager {
    config: SegmentManagerConfig,
    storage: Arc<dyn Storage>,
    segments: Arc<RwLock<Vec<ManagedSegmentInfo>>>,
    next_segment_id: Arc<RwLock<u64>>,
    /// Last WAL sequence number recorded in the manifest (see
    /// [`SegmentManifest::last_wal_seq`]). Persisted by the next save.
    last_wal_seq: AtomicU64,
}

impl SegmentManager {
    /// Create a new segment manager with the given configuration.
    ///
    /// Loads the `segments.json` manifest when present and sweeps orphaned
    /// segment files (a torn flush can leave a `segment_*.hnsw` on storage
    /// that never made it into the manifest — its documents are covered by
    /// WAL replay, so the file is garbage).
    ///
    /// # Errors
    ///
    /// Returns an error if an existing manifest fails to load or fails its
    /// CRC check — a corrupted registry must fail the open loudly rather
    /// than silently starting empty (which would orphan, and then sweep,
    /// every live segment).
    pub fn new(config: SegmentManagerConfig, storage: Arc<dyn Storage>) -> Result<Self> {
        let manager = Self {
            config,
            storage,
            segments: Arc::new(RwLock::new(Vec::new())),
            next_segment_id: Arc::new(RwLock::new(0)),
            last_wal_seq: AtomicU64::new(0),
        };

        manager.load_state()?;
        manager.cleanup_orphans();

        Ok(manager)
    }

    fn load_state(&self) -> Result<()> {
        let mut reader = match self.storage.open_input(MANIFEST_FILE) {
            Ok(r) => r,
            Err(_) => return Ok(()),
        };

        let mut content = Vec::new();
        reader.read_to_end(&mut content)?;
        // If empty file, ignore
        if content.is_empty() {
            return Ok(());
        }

        // Legacy format: a bare pretty-printed JSON array written in place
        // (no framing, no checksum). Detect it by the leading byte and keep
        // reading it; the next save upgrades to the versioned format.
        let first = content
            .iter()
            .copied()
            .find(|b| !b.is_ascii_whitespace())
            .unwrap_or(0);
        let (segments_info, next_id, wal_seq) = if first == b'[' {
            let segments: Vec<ManagedSegmentInfo> = serde_json::from_slice(&content)?;
            (segments, 0, 0)
        } else {
            // Versioned format: length-prefixed JSON + CRC-32 trailer
            // (the StructWriter/StructReader framing).
            let reader = self.storage.open_input(MANIFEST_FILE)?;
            let mut struct_reader = StructReader::new(reader)?;
            let json = struct_reader.read_bytes()?;
            if !struct_reader.verify_checksum()? {
                return Err(LaurusError::index(
                    "segments.json checksum mismatch — manifest is corrupted",
                ));
            }
            let manifest: SegmentManifest = serde_json::from_slice(&json)?;
            (
                manifest.segments,
                manifest.next_segment_id,
                manifest.last_wal_seq,
            )
        };

        let mut segments = self.segments.write();
        *segments = segments_info;

        // Zero generations in a loaded manifest need stamping, or the
        // newest-generation-wins dedup ordering is meaningless (Issue #880).
        // Two populations exist:
        // - Legacy all-zero manifests (pre-#879 nothing stamped): list
        //   position is flush order, so stamp 1..N in list order.
        // - MIXED manifests (stamped flush segments + a zero-generation
        //   merged segment from the brief window where `apply_merge` did
        //   not stamp): the zero entry's true age is unrecoverable, so
        //   stamp it BELOW every stamped generation (shift the stamped ones
        //   up). Treating it as oldest is the loss-minimizing direction —
        //   stamping it newest would let a stale copy inside it claim the
        //   `(doc_id, field)` key at the next merge and physically drop the
        //   genuinely newer copy.
        let zero_count = segments.iter().filter(|s| s.generation == 0).count() as u64;
        if zero_count > 0 {
            let all_zero = zero_count == segments.len() as u64;
            if !all_zero {
                for segment in segments.iter_mut() {
                    if segment.generation != 0 {
                        segment.generation += zero_count;
                    }
                }
            }
            let mut next = 0u64;
            for segment in segments.iter_mut() {
                if segment.generation == 0 {
                    next += 1;
                    segment.generation = next;
                }
            }
        }

        // The persisted counter wins over the max-scan, but never goes
        // backwards: a merged-away highest segment must not cause ID reuse.
        let max_id = segments
            .iter()
            .filter_map(|s| s.segment_id.strip_prefix("segment_"))
            .filter_map(|s| s.parse::<u64>().ok())
            .map(|id| id + 1)
            .max()
            .unwrap_or(0);
        *self.next_segment_id.write() = next_id.max(max_id);
        self.last_wal_seq.store(wal_seq, Ordering::Release);

        Ok(())
    }

    /// Persist the manifest atomically: temp file + length-prefixed JSON +
    /// CRC-32 trailer + fsync + rename (#784/#786 parity).
    ///
    /// Takes the segment list as a parameter so mutators can call it while
    /// still holding the `segments` write lock — saving after dropping the
    /// lock let two racing mutators publish manifests out of order.
    fn save_state_locked(&self, segments: &[ManagedSegmentInfo]) -> Result<()> {
        let manifest = SegmentManifest {
            version: MANIFEST_VERSION,
            next_segment_id: *self.next_segment_id.read(),
            last_wal_seq: self.last_wal_seq.load(Ordering::Acquire),
            segments: segments.to_vec(),
        };
        let json = serde_json::to_vec(&manifest)
            .map_err(|e| LaurusError::index(format!("failed to serialize manifest: {e}")))?;

        // `write_bytes` records the payload's CRC-32 and `close` writes it as
        // the file trailer; the reader verifies it via `verify_checksum`.
        let output = self.storage.create_output(MANIFEST_TMP_FILE)?;
        let mut writer = StructWriter::new(output);
        writer.write_bytes(&json)?;
        writer.close()?;

        self.storage.rename_file(MANIFEST_TMP_FILE, MANIFEST_FILE)?;
        // Make the rename durable/visible (directory metadata) before the
        // caller treats the new manifest as published.
        self.storage.sync()?;
        Ok(())
    }

    /// Persist the current manifest state.
    ///
    /// # Errors
    ///
    /// Returns an error if serializing or writing the manifest fails; the
    /// previously published manifest stays intact (atomic rename).
    pub fn save_state(&self) -> Result<()> {
        let segments = self.segments.read();
        self.save_state_locked(&segments)
    }

    /// Record the last applied WAL sequence number; persisted by the next
    /// save (see [`SegmentManifest::last_wal_seq`]).
    pub fn set_last_wal_seq(&self, seq: u64) {
        self.last_wal_seq.store(seq, Ordering::Release);
    }

    /// Last WAL sequence number recorded in the manifest.
    pub fn last_wal_seq(&self) -> u64 {
        self.last_wal_seq.load(Ordering::Acquire)
    }

    /// Best-effort sweep of files that belong to no registered segment.
    ///
    /// Only touches names this manager itself generates
    /// (`segment_NNNNNN.hnsw` plus its `.hnsw.f32` / `.delmap` sidecars and
    /// `.hnsw.tmp` staging files) and the manifest temp file, so foreign
    /// files — e.g. a monolithic `vector_index.hnsw` in the same directory —
    /// are never affected. Runs only after a successful manifest load.
    fn cleanup_orphans(&self) {
        let Ok(files) = self.storage.list_files() else {
            return;
        };
        let segments = self.segments.read();
        for file in files {
            if file == MANIFEST_TMP_FILE {
                let _ = self.storage.delete_file(&file);
                continue;
            }
            // Strip a known suffix; skip files that are not segment-shaped.
            let Some(stem) = file
                .strip_suffix(".hnsw.tmp")
                .or_else(|| file.strip_suffix(".hnsw.f32"))
                .or_else(|| file.strip_suffix(".delmap"))
                .or_else(|| file.strip_suffix(".hnsw"))
            else {
                continue;
            };
            let Some(ordinal) = stem.strip_prefix("segment_") else {
                continue;
            };
            if ordinal.is_empty() || !ordinal.bytes().all(|b| b.is_ascii_digit()) {
                continue;
            }
            if !segments.iter().any(|s| s.segment_id == stem) {
                let _ = self.storage.delete_file(&file);
            }
        }
    }

    /// Add a new segment.
    ///
    /// Fills in bookkeeping metadata the caller did not provide (#879):
    /// a zero `size_bytes` is replaced by the on-storage size of the segment
    /// file plus its sidecars, and a zero `generation` is stamped with the
    /// next generation after the currently registered maximum — both feed
    /// the metadata-driven merge policy.
    ///
    /// # Errors
    ///
    /// Returns an error if persisting the manifest fails.
    pub fn add_segment(&self, mut info: ManagedSegmentInfo) -> Result<()> {
        if info.size_bytes == 0 {
            info.size_bytes = self.measure_segment_size(&info.segment_id);
        }
        let mut segments = self.segments.write();
        if info.generation == 0 {
            info.generation = segments.iter().map(|s| s.generation).max().unwrap_or(0) + 1;
        }
        segments.push(info);
        self.save_state_locked(&segments)
    }

    /// Sum the on-storage sizes of a segment's file and sidecars.
    fn measure_segment_size(&self, segment_id: &str) -> u64 {
        [
            format!("{segment_id}.hnsw"),
            format!("{segment_id}.hnsw.f32"),
            format!("{segment_id}.delmap"),
        ]
        .iter()
        .filter_map(|name| self.storage.file_size(name).ok())
        .sum()
    }

    /// Remove a segment.
    pub fn remove_segment(&self, segment_id: &str) -> Result<()> {
        let mut segments = self.segments.write();
        if let Some(pos) = segments.iter().position(|s| s.segment_id == segment_id) {
            segments.remove(pos);
            self.save_state_locked(&segments)
        } else {
            Ok(())
        }
    }

    /// Delete physical files associated with a segment, including its
    /// rerank sidecar (`.hnsw.f32`) and deletion bitmap (`.delmap`) —
    /// leaving them behind orphans storage after every merge (#879).
    pub fn delete_segment_files(&self, segment_id: &str) -> Result<()> {
        // Best effort deletion - ignore if a file doesn't exist
        let _ = self.storage.delete_file(&format!("{segment_id}.hnsw"));
        let _ = self.storage.delete_file(&format!("{segment_id}.hnsw.f32"));
        let _ = self.storage.delete_file(&format!("{segment_id}.delmap"));
        Ok(())
    }

    /// Flag every registered segment as containing deletions (Issue #880).
    ///
    /// The vector-side [`ManagedSegmentInfo`] carries no per-segment doc-id
    /// range, so a delete that may target a sealed segment cannot be
    /// attributed to a specific one; flagging all of them is conservative
    /// but correct — the flag only feeds the merge policy's prioritization,
    /// while actual filtering always goes through the shared deletion
    /// bitmap. Saves the manifest once (not per segment). No-op when every
    /// segment is already flagged.
    ///
    /// # Errors
    ///
    /// Returns an error if persisting the manifest fails.
    pub fn mark_all_has_deletions(&self) -> Result<()> {
        let mut segments = self.segments.write();
        let mut changed = false;
        for segment in segments.iter_mut() {
            if !segment.has_deletions {
                segment.has_deletions = true;
                changed = true;
            }
        }
        if changed {
            self.save_state_locked(&segments)
        } else {
            Ok(())
        }
    }

    /// Update a segment info.
    pub fn update_segment(&self, info: ManagedSegmentInfo) -> Result<()> {
        let mut segments = self.segments.write();
        if let Some(idx) = segments
            .iter()
            .position(|s| s.segment_id == info.segment_id)
        {
            segments[idx] = info;
        }
        self.save_state_locked(&segments)
    }

    /// Get segment information.
    pub fn get_segment(&self, segment_id: &str) -> Option<ManagedSegmentInfo> {
        let segments = self.segments.read();
        segments
            .iter()
            .find(|s| s.segment_id == segment_id)
            .cloned()
    }

    /// List all segments.
    pub fn list_segments(&self) -> Vec<ManagedSegmentInfo> {
        let segments = self.segments.read();
        segments.clone()
    }

    /// Check if any segments need merging.
    pub fn check_merge(&self, policy: &dyn MergePolicy) -> Option<MergeCandidate> {
        let segments_lock = self.segments.read();

        if let Some(candidate_ids) = policy.candidates(&segments_lock, &self.config) {
            let mut total_vectors = 0;
            let mut total_size = 0;
            let mut candidates = Vec::new();

            for id in &candidate_ids {
                if let Some(segment) = segments_lock.iter().find(|s| s.segment_id == *id) {
                    total_vectors += segment.vector_count;
                    total_size += segment.size_bytes;
                    candidates.push(segment.clone());
                }
            }

            return Some(MergeCandidate {
                segments: candidates,
                total_vectors,
                total_size,
            });
        }
        None
    }

    /// Apply a merge result by replacing source segments with the merged segment.
    pub fn apply_merge(
        &self,
        candidate: MergeCandidate,
        mut merged_segment: ManagedSegmentInfo,
    ) -> Result<()> {
        // Replace the engine's estimated size with the on-storage size of
        // the merged segment file + sidecars (mirrors `add_segment`, #879).
        merged_segment.size_bytes = self.measure_segment_size(&merged_segment.segment_id);
        let mut segments_lock = self.segments.write();

        // 1. Remove source segments
        let ids_to_remove: std::collections::HashSet<_> =
            candidate.segments.iter().map(|s| &s.segment_id).collect();

        segments_lock.retain(|s| !ids_to_remove.contains(&s.segment_id));

        // 2. Add new segment
        segments_lock.push(merged_segment);

        // 3. Save state (lock held — see `save_state_locked`)
        self.save_state_locked(&segments_lock)?;
        drop(segments_lock);

        // 4. Cleanup physical files of source segments
        for segment in candidate.segments {
            self.delete_segment_files(&segment.segment_id)?;
        }

        Ok(())
    }
    pub fn total_vectors(&self) -> u64 {
        self.segments.read().iter().map(|s| s.vector_count).sum()
    }

    /// Generate a new segment ID.
    pub fn generate_segment_id(&self) -> String {
        let mut next_id = self.next_segment_id.write();
        let id = *next_id;
        *next_id += 1;
        format!("segment_{:06}", id)
    }

    /// Check if merging is needed.
    pub fn needs_merge(&self) -> bool {
        let segments = self.segments.read();
        segments.len() as u32 > self.config.max_segments
    }

    /// Create a merge plan.
    pub fn create_merge_plan(&self, strategy: MergeStrategy) -> Option<MergePlan> {
        let segments = self.segments.read();

        if segments.len() <= 1 {
            return None;
        }

        let mut segment_list: Vec<_> = segments.iter().cloned().collect();

        // Sort based on strategy
        match strategy {
            MergeStrategy::Smallest => {
                segment_list.sort_by_key(|s| s.vector_count);
            }
            MergeStrategy::MostDeletions => {
                segment_list.sort_by_key(|s| std::cmp::Reverse(s.has_deletions));
            }
            MergeStrategy::Adjacent => {
                segment_list.sort_by_key(|s| s.vector_offset);
            }
        }

        // Select segments to merge
        let merge_count = self.config.merge_factor.min(segment_list.len() as u32) as usize;
        let to_merge = &segment_list[..merge_count];

        let candidate = MergeCandidate {
            segments: to_merge.to_vec(),
            total_vectors: to_merge.iter().map(|s| s.vector_count).sum(),
            total_size: to_merge.iter().map(|s| s.size_bytes).sum(),
        };

        // Determine urgency
        let urgency = if segments.len() as u32 > self.config.max_segments * 2 {
            MergeUrgency::High
        } else if segments.len() as u32 > self.config.max_segments {
            MergeUrgency::Medium
        } else {
            MergeUrgency::Low
        };

        Some(MergePlan {
            candidates: vec![candidate],
            strategy,
            urgency,
        })
    }

    /// Get statistics.
    pub fn stats(&self) -> SegmentManagerStats {
        let segments = self.segments.read();
        let segment_count = segments.len() as u32;
        let total_vectors: u64 = segments.iter().map(|s| s.vector_count).sum();
        let total_size: u64 = segments.iter().map(|s| s.size_bytes).sum();
        let segments_with_deletions = segments.iter().filter(|s| s.has_deletions).count() as u32;
        let avg_vectors_per_segment = if segment_count > 0 {
            total_vectors as f64 / segment_count as f64
        } else {
            0.0
        };

        SegmentManagerStats {
            segment_count,
            total_vectors,
            total_size,
            segments_with_deletions,
            avg_vectors_per_segment,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};
    use crate::vector::index::hnsw::segment::merge_policy::SimpleMergePolicy;

    fn create_info(id: &str, count: u64) -> ManagedSegmentInfo {
        ManagedSegmentInfo {
            segment_id: id.to_string(),
            vector_count: count,
            vector_offset: 0,
            generation: 1,
            has_deletions: false,
            size_bytes: count * 100,
        }
    }

    #[test]
    fn test_segment_manager_basic() {
        let config = SegmentManagerConfig::default();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let manager = SegmentManager::new(config, storage).unwrap();

        let segment_id = manager.generate_segment_id();
        assert_eq!(segment_id, "segment_000000");

        let info = ManagedSegmentInfo::new(segment_id.clone(), 1000, 0, 0);
        manager.add_segment(info.clone()).unwrap();

        let retrieved = manager.get_segment(&segment_id).unwrap();
        assert_eq!(retrieved.vector_count, 1000);
    }

    // Additional tests for persistence?
    #[test]
    fn test_persistence() {
        let config = SegmentManagerConfig::default();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        {
            let manager = SegmentManager::new(config.clone(), storage.clone()).unwrap();
            let info = ManagedSegmentInfo::new("segment_000000".to_string(), 1000, 0, 0);
            manager.add_segment(info).unwrap();
            // Saves automatically
        }

        // Reload
        {
            let manager = SegmentManager::new(config, storage.clone()).unwrap();
            let segments = manager.list_segments();
            assert_eq!(segments.len(), 1);
            assert_eq!(segments[0].segment_id, "segment_000000");
        }
    }

    /// #879: the versioned manifest round-trips segments, the ID counter,
    /// and the WAL checkpoint through save + reload.
    #[test]
    fn test_versioned_manifest_round_trip() {
        let config = SegmentManagerConfig::default();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        {
            let manager = SegmentManager::new(config.clone(), storage.clone()).unwrap();
            let id0 = manager.generate_segment_id();
            let id1 = manager.generate_segment_id();
            manager
                .add_segment(ManagedSegmentInfo::new(id0, 100, 0, 0))
                .unwrap();
            manager
                .add_segment(ManagedSegmentInfo::new(id1.clone(), 200, 100, 0))
                .unwrap();
            manager.set_last_wal_seq(42);
            // Remove the highest-numbered segment: the persisted counter must
            // still prevent its ID from being reused after reload.
            manager.remove_segment(&id1).unwrap();
            manager.save_state().unwrap();
        }

        {
            let manager = SegmentManager::new(config, storage).unwrap();
            let segments = manager.list_segments();
            assert_eq!(segments.len(), 1);
            assert_eq!(segments[0].segment_id, "segment_000000");
            assert_eq!(manager.last_wal_seq(), 42, "WAL checkpoint must persist");
            assert_eq!(
                manager.generate_segment_id(),
                "segment_000002",
                "the persisted counter must win over the max-scan (no ID reuse)"
            );
        }
    }

    /// #879: the legacy bare-JSON-array manifest still loads and is upgraded
    /// to the versioned format by the next save.
    #[test]
    fn test_legacy_manifest_loads_and_upgrades() {
        let config = SegmentManagerConfig::default();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        // Hand-write the pre-#879 format: a bare pretty JSON array.
        let legacy = serde_json::to_vec_pretty(&vec![create_info("segment_000007", 100)]).unwrap();
        {
            let mut out = storage.create_output("segments.json").unwrap();
            std::io::Write::write_all(&mut out, &legacy).unwrap();
            out.close().unwrap();
        }

        let manager = SegmentManager::new(config.clone(), storage.clone()).unwrap();
        assert_eq!(manager.list_segments().len(), 1);
        assert_eq!(
            manager.generate_segment_id(),
            "segment_000008",
            "legacy load must derive the counter from the max scan"
        );

        // A save upgrades the file; a reload parses the versioned format.
        manager.save_state().unwrap();
        let manager = SegmentManager::new(config, storage).unwrap();
        assert_eq!(manager.list_segments().len(), 1);
        assert_eq!(manager.list_segments()[0].segment_id, "segment_000007");
    }

    /// #879: a corrupted manifest fails the open loudly — silently starting
    /// empty would orphan (and then sweep) every live segment.
    #[test]
    fn test_corrupted_manifest_fails_open() {
        let config = SegmentManagerConfig::default();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        {
            let manager = SegmentManager::new(config.clone(), storage.clone()).unwrap();
            manager
                .add_segment(create_info("segment_000000", 100))
                .unwrap();
        }

        // Flip one byte inside the JSON payload (skip the length prefix).
        let mut content = Vec::new();
        {
            let mut input = storage.open_input("segments.json").unwrap();
            input.read_to_end(&mut content).unwrap();
        }
        let mid = content.len() / 2;
        content[mid] ^= 0xFF;
        {
            let mut out = storage.create_output("segments.json").unwrap();
            std::io::Write::write_all(&mut out, &content).unwrap();
            out.close().unwrap();
        }

        assert!(
            SegmentManager::new(config, storage).is_err(),
            "a corrupted manifest must fail the open, not silently start empty"
        );
    }

    /// #880: legacy manifests carry all-zero generations; the load stamps
    /// them in list order (flush order = age) so newest-generation-wins
    /// dedup has a meaningful ordering.
    #[test]
    fn test_legacy_zero_generations_are_stamped_in_list_order() {
        let config = SegmentManagerConfig::default();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        // Hand-write the pre-#879 format with generation 0 everywhere.
        let mut a = create_info("segment_000000", 100);
        a.generation = 0;
        let mut b = create_info("segment_000001", 100);
        b.generation = 0;
        let legacy = serde_json::to_vec_pretty(&vec![a, b]).unwrap();
        {
            let mut out = storage.create_output("segments.json").unwrap();
            std::io::Write::write_all(&mut out, &legacy).unwrap();
            out.close().unwrap();
        }

        let manager = SegmentManager::new(config, storage).unwrap();
        let segments = manager.list_segments();
        assert_eq!(segments[0].generation, 1, "stamped in list order");
        assert_eq!(segments[1].generation, 2, "monotone with position");
    }

    /// #880: a MIXED manifest (stamped flush segments + a zero-generation
    /// merged segment from the pre-#880 `apply_merge`) stamps the zero
    /// entry as the OLDEST — its true age is unrecoverable, and treating it
    /// as newest would let a stale copy inside it physically drop a newer
    /// copy at the next merge.
    #[test]
    fn test_mixed_manifest_zero_generation_stamped_oldest() {
        let config = SegmentManagerConfig::default();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        let mut flushed = create_info("segment_000002", 100);
        flushed.generation = 5;
        let mut merged = create_info("segment_000003", 100);
        merged.generation = 0;
        let legacy = serde_json::to_vec_pretty(&vec![flushed, merged]).unwrap();
        {
            let mut out = storage.create_output("segments.json").unwrap();
            std::io::Write::write_all(&mut out, &legacy).unwrap();
            out.close().unwrap();
        }

        let manager = SegmentManager::new(config, storage).unwrap();
        let segments = manager.list_segments();
        let gen_of = |id: &str| {
            segments
                .iter()
                .find(|s| s.segment_id == id)
                .unwrap()
                .generation
        };
        assert!(
            gen_of("segment_000003") < gen_of("segment_000002"),
            "the zero-generation merged segment must be stamped OLDEST, got {:?}",
            segments
                .iter()
                .map(|s| (s.segment_id.clone(), s.generation))
                .collect::<Vec<_>>()
        );
    }

    /// #879: the CRC verify itself is load-bearing — corrupting only the
    /// trailer (payload still valid JSON) must fail the open.
    #[test]
    fn test_crc_trailer_mismatch_fails_open() {
        let config = SegmentManagerConfig::default();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        {
            let manager = SegmentManager::new(config.clone(), storage.clone()).unwrap();
            manager
                .add_segment(create_info("segment_000000", 100))
                .unwrap();
        }

        let mut content = Vec::new();
        {
            let mut input = storage.open_input("segments.json").unwrap();
            input.read_to_end(&mut content).unwrap();
        }
        // Flip a bit in the 4-byte CRC trailer only — the JSON payload stays
        // intact, so only the checksum comparison can catch this.
        let last = content.len() - 1;
        content[last] ^= 0x01;
        {
            let mut out = storage.create_output("segments.json").unwrap();
            std::io::Write::write_all(&mut out, &content).unwrap();
            out.close().unwrap();
        }

        let err = SegmentManager::new(config, storage).unwrap_err();
        assert!(
            err.to_string().contains("checksum mismatch"),
            "the CRC verify must reject a trailer-only corruption, got: {err}"
        );
    }

    /// #879: a torn manifest write (garbage left in the staging file) must
    /// not affect the published manifest; the temp file is swept on open.
    #[test]
    fn test_torn_manifest_write_survives() {
        let config = SegmentManagerConfig::default();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        {
            let manager = SegmentManager::new(config.clone(), storage.clone()).unwrap();
            manager
                .add_segment(create_info("segment_000000", 100))
                .unwrap();
        }
        // Simulate a crash mid-save: garbage staged, rename never happened.
        {
            let mut out = storage.create_output("segments.json.tmp").unwrap();
            std::io::Write::write_all(&mut out, b"\x00garbage").unwrap();
            out.close().unwrap();
        }

        let manager = SegmentManager::new(config, storage.clone()).unwrap();
        assert_eq!(
            manager.list_segments().len(),
            1,
            "published manifest intact"
        );
        assert!(
            !storage.file_exists("segments.json.tmp"),
            "the torn staging file must be swept on open"
        );
    }

    /// #879: files belonging to no registered segment are swept on open;
    /// foreign files (e.g. a monolithic index in the same directory) are
    /// never touched.
    #[test]
    fn test_orphan_sweep_spares_foreign_files() {
        let config = SegmentManagerConfig::default();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        let write = |name: &str| {
            let mut out = storage.create_output(name).unwrap();
            std::io::Write::write_all(&mut out, b"x").unwrap();
            out.close().unwrap();
        };

        {
            let manager = SegmentManager::new(config.clone(), storage.clone()).unwrap();
            manager
                .add_segment(create_info("segment_000000", 100))
                .unwrap();
        }
        // Registered segment's files: must survive.
        write("segment_000000.hnsw");
        write("segment_000000.hnsw.f32");
        // Orphans from a torn flush: must be swept.
        write("segment_000042.hnsw");
        write("segment_000042.hnsw.f32");
        write("segment_000042.delmap");
        write("segment_000042.hnsw.tmp");
        // Foreign files: must never be touched.
        write("vector_index.hnsw");
        write("not_a_segment.hnsw");

        let _manager = SegmentManager::new(config, storage.clone()).unwrap();
        assert!(storage.file_exists("segment_000000.hnsw"));
        assert!(storage.file_exists("segment_000000.hnsw.f32"));
        assert!(!storage.file_exists("segment_000042.hnsw"), "orphan swept");
        assert!(!storage.file_exists("segment_000042.hnsw.f32"));
        assert!(!storage.file_exists("segment_000042.delmap"));
        assert!(!storage.file_exists("segment_000042.hnsw.tmp"));
        assert!(storage.file_exists("vector_index.hnsw"), "foreign spared");
        assert!(storage.file_exists("not_a_segment.hnsw"), "foreign spared");
    }

    /// #879: deleting a segment's files removes the rerank sidecar and the
    /// deletion bitmap along with the index file.
    #[test]
    fn test_delete_segment_files_includes_sidecars() {
        let config = SegmentManagerConfig::default();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let manager = SegmentManager::new(config, storage.clone()).unwrap();

        for name in [
            "segment_000001.hnsw",
            "segment_000001.hnsw.f32",
            "segment_000001.delmap",
        ] {
            let mut out = storage.create_output(name).unwrap();
            std::io::Write::write_all(&mut out, b"x").unwrap();
            out.close().unwrap();
        }

        manager.delete_segment_files("segment_000001").unwrap();
        assert!(!storage.file_exists("segment_000001.hnsw"));
        assert!(
            !storage.file_exists("segment_000001.hnsw.f32"),
            "sidecar GC"
        );
        assert!(!storage.file_exists("segment_000001.delmap"), "delmap GC");
    }

    /// #879: `add_segment` stamps a zero generation with max+1 and measures a
    /// zero `size_bytes` from storage — the merge policy's inputs are real.
    #[test]
    fn test_add_segment_fills_metadata() {
        let config = SegmentManagerConfig::default();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let manager = SegmentManager::new(config, storage.clone()).unwrap();

        {
            let mut out = storage.create_output("segment_000000.hnsw").unwrap();
            std::io::Write::write_all(&mut out, &[0u8; 128]).unwrap();
            out.close().unwrap();
        }

        manager
            .add_segment(ManagedSegmentInfo::new(
                "segment_000000".to_string(),
                10,
                0,
                0,
            ))
            .unwrap();
        manager
            .add_segment(ManagedSegmentInfo::new(
                "segment_000001".to_string(),
                10,
                10,
                0,
            ))
            .unwrap();

        let segments = manager.list_segments();
        assert_eq!(segments[0].generation, 1, "zero generation stamped");
        assert_eq!(segments[1].generation, 2, "monotonically increasing");
        assert_eq!(
            segments[0].size_bytes, 128,
            "zero size_bytes measured from storage"
        );
    }

    #[test]
    fn test_check_merge() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let config = SegmentManagerConfig {
            max_segments: 5,
            merge_factor: 3,
            ..Default::default()
        };

        // We use a temporary config for the manager
        let manager = SegmentManager::new(config, storage).unwrap();

        // 1. Add segments (not enough for merge)
        manager.add_segment(create_info("1", 100)).unwrap();
        manager.add_segment(create_info("2", 100)).unwrap();

        assert!(manager.check_merge(&SimpleMergePolicy::new()).is_none());

        // 2. Add more segments to trigger merge
        manager.add_segment(create_info("3", 100)).unwrap();
        manager.add_segment(create_info("4", 100)).unwrap();
        manager.add_segment(create_info("5", 100)).unwrap();
        manager.add_segment(create_info("6", 100)).unwrap(); // Total 6 > 5

        let candidate = manager.check_merge(&SimpleMergePolicy::new());
        assert!(candidate.is_some());

        let candidate = candidate.unwrap();
        assert_eq!(candidate.segments.len(), 3);
        // Expect smallest: 1, 2, 3, 4, 5, 6 are all 100?
        // Wait, simple policy sort by vector_count.
        // If all equal, it picks stable sort order? Or arbitrary.
        // SimpleMergePolicy uses `segments.iter().enumerate()` then sort_by_key.
        // `sort_by_key` is stable. So it picks first 3: 1, 2, 3.

        let ids: Vec<String> = candidate
            .segments
            .iter()
            .map(|s| s.segment_id.clone())
            .collect();
        assert!(ids.contains(&"1".to_string()));
        assert!(ids.contains(&"2".to_string()));
        assert!(ids.contains(&"3".to_string()));
    }
}
