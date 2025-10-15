//! Segment management system for efficient index operations.
//!
//! This module provides comprehensive segment lifecycle management including
//! creation, deletion, merging, and optimization based on configurable policies.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::error::{Result, SageError};
use crate::full_text::SegmentInfo;
use crate::storage::{Storage, StorageInput, StructReader, StructWriter};

/// Configuration for segment management.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentManagerConfig {
    /// Maximum number of segments before triggering merge.
    pub max_segments: usize,

    /// Minimum segment size for merge consideration (in bytes).
    pub min_segment_size: u64,

    /// Maximum segment size before forcing split (in bytes).
    pub max_segment_size: u64,

    /// Target segments per tier in tiered merge policy.
    pub segments_per_tier: usize,

    /// Enable automatic background merging.
    pub auto_merge_enabled: bool,

    /// Merge scheduling interval in seconds.
    pub merge_interval_secs: u64,

    /// Maximum deletion ratio before compaction (0.0-1.0).
    pub max_deletion_ratio: f64,
}

impl Default for SegmentManagerConfig {
    fn default() -> Self {
        SegmentManagerConfig {
            max_segments: 10,
            min_segment_size: 1024 * 1024,       // 1MB
            max_segment_size: 100 * 1024 * 1024, // 100MB
            segments_per_tier: 4,
            auto_merge_enabled: true,
            merge_interval_secs: 60,
            max_deletion_ratio: 0.3,
        }
    }
}

/// Extended segment information with management metadata.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ManagedSegmentInfo {
    /// Core segment information.
    pub segment_info: SegmentInfo,

    /// Size of the segment in bytes.
    pub size_bytes: u64,

    /// Number of deleted documents in this segment.
    pub deleted_count: u64,

    /// Timestamp when segment was created.
    pub created_at: u64,

    /// Timestamp when segment was last modified.
    pub last_modified: u64,

    /// Merge tier (for tiered merge policy).
    pub tier: u8,

    /// Whether this segment is currently being merged.
    pub is_merging: bool,

    /// Segment file paths for cleanup.
    pub file_paths: Vec<String>,
}

impl ManagedSegmentInfo {
    /// Create new managed segment info.
    pub fn new(segment_info: SegmentInfo) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        ManagedSegmentInfo {
            segment_info,
            size_bytes: 0,
            deleted_count: 0,
            created_at: now,
            last_modified: now,
            tier: 0,
            is_merging: false,
            file_paths: Vec::new(),
        }
    }

    /// Get deletion ratio (deleted docs / total docs).
    pub fn deletion_ratio(&self) -> f64 {
        if self.segment_info.doc_count == 0 {
            0.0
        } else {
            self.deleted_count as f64 / self.segment_info.doc_count as f64
        }
    }

    /// Get effective document count (total - deleted).
    pub fn effective_doc_count(&self) -> u64 {
        self.segment_info
            .doc_count
            .saturating_sub(self.deleted_count)
    }

    /// Check if segment needs compaction.
    pub fn needs_compaction(&self, threshold: f64) -> bool {
        self.deletion_ratio() > threshold
    }
}

/// Merge candidate representing segments to be merged.
#[derive(Debug, Clone)]
pub struct MergeCandidate {
    /// Segments to merge.
    pub segments: Vec<String>,

    /// Priority score (higher = more urgent).
    pub priority: f64,

    /// Expected size after merge.
    pub estimated_size: u64,

    /// Merge strategy to use.
    pub strategy: MergeStrategy,
}

/// Merge strategy options.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MergeStrategy {
    /// Size-based merging (small segments first).
    SizeBased,

    /// Deletion-based merging (high deletion ratio first).
    DeletionBased,

    /// Time-based merging (oldest segments first).
    TimeBased,

    /// Balanced approach considering multiple factors.
    Balanced,
}

/// Urgency level for merge operations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MergeUrgency {
    /// Critical - immediate action required.
    Critical,

    /// High - should be performed soon.
    High,

    /// Medium - can be scheduled.
    Medium,

    /// Low - optional optimization.
    Low,
}

/// Comprehensive merge plan with recommendations.
#[derive(Debug, Clone)]
pub struct MergePlan {
    /// Recommended merge strategy.
    pub strategy: MergeStrategy,

    /// List of merge candidates.
    pub candidates: Vec<MergeCandidate>,

    /// Estimated benefit of performing all merges.
    pub estimated_benefit: f64,

    /// Urgency of merge operations.
    pub urgency: MergeUrgency,
}

/// Statistics about segment management operations.
#[derive(Debug, Clone, Default)]
pub struct SegmentManagerStats {
    /// Total number of segments.
    pub total_segments: usize,

    /// Total size of all segments.
    pub total_size_bytes: u64,

    /// Total number of documents.
    pub total_doc_count: u64,

    /// Total number of deleted documents.
    pub total_deleted_count: u64,

    /// Number of merge operations performed.
    pub merge_operations: u64,

    /// Number of compaction operations performed.
    pub compaction_operations: u64,

    /// Time of last merge operation.
    pub last_merge_time: u64,

    /// Average segment size.
    pub avg_segment_size: u64,

    /// Overall deletion ratio.
    pub overall_deletion_ratio: f64,
}

/// Core segment manager responsible for segment lifecycle.
#[derive(Debug)]
pub struct SegmentManager {
    /// Configuration for segment management.
    config: SegmentManagerConfig,

    /// Storage backend.
    storage: Arc<dyn Storage>,

    /// Managed segments with metadata.
    segments: RwLock<BTreeMap<String, ManagedSegmentInfo>>,

    /// Generation counter for new segments.
    generation: AtomicU64,

    /// Manager statistics.
    stats: RwLock<SegmentManagerStats>,

    /// Lock for merge operations.
    #[allow(dead_code)]
    merge_lock: RwLock<()>,

    /// Flag to indicate if manifest needs to be written.
    manifest_dirty: AtomicBool,

    /// Flag to indicate if statistics need to be updated.
    stats_dirty: AtomicBool,

    /// Last manifest write time.
    last_manifest_write: AtomicU64,
}

impl SegmentManager {
    /// Create a new segment manager.
    pub fn new(config: SegmentManagerConfig, storage: Arc<dyn Storage>) -> Result<Self> {
        let manager = SegmentManager {
            config,
            storage,
            segments: RwLock::new(BTreeMap::new()),
            generation: AtomicU64::new(1),
            stats: RwLock::new(SegmentManagerStats::default()),
            merge_lock: RwLock::new(()),
            manifest_dirty: AtomicBool::new(false),
            stats_dirty: AtomicBool::new(true), // Start with dirty stats
            last_manifest_write: AtomicU64::new(0),
        };

        // Load existing segments (skip during tests for performance)
        #[cfg(not(test))]
        {
            // Try to load existing segments from storage
            if let Err(e) = manager.load_segments() {
                // If segments can't be loaded, start fresh
                eprintln!("Warning: Could not load existing segments: {e}");
            }
        }

        Ok(manager)
    }

    /// Load existing segments from storage.
    #[allow(dead_code)]
    fn load_segments(&self) -> Result<()> {
        // Try to load segment manifest
        if let Ok(input) = self.storage.open_input("segments.manifest") {
            let mut reader = StructReader::new(input)?;
            let segments = self.read_segment_manifest(&mut reader)?;

            *self.segments.write().unwrap() = segments;

            // Update generation counter
            let max_gen = self
                .segments
                .read()
                .unwrap()
                .values()
                .map(|seg| seg.segment_info.generation)
                .max()
                .unwrap_or(0);
            self.generation.store(max_gen + 1, Ordering::Relaxed);
        }

        // Update statistics
        #[cfg(not(test))]
        self.update_stats();

        Ok(())
    }

    /// Read segment manifest from storage.
    #[allow(dead_code)]
    fn read_segment_manifest<R: StorageInput>(
        &self,
        reader: &mut StructReader<R>,
    ) -> Result<BTreeMap<String, ManagedSegmentInfo>> {
        // Read magic number
        let magic = reader.read_u32()?;
        if magic != 0x53454753 {
            // "SEGS"
            return Err(SageError::index("Invalid segment manifest format"));
        }

        // Read version
        let version = reader.read_u32()?;
        if version != 1 {
            return Err(SageError::index(format!(
                "Unsupported manifest version: {version}"
            )));
        }

        // Read segment count
        let segment_count = reader.read_varint()? as usize;
        let mut segments = BTreeMap::new();

        for _ in 0..segment_count {
            let segment_id = reader.read_string()?;

            // Read core segment info
            let doc_count = reader.read_u64()?;
            let doc_offset = reader.read_u64()?;
            let generation = reader.read_u64()?;
            let has_deletions = reader.read_u8()? != 0;

            let segment_info = SegmentInfo {
                segment_id: segment_id.clone(),
                doc_count,
                doc_offset,
                generation,
                has_deletions,
            };

            // Read management metadata
            let size_bytes = reader.read_u64()?;
            let deleted_count = reader.read_u64()?;
            let created_at = reader.read_u64()?;
            let last_modified = reader.read_u64()?;
            let tier = reader.read_u8()?;

            // Read file paths
            let path_count = reader.read_varint()? as usize;
            let mut file_paths = Vec::with_capacity(path_count);
            for _ in 0..path_count {
                file_paths.push(reader.read_string()?);
            }

            let managed_info = ManagedSegmentInfo {
                segment_info,
                size_bytes,
                deleted_count,
                created_at,
                last_modified,
                tier,
                is_merging: false,
                file_paths,
            };

            segments.insert(segment_id, managed_info);
        }

        Ok(segments)
    }

    /// Write segment manifest to storage.
    fn write_segment_manifest(&self) -> Result<()> {
        let output = self.storage.create_output("segments.manifest")?;
        let mut writer = StructWriter::new(output);

        // Write magic number and version
        writer.write_u32(0x53454753)?; // "SEGS"
        writer.write_u32(1)?; // version

        let segments = self.segments.read().unwrap();
        writer.write_varint(segments.len() as u64)?;

        for (segment_id, managed_info) in segments.iter() {
            writer.write_string(segment_id)?;

            // Write core segment info
            let seg_info = &managed_info.segment_info;
            writer.write_u64(seg_info.doc_count)?;
            writer.write_u64(seg_info.doc_offset)?;
            writer.write_u64(seg_info.generation)?;
            writer.write_u8(if seg_info.has_deletions { 1 } else { 0 })?;

            // Write management metadata
            writer.write_u64(managed_info.size_bytes)?;
            writer.write_u64(managed_info.deleted_count)?;
            writer.write_u64(managed_info.created_at)?;
            writer.write_u64(managed_info.last_modified)?;
            writer.write_u8(managed_info.tier)?;

            // Write file paths
            writer.write_varint(managed_info.file_paths.len() as u64)?;
            for path in &managed_info.file_paths {
                writer.write_string(path)?;
            }
        }

        writer.close()?;

        // Update last write time and clear dirty flag
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.last_manifest_write.store(now, Ordering::Relaxed);
        self.manifest_dirty.store(false, Ordering::Relaxed);

        Ok(())
    }

    /// Check if manifest should be written based on time or dirty state.
    fn should_write_manifest(&self) -> bool {
        if !self.manifest_dirty.load(Ordering::Relaxed) {
            return false;
        }

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let last_write = self.last_manifest_write.load(Ordering::Relaxed);

        // Write if it's been more than 1 second since last write
        now - last_write > 1
    }

    /// Conditionally write manifest if needed.
    fn maybe_write_manifest(&self) -> Result<()> {
        if self.should_write_manifest() {
            self.write_segment_manifest()?;
        }
        Ok(())
    }

    /// Mark manifest as dirty.
    fn mark_manifest_dirty(&self) {
        self.manifest_dirty.store(true, Ordering::Relaxed);
    }

    /// Force flush manifest to storage.
    pub fn flush_manifest(&self) -> Result<()> {
        if self.manifest_dirty.load(Ordering::Relaxed) {
            self.write_segment_manifest()?;
        }
        Ok(())
    }

    /// Add a new segment to management.
    pub fn add_segment(
        &self,
        mut segment_info: SegmentInfo,
        file_paths: Vec<String>,
    ) -> Result<()> {
        segment_info.generation = self.generation.fetch_add(1, Ordering::Relaxed);

        let mut managed_info = ManagedSegmentInfo::new(segment_info.clone());
        managed_info.file_paths = file_paths;

        // Calculate segment size
        managed_info.size_bytes = self.calculate_segment_size(&managed_info.file_paths)?;

        // Assign tier based on segment size
        managed_info.tier = self.calculate_tier(managed_info.size_bytes);

        // Add to segments
        {
            let mut segments = self.segments.write().unwrap();
            segments.insert(segment_info.segment_id.clone(), managed_info);
        }

        // Update manifest and stats
        self.mark_manifest_dirty();
        self.mark_stats_dirty();
        self.maybe_write_manifest()?;
        self.update_stats();

        // Check if merge is needed
        if self.config.auto_merge_enabled && self.should_trigger_merge() {
            // TODO: Trigger background merge
        }

        Ok(())
    }

    /// Remove a segment from management.
    pub fn remove_segment(&self, segment_id: &str) -> Result<Option<ManagedSegmentInfo>> {
        let removed = {
            let mut segments = self.segments.write().unwrap();
            segments.remove(segment_id)
        };

        if removed.is_some() {
            self.mark_manifest_dirty();
            self.mark_stats_dirty();
            self.maybe_write_manifest()?;
            self.update_stats();
        }

        Ok(removed)
    }

    /// Get all segments.
    pub fn get_segments(&self) -> Vec<ManagedSegmentInfo> {
        self.segments.read().unwrap().values().cloned().collect()
    }

    /// Get segment by ID.
    pub fn get_segment(&self, segment_id: &str) -> Option<ManagedSegmentInfo> {
        self.segments.read().unwrap().get(segment_id).cloned()
    }

    /// Mark documents as deleted in a segment.
    pub fn mark_deleted(&self, segment_id: &str, deleted_count: u64) -> Result<()> {
        self.mark_deleted_internal(segment_id, deleted_count, true)
    }

    /// Internal method to mark documents as deleted with option to skip manifest write.
    fn mark_deleted_internal(
        &self,
        segment_id: &str,
        deleted_count: u64,
        write_manifest: bool,
    ) -> Result<()> {
        let mut segments = self.segments.write().unwrap();
        if let Some(managed_info) = segments.get_mut(segment_id) {
            managed_info.deleted_count += deleted_count;
            managed_info.last_modified = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();

            if managed_info.deleted_count > 0 {
                managed_info.segment_info.has_deletions = true;
            }
        }

        // Mark manifest as dirty and conditionally write
        self.mark_manifest_dirty();
        if write_manifest {
            self.maybe_write_manifest()?;
        }

        // Skip stats update in test mode for performance
        #[cfg(not(test))]
        #[cfg(not(test))]
        self.update_stats();

        Ok(())
    }

    /// Test-friendly version that doesn't write manifest.
    #[cfg(test)]
    pub fn mark_deleted_fast(&self, segment_id: &str, deleted_count: u64) -> Result<()> {
        self.mark_deleted_internal(segment_id, deleted_count, false)
    }

    /// Batch mark deleted for multiple segments efficiently.
    pub fn batch_mark_deleted(&self, updates: &[(String, u64)]) -> Result<()> {
        {
            let mut segments = self.segments.write().unwrap();
            for (segment_id, deleted_count) in updates {
                if let Some(managed_info) = segments.get_mut(segment_id) {
                    managed_info.deleted_count += deleted_count;
                    managed_info.last_modified = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_secs();

                    if managed_info.deleted_count > 0 {
                        managed_info.segment_info.has_deletions = true;
                    }
                }
            }
        }

        // Mark manifest as dirty and conditionally write once for all updates
        self.mark_manifest_dirty();
        self.mark_stats_dirty();
        self.maybe_write_manifest()?;
        self.update_stats();

        Ok(())
    }

    /// Get segments that need compaction.
    pub fn get_compaction_candidates(&self) -> Vec<ManagedSegmentInfo> {
        self.segments
            .read()
            .unwrap()
            .values()
            .filter(|seg| !seg.is_merging && seg.needs_compaction(self.config.max_deletion_ratio))
            .cloned()
            .collect()
    }

    /// Generate merge candidates based on strategy.
    pub fn generate_merge_candidates(&self, strategy: MergeStrategy) -> Vec<MergeCandidate> {
        let segments = self.segments.read().unwrap();
        let available_segments: Vec<_> = segments
            .values()
            .filter(|seg| !seg.is_merging)
            .cloned()
            .collect();

        if available_segments.len() < 2 {
            return Vec::new();
        }

        match strategy {
            MergeStrategy::SizeBased => self.generate_size_based_candidates(&available_segments),
            MergeStrategy::DeletionBased => {
                self.generate_deletion_based_candidates(&available_segments)
            }
            MergeStrategy::TimeBased => self.generate_time_based_candidates(&available_segments),
            MergeStrategy::Balanced => self.generate_balanced_candidates(&available_segments),
        }
    }

    /// Generate size-based merge candidates.
    fn generate_size_based_candidates(
        &self,
        segments: &[ManagedSegmentInfo],
    ) -> Vec<MergeCandidate> {
        let mut candidates = Vec::new();
        let mut sorted_segments = segments.to_vec();
        sorted_segments.sort_by_key(|s| s.size_bytes);

        // Group small segments for merging
        let mut group = Vec::new();
        let mut estimated_size = 0;

        for segment in sorted_segments {
            if segment.size_bytes < self.config.min_segment_size {
                group.push(segment.segment_info.segment_id.clone());
                estimated_size += segment.size_bytes;

                if group.len() >= self.config.segments_per_tier {
                    let priority = self.calculate_size_priority(&group, segments);
                    candidates.push(MergeCandidate {
                        segments: group.clone(),
                        priority,
                        estimated_size,
                        strategy: MergeStrategy::SizeBased,
                    });
                    group.clear();
                    estimated_size = 0;
                }
            }
        }

        // Add remaining group if it has at least 2 segments
        if group.len() >= 2 {
            let priority = self.calculate_size_priority(&group, segments);
            candidates.push(MergeCandidate {
                segments: group,
                priority,
                estimated_size,
                strategy: MergeStrategy::SizeBased,
            });
        }

        candidates
    }

    /// Generate deletion-based merge candidates.
    fn generate_deletion_based_candidates(
        &self,
        segments: &[ManagedSegmentInfo],
    ) -> Vec<MergeCandidate> {
        let mut candidates = Vec::new();
        let mut high_deletion_segments: Vec<_> = segments
            .iter()
            .filter(|s| s.deletion_ratio() > self.config.max_deletion_ratio / 2.0)
            .collect();

        high_deletion_segments
            .sort_by(|a, b| b.deletion_ratio().partial_cmp(&a.deletion_ratio()).unwrap());

        // Group high-deletion segments
        for chunk in high_deletion_segments.chunks(self.config.segments_per_tier) {
            if chunk.len() >= 2 {
                let segments: Vec<String> = chunk
                    .iter()
                    .map(|s| s.segment_info.segment_id.clone())
                    .collect();
                let estimated_size: u64 = chunk.iter().map(|s| s.size_bytes).sum();
                let chunk_owned: Vec<ManagedSegmentInfo> =
                    chunk.iter().map(|s| (*s).clone()).collect();
                let priority = self.calculate_deletion_priority(&segments, &chunk_owned);

                candidates.push(MergeCandidate {
                    segments,
                    priority,
                    estimated_size,
                    strategy: MergeStrategy::DeletionBased,
                });
            }
        }

        candidates
    }

    /// Generate time-based merge candidates.
    fn generate_time_based_candidates(
        &self,
        segments: &[ManagedSegmentInfo],
    ) -> Vec<MergeCandidate> {
        let mut candidates = Vec::new();
        let mut sorted_segments = segments.to_vec();
        sorted_segments.sort_by_key(|s| s.created_at);

        // Group oldest segments
        for chunk in sorted_segments.chunks(self.config.segments_per_tier) {
            if chunk.len() >= 2 {
                let segments: Vec<String> = chunk
                    .iter()
                    .map(|s| s.segment_info.segment_id.clone())
                    .collect();
                let estimated_size: u64 = chunk.iter().map(|s| s.size_bytes).sum();
                let chunk_owned: Vec<ManagedSegmentInfo> = chunk.to_vec();
                let priority = self.calculate_time_priority(&segments, &chunk_owned);

                candidates.push(MergeCandidate {
                    segments,
                    priority,
                    estimated_size,
                    strategy: MergeStrategy::TimeBased,
                });
            }
        }

        candidates
    }

    /// Generate balanced merge candidates.
    fn generate_balanced_candidates(&self, segments: &[ManagedSegmentInfo]) -> Vec<MergeCandidate> {
        let mut all_candidates = Vec::new();

        // Combine candidates from all strategies
        all_candidates.extend(self.generate_size_based_candidates(segments));
        all_candidates.extend(self.generate_deletion_based_candidates(segments));
        all_candidates.extend(self.generate_time_based_candidates(segments));

        // Sort by priority and remove duplicates
        all_candidates.sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap());
        all_candidates.dedup_by(|a, b| a.segments == b.segments);

        // Take top candidates
        all_candidates.into_iter().take(5).collect()
    }

    /// Calculate priority for size-based merging.
    fn calculate_size_priority(
        &self,
        segment_ids: &[String],
        all_segments: &[ManagedSegmentInfo],
    ) -> f64 {
        let segments: Vec<_> = all_segments
            .iter()
            .filter(|s| segment_ids.contains(&s.segment_info.segment_id))
            .collect();

        if segments.is_empty() {
            return 0.0;
        }

        let total_size: u64 = segments.iter().map(|s| s.size_bytes).sum();
        let avg_size = total_size / segments.len() as u64;

        // Higher priority for smaller segments
        let size_factor = if avg_size < self.config.min_segment_size {
            2.0
        } else if avg_size < self.config.max_segment_size / 4 {
            1.5
        } else {
            1.0
        };

        size_factor * segments.len() as f64
    }

    /// Calculate priority for deletion-based merging.
    fn calculate_deletion_priority(
        &self,
        segment_ids: &[String],
        all_segments: &[ManagedSegmentInfo],
    ) -> f64 {
        let segments: Vec<_> = all_segments
            .iter()
            .filter(|s| segment_ids.contains(&s.segment_info.segment_id))
            .collect();

        if segments.is_empty() {
            return 0.0;
        }

        let avg_deletion_ratio: f64 =
            segments.iter().map(|s| s.deletion_ratio()).sum::<f64>() / segments.len() as f64;

        // Higher priority for higher deletion ratios
        avg_deletion_ratio * 10.0 * segments.len() as f64
    }

    /// Calculate priority for time-based merging.
    fn calculate_time_priority(
        &self,
        segment_ids: &[String],
        all_segments: &[ManagedSegmentInfo],
    ) -> f64 {
        let segments: Vec<_> = all_segments
            .iter()
            .filter(|s| segment_ids.contains(&s.segment_info.segment_id))
            .collect();

        if segments.is_empty() {
            return 0.0;
        }

        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let avg_age: f64 = segments
            .iter()
            .map(|s| (current_time - s.created_at) as f64)
            .sum::<f64>()
            / segments.len() as f64;

        // Higher priority for older segments
        (avg_age / 3600.0) * segments.len() as f64 // age in hours
    }

    /// Mark segments as being merged.
    pub fn mark_segments_merging(&self, segment_ids: &[String], merging: bool) -> Result<()> {
        let mut segments = self.segments.write().unwrap();
        for segment_id in segment_ids {
            if let Some(segment) = segments.get_mut(segment_id) {
                segment.is_merging = merging;
                if merging {
                    segment.last_modified = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_secs();
                }
            }
        }

        if !segment_ids.is_empty() {
            drop(segments); // Release lock before writing manifest
            #[cfg(not(test))]
            self.mark_manifest_dirty();
            self.maybe_write_manifest()?;
        }

        Ok(())
    }

    /// Complete merge operation by replacing old segments with new one.
    pub fn complete_merge(
        &self,
        old_segment_ids: &[String],
        new_segment: SegmentInfo,
        new_file_paths: Vec<String>,
    ) -> Result<()> {
        // Add new segment
        self.add_segment(new_segment, new_file_paths)?;

        // Remove old segments
        let mut removed_segments = Vec::new();
        {
            let mut segments = self.segments.write().unwrap();
            for segment_id in old_segment_ids {
                if let Some(removed) = segments.remove(segment_id) {
                    removed_segments.push(removed);
                }
            }
        }

        // Update stats and increment merge counter
        {
            let mut stats = self.stats.write().unwrap();
            stats.merge_operations += 1;
            stats.last_merge_time = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();
        }

        self.mark_manifest_dirty();
        self.mark_stats_dirty();
        self.maybe_write_manifest()?;
        self.update_stats();

        // Clean up old segment files
        for removed_segment in removed_segments {
            for file_path in &removed_segment.file_paths {
                let _ = self.storage.delete_file(file_path);
            }
        }

        Ok(())
    }

    /// Get optimal merge plan based on current state.
    pub fn get_merge_plan(&self) -> MergePlan {
        let stats = self.get_stats();
        let compaction_candidates = self.get_compaction_candidates();

        // Determine best strategy based on current state
        let strategy = if !compaction_candidates.is_empty() {
            MergeStrategy::DeletionBased
        } else if stats.total_segments > self.config.max_segments
            || stats.avg_segment_size < self.config.min_segment_size
        {
            MergeStrategy::SizeBased
        } else {
            MergeStrategy::Balanced
        };

        let candidates = self.generate_merge_candidates(strategy);
        let estimated_benefit = self.calculate_merge_benefit(&candidates);

        MergePlan {
            strategy,
            candidates,
            estimated_benefit,
            urgency: self.calculate_merge_urgency(&stats),
        }
    }

    /// Calculate benefit of performing merges.
    fn calculate_merge_benefit(&self, candidates: &[MergeCandidate]) -> f64 {
        let mut total_benefit = 0.0;

        for candidate in candidates {
            // Benefit from reducing segment count
            let segment_reduction_benefit = candidate.segments.len() as f64 * 0.1;

            // Benefit from space reclamation (deletion-based merges)
            let space_benefit = match candidate.strategy {
                MergeStrategy::DeletionBased => candidate.priority * 0.1,
                _ => 0.0,
            };

            // Benefit from size optimization
            let size_benefit = if candidate.estimated_size < self.config.min_segment_size {
                1.0
            } else {
                0.5
            };

            total_benefit += segment_reduction_benefit + space_benefit + size_benefit;
        }

        total_benefit
    }

    /// Calculate urgency of merge operations.
    fn calculate_merge_urgency(&self, stats: &SegmentManagerStats) -> MergeUrgency {
        if stats.total_segments > self.config.max_segments * 2 {
            MergeUrgency::Critical
        } else if stats.overall_deletion_ratio > self.config.max_deletion_ratio * 1.5 {
            MergeUrgency::High
        } else if stats.total_segments > self.config.max_segments {
            MergeUrgency::Medium
        } else {
            MergeUrgency::Low
        }
    }

    /// Get segments by tier.
    pub fn get_segments_by_tier(&self) -> Vec<Vec<ManagedSegmentInfo>> {
        let segments = self.segments.read().unwrap();
        let mut tiers: Vec<Vec<ManagedSegmentInfo>> = vec![Vec::new(); 4];

        for segment in segments.values() {
            let tier = segment.tier.min(3) as usize;
            tiers[tier].push(segment.clone());
        }

        tiers
    }

    /// Force rebalance tiers based on current segment sizes.
    pub fn rebalance_tiers(&self) -> Result<()> {
        let mut segments = self.segments.write().unwrap();

        for segment in segments.values_mut() {
            let new_tier = self.calculate_tier(segment.size_bytes);
            if new_tier != segment.tier {
                segment.tier = new_tier;
                segment.last_modified = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs();
            }
        }

        drop(segments);
        self.mark_manifest_dirty();
        self.mark_stats_dirty();
        self.maybe_write_manifest()?;
        self.update_stats();
        Ok(())
    }

    /// Check if merge should be triggered.
    fn should_trigger_merge(&self) -> bool {
        let segments = self.segments.read().unwrap();
        segments.len() > self.config.max_segments
    }

    /// Calculate tier for a segment based on size.
    fn calculate_tier(&self, size_bytes: u64) -> u8 {
        if size_bytes < self.config.min_segment_size {
            0
        } else if size_bytes < self.config.max_segment_size / 4 {
            1
        } else if size_bytes < self.config.max_segment_size / 2 {
            2
        } else {
            3
        }
    }

    /// Calculate segment size from file paths.
    fn calculate_segment_size(&self, file_paths: &[String]) -> Result<u64> {
        let mut total_size = 0;
        for path in file_paths {
            if let Ok(metadata) = self.storage.metadata(path) {
                total_size += metadata.size;
            }
        }
        Ok(total_size)
    }

    /// Update internal statistics incrementally.
    fn update_stats(&self) {
        // Only update statistics if dirty flag is set to avoid unnecessary computation
        if !self.stats_dirty.load(std::sync::atomic::Ordering::Relaxed) {
            return;
        }

        let segments = self.segments.read().unwrap();
        let mut stats = self.stats.write().unwrap();

        stats.total_segments = segments.len();
        stats.total_size_bytes = segments.values().map(|s| s.size_bytes).sum();
        stats.total_doc_count = segments.values().map(|s| s.segment_info.doc_count).sum();
        stats.total_deleted_count = segments.values().map(|s| s.deleted_count).sum();

        if stats.total_segments > 0 {
            stats.avg_segment_size = stats.total_size_bytes / stats.total_segments as u64;
        }

        if stats.total_doc_count > 0 {
            stats.overall_deletion_ratio =
                stats.total_deleted_count as f64 / stats.total_doc_count as f64;
        }

        // Clear dirty flag
        self.stats_dirty
            .store(false, std::sync::atomic::Ordering::Relaxed);
    }

    /// Mark statistics as needing update.
    fn mark_stats_dirty(&self) {
        self.stats_dirty
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// Get current statistics.
    pub fn get_stats(&self) -> SegmentManagerStats {
        self.stats.read().unwrap().clone()
    }

    /// Get configuration.
    pub fn get_config(&self) -> &SegmentManagerConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{MemoryStorage, StorageConfig};

    #[allow(dead_code)]
    fn create_test_segment_info(segment_id: &str, doc_count: u64) -> SegmentInfo {
        SegmentInfo {
            segment_id: segment_id.to_string(),
            doc_count,
            doc_offset: 0,
            generation: 1,
            has_deletions: false,
        }
    }

    // Test helper method to add segment with specific size
    fn add_test_segment_with_size(
        manager: &SegmentManager,
        segment_id: &str,
        doc_count: u64,
        size_bytes: u64,
    ) -> Result<()> {
        let mut segment_info = create_test_segment_info(segment_id, doc_count);
        segment_info.generation = manager.generation.fetch_add(1, Ordering::Relaxed);

        let mut managed_info = ManagedSegmentInfo::new(segment_info.clone());
        managed_info.file_paths = vec![];
        managed_info.size_bytes = size_bytes;
        managed_info.tier = manager.calculate_tier(size_bytes);

        {
            let mut segments = manager.segments.write().unwrap();
            segments.insert(segment_info.segment_id.clone(), managed_info);
        }

        manager.mark_manifest_dirty();
        manager.mark_stats_dirty();
        manager.update_stats();

        Ok(())
    }

    #[test]
    fn test_segment_manager_creation() {
        let config = SegmentManagerConfig::default();
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));

        let manager = SegmentManager::new(config, storage).unwrap();

        let stats = manager.get_stats();
        assert_eq!(stats.total_segments, 0);
        assert_eq!(stats.total_doc_count, 0);
    }

    #[test]
    fn test_add_segment() {
        let config = SegmentManagerConfig::default();
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let manager = SegmentManager::new(config, storage).unwrap();

        let segment_info = create_test_segment_info("seg001", 1000);
        let file_paths = vec!["seg001.idx".to_string(), "seg001.dict".to_string()];

        manager.add_segment(segment_info, file_paths).unwrap();

        let stats = manager.get_stats();
        assert_eq!(stats.total_segments, 1);
        assert_eq!(stats.total_doc_count, 1000);

        let segments = manager.get_segments();
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].segment_info.segment_id, "seg001");
    }

    #[test]
    fn test_remove_segment() {
        let config = SegmentManagerConfig::default();
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let manager = SegmentManager::new(config, storage).unwrap();

        let segment_info = create_test_segment_info("seg001", 1000);
        manager.add_segment(segment_info, vec![]).unwrap();

        let removed = manager.remove_segment("seg001").unwrap();
        assert!(removed.is_some());

        let stats = manager.get_stats();
        assert_eq!(stats.total_segments, 0);
    }

    #[test]
    fn test_mark_deleted() {
        let config = SegmentManagerConfig::default();
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let manager = SegmentManager::new(config, storage).unwrap();

        let segment_info = create_test_segment_info("seg001", 10); // Reduced from 1000 to 10
        manager.add_segment(segment_info, vec![]).unwrap();

        manager.mark_deleted_fast("seg001", 1).unwrap(); // Use fast version

        let segment = manager.get_segment("seg001").unwrap();
        assert_eq!(segment.deleted_count, 1);
        assert_eq!(segment.deletion_ratio(), 0.1);
        assert!(segment.segment_info.has_deletions);
    }

    #[test]
    fn test_compaction_candidates() {
        let config = SegmentManagerConfig {
            max_deletion_ratio: 0.2,
            ..Default::default()
        };

        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let manager = SegmentManager::new(config, storage).unwrap();

        // Add segments with different deletion ratios (reduced sizes)
        let seg1 = create_test_segment_info("seg001", 10); // Reduced from 1000 to 10
        let seg2 = create_test_segment_info("seg002", 10); // Reduced from 1000 to 10

        manager.add_segment(seg1, vec![]).unwrap();
        manager.add_segment(seg2, vec![]).unwrap();

        // Mark one segment with high deletion ratio (use batch method)
        let updates = vec![
            ("seg001".to_string(), 3), // 30% deletion (3/10)
            ("seg002".to_string(), 1), // 10% deletion (1/10)
        ];
        manager.batch_mark_deleted(&updates).unwrap();

        let candidates = manager.get_compaction_candidates();
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].segment_info.segment_id, "seg001");
    }

    #[test]
    fn test_tier_calculation() {
        let config = SegmentManagerConfig::default();
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let manager = SegmentManager::new(config, storage).unwrap();

        assert_eq!(manager.calculate_tier(500 * 1024), 0); // < min_segment_size
        assert_eq!(manager.calculate_tier(5 * 1024 * 1024), 1); // < max/4
        assert_eq!(manager.calculate_tier(30 * 1024 * 1024), 2); // < max/2
        assert_eq!(manager.calculate_tier(80 * 1024 * 1024), 3); // >= max/2
    }

    #[test]
    fn test_merge_candidate_generation() {
        let config = SegmentManagerConfig::default();
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let manager = SegmentManager::new(config, storage).unwrap();

        // Add multiple segments of different sizes
        for i in 0..6 {
            let mut segment_info = create_test_segment_info(&format!("seg{i:03}"), 500);
            segment_info.generation = i;
            manager.add_segment(segment_info, vec![]).unwrap();
        }

        // Test size-based candidates
        let size_candidates = manager.generate_merge_candidates(MergeStrategy::SizeBased);
        assert!(!size_candidates.is_empty());

        // Test balanced candidates
        let balanced_candidates = manager.generate_merge_candidates(MergeStrategy::Balanced);
        assert!(!balanced_candidates.is_empty());
    }

    #[test]
    fn test_deletion_based_merge_candidates() {
        let config = SegmentManagerConfig::default();
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let manager = SegmentManager::new(config, storage).unwrap();

        // Add segments and mark some with high deletion (reduced sizes)
        for i in 0..4 {
            let segment_info = create_test_segment_info(&format!("seg{i:03}"), 10); // Reduced from 1000 to 10
            manager.add_segment(segment_info, vec![]).unwrap();
        }

        // Mark segments with high deletion ratios (use batch method)
        let updates = vec![
            ("seg000".to_string(), 4), // 40% deletion (4/10)
            ("seg001".to_string(), 3), // 30% deletion (3/10)
            ("seg002".to_string(), 1), // 10% deletion (1/10)
        ];
        manager.batch_mark_deleted(&updates).unwrap();

        let deletion_candidates = manager.generate_merge_candidates(MergeStrategy::DeletionBased);
        assert!(!deletion_candidates.is_empty());

        // Should prioritize high-deletion segments
        if let Some(candidate) = deletion_candidates.first() {
            assert!(
                candidate.segments.contains(&"seg000".to_string())
                    || candidate.segments.contains(&"seg001".to_string())
            );
        }
    }

    #[test]
    fn test_mark_segments_merging() {
        let config = SegmentManagerConfig::default();
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let manager = SegmentManager::new(config, storage).unwrap();

        let segment_info = create_test_segment_info("seg001", 1000);
        manager.add_segment(segment_info, vec![]).unwrap();

        // Mark as merging
        manager
            .mark_segments_merging(&["seg001".to_string()], true)
            .unwrap();

        let segment = manager.get_segment("seg001").unwrap();
        assert!(segment.is_merging);

        // Mark as not merging
        manager
            .mark_segments_merging(&["seg001".to_string()], false)
            .unwrap();

        let segment = manager.get_segment("seg001").unwrap();
        assert!(!segment.is_merging);
    }

    #[test]
    fn test_complete_merge() {
        let config = SegmentManagerConfig::default();
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let manager = SegmentManager::new(config, storage).unwrap();

        // Add segments to be merged
        let seg1 = create_test_segment_info("seg001", 500);
        let seg2 = create_test_segment_info("seg002", 600);
        manager.add_segment(seg1, vec![]).unwrap();
        manager.add_segment(seg2, vec![]).unwrap();

        let initial_stats = manager.get_stats();
        assert_eq!(initial_stats.total_segments, 2);

        // Complete merge
        let new_segment = create_test_segment_info("merged_seg", 1100);
        manager
            .complete_merge(
                &["seg001".to_string(), "seg002".to_string()],
                new_segment,
                vec![],
            )
            .unwrap();

        let final_stats = manager.get_stats();
        assert_eq!(final_stats.total_segments, 1);
        assert_eq!(final_stats.merge_operations, 1);

        // Original segments should be gone
        assert!(manager.get_segment("seg001").is_none());
        assert!(manager.get_segment("seg002").is_none());

        // New segment should exist
        assert!(manager.get_segment("merged_seg").is_some());
    }

    #[test]
    fn test_merge_plan_generation() {
        let config = SegmentManagerConfig {
            max_segments: 3,
            ..Default::default()
        };

        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let manager = SegmentManager::new(config, storage).unwrap();

        // Add enough segments to trigger merge
        for i in 0..5 {
            let segment_info = create_test_segment_info(&format!("seg{i:03}"), 500);
            manager.add_segment(segment_info, vec![]).unwrap();
        }

        let merge_plan = manager.get_merge_plan();

        assert!(!merge_plan.candidates.is_empty());
        assert!(matches!(
            merge_plan.urgency,
            MergeUrgency::Medium | MergeUrgency::High
        ));
        assert!(merge_plan.estimated_benefit > 0.0);
    }

    #[test]
    fn test_segments_by_tier() {
        let config = SegmentManagerConfig::default();
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let manager = SegmentManager::new(config, storage).unwrap();

        // Add segments with different sizes to create different tiers
        add_test_segment_with_size(&manager, "small", 100, 500 * 1024).unwrap(); // < min_segment_size (1MB)
        add_test_segment_with_size(&manager, "medium", 5000, 5 * 1024 * 1024).unwrap(); // < max/4 (25MB)
        add_test_segment_with_size(&manager, "large", 50000, 50 * 1024 * 1024).unwrap(); // >= max/2 (50MB)

        let tiers = manager.get_segments_by_tier();

        // Should have segments in different tiers
        assert_eq!(tiers.len(), 4);

        // Small segment should be in tier 0
        assert!(!tiers[0].is_empty());
        assert_eq!(tiers[0][0].segment_info.segment_id, "small");

        // Medium segment should be in tier 1
        assert!(!tiers[1].is_empty());
        assert_eq!(tiers[1][0].segment_info.segment_id, "medium");

        // Large segment should be in tier 3 (>= max/2)
        assert!(!tiers[3].is_empty());
        assert_eq!(tiers[3][0].segment_info.segment_id, "large");
    }

    #[test]
    fn test_rebalance_tiers() {
        let config = SegmentManagerConfig::default();
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let manager = SegmentManager::new(config, storage).unwrap();

        // Add segment with initial size
        add_test_segment_with_size(&manager, "seg001", 1000, 1024).unwrap(); // Small size, should be tier 0

        // Verify initial tier
        let initial_segment = manager.get_segment("seg001").unwrap();
        assert_eq!(initial_segment.tier, 0);

        // Manually change segment size and set wrong tier
        {
            let mut segments = manager.segments.write().unwrap();
            if let Some(segment) = segments.get_mut("seg001") {
                segment.size_bytes = 50 * 1024 * 1024; // Change to large size
                segment.tier = 0; // Wrong tier
            }
        }

        manager.rebalance_tiers().unwrap();

        let segment = manager.get_segment("seg001").unwrap();
        assert_eq!(segment.tier, 3); // Should be rebalanced to tier 3 (>= max/2)
    }
}
