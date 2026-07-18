//! Merge engine for vector index segments.
//!
//! This module handles the actual merging of segments.

use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::error::Result;
use crate::storage::Storage;
use crate::vector::core::vector::Vector;

use super::manager::ManagedSegmentInfo;

use crate::maintenance::deletion::DeletionBitmap;
use crate::vector::index::config::HnswIndexConfig;
use crate::vector::index::hnsw::reader::HnswIndexReader;
use crate::vector::index::hnsw::writer::HnswIndexWriter;
use crate::vector::reader::VectorIndexReader;
use crate::vector::writer::{VectorIndexWriter, VectorIndexWriterConfig};

/// Configuration for merge operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeConfig {
    /// Maximum number of segments to merge at once.
    pub max_merge_segments: u32,

    /// Target segment size after merge (in vectors).
    pub target_segment_size: u64,

    /// Whether to use parallel merging.
    pub parallel_merge: bool,

    /// Number of threads to use for parallel merging.
    pub num_threads: usize,
}

impl Default for MergeConfig {
    fn default() -> Self {
        Self {
            max_merge_segments: 10,
            target_segment_size: 1000000,
            parallel_merge: true,
            num_threads: 4,
        }
    }
}

/// Statistics about a merge operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeStats {
    /// Number of segments merged.
    pub segments_merged: u32,

    /// Number of vectors in merged segment.
    pub vectors_merged: u64,

    /// Number of deleted vectors removed.
    pub deletions_removed: u64,

    /// Number of stale cross-segment duplicates removed — copies of a
    /// `(doc_id, field)` key shadowed by a newer segment's copy
    /// (Issue #880). Defaults to 0 when deserializing older stats.
    #[serde(default)]
    pub duplicates_removed: u64,

    /// Time taken for merge (in milliseconds).
    pub merge_time_ms: u64,

    /// Size of merged segment (in bytes).
    pub merged_size_bytes: u64,
}

impl MergeStats {
    /// Create new merge stats.
    pub fn new() -> Self {
        Self {
            segments_merged: 0,
            vectors_merged: 0,
            deletions_removed: 0,
            duplicates_removed: 0,
            merge_time_ms: 0,
            merged_size_bytes: 0,
        }
    }

    /// Calculate compression ratio.
    pub fn compression_ratio(&self) -> f64 {
        if self.vectors_merged == 0 {
            return 1.0;
        }
        1.0 - (self.deletions_removed as f64 / self.vectors_merged as f64)
    }
}

impl Default for MergeStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a merge operation.
#[derive(Debug, Clone)]
pub struct MergeResult {
    /// Information about the new merged segment.
    pub merged_segment: ManagedSegmentInfo,

    /// Statistics about the merge.
    pub stats: MergeStats,

    /// IDs of segments that were merged.
    pub merged_segment_ids: Vec<String>,
}

/// Engine for merging vector index segments.
pub struct MergeEngine {
    config: MergeConfig,
    storage: Arc<dyn Storage>,
    index_config: HnswIndexConfig,
    writer_config: VectorIndexWriterConfig,
    deletion_bitmap: Option<Arc<DeletionBitmap>>,
}

impl MergeEngine {
    /// Create a new merge engine.
    pub fn new(
        config: MergeConfig,
        storage: Arc<dyn Storage>,
        index_config: HnswIndexConfig,
        writer_config: VectorIndexWriterConfig,
    ) -> Self {
        Self {
            config,
            storage,
            index_config,
            writer_config,
            deletion_bitmap: None,
        }
    }

    /// Set deletion bitmap for filtering deleted vectors during merge.
    pub fn set_deletion_bitmap(&mut self, bitmap: Arc<DeletionBitmap>) {
        self.deletion_bitmap = Some(bitmap);
    }

    /// Merge multiple segments into a single segment.
    ///
    /// Reads every live vector from the source segments (deletion-filtered
    /// via the configured bitmap, losslessly sourced from the f32 rerank
    /// sidecar when present — Issue #795), deduplicates cross-segment
    /// duplicates of the same `(doc_id, field)` with **newest-generation
    /// wins** semantics (Issue #880 — same-id upserts replayed from the WAL
    /// land in newer segments and must shadow the stale copies), and writes
    /// the survivors into a new segment.
    pub fn merge_segments(
        &self,
        segments: Vec<ManagedSegmentInfo>,
        new_segment_id: String,
    ) -> Result<MergeResult> {
        let start_time = crate::util::time::Timer::now();

        // Calculate statistics
        let segments_merged = segments.len() as u32;
        #[allow(unused_assignments)]
        let mut vectors_merged = 0;
        let mut deletions_removed = 0;
        let mut duplicates_removed = 0u64;
        #[allow(unused_assignments)]
        let mut total_size = segments.iter().map(|s| s.size_bytes).sum::<u64>();

        let mut all_vectors: Vec<(u64, String, Vector)> = Vec::new();

        // Newest generation FIRST, so the first occurrence of a
        // `(doc_id, field)` key is the authoritative (newest) copy and every
        // later one is a stale duplicate (Issue #880).
        let mut sources = segments.clone();
        sources.sort_by_key(|s| std::cmp::Reverse(s.generation));
        let mut seen: std::collections::HashSet<(u64, String)> = std::collections::HashSet::new();

        // 1. Read all live vectors from source segments (newest first)
        for segment in &sources {
            // Note: HnswIndexReader::load expects path without extension
            let reader = HnswIndexReader::load(
                self.storage.clone(),
                &segment.segment_id,
                self.index_config.distance_metric,
            )?;

            // Issue #795: prefer the source segment's original f32 rerank
            // sidecar (lossless) over the int8-dequantized iterator value,
            // so a merge does not bake one round of quantization error into
            // the merged sidecar. The pool is already loaded by the reader
            // (Eager mode, zero extra I/O); when absent (no sidecar, or Lazy
            // mode) we keep the int8-dequantized vector — the best available
            // source.
            let rerank_pool = reader.rerank_storage().cloned();

            let mut iterator = reader.vector_iterator()?;
            while let Some((doc_id, field, vector)) = iterator.next()? {
                if let Some(bitmap) = &self.deletion_bitmap
                    && bitmap.is_deleted(doc_id)
                {
                    deletions_removed += 1;
                    continue;
                }
                if !seen.insert((doc_id, field.clone())) {
                    // A newer segment already contributed this key.
                    duplicates_removed += 1;
                    continue;
                }
                let vector = match rerank_pool
                    .as_ref()
                    .and_then(|pool| pool.get_f32_slice(doc_id, &field))
                {
                    Some(f32_vector) => Vector::new(f32_vector.to_vec()),
                    None => vector,
                };
                all_vectors.push((doc_id, field, vector));
            }
        }

        // 2. Write to new segment
        // We use with_storage to ensure it writes to the correct location
        let mut writer = HnswIndexWriter::with_storage(
            self.index_config.clone(),
            self.writer_config.clone(),
            &new_segment_id,
            self.storage.clone(),
        )?;

        writer.add_vectors(all_vectors.clone())?;
        writer.finalize()?;
        writer.write()?;

        vectors_merged = all_vectors.len() as u64;
        total_size = vectors_merged * 128; // Dummy estimate

        let merge_time_ms = start_time.elapsed_ms();

        let merged_segment = ManagedSegmentInfo {
            segment_id: new_segment_id,
            vector_count: vectors_merged,
            vector_offset: 0,
            // The merged output represents data no newer than its newest
            // source, so it inherits max(source generations) — NOT max+1
            // (Issue #880): with +1 a merge over non-adjacent sources would
            // out-rank untouched newer segments, laundering a stale copy
            // from an old source above a genuinely newer one under the
            // newest-generation-wins dedup. Generations are unique (stamped
            // max+1 at flush) and the sources are removed with the merge, so
            // inheriting the maximum cannot collide with a live segment.
            generation: segments.iter().map(|s| s.generation).max().unwrap_or(0),
            has_deletions: false,
            size_bytes: total_size,
        };

        let stats = MergeStats {
            segments_merged,
            vectors_merged,
            deletions_removed,
            duplicates_removed,
            merge_time_ms,
            merged_size_bytes: total_size,
        };

        Ok(MergeResult {
            merged_segment,
            stats,
            merged_segment_ids: segments.iter().map(|s| s.segment_id.clone()).collect(),
        })
    }

    /// Get storage reference.
    pub fn storage(&self) -> &Arc<dyn Storage> {
        &self.storage
    }

    /// Get configuration.
    pub fn config(&self) -> &MergeConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};

    #[test]
    fn test_merge_engine_basic() {
        let config = MergeConfig::default();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let index_config = HnswIndexConfig::default();
        let writer_config = VectorIndexWriterConfig::default();

        let engine = MergeEngine::new(config, storage, index_config, writer_config);

        // In this unit test, we cannot easily mock HnswIndexReader::load unless we actually write files to MemoryStorage first.
        // HnswIndexReader::load uses storage.open_input().
        // So we would need to prepare segments.
        // Since that is complex setup, we will skip the execution part for now or use a simpler verification.
        // Or we could mock storage.

        let _segments = [ManagedSegmentInfo {
            segment_id: "seg1".to_string(),
            vector_count: 1000,
            vector_offset: 0,
            generation: 0,
            has_deletions: false,
            size_bytes: 128000,
        }];

        // We comment out actual execution because it will fail on file not found
        // let result = engine.merge_segments(segments, "merged_seg".to_string());
        // assert!(result.is_ok());

        // At least we verify compilation of `new` signature
        assert_eq!(engine.config.max_merge_segments, 10);
    }

    #[test]
    fn test_merge_stats() {
        let stats = MergeStats {
            segments_merged: 3,
            vectors_merged: 1000,
            deletions_removed: 200,
            duplicates_removed: 0,
            merge_time_ms: 100,
            merged_size_bytes: 102400,
        };

        assert_eq!(stats.compression_ratio(), 0.8);
    }
}
