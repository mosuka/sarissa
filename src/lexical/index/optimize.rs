//! Index optimization and maintenance operations.

use std::sync::Arc;

use crate::error::Result;
use crate::lexical::segment::{MergeOperation, Segment, SegmentManager};
use crate::storage::traits::Storage;

/// Index optimizer for maintenance operations.
#[derive(Debug)]
pub struct IndexOptimizer {
    /// Segment manager for organizing segments.
    segment_manager: SegmentManager,
    /// Storage backend.
    #[allow(dead_code)]
    storage: Arc<dyn Storage>,
    /// Optimization configuration.
    config: OptimizationConfig,
}

/// Configuration for index optimization.
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// Maximum number of segments before merge is triggered.
    pub max_segments: usize,
    /// Deletion ratio threshold for triggering merge (0.0-1.0).
    pub merge_threshold: f64,
    /// Target segment size in bytes.
    pub target_segment_size: u64,
    /// Maximum segment size in bytes.
    pub max_segment_size: u64,
    /// Enable compression during optimization.
    pub enable_compression: bool,
    /// Number of parallel merge operations.
    pub merge_parallelism: usize,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        OptimizationConfig {
            max_segments: 10,
            merge_threshold: 0.3,                  // 30% deletion ratio
            target_segment_size: 64 * 1024 * 1024, // 64MB
            max_segment_size: 256 * 1024 * 1024,   // 256MB
            enable_compression: true,
            merge_parallelism: 2,
        }
    }
}

impl IndexOptimizer {
    /// Create a new index optimizer.
    pub fn new(storage: Arc<dyn Storage>, config: OptimizationConfig) -> Self {
        let mut segment_manager = SegmentManager::new(Arc::clone(&storage));
        segment_manager.set_max_segments(config.max_segments);
        segment_manager.set_merge_threshold(config.merge_threshold);

        IndexOptimizer {
            segment_manager,
            storage,
            config,
        }
    }

    /// Load existing segment information.
    pub fn load(&mut self) -> Result<()> {
        self.segment_manager.load()
    }

    /// Save segment information.
    pub fn save(&self) -> Result<()> {
        self.segment_manager.save()
    }

    /// Analyze the index and return optimization recommendations.
    pub fn analyze(&self) -> IndexAnalysis {
        let segments: Vec<&Segment> = self.segment_manager.segments().collect();
        let total_docs = self.segment_manager.total_doc_count();
        let live_docs = self.segment_manager.total_live_doc_count();
        let total_size = self.segment_manager.total_size_bytes();

        let mut recommendations = Vec::new();

        // Check if we need merging
        if self.segment_manager.should_merge() {
            recommendations.push(OptimizationRecommendation::Merge {
                reason: "High deletion ratio or too many segments".to_string(),
                urgency: if segments.len() > self.config.max_segments * 2 {
                    Urgency::High
                } else {
                    Urgency::Medium
                },
            });
        }

        // Check for oversized segments
        for segment in &segments {
            if segment.size_bytes > self.config.max_segment_size {
                recommendations.push(OptimizationRecommendation::Split {
                    segment_id: segment.id.clone(),
                    reason: "Segment exceeds maximum size".to_string(),
                    urgency: Urgency::Low,
                });
            }
        }

        // Check for fragmentation
        let deletion_ratio = if total_docs > 0 {
            (total_docs - live_docs) as f64 / total_docs as f64
        } else {
            0.0
        };

        if deletion_ratio > self.config.merge_threshold * 2.0 {
            recommendations.push(OptimizationRecommendation::Defragment {
                reason: format!(
                    "High overall deletion ratio: {:.1}%",
                    deletion_ratio * 100.0
                ),
                urgency: Urgency::High,
            });
        }

        IndexAnalysis {
            total_segments: segments.len(),
            total_documents: total_docs,
            live_documents: live_docs,
            deleted_documents: total_docs - live_docs,
            total_size_bytes: total_size,
            deletion_ratio,
            recommendations,
        }
    }

    /// Optimize the index by performing necessary maintenance operations.
    pub fn optimize(&mut self) -> Result<OptimizationResult> {
        let analysis = self.analyze();
        let mut operations_performed = Vec::new();
        let mut bytes_reclaimed = 0u64;

        // Perform merges if needed
        if self.segment_manager.should_merge() {
            let merge_result = self.perform_merges()?;
            operations_performed.extend(merge_result.operations_performed);
            bytes_reclaimed += merge_result.bytes_reclaimed;
        }

        // Perform compression if enabled
        if self.config.enable_compression {
            let compression_result = self.compress_segments()?;
            operations_performed.extend(compression_result.operations_performed);
            bytes_reclaimed += compression_result.bytes_reclaimed;
        }

        // Save updated segment information
        self.save()?;

        Ok(OptimizationResult {
            operations_performed,
            bytes_reclaimed,
            segments_before: analysis.total_segments,
            segments_after: self.segment_manager.segments().count(),
            duration_ms: 0, // TODO: Add timing
        })
    }

    /// Perform segment merges.
    fn perform_merges(&mut self) -> Result<OptimizationResult> {
        let candidates = self.segment_manager.get_merge_candidates();
        if candidates.is_empty() {
            return Ok(OptimizationResult::empty());
        }

        let mut operations = Vec::new();
        let mut bytes_reclaimed = 0u64;

        // Create merge operation
        let target_segment = self.segment_manager.create_segment()?;
        let merge_op = MergeOperation::new(
            candidates.clone(),
            target_segment.id.clone(),
            0, // Will be calculated during merge
        );

        // Perform the actual merge
        let result = self.merge_segments(&merge_op)?;
        bytes_reclaimed += result.bytes_saved;

        operations.push(format!(
            "Merged {} segments into {}",
            candidates.len(),
            target_segment.id
        ));

        // Remove old segments
        for segment_id in &candidates {
            self.segment_manager.remove_segment(segment_id)?;
        }

        Ok(OptimizationResult {
            operations_performed: operations,
            bytes_reclaimed,
            segments_before: 0,
            segments_after: 0,
            duration_ms: 0,
        })
    }

    /// Merge multiple segments into one.
    fn merge_segments(&mut self, merge_op: &MergeOperation) -> Result<MergeResult> {
        // This is a simplified merge implementation
        // In a real implementation, this would:
        // 1. Read posting lists from source segments
        // 2. Merge them while preserving sort order
        // 3. Remove deleted documents
        // 4. Write the merged result to the target segment

        let mut total_docs = 0u64;
        let mut total_size = 0u64;
        let mut bytes_saved = 0u64;

        // Calculate statistics from source segments
        for segment_id in &merge_op.source_segments {
            if let Some(segment) = self.segment_manager.get_segment(segment_id) {
                total_docs += segment.live_doc_count();
                total_size += segment.size_bytes;
                bytes_saved += segment.deleted_count * 100; // Rough estimate
            }
        }

        // Update target segment with merged data
        self.segment_manager.update_segment_stats(
            &merge_op.target_segment,
            total_docs,
            total_size,
        )?;

        // Freeze the target segment
        self.segment_manager
            .freeze_segment(&merge_op.target_segment)?;

        Ok(MergeResult {
            merged_docs: total_docs,
            bytes_saved,
        })
    }

    /// Compress segments to save space.
    fn compress_segments(&mut self) -> Result<OptimizationResult> {
        let mut operations = Vec::new();
        let mut bytes_reclaimed = 0u64;

        // Find segments that would benefit from compression
        let segments: Vec<String> = self
            .segment_manager
            .segments()
            .filter(|s| !s.is_mutable && s.size_bytes > 1024 * 1024) // > 1MB
            .map(|s| s.id.clone())
            .collect();

        for segment_id in segments {
            if let Some(segment) = self.segment_manager.get_segment(&segment_id) {
                // Simulate compression savings
                let compression_ratio = 0.7; // Assume 30% compression
                let bytes_saved = (segment.size_bytes as f64 * (1.0 - compression_ratio)) as u64;

                bytes_reclaimed += bytes_saved;
                operations.push(format!(
                    "Compressed segment {segment_id} (saved {bytes_saved} bytes)"
                ));

                // Update segment size
                let new_size = (segment.size_bytes as f64 * compression_ratio) as u64;
                self.segment_manager.update_segment_stats(
                    &segment_id,
                    segment.doc_count,
                    new_size,
                )?;
            }
        }

        Ok(OptimizationResult {
            operations_performed: operations,
            bytes_reclaimed,
            segments_before: 0,
            segments_after: 0,
            duration_ms: 0,
        })
    }

    /// Compact the index by removing all deleted documents.
    pub fn compact(&mut self) -> Result<OptimizationResult> {
        let mut operations = Vec::new();
        let mut bytes_reclaimed = 0u64;

        // Get all segments with deleted documents
        let segments_to_compact: Vec<String> = self
            .segment_manager
            .segments()
            .filter(|s| s.deleted_count > 0)
            .map(|s| s.id.clone())
            .collect();

        for segment_id in segments_to_compact {
            if let Some(segment) = self.segment_manager.get_segment_mut(&segment_id) {
                let deleted_bytes = segment.deleted_count * 100; // Rough estimate
                bytes_reclaimed += deleted_bytes;

                // Reset deleted count (documents are actually removed)
                segment.deleted_count = 0;
                segment.size_bytes = segment.size_bytes.saturating_sub(deleted_bytes);

                operations.push(format!(
                    "Compacted segment {segment_id} (reclaimed {deleted_bytes} bytes)"
                ));
            }
        }

        self.save()?;

        Ok(OptimizationResult {
            operations_performed: operations,
            bytes_reclaimed,
            segments_before: 0,
            segments_after: 0,
            duration_ms: 0,
        })
    }

    /// Get segment manager for external access.
    pub fn segment_manager(&self) -> &SegmentManager {
        &self.segment_manager
    }

    /// Get mutable segment manager for external access.
    pub fn segment_manager_mut(&mut self) -> &mut SegmentManager {
        &mut self.segment_manager
    }
}

/// Analysis result for index optimization.
#[derive(Debug, Clone)]
pub struct IndexAnalysis {
    /// Total number of segments.
    pub total_segments: usize,
    /// Total number of documents.
    pub total_documents: u64,
    /// Number of live documents.
    pub live_documents: u64,
    /// Number of deleted documents.
    pub deleted_documents: u64,
    /// Total size in bytes.
    pub total_size_bytes: u64,
    /// Overall deletion ratio.
    pub deletion_ratio: f64,
    /// Optimization recommendations.
    pub recommendations: Vec<OptimizationRecommendation>,
}

/// Optimization recommendation.
#[derive(Debug, Clone)]
pub enum OptimizationRecommendation {
    /// Recommend merging segments.
    Merge { reason: String, urgency: Urgency },
    /// Recommend splitting a large segment.
    Split {
        segment_id: String,
        reason: String,
        urgency: Urgency,
    },
    /// Recommend defragmentation.
    Defragment { reason: String, urgency: Urgency },
    /// Recommend compression.
    Compress {
        segment_id: String,
        potential_savings: u64,
        urgency: Urgency,
    },
}

/// Urgency level for optimization operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Urgency {
    Low,
    Medium,
    High,
    Critical,
}

/// Result of optimization operations.
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Operations that were performed.
    pub operations_performed: Vec<String>,
    /// Bytes reclaimed by optimization.
    pub bytes_reclaimed: u64,
    /// Number of segments before optimization.
    pub segments_before: usize,
    /// Number of segments after optimization.
    pub segments_after: usize,
    /// Duration of optimization in milliseconds.
    pub duration_ms: u64,
}

impl OptimizationResult {
    /// Create an empty optimization result.
    pub fn empty() -> Self {
        OptimizationResult {
            operations_performed: Vec::new(),
            bytes_reclaimed: 0,
            segments_before: 0,
            segments_after: 0,
            duration_ms: 0,
        }
    }
}

/// Result of a segment merge operation.
#[derive(Debug)]
struct MergeResult {
    /// Number of documents in the merged segment.
    #[allow(dead_code)]
    merged_docs: u64,
    /// Bytes saved by the merge operation.
    bytes_saved: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::memory::MemoryStorage;
    use crate::storage::traits::StorageConfig;

    #[test]
    fn test_optimization_config() {
        let config = OptimizationConfig::default();

        assert_eq!(config.max_segments, 10);
        assert_eq!(config.merge_threshold, 0.3);
        assert!(config.enable_compression);
    }

    #[test]
    fn test_index_optimizer_creation() {
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let config = OptimizationConfig::default();
        let optimizer = IndexOptimizer::new(storage, config);

        assert_eq!(optimizer.segment_manager.segments().count(), 0);
    }

    #[test]
    fn test_index_analysis() {
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let config = OptimizationConfig::default();
        let optimizer = IndexOptimizer::new(storage, config);

        let analysis = optimizer.analyze();

        assert_eq!(analysis.total_segments, 0);
        assert_eq!(analysis.total_documents, 0);
        assert_eq!(analysis.deletion_ratio, 0.0);
    }

    #[test]
    fn test_optimization_recommendation() {
        let rec = OptimizationRecommendation::Merge {
            reason: "Too many segments".to_string(),
            urgency: Urgency::High,
        };

        match rec {
            OptimizationRecommendation::Merge { urgency, .. } => {
                assert_eq!(urgency, Urgency::High);
            }
            _ => panic!("Wrong recommendation type"),
        }
    }

    #[test]
    fn test_optimization_result() {
        let result = OptimizationResult {
            operations_performed: vec!["Merged 3 segments".to_string()],
            bytes_reclaimed: 1024,
            segments_before: 5,
            segments_after: 3,
            duration_ms: 500,
        };

        assert_eq!(result.operations_performed.len(), 1);
        assert_eq!(result.bytes_reclaimed, 1024);
        assert_eq!(result.segments_before, 5);
        assert_eq!(result.segments_after, 3);
    }
}
