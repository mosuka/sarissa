//! Index optimization algorithms for efficient search performance.
//!
//! This module provides advanced optimization strategies for index structure,
//! segment organization, and query performance improvements.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::error::{SarissaError, Result};
use crate::lexical::index::inverted::maintenance::deletion::DeletionManager;
use crate::lexical::index::inverted::segment::manager::{
    ManagedSegmentInfo, MergeStrategy, SegmentManager,
};
use crate::lexical::index::inverted::segment::merge_engine::{
    MergeConfig, MergeEngine, MergeResult,
};
use crate::storage::Storage;

/// Optimization strategy types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OptimizationLevel {
    /// Aggressive optimization - maximum compression and merging.
    Aggressive,
    /// Balanced optimization - good performance with reasonable resource usage.
    #[default]
    Balanced,
    /// Conservative optimization - minimal impact on ongoing operations.
    Conservative,
    /// Custom optimization with specific parameters.
    Custom,
}

/// Configuration for index optimization.
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// Optimization strategy to use.
    pub strategy: OptimizationLevel,

    /// Maximum number of segments to merge in one operation.
    pub max_merge_segments: usize,

    /// Target number of segments after optimization.
    pub target_segment_count: usize,

    /// Minimum deletion ratio to trigger segment compaction.
    pub compaction_threshold: f64,

    /// Maximum memory usage during optimization (in MB).
    pub max_memory_mb: u64,

    /// Whether to optimize term dictionaries.
    pub optimize_dictionaries: bool,

    /// Whether to rebuild posting lists for better compression.
    pub rebuild_postings: bool,

    /// Whether to reorder documents for better locality.
    pub reorder_documents: bool,

    /// Maximum time to spend on optimization (in seconds).
    pub max_optimization_time_secs: u64,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        OptimizationConfig {
            strategy: OptimizationLevel::Balanced,
            max_merge_segments: 10,
            target_segment_count: 5,
            compaction_threshold: 0.1, // 10%
            max_memory_mb: 512,
            optimize_dictionaries: true,
            rebuild_postings: true,
            reorder_documents: false,
            max_optimization_time_secs: 300, // 5 minutes
        }
    }
}

/// Results of an optimization operation.
#[derive(Debug, Default)]
pub struct OptimizationResult {
    /// Number of segments before optimization.
    pub segments_before: usize,

    /// Number of segments after optimization.
    pub segments_after: usize,

    /// Total size before optimization (bytes).
    pub size_before: u64,

    /// Total size after optimization (bytes).
    pub size_after: u64,

    /// Number of deleted documents removed.
    pub deleted_docs_removed: u64,

    /// Number of merge operations performed.
    pub merge_operations: u64,

    /// Time taken for optimization (milliseconds).
    pub optimization_time_ms: u64,

    /// Space savings achieved (percentage).
    pub space_savings_percent: f64,

    /// Whether optimization completed successfully.
    pub completed: bool,

    /// Detailed merge results.
    pub merge_results: Vec<MergeResult>,
}

impl OptimizationResult {
    /// Calculate compression ratio.
    pub fn compression_ratio(&self) -> f64 {
        if self.size_before == 0 {
            1.0
        } else {
            self.size_after as f64 / self.size_before as f64
        }
    }

    /// Calculate segment reduction ratio.
    pub fn segment_reduction_ratio(&self) -> f64 {
        if self.segments_before == 0 {
            1.0
        } else {
            self.segments_after as f64 / self.segments_before as f64
        }
    }
}

/// Advanced index optimizer.
#[derive(Debug)]
pub struct IndexOptimizer {
    /// Optimization configuration.
    config: OptimizationConfig,

    /// Merge engine for segment operations.
    merge_engine: MergeEngine,

    /// Storage backend.
    #[allow(dead_code)]
    storage: Arc<dyn Storage>,
}

impl IndexOptimizer {
    /// Create a new index optimizer (schema-less mode).
    pub fn new(config: OptimizationConfig, storage: Arc<dyn Storage>) -> Self {
        let merge_config = MergeConfig {
            max_memory_mb: config.max_memory_mb,
            batch_size: 50000,
            enable_compression: true,
            remove_deleted_docs: true,
            sort_by_doc_id: config.reorder_documents,
            verify_after_merge: true,
        };

        let merge_engine = MergeEngine::new(merge_config, storage.clone());

        IndexOptimizer {
            config,
            merge_engine,
            storage,
        }
    }

    /// Perform full index optimization.
    pub fn optimize_index(
        &self,
        segment_manager: &mut SegmentManager,
        deletion_manager: &mut DeletionManager,
    ) -> Result<OptimizationResult> {
        let start_time = SystemTime::now();
        let mut result = OptimizationResult::default();

        // Collect initial statistics
        let initial_segments = segment_manager.get_segments();
        result.segments_before = initial_segments.len();
        result.size_before = initial_segments.iter().map(|s| s.size_bytes).sum();

        // Perform optimization based on strategy
        let optimization_result = match self.config.strategy {
            OptimizationLevel::Aggressive => {
                self.optimize_aggressive(segment_manager, deletion_manager)?
            }
            OptimizationLevel::Balanced => {
                self.optimize_balanced(segment_manager, deletion_manager)?
            }
            OptimizationLevel::Conservative => {
                self.optimize_conservative(segment_manager, deletion_manager)?
            }
            OptimizationLevel::Custom => self.optimize_custom(segment_manager, deletion_manager)?,
        };

        // Update result with optimization outcomes
        result.merge_results = optimization_result.merge_results;
        result.merge_operations = optimization_result.merge_operations;
        result.deleted_docs_removed = optimization_result.deleted_docs_removed;

        // Collect final statistics
        let final_segments = segment_manager.get_segments();
        result.segments_after = final_segments.len();
        result.size_after = final_segments.iter().map(|s| s.size_bytes).sum();

        // Calculate derived metrics
        if result.size_before > 0 {
            result.space_savings_percent = ((result.size_before - result.size_after) as f64
                / result.size_before as f64)
                * 100.0;
        }

        result.optimization_time_ms = start_time.elapsed().unwrap_or_default().as_millis() as u64;

        result.completed = true;

        Ok(result)
    }

    /// Aggressive optimization - maximum compression and merging.
    fn optimize_aggressive(
        &self,
        segment_manager: &mut SegmentManager,
        _deletion_manager: &mut DeletionManager,
    ) -> Result<OptimizationSubResult> {
        let mut sub_result = OptimizationSubResult::default();

        // Get all segments sorted by deletion ratio (highest first)
        let mut segments = segment_manager.get_segments();
        segments.sort_by(|a, b| b.deletion_ratio().partial_cmp(&a.deletion_ratio()).unwrap());

        // Merge segments with any deletions first
        let segments_with_deletions: Vec<_> = segments
            .iter()
            .filter(|s| s.deletion_ratio() > 0.0)
            .take(self.config.max_merge_segments)
            .cloned()
            .collect();

        if !segments_with_deletions.is_empty() {
            let merge_result = self.merge_segments_with_strategy(
                &segments_with_deletions,
                MergeStrategy::DeletionBased,
                segment_manager,
            )?;

            sub_result.add_merge_result(merge_result);
        }

        // Merge remaining segments to reach target count
        let remaining_segments = segment_manager.get_segments();
        if remaining_segments.len() > self.config.target_segment_count {
            let take_count = self
                .config
                .max_merge_segments
                .min(remaining_segments.len() - self.config.target_segment_count + 1);
            let segments_to_merge: Vec<_> =
                remaining_segments.into_iter().take(take_count).collect();

            if segments_to_merge.len() >= 2 {
                let merge_result = self.merge_segments_with_strategy(
                    &segments_to_merge,
                    MergeStrategy::SizeBased,
                    segment_manager,
                )?;

                sub_result.add_merge_result(merge_result);
            }
        }

        Ok(sub_result)
    }

    /// Balanced optimization - good performance with reasonable resource usage.
    fn optimize_balanced(
        &self,
        segment_manager: &mut SegmentManager,
        _deletion_manager: &mut DeletionManager,
    ) -> Result<OptimizationSubResult> {
        let mut sub_result = OptimizationSubResult::default();

        // Focus on segments with high deletion ratios first
        let segments = segment_manager.get_segments();
        let high_deletion_segments: Vec<_> = segments
            .iter()
            .filter(|s| s.deletion_ratio() >= self.config.compaction_threshold)
            .take(self.config.max_merge_segments / 2)
            .cloned()
            .collect();

        if !high_deletion_segments.is_empty() {
            let merge_result = self.merge_segments_with_strategy(
                &high_deletion_segments,
                MergeStrategy::Balanced,
                segment_manager,
            )?;

            sub_result.add_merge_result(merge_result);
        }

        // Merge smaller segments for efficiency
        let remaining_segments = segment_manager.get_segments();
        if remaining_segments.len() > self.config.target_segment_count * 2 {
            let small_segments: Vec<_> = remaining_segments
                .iter()
                .filter(|s| s.size_bytes < 1024 * 1024) // < 1MB
                .take(self.config.max_merge_segments / 2)
                .cloned()
                .collect();

            if small_segments.len() >= 2 {
                let merge_result = self.merge_segments_with_strategy(
                    &small_segments,
                    MergeStrategy::SizeBased,
                    segment_manager,
                )?;

                sub_result.add_merge_result(merge_result);
            }
        }

        Ok(sub_result)
    }

    /// Conservative optimization - minimal impact on ongoing operations.
    fn optimize_conservative(
        &self,
        segment_manager: &mut SegmentManager,
        _deletion_manager: &mut DeletionManager,
    ) -> Result<OptimizationSubResult> {
        let mut sub_result = OptimizationSubResult::default();

        // Only merge segments with very high deletion ratios
        let segments = segment_manager.get_segments();
        let urgent_segments: Vec<_> = segments
            .iter()
            .filter(|s| s.deletion_ratio() >= 0.5) // 50% or more deletions
            .take(3) // Conservative limit
            .cloned()
            .collect();

        if urgent_segments.len() >= 2 {
            let merge_result = self.merge_segments_with_strategy(
                &urgent_segments,
                MergeStrategy::DeletionBased,
                segment_manager,
            )?;

            sub_result.add_merge_result(merge_result);
        }

        Ok(sub_result)
    }

    /// Custom optimization with specific parameters.
    fn optimize_custom(
        &self,
        segment_manager: &mut SegmentManager,
        deletion_manager: &mut DeletionManager,
    ) -> Result<OptimizationSubResult> {
        // For custom optimization, use balanced approach as baseline
        self.optimize_balanced(segment_manager, deletion_manager)
    }

    /// Merge segments with a specific strategy.
    fn merge_segments_with_strategy(
        &self,
        segments: &[ManagedSegmentInfo],
        strategy: MergeStrategy,
        segment_manager: &mut SegmentManager,
    ) -> Result<MergeResult> {
        if segments.len() < 2 {
            return Err(SarissaError::index("Need at least 2 segments to merge"));
        }

        // Create merge candidate
        let segment_ids: Vec<_> = segments
            .iter()
            .map(|s| s.segment_info.segment_id.clone())
            .collect();

        let estimated_size = segments.iter().map(|s| s.size_bytes).sum();

        let merge_candidate = crate::lexical::index::inverted::segment::manager::MergeCandidate {
            segments: segment_ids.clone(),
            priority: 1.0,
            strategy,
            estimated_size,
        };

        // Generate next generation ID
        let next_generation = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        // Perform merge
        let merge_result =
            self.merge_engine
                .merge_segments(&merge_candidate, segments, next_generation)?;

        // Update segment manager
        segment_manager.complete_merge(
            &segment_ids,
            merge_result.new_segment.segment_info.clone(),
            merge_result.file_paths.clone(),
        )?;

        Ok(merge_result)
    }

    /// Get optimization recommendations.
    pub fn get_optimization_recommendations(
        &self,
        segment_manager: &SegmentManager,
        deletion_manager: &DeletionManager,
    ) -> OptimizationRecommendations {
        let segments = segment_manager.get_segments();
        let global_deletion_state = deletion_manager.get_global_state();

        let mut recommendations = OptimizationRecommendations::default();

        // Analyze segment count
        if segments.len() > self.config.target_segment_count * 2 {
            recommendations.should_merge_segments = true;
            recommendations.priority = RecommendationPriority::High;
            recommendations
                .reasons
                .push("Too many segments - performance may be degraded".to_string());
        }

        // Analyze deletion ratios
        if global_deletion_state.global_deletion_ratio > 0.3 {
            recommendations.should_compact = true;
            recommendations.priority = RecommendationPriority::High;
            recommendations.reasons.push(format!(
                "High global deletion ratio: {:.1}%",
                global_deletion_state.global_deletion_ratio * 100.0
            ));
        }

        // Analyze individual segments
        let high_deletion_segments = segments.iter().filter(|s| s.deletion_ratio() > 0.5).count();

        if high_deletion_segments > 0 {
            recommendations.should_compact = true;
            recommendations.reasons.push(format!(
                "{high_deletion_segments} segments have >50% deletions"
            ));
        }

        // Size-based recommendations (only for non-empty indexes)
        if !segments.is_empty() {
            let total_size: u64 = segments.iter().map(|s| s.size_bytes).sum();
            let avg_segment_size = total_size / segments.len() as u64;

            if avg_segment_size < 512 * 1024 {
                // < 512KB
                recommendations.should_merge_segments = true;
                recommendations.reasons.push(
                    "Small average segment size - merging would improve efficiency".to_string(),
                );
            }
        }

        // Set overall recommendation
        if recommendations.should_compact || recommendations.should_merge_segments {
            recommendations.recommended_strategy = match recommendations.priority {
                RecommendationPriority::High => OptimizationLevel::Aggressive,
                RecommendationPriority::Medium => OptimizationLevel::Balanced,
                RecommendationPriority::Low => OptimizationLevel::Conservative,
            };
        }

        recommendations
    }
}

/// Internal result type for optimization sub-operations.
#[derive(Debug, Default)]
struct OptimizationSubResult {
    merge_results: Vec<MergeResult>,
    merge_operations: u64,
    deleted_docs_removed: u64,
}

impl OptimizationSubResult {
    fn add_merge_result(&mut self, result: MergeResult) {
        self.deleted_docs_removed += result.stats.deleted_docs_removed;
        self.merge_operations += 1;
        self.merge_results.push(result);
    }
}

/// Optimization recommendations.
#[derive(Debug, Default)]
pub struct OptimizationRecommendations {
    /// Whether segments should be merged.
    pub should_merge_segments: bool,

    /// Whether compaction is needed.
    pub should_compact: bool,

    /// Recommended optimization strategy.
    pub recommended_strategy: OptimizationLevel,

    /// Priority level of the recommendation.
    pub priority: RecommendationPriority,

    /// Reasons for the recommendations.
    pub reasons: Vec<String>,

    /// Estimated space savings (percentage).
    pub estimated_space_savings: f64,

    /// Estimated time for optimization (seconds).
    pub estimated_time_secs: u64,
}

/// Priority level for optimization recommendations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RecommendationPriority {
    /// Low priority - optimization would be beneficial but not urgent.
    #[default]
    Low,
    /// Medium priority - optimization recommended for better performance.
    Medium,
    /// High priority - optimization strongly recommended.
    High,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexical::index::inverted::maintenance::deletion::DeletionConfig;
    use crate::lexical::index::inverted::segment::manager::SegmentManagerConfig;

    use crate::storage::memory::MemoryStorage;
    use crate::storage::memory::MemoryStorageConfig;

    #[allow(dead_code)]
    #[test]
    fn test_optimization_config_default() {
        let config = OptimizationConfig::default();

        assert_eq!(config.strategy, OptimizationLevel::Balanced);
        assert_eq!(config.max_merge_segments, 10);
        assert_eq!(config.target_segment_count, 5);
        assert_eq!(config.compaction_threshold, 0.1);
        assert!(config.optimize_dictionaries);
        assert!(config.rebuild_postings);
    }

    #[test]
    fn test_optimization_result_metrics() {
        let mut result = OptimizationResult {
            segments_before: 10,
            segments_after: 5,
            size_before: 1000,
            size_after: 800,
            ..Default::default()
        };

        assert_eq!(result.compression_ratio(), 0.8);
        assert_eq!(result.segment_reduction_ratio(), 0.5);

        result.size_before = 0;
        assert_eq!(result.compression_ratio(), 1.0);
    }

    #[test]
    fn test_index_optimizer_creation() {
        let config = OptimizationConfig::default();

        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        let optimizer = IndexOptimizer::new(config, storage);
        assert_eq!(optimizer.config.strategy, OptimizationLevel::Balanced);
    }

    #[test]
    fn test_optimization_recommendations() {
        let config = OptimizationConfig::default();

        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        let optimizer = IndexOptimizer::new(config, storage.clone());

        let segment_manager =
            SegmentManager::new(SegmentManagerConfig::default(), storage.clone()).unwrap();

        let deletion_manager = DeletionManager::new(DeletionConfig::default(), storage).unwrap();

        let recommendations =
            optimizer.get_optimization_recommendations(&segment_manager, &deletion_manager);

        // Should not recommend optimization for empty index
        assert!(!recommendations.should_merge_segments);
        assert!(!recommendations.should_compact);
    }

    #[test]
    fn test_recommendation_priority() {
        let priorities = [
            RecommendationPriority::Low,
            RecommendationPriority::Medium,
            RecommendationPriority::High,
        ];

        for priority in priorities {
            // Test that priorities can be compared
            assert!(priority == priority);
        }
    }
}
