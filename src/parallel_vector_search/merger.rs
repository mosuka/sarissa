//! Result merging functionality for parallel vector searches.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use serde::{Deserialize, Serialize};

use crate::error::{Result, SageError};
use crate::vector::types::{VectorSearchResult, VectorSearchResults};

/// Strategy for merging search results from multiple sources.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MergeStrategy {
    /// Merge based on similarity scores.
    ScoreBased,
    /// Merge using document frequency weighting.
    DocumentFrequency,
    /// Round-robin merging from all sources.
    RoundRobin,
    /// Weighted combination of results.
    Weighted,
}

/// Statistics for result merging operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultMergeStats {
    /// Total merge operations performed.
    pub total_merges: usize,
    /// Total time spent merging (milliseconds).
    pub total_merge_time_ms: f64,
    /// Average merge time (milliseconds).
    pub avg_merge_time_ms: f64,
    /// Total results merged.
    pub total_results_merged: usize,
    /// Average input sources per merge.
    pub avg_input_sources: f32,
    /// Average output size per merge.
    pub avg_output_size: f32,
}

impl Default for ResultMergeStats {
    fn default() -> Self {
        Self {
            total_merges: 0,
            total_merge_time_ms: 0.0,
            avg_merge_time_ms: 0.0,
            total_results_merged: 0,
            avg_input_sources: 0.0,
            avg_output_size: 0.0,
        }
    }
}

/// Result merger for combining search results from multiple sources.
pub struct VectorResultMerger {
    strategy: MergeStrategy,
    stats: ResultMergeStats,
}

/// Wrapper for VectorSearchResult to implement Ord for BinaryHeap.
#[derive(Debug, Clone)]
struct HeapScoredVector {
    scored_vector: VectorSearchResult,
    _source_id: usize,
}

impl PartialEq for HeapScoredVector {
    fn eq(&self, other: &Self) -> bool {
        self.scored_vector.similarity == other.scored_vector.similarity
    }
}

impl Eq for HeapScoredVector {}

impl PartialOrd for HeapScoredVector {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapScoredVector {
    fn cmp(&self, other: &Self) -> Ordering {
        // For max-heap behavior with highest scores first
        other
            .scored_vector
            .similarity
            .partial_cmp(&self.scored_vector.similarity)
            .unwrap_or(Ordering::Equal)
    }
}

impl VectorResultMerger {
    /// Create a new result merger.
    pub fn new(strategy: MergeStrategy) -> Result<Self> {
        Ok(Self {
            strategy,
            stats: ResultMergeStats::default(),
        })
    }

    /// Merge multiple search results into a single result.
    pub fn merge_results(
        &mut self,
        results: Vec<VectorSearchResults>,
        target_size: usize,
    ) -> Result<VectorSearchResults> {
        if results.is_empty() {
            return Ok(VectorSearchResults {
                results: Vec::new(),
                candidates_examined: 0,
                search_time_ms: 0.0,
                query_metadata: std::collections::HashMap::new(),
            });
        }

        if results.len() == 1 {
            return Ok(results.into_iter().next().unwrap());
        }

        let start_time = std::time::Instant::now();

        // Store original results info for stats before they are moved
        let original_results_count = results.len();
        let original_total_results: usize = results.iter().map(|r| r.results.len()).sum();

        let merged_result = match self.strategy {
            MergeStrategy::ScoreBased => self.merge_score_based(results, target_size)?,
            MergeStrategy::DocumentFrequency => {
                self.merge_document_frequency_based(results, target_size)?
            }
            MergeStrategy::RoundRobin => self.merge_round_robin(results, target_size)?,
            MergeStrategy::Weighted => self.merge_weighted(results, target_size)?,
        };

        // Update statistics (use simplified version since results have been moved)
        self.update_merge_stats_simple(
            original_results_count,
            original_total_results,
            &merged_result,
            start_time.elapsed(),
        );

        Ok(merged_result)
    }

    /// Merge results with custom weights.
    pub fn merge_results_weighted(
        &mut self,
        results: Vec<VectorSearchResults>,
        weights: Vec<f32>,
        target_size: usize,
    ) -> Result<VectorSearchResults> {
        if results.len() != weights.len() {
            return Err(SageError::InvalidOperation(
                "Number of results and weights must match".to_string(),
            ));
        }

        let start_time = std::time::Instant::now();
        let merged_result = self.merge_weighted_with_weights(results, weights, target_size)?;
        self.update_merge_stats(&[], &merged_result, start_time.elapsed());

        Ok(merged_result)
    }

    /// Get merge statistics.
    pub fn stats(&self) -> &ResultMergeStats {
        &self.stats
    }

    /// Merge results based on similarity scores.
    fn merge_score_based(
        &self,
        results: Vec<VectorSearchResults>,
        target_size: usize,
    ) -> Result<VectorSearchResults> {
        let mut heap = BinaryHeap::new();

        // Add all scored vectors to the heap
        for (source_id, result) in results.iter().enumerate() {
            for scored_vector in &result.results {
                heap.push(HeapScoredVector {
                    scored_vector: scored_vector.clone(),
                    _source_id: source_id,
                });
            }
        }

        // Extract top results
        let mut merged_results = Vec::new();
        let mut seen_docs = std::collections::HashSet::new();

        while let Some(heap_item) = heap.pop() {
            // Avoid duplicates by document ID
            if seen_docs.insert(heap_item.scored_vector.doc_id) {
                merged_results.push(heap_item.scored_vector);
                if merged_results.len() >= target_size {
                    break;
                }
            }
        }

        // Aggregate metadata
        let total_candidates_examined: usize = results.iter().map(|r| r.candidates_examined).sum();
        let total_search_time_ms: f64 = results.iter().map(|r| r.search_time_ms).sum();

        Ok(VectorSearchResults {
            results: merged_results,
            candidates_examined: total_candidates_examined,
            search_time_ms: total_search_time_ms,
            query_metadata: std::collections::HashMap::new(),
        })
    }

    /// Merge results using document frequency weighting.
    fn merge_document_frequency_based(
        &self,
        results: Vec<VectorSearchResults>,
        target_size: usize,
    ) -> Result<VectorSearchResults> {
        // Count how many times each document appears across results
        let mut doc_frequency = std::collections::HashMap::new();
        let mut doc_scores = std::collections::HashMap::new();

        for result in &results {
            for scored_vector in &result.results {
                let doc_id = scored_vector.doc_id;
                *doc_frequency.entry(doc_id).or_insert(0) += 1;

                // Keep the best score for each document
                let current_score = doc_scores.get(&doc_id).copied().unwrap_or(0.0);
                if scored_vector.similarity > current_score {
                    doc_scores.insert(doc_id, scored_vector.similarity);
                }
            }
        }

        // Create weighted scores (frequency * original score)
        let mut weighted_results = Vec::new();
        for (doc_id, frequency) in doc_frequency {
            if let Some(score) = doc_scores.get(&doc_id) {
                let weighted_score = score * (frequency as f32).sqrt(); // Square root to dampen frequency effect

                // Find the original ScoredVector
                for result in &results {
                    if let Some(scored_vector) =
                        result.results.iter().find(|sv| sv.doc_id == doc_id)
                    {
                        let mut weighted_vector = scored_vector.clone();
                        weighted_vector.similarity = weighted_score;
                        weighted_results.push(weighted_vector);
                        break;
                    }
                }
            }
        }

        // Sort by weighted score and take top results
        weighted_results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(Ordering::Equal)
        });
        weighted_results.truncate(target_size);

        let total_candidates_examined: usize = results.iter().map(|r| r.candidates_examined).sum();
        let total_search_time_ms: f64 = results.iter().map(|r| r.search_time_ms).sum();

        Ok(VectorSearchResults {
            results: weighted_results,
            candidates_examined: total_candidates_examined,
            search_time_ms: total_search_time_ms,
            query_metadata: std::collections::HashMap::new(),
        })
    }

    /// Merge results using round-robin strategy.
    fn merge_round_robin(
        &self,
        results: Vec<VectorSearchResults>,
        target_size: usize,
    ) -> Result<VectorSearchResults> {
        let mut merged_results = Vec::new();
        let mut indices = vec![0; results.len()];
        let mut seen_docs = std::collections::HashSet::new();

        while merged_results.len() < target_size {
            let mut added_any = false;

            for (source_idx, result) in results.iter().enumerate() {
                if indices[source_idx] < result.results.len() {
                    let scored_vector = &result.results[indices[source_idx]];

                    // Avoid duplicates
                    if seen_docs.insert(scored_vector.doc_id) {
                        merged_results.push(scored_vector.clone());
                        if merged_results.len() >= target_size {
                            break;
                        }
                    }

                    indices[source_idx] += 1;
                    added_any = true;
                }
            }

            if !added_any {
                break; // No more results to add
            }
        }

        let total_candidates_examined: usize = results.iter().map(|r| r.candidates_examined).sum();
        let total_search_time_ms: f64 = results.iter().map(|r| r.search_time_ms).sum();

        Ok(VectorSearchResults {
            results: merged_results,
            candidates_examined: total_candidates_examined,
            search_time_ms: total_search_time_ms,
            query_metadata: std::collections::HashMap::new(),
        })
    }

    /// Merge results using weighted combination.
    fn merge_weighted(
        &self,
        results: Vec<VectorSearchResults>,
        target_size: usize,
    ) -> Result<VectorSearchResults> {
        // Default equal weights
        let weights = vec![1.0; results.len()];
        self.merge_weighted_with_weights(results, weights, target_size)
    }

    /// Merge results with specific weights.
    fn merge_weighted_with_weights(
        &self,
        results: Vec<VectorSearchResults>,
        weights: Vec<f32>,
        target_size: usize,
    ) -> Result<VectorSearchResults> {
        let mut weighted_heap = BinaryHeap::new();

        // Apply weights to scores and add to heap
        for (source_id, (result, weight)) in results.iter().zip(weights.iter()).enumerate() {
            for scored_vector in &result.results {
                let mut weighted_vector = scored_vector.clone();
                weighted_vector.similarity *= weight;

                weighted_heap.push(HeapScoredVector {
                    scored_vector: weighted_vector,
                    _source_id: source_id,
                });
            }
        }

        // Extract top weighted results
        let mut merged_results = Vec::new();
        let mut seen_docs = std::collections::HashSet::new();

        while let Some(heap_item) = weighted_heap.pop() {
            if seen_docs.insert(heap_item.scored_vector.doc_id) {
                merged_results.push(heap_item.scored_vector);
                if merged_results.len() >= target_size {
                    break;
                }
            }
        }

        let total_candidates_examined: usize = results.iter().map(|r| r.candidates_examined).sum();
        let total_search_time_ms: f64 = results.iter().map(|r| r.search_time_ms).sum();

        Ok(VectorSearchResults {
            results: merged_results,
            candidates_examined: total_candidates_examined,
            search_time_ms: total_search_time_ms,
            query_metadata: std::collections::HashMap::new(),
        })
    }

    /// Update merge statistics.
    fn update_merge_stats(
        &mut self,
        input_results: &[VectorSearchResults],
        merged_result: &VectorSearchResults,
        duration: std::time::Duration,
    ) {
        let merge_time_ms = duration.as_secs_f64() * 1000.0;
        let total_input_results: usize = input_results.iter().map(|r| r.results.len()).sum();

        self.stats.total_merges += 1;
        self.stats.total_merge_time_ms += merge_time_ms;
        self.stats.avg_merge_time_ms =
            self.stats.total_merge_time_ms / self.stats.total_merges as f64;
        self.stats.total_results_merged += total_input_results;

        let total_merges = self.stats.total_merges as f32;
        self.stats.avg_input_sources = (self.stats.avg_input_sources * (total_merges - 1.0)
            + input_results.len() as f32)
            / total_merges;
        self.stats.avg_output_size = (self.stats.avg_output_size * (total_merges - 1.0)
            + merged_result.results.len() as f32)
            / total_merges;
    }

    /// Update merge statistics (simplified version when input results are not available).
    fn update_merge_stats_simple(
        &mut self,
        input_count: usize,
        total_input_results: usize,
        merged_result: &VectorSearchResults,
        duration: std::time::Duration,
    ) {
        let merge_time_ms = duration.as_secs_f64() * 1000.0;

        self.stats.total_merges += 1;
        self.stats.total_merge_time_ms += merge_time_ms;
        self.stats.avg_merge_time_ms =
            self.stats.total_merge_time_ms / self.stats.total_merges as f64;
        self.stats.total_results_merged += total_input_results;

        let total_merges = self.stats.total_merges as f32;
        self.stats.avg_input_sources = (self.stats.avg_input_sources * (total_merges - 1.0)
            + input_count as f32)
            / total_merges;
        self.stats.avg_output_size = (self.stats.avg_output_size * (total_merges - 1.0)
            + merged_result.results.len() as f32)
            / total_merges;
    }
}
