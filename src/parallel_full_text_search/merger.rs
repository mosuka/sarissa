//! Result merging strategies for parallel search.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::error::Result;
use crate::parallel_full_text_search::search_task::TaskResult;
use crate::query::{SearchHit, SearchResults};

/// Trait for merging search results from multiple indices.
pub trait MergeStrategy: Send + Sync {
    /// Merge multiple search results into a single result set.
    fn merge(
        &self,
        results: Vec<TaskResult>,
        max_docs: usize,
        min_score: Option<f32>,
    ) -> Result<SearchResults>;

    /// Get the name of this merge strategy.
    fn name(&self) -> &str;
}

/// Score-based merger that combines results by document score.
#[derive(Debug, Default)]
pub struct ScoreBasedMerger {
    /// Whether to normalize scores across indices.
    normalize_scores: bool,
}

impl ScoreBasedMerger {
    /// Create a new score-based merger.
    pub fn new() -> Self {
        Self {
            normalize_scores: false,
        }
    }

    /// Enable score normalization.
    pub fn with_normalization(mut self) -> Self {
        self.normalize_scores = true;
        self
    }
}

impl MergeStrategy for ScoreBasedMerger {
    fn merge(
        &self,
        results: Vec<TaskResult>,
        max_docs: usize,
        min_score: Option<f32>,
    ) -> Result<SearchResults> {
        // Use a max heap to efficiently get top documents
        let mut heap = BinaryHeap::new();
        let mut total_hits = 0u64;
        let mut max_score = 0.0f32;

        // Process each result
        for task_result in results {
            if let Some(search_results) = task_result.results {
                total_hits += search_results.total_hits;

                // Track global max score
                if search_results.max_score > max_score {
                    max_score = search_results.max_score;
                }

                // Add hits to heap
                for hit in search_results.hits {
                    // Apply min score filter if specified
                    if let Some(min) = min_score {
                        if hit.score < min {
                            continue;
                        }
                    }

                    // Wrap hit for heap ordering
                    let scored_hit = ScoredHit { hit };
                    heap.push(scored_hit);
                }
            }
        }

        // Extract top documents
        let mut hits = Vec::with_capacity(max_docs.min(heap.len()));
        let mut seen_docs = HashSet::new();

        while hits.len() < max_docs && !heap.is_empty() {
            let scored_hit = heap.pop().unwrap();

            // Skip duplicates (same doc_id from different indices)
            if seen_docs.insert(scored_hit.hit.doc_id) {
                hits.push(scored_hit.hit);
            }
        }

        Ok(SearchResults {
            hits,
            total_hits,
            max_score,
        })
    }

    fn name(&self) -> &str {
        "ScoreBasedMerger"
    }
}

/// Weighted merger that applies index weights to scores.
#[derive(Debug)]
pub struct WeightedMerger {
    /// Map of index ID to weight.
    index_weights: HashMap<String, f32>,
}

impl WeightedMerger {
    /// Create a new weighted merger.
    pub fn new() -> Self {
        Self {
            index_weights: HashMap::new(),
        }
    }

    /// Set weight for an index.
    pub fn with_weight(mut self, index_id: String, weight: f32) -> Self {
        self.index_weights.insert(index_id, weight);
        self
    }

    /// Set weights from a map.
    pub fn with_weights(mut self, weights: HashMap<String, f32>) -> Self {
        self.index_weights = weights;
        self
    }
}

impl MergeStrategy for WeightedMerger {
    fn merge(
        &self,
        results: Vec<TaskResult>,
        max_docs: usize,
        min_score: Option<f32>,
    ) -> Result<SearchResults> {
        let mut heap = BinaryHeap::new();
        let mut total_hits = 0u64;
        let mut max_score = 0.0f32;

        for task_result in results {
            if let Some(search_results) = task_result.results {
                total_hits += search_results.total_hits;

                // Get weight for this index
                let weight = self
                    .index_weights
                    .get(&task_result.index_id)
                    .copied()
                    .unwrap_or(1.0);

                for mut hit in search_results.hits {
                    // Apply weight to score
                    hit.score *= weight;

                    // Track max weighted score
                    if hit.score > max_score {
                        max_score = hit.score;
                    }

                    // Apply min score filter
                    if let Some(min) = min_score {
                        if hit.score < min {
                            continue;
                        }
                    }

                    let scored_hit = ScoredHit { hit };
                    heap.push(scored_hit);
                }
            }
        }

        // Extract top documents
        let mut hits = Vec::with_capacity(max_docs.min(heap.len()));
        let mut seen_docs = HashSet::new();

        while hits.len() < max_docs && !heap.is_empty() {
            let scored_hit = heap.pop().unwrap();
            if seen_docs.insert(scored_hit.hit.doc_id) {
                hits.push(scored_hit.hit);
            }
        }

        Ok(SearchResults {
            hits,
            total_hits,
            max_score,
        })
    }

    fn name(&self) -> &str {
        "WeightedMerger"
    }
}

impl Default for WeightedMerger {
    fn default() -> Self {
        Self::new()
    }
}

/// Round-robin merger for result diversity.
#[derive(Debug, Default)]
pub struct RoundRobinMerger {
    /// Whether to apply score threshold.
    use_score_threshold: bool,
}

impl RoundRobinMerger {
    /// Create a new round-robin merger.
    pub fn new() -> Self {
        Self {
            use_score_threshold: true,
        }
    }

    /// Disable score threshold checking.
    pub fn without_score_threshold(mut self) -> Self {
        self.use_score_threshold = false;
        self
    }
}

impl MergeStrategy for RoundRobinMerger {
    fn merge(
        &self,
        results: Vec<TaskResult>,
        max_docs: usize,
        min_score: Option<f32>,
    ) -> Result<SearchResults> {
        // Prepare iterators for each result set
        let mut iterators: Vec<_> = results
            .into_iter()
            .filter_map(|r| r.results.map(|res| res.hits.into_iter()))
            .collect();

        let mut merged_hits = Vec::with_capacity(max_docs);
        let mut seen_docs = HashSet::new();
        let mut total_hits = 0u64;
        let mut max_score = 0.0f32;

        // Round-robin through results
        let mut any_remaining = true;
        while merged_hits.len() < max_docs && any_remaining {
            any_remaining = false;

            for iter in &mut iterators {
                if let Some(hit) = iter.next() {
                    any_remaining = true;

                    // Update max score
                    if hit.score > max_score {
                        max_score = hit.score;
                    }

                    // Apply score filter
                    if self.use_score_threshold {
                        if let Some(min) = min_score {
                            if hit.score < min {
                                continue;
                            }
                        }
                    }

                    // Skip duplicates
                    if seen_docs.insert(hit.doc_id) {
                        merged_hits.push(hit);
                        if merged_hits.len() >= max_docs {
                            break;
                        }
                    }
                }
            }
        }

        // Count remaining hits for total
        for iter in iterators {
            total_hits += iter.count() as u64;
        }
        total_hits += merged_hits.len() as u64;

        Ok(SearchResults {
            hits: merged_hits,
            total_hits,
            max_score,
        })
    }

    fn name(&self) -> &str {
        "RoundRobinMerger"
    }
}

/// Helper struct for heap-based merging.
struct ScoredHit {
    hit: SearchHit,
}

impl Ord for ScoredHit {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order for max heap
        self.hit
            .score
            .partial_cmp(&other.hit.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.hit.doc_id.cmp(&self.hit.doc_id))
    }
}

impl PartialOrd for ScoredHit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for ScoredHit {}

impl PartialEq for ScoredHit {
    fn eq(&self, other: &Self) -> bool {
        self.hit.score == other.hit.score && self.hit.doc_id == other.hit.doc_id
    }
}

/// Factory for creating merge strategies.
pub struct MergerFactory;

impl MergerFactory {
    /// Create a merger based on the strategy type.
    pub fn create(
        strategy: crate::parallel_full_text_search::config::MergeStrategyType,
    ) -> Box<dyn MergeStrategy> {
        match strategy {
            crate::parallel_full_text_search::config::MergeStrategyType::ScoreBased => {
                Box::new(ScoreBasedMerger::new())
            }
            crate::parallel_full_text_search::config::MergeStrategyType::RoundRobin => {
                Box::new(RoundRobinMerger::new())
            }
            crate::parallel_full_text_search::config::MergeStrategyType::Weighted => {
                Box::new(WeightedMerger::new())
            }
            crate::parallel_full_text_search::config::MergeStrategyType::Custom => {
                // Default to score-based for custom
                Box::new(ScoreBasedMerger::new())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::Document;

    fn create_test_results() -> Vec<TaskResult> {
        vec![
            TaskResult::success(
                "task1".to_string(),
                "index1".to_string(),
                SearchResults {
                    hits: vec![
                        SearchHit {
                            doc_id: 1,
                            score: 0.9,
                            document: Some(Document::new()),
                        },
                        SearchHit {
                            doc_id: 2,
                            score: 0.7,
                            document: Some(Document::new()),
                        },
                    ],
                    total_hits: 2,
                    max_score: 0.9,
                },
                std::time::Duration::from_millis(10),
            ),
            TaskResult::success(
                "task2".to_string(),
                "index2".to_string(),
                SearchResults {
                    hits: vec![
                        SearchHit {
                            doc_id: 3,
                            score: 0.8,
                            document: Some(Document::new()),
                        },
                        SearchHit {
                            doc_id: 1, // Duplicate
                            score: 0.85,
                            document: Some(Document::new()),
                        },
                    ],
                    total_hits: 2,
                    max_score: 0.85,
                },
                std::time::Duration::from_millis(15),
            ),
        ]
    }

    #[test]
    fn test_score_based_merger() {
        let merger = ScoreBasedMerger::new();
        let results = create_test_results();

        let merged = merger.merge(results, 10, None).unwrap();

        // Should have 3 unique documents (doc_id 1 is deduplicated)
        assert_eq!(merged.hits.len(), 3);
        assert_eq!(merged.total_hits, 4);
        assert_eq!(merged.max_score, 0.9);

        // Should be sorted by score descending
        assert_eq!(merged.hits[0].doc_id, 1); // score 0.9
        assert_eq!(merged.hits[1].doc_id, 3); // score 0.8
        assert_eq!(merged.hits[2].doc_id, 2); // score 0.7
    }

    #[test]
    fn test_weighted_merger() {
        let mut merger = WeightedMerger::new();
        merger.index_weights.insert("index1".to_string(), 1.0);
        merger.index_weights.insert("index2".to_string(), 2.0);

        let results = create_test_results();
        let merged = merger.merge(results, 10, None).unwrap();

        // Scores should be weighted
        // index2 docs should have higher scores due to 2.0 weight
        assert_eq!(merged.hits.len(), 3);
        assert!(merged.hits[0].score > 1.0); // Weighted score
    }

    #[test]
    fn test_round_robin_merger() {
        let merger = RoundRobinMerger::new();
        let results = create_test_results();

        let merged = merger.merge(results, 10, None).unwrap();

        // Should alternate between indices
        assert_eq!(merged.hits.len(), 3);
        assert_eq!(merged.total_hits, 3); // After deduplication
    }

    #[test]
    fn test_min_score_filter() {
        let merger = ScoreBasedMerger::new();
        let results = create_test_results();

        let merged = merger.merge(results, 10, Some(0.8)).unwrap();

        // Only docs with score >= 0.8 should be included
        assert_eq!(merged.hits.len(), 2);
        assert!(merged.hits.iter().all(|h| h.score >= 0.8));
    }
}
