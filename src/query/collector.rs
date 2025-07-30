//! Collector implementations for gathering search results.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fmt::Debug;

use crate::error::Result;
use crate::query::SearchHit;

/// Trait for collecting search results.
pub trait Collector: Send + Debug {
    /// Collect a document hit.
    fn collect(&mut self, doc_id: u64, score: f32) -> Result<()>;

    /// Get the final results.
    fn results(&self) -> Vec<SearchHit>;

    /// Get the total number of hits collected.
    fn total_hits(&self) -> u64;

    /// Check if this collector needs more results.
    fn needs_more(&self) -> bool;

    /// Get the minimum score threshold.
    fn min_score(&self) -> f32;

    /// Reset the collector for a new search.
    fn reset(&mut self);
}

/// A collector that keeps the top N documents by score.
#[derive(Debug)]
pub struct TopDocsCollector {
    /// Maximum number of documents to collect.
    max_docs: usize,
    /// Minimum score threshold.
    min_score: f32,
    /// Collected hits (min-heap based on score).
    hits: BinaryHeap<ScoredDoc>,
    /// Total number of documents processed.
    total_hits: u64,
}

/// A scored document for use in the heap.
#[derive(Debug, Clone)]
struct ScoredDoc {
    doc_id: u64,
    score: f32,
}

impl PartialEq for ScoredDoc {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.doc_id == other.doc_id
    }
}

impl Eq for ScoredDoc {}

impl PartialOrd for ScoredDoc {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredDoc {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: lower scores come first
        other
            .score
            .partial_cmp(&self.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.doc_id.cmp(&self.doc_id))
    }
}

impl TopDocsCollector {
    /// Create a new top docs collector.
    pub fn new(max_docs: usize) -> Self {
        TopDocsCollector {
            max_docs,
            min_score: 0.0,
            hits: BinaryHeap::new(),
            total_hits: 0,
        }
    }

    /// Create a new top docs collector with minimum score threshold.
    pub fn with_min_score(max_docs: usize, min_score: f32) -> Self {
        TopDocsCollector {
            max_docs,
            min_score,
            hits: BinaryHeap::new(),
            total_hits: 0,
        }
    }

    /// Get the maximum number of documents to collect.
    pub fn max_docs(&self) -> usize {
        self.max_docs
    }

    /// Get the current minimum score in the collection.
    pub fn current_min_score(&self) -> f32 {
        if self.hits.len() < self.max_docs {
            self.min_score
        } else {
            self.hits
                .peek()
                .map(|doc| doc.score)
                .unwrap_or(self.min_score)
        }
    }
}

impl Collector for TopDocsCollector {
    fn collect(&mut self, doc_id: u64, score: f32) -> Result<()> {
        self.total_hits += 1;

        // Check minimum score threshold
        if score < self.min_score {
            return Ok(());
        }

        let scored_doc = ScoredDoc { doc_id, score };

        if self.hits.len() < self.max_docs {
            // We have space, just add it
            self.hits.push(scored_doc);
        } else {
            // Check if this score is better than the worst score
            if let Some(worst) = self.hits.peek() {
                if score > worst.score {
                    // Replace the worst document
                    self.hits.pop();
                    self.hits.push(scored_doc);
                }
            }
        }

        Ok(())
    }

    fn results(&self) -> Vec<SearchHit> {
        let mut results: Vec<_> = self
            .hits
            .iter()
            .map(|doc| SearchHit {
                doc_id: doc.doc_id,
                score: doc.score,
                document: None,
            })
            .collect();

        // Sort by score descending
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));

        results
    }

    fn total_hits(&self) -> u64 {
        self.total_hits
    }

    fn needs_more(&self) -> bool {
        self.hits.len() < self.max_docs
    }

    fn min_score(&self) -> f32 {
        self.current_min_score()
    }

    fn reset(&mut self) {
        self.hits.clear();
        self.total_hits = 0;
    }
}

/// A collector that just counts the number of matching documents.
#[derive(Debug)]
pub struct CountCollector {
    /// Total number of documents that matched.
    count: u64,
    /// Minimum score threshold.
    min_score: f32,
}

impl CountCollector {
    /// Create a new count collector.
    pub fn new() -> Self {
        CountCollector {
            count: 0,
            min_score: 0.0,
        }
    }

    /// Create a new count collector with minimum score threshold.
    pub fn with_min_score(min_score: f32) -> Self {
        CountCollector {
            count: 0,
            min_score,
        }
    }

    /// Get the current count.
    pub fn count(&self) -> u64 {
        self.count
    }
}

impl Default for CountCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl Collector for CountCollector {
    fn collect(&mut self, _doc_id: u64, score: f32) -> Result<()> {
        if score >= self.min_score {
            self.count += 1;
        }
        Ok(())
    }

    fn results(&self) -> Vec<SearchHit> {
        // Count collector doesn't return actual documents
        Vec::new()
    }

    fn total_hits(&self) -> u64 {
        self.count
    }

    fn needs_more(&self) -> bool {
        // Count collector always needs more to get the full count
        true
    }

    fn min_score(&self) -> f32 {
        self.min_score
    }

    fn reset(&mut self) {
        self.count = 0;
    }
}

/// A collector that collects all matching documents.
#[derive(Debug)]
pub struct AllDocsCollector {
    /// All collected hits.
    hits: Vec<SearchHit>,
    /// Minimum score threshold.
    min_score: f32,
}

impl AllDocsCollector {
    /// Create a new all docs collector.
    pub fn new() -> Self {
        AllDocsCollector {
            hits: Vec::new(),
            min_score: 0.0,
        }
    }

    /// Create a new all docs collector with minimum score threshold.
    pub fn with_min_score(min_score: f32) -> Self {
        AllDocsCollector {
            hits: Vec::new(),
            min_score,
        }
    }
}

impl Default for AllDocsCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl Collector for AllDocsCollector {
    fn collect(&mut self, doc_id: u64, score: f32) -> Result<()> {
        if score >= self.min_score {
            self.hits.push(SearchHit {
                doc_id,
                score,
                document: None,
            });
        }
        Ok(())
    }

    fn results(&self) -> Vec<SearchHit> {
        let mut results = self.hits.clone();
        // Sort by score descending
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        results
    }

    fn total_hits(&self) -> u64 {
        self.hits.len() as u64
    }

    fn needs_more(&self) -> bool {
        // All docs collector always needs more
        true
    }

    fn min_score(&self) -> f32 {
        self.min_score
    }

    fn reset(&mut self) {
        self.hits.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_top_docs_collector() {
        let mut collector = TopDocsCollector::new(3);

        assert_eq!(collector.max_docs(), 3);
        assert_eq!(collector.total_hits(), 0);
        assert!(collector.needs_more());

        // Add some documents
        collector.collect(1, 0.5).unwrap();
        collector.collect(2, 0.8).unwrap();
        collector.collect(3, 0.3).unwrap();

        assert_eq!(collector.total_hits(), 3);
        assert!(!collector.needs_more());

        // Add a better document - should replace the worst
        collector.collect(4, 0.9).unwrap();

        assert_eq!(collector.total_hits(), 4);

        let results = collector.results();
        assert_eq!(results.len(), 3);

        // Results should be sorted by score descending
        assert!(results[0].score >= results[1].score);
        assert!(results[1].score >= results[2].score);

        // The best document should be first
        assert_eq!(results[0].doc_id, 4);
        assert_eq!(results[0].score, 0.9);
    }

    #[test]
    fn test_top_docs_collector_with_min_score() {
        let mut collector = TopDocsCollector::with_min_score(3, 0.5);

        assert_eq!(collector.min_score(), 0.5);

        // Add documents, some below threshold
        collector.collect(1, 0.3).unwrap(); // Below threshold
        collector.collect(2, 0.8).unwrap(); // Above threshold
        collector.collect(3, 0.6).unwrap(); // Above threshold

        assert_eq!(collector.total_hits(), 3);

        let results = collector.results();
        assert_eq!(results.len(), 2); // Only 2 above threshold

        // Check that low-score document was filtered out
        assert!(!results.iter().any(|hit| hit.score == 0.3));
    }

    #[test]
    fn test_count_collector() {
        let mut collector = CountCollector::new();

        assert_eq!(collector.count(), 0);
        assert_eq!(collector.total_hits(), 0);
        assert!(collector.needs_more());

        // Add some documents
        collector.collect(1, 0.5).unwrap();
        collector.collect(2, 0.8).unwrap();
        collector.collect(3, 0.3).unwrap();

        assert_eq!(collector.count(), 3);
        assert_eq!(collector.total_hits(), 3);

        // Results should be empty for count collector
        let results = collector.results();
        assert!(results.is_empty());
    }

    #[test]
    fn test_count_collector_with_min_score() {
        let mut collector = CountCollector::with_min_score(0.5);

        // Add documents, some below threshold
        collector.collect(1, 0.3).unwrap(); // Below threshold
        collector.collect(2, 0.8).unwrap(); // Above threshold
        collector.collect(3, 0.6).unwrap(); // Above threshold

        assert_eq!(collector.count(), 2); // Only 2 above threshold
        assert_eq!(collector.total_hits(), 2);
    }

    #[test]
    fn test_all_docs_collector() {
        let mut collector = AllDocsCollector::new();

        assert_eq!(collector.total_hits(), 0);
        assert!(collector.needs_more());

        // Add some documents
        collector.collect(1, 0.5).unwrap();
        collector.collect(2, 0.8).unwrap();
        collector.collect(3, 0.3).unwrap();

        assert_eq!(collector.total_hits(), 3);

        let results = collector.results();
        assert_eq!(results.len(), 3);

        // Results should be sorted by score descending
        assert!(results[0].score >= results[1].score);
        assert!(results[1].score >= results[2].score);

        // Check specific order
        assert_eq!(results[0].doc_id, 2); // score 0.8
        assert_eq!(results[1].doc_id, 1); // score 0.5
        assert_eq!(results[2].doc_id, 3); // score 0.3
    }

    #[test]
    fn test_all_docs_collector_with_min_score() {
        let mut collector = AllDocsCollector::with_min_score(0.5);

        // Add documents, some below threshold
        collector.collect(1, 0.3).unwrap(); // Below threshold
        collector.collect(2, 0.8).unwrap(); // Above threshold
        collector.collect(3, 0.6).unwrap(); // Above threshold

        assert_eq!(collector.total_hits(), 2); // Only 2 above threshold

        let results = collector.results();
        assert_eq!(results.len(), 2);

        // Check that low-score document was filtered out
        assert!(!results.iter().any(|hit| hit.score == 0.3));
    }

    #[test]
    fn test_collector_reset() {
        let mut collector = TopDocsCollector::new(3);

        // Add some documents
        collector.collect(1, 0.5).unwrap();
        collector.collect(2, 0.8).unwrap();

        assert_eq!(collector.total_hits(), 2);
        assert_eq!(collector.results().len(), 2);

        // Reset collector
        collector.reset();

        assert_eq!(collector.total_hits(), 0);
        assert_eq!(collector.results().len(), 0);
        assert!(collector.needs_more());
    }

    #[test]
    fn test_scored_doc_ordering() {
        let doc1 = ScoredDoc {
            doc_id: 1,
            score: 0.5,
        };
        let doc2 = ScoredDoc {
            doc_id: 2,
            score: 0.8,
        };
        let doc3 = ScoredDoc {
            doc_id: 3,
            score: 0.8,
        };

        // Higher score should be "less" (for min-heap)
        assert!(doc2 < doc1);

        // Same score should compare by doc_id
        assert!(doc3 < doc2);

        // Test in heap
        let mut heap = BinaryHeap::new();
        heap.push(doc1);
        heap.push(doc2);
        heap.push(doc3);

        // Min-heap: lowest score should be at top
        assert_eq!(heap.peek().unwrap().score, 0.5);
    }
}
