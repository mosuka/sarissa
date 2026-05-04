//! Collector implementations for gathering search results.

use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fmt::Debug;

use crate::error::Result;
use crate::lexical::query::SearchHit;

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

    /// The current "competitive" score floor — a candidate whose
    /// upper-bound score is at or below this value cannot enter the
    /// final result set, so the searcher is free to skip it.
    ///
    /// Top-K collectors override this to expose the K-th heap score
    /// once the heap is full, enabling MaxScore / WAND-style early
    /// termination (#403). The default returns `f32::NEG_INFINITY`,
    /// which disables the optimisation for collectors that do not have
    /// a meaningful upper-bound concept (count-only, all-docs, etc.).
    ///
    /// # Returns
    ///
    /// The score that an incoming candidate's upper bound must exceed
    /// to be worth collecting.
    fn min_competitive(&self) -> f32 {
        f32::NEG_INFINITY
    }
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

/// A document with field value for field-based sorting.
#[derive(Debug, Clone)]
struct FieldScoredDoc {
    doc_id: u64,
    score: f32,
    field_value: crate::lexical::core::field::FieldValue,
    ascending: bool,
}

/// A collector that keeps the top N documents sorted by a field value.
/// This performs sorting during collection (Lucene-style) rather than after.
#[derive(Debug)]
pub struct TopFieldCollector<'a> {
    /// Maximum number of documents to collect.
    max_docs: usize,
    /// Minimum score threshold.
    min_score: f32,
    /// Field name to sort by.
    field_name: String,
    /// Sort order (true for ascending, false for descending).
    ascending: bool,
    /// Collected hits (min-heap for ascending, needs reverse comparison).
    hits: BinaryHeap<FieldScoredDoc>,
    /// Total number of documents processed.
    total_hits: u64,
    /// Reference to the index reader for accessing field values.
    reader: &'a dyn crate::lexical::reader::LexicalIndexReader,
}

impl<'a> TopFieldCollector<'a> {
    /// Create a new top field collector.
    pub fn new(
        max_docs: usize,
        field_name: String,
        ascending: bool,
        reader: &'a dyn crate::lexical::reader::LexicalIndexReader,
    ) -> Self {
        TopFieldCollector {
            max_docs,
            min_score: 0.0,
            field_name,
            ascending,
            hits: BinaryHeap::new(),
            total_hits: 0,
            reader,
        }
    }

    /// Create a new top field collector with minimum score threshold.
    pub fn with_min_score(
        max_docs: usize,
        min_score: f32,
        field_name: String,
        ascending: bool,
        reader: &'a dyn crate::lexical::reader::LexicalIndexReader,
    ) -> Self {
        TopFieldCollector {
            max_docs,
            min_score,
            field_name,
            ascending,
            hits: BinaryHeap::new(),
            total_hits: 0,
            reader,
        }
    }

    /// Get the field value for a document using DocValues.
    /// DocValues provide efficient column-oriented storage for field values.
    fn get_field_value(&self, doc_id: u64) -> crate::lexical::core::field::FieldValue {
        // Get field value from DocValues (efficient column-oriented storage)
        if let Ok(Some(value)) = self.reader.get_doc_value(&self.field_name, doc_id) {
            value
        } else {
            // Return Null if field not found
            crate::lexical::core::field::FieldValue::Null
        }
    }

    /// Compare two field values based on sort order.
    fn compare_for_heap(&self, a: &FieldScoredDoc, b: &FieldScoredDoc) -> Ordering {
        if self.ascending {
            // For ascending: heap comparison is already reversed (min-heap behavior)
            a.cmp(b)
        } else {
            // For descending: reverse the comparison
            b.cmp(a)
        }
    }

    /// Check if a new document should be collected based on field value.
    fn should_collect(&self, new_doc: &FieldScoredDoc) -> bool {
        if self.hits.len() < self.max_docs {
            return true;
        }

        if let Some(worst) = self.hits.peek() {
            self.compare_for_heap(new_doc, worst) == Ordering::Less
        } else {
            true
        }
    }
}

impl<'a> Collector for TopFieldCollector<'a> {
    fn collect(&mut self, doc_id: u64, score: f32) -> Result<()> {
        self.total_hits += 1;

        // Check minimum score threshold
        if score < self.min_score {
            return Ok(());
        }

        // Get field value during collection (Lucene-style)
        let field_value = self.get_field_value(doc_id);

        let scored_doc = FieldScoredDoc {
            doc_id,
            score,
            field_value,
            ascending: self.ascending,
        };

        if self.hits.len() < self.max_docs {
            // We have space, just add it
            self.hits.push(scored_doc);
        } else {
            // Check if this document should replace the worst one
            if self.should_collect(&scored_doc) {
                self.hits.pop();
                self.hits.push(scored_doc);
            }
        }

        Ok(())
    }

    fn results(&self) -> Vec<SearchHit> {
        // Convert heap to vector and sort by field values
        let mut sorted_docs: Vec<_> = self.hits.iter().cloned().collect();

        // Sort based on the sort order
        use crate::lexical::core::field::FieldValue;
        if self.ascending {
            // Ascending: compare field values directly
            sorted_docs.sort_unstable_by(|a, b| match (&a.field_value, &b.field_value) {
                (FieldValue::Text(av), FieldValue::Text(bv)) => av.cmp(bv),
                (FieldValue::Int64(av), FieldValue::Int64(bv)) => av.cmp(bv),
                (FieldValue::Float64(av), FieldValue::Float64(bv)) => {
                    av.partial_cmp(bv).unwrap_or(Ordering::Equal)
                }
                (FieldValue::Bool(av), FieldValue::Bool(bv)) => av.cmp(bv),
                (FieldValue::DateTime(av), FieldValue::DateTime(bv)) => av.cmp(bv),
                (FieldValue::Geo(a), FieldValue::Geo(b)) => {
                    let lat_cmp = a.lat.partial_cmp(&b.lat).unwrap_or(Ordering::Equal);
                    if lat_cmp != Ordering::Equal {
                        lat_cmp
                    } else {
                        a.lon.partial_cmp(&b.lon).unwrap_or(Ordering::Equal)
                    }
                }
                (FieldValue::Bytes(av, _), FieldValue::Bytes(bv, _)) => av.cmp(bv),
                (FieldValue::Null, FieldValue::Null) => Ordering::Equal,
                (FieldValue::Null, _) => Ordering::Greater,
                (_, FieldValue::Null) => Ordering::Less,
                _ => Ordering::Equal,
            });
        } else {
            // Descending: reverse comparison
            sorted_docs.sort_unstable_by(|a, b| match (&a.field_value, &b.field_value) {
                (FieldValue::Text(av), FieldValue::Text(bv)) => bv.cmp(av),
                (FieldValue::Int64(av), FieldValue::Int64(bv)) => bv.cmp(av),
                (FieldValue::Float64(av), FieldValue::Float64(bv)) => {
                    bv.partial_cmp(av).unwrap_or(Ordering::Equal)
                }
                (FieldValue::Bool(av), FieldValue::Bool(bv)) => bv.cmp(av),
                (FieldValue::DateTime(av), FieldValue::DateTime(bv)) => bv.cmp(av),
                (FieldValue::Geo(a), FieldValue::Geo(b)) => {
                    let lat_cmp = b.lat.partial_cmp(&a.lat).unwrap_or(Ordering::Equal);
                    if lat_cmp != Ordering::Equal {
                        lat_cmp
                    } else {
                        b.lon.partial_cmp(&a.lon).unwrap_or(Ordering::Equal)
                    }
                }
                (FieldValue::Bytes(av, _), FieldValue::Bytes(bv, _)) => bv.cmp(av),
                (FieldValue::Null, FieldValue::Null) => Ordering::Equal,
                (FieldValue::Null, _) => Ordering::Less,
                (_, FieldValue::Null) => Ordering::Greater,
                _ => Ordering::Equal,
            });
        }

        // Convert to SearchHit
        sorted_docs
            .into_iter()
            .map(|doc| SearchHit {
                doc_id: doc.doc_id,
                score: doc.score,
                document: None,
            })
            .collect()
    }

    fn total_hits(&self) -> u64 {
        self.total_hits
    }

    fn needs_more(&self) -> bool {
        self.hits.len() < self.max_docs
    }

    fn min_score(&self) -> f32 {
        self.min_score
    }

    fn reset(&mut self) {
        self.hits.clear();
        self.total_hits = 0;
    }
}

impl PartialEq for FieldScoredDoc {
    fn eq(&self, other: &Self) -> bool {
        self.field_value == other.field_value && self.doc_id == other.doc_id
    }
}

impl Eq for FieldScoredDoc {}

impl PartialOrd for FieldScoredDoc {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FieldScoredDoc {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap, but we want min-heap behavior for ascending sorts
        // and max-heap behavior for descending sorts.
        //
        // For ascending: We want lowest values to be popped first (min-heap).
        //   So we reverse comparison: b.cmp(a) makes heap pop smallest first.
        // For descending: We want highest values to be popped first (max-heap).
        //   So we use normal comparison: a.cmp(b) makes heap pop largest first.
        use crate::lexical::core::field::FieldValue;

        let field_cmp = if self.ascending {
            // Ascending: reverse comparison for min-heap behavior
            match (&self.field_value, &other.field_value) {
                (FieldValue::Text(a), FieldValue::Text(b)) => b.cmp(a),
                (FieldValue::Int64(a), FieldValue::Int64(b)) => b.cmp(a),
                (FieldValue::Float64(a), FieldValue::Float64(b)) => {
                    b.partial_cmp(a).unwrap_or(Ordering::Equal)
                }
                (FieldValue::Bool(a), FieldValue::Bool(b)) => b.cmp(a),
                (FieldValue::DateTime(a), FieldValue::DateTime(b)) => b.cmp(a),
                (FieldValue::Geo(a), FieldValue::Geo(b)) => {
                    let lat_cmp = b.lat.partial_cmp(&a.lat).unwrap_or(Ordering::Equal);
                    if lat_cmp != Ordering::Equal {
                        lat_cmp
                    } else {
                        b.lon.partial_cmp(&a.lon).unwrap_or(Ordering::Equal)
                    }
                }
                (FieldValue::Bytes(a, _), FieldValue::Bytes(b, _)) => b.cmp(a),
                (FieldValue::Null, FieldValue::Null) => Ordering::Equal,
                (FieldValue::Null, _) => Ordering::Greater,
                (_, FieldValue::Null) => Ordering::Less,
                _ => Ordering::Equal,
            }
        } else {
            // Descending: normal comparison for max-heap behavior
            match (&self.field_value, &other.field_value) {
                (FieldValue::Text(a), FieldValue::Text(b)) => a.cmp(b),
                (FieldValue::Int64(a), FieldValue::Int64(b)) => a.cmp(b),
                (FieldValue::Float64(a), FieldValue::Float64(b)) => {
                    a.partial_cmp(b).unwrap_or(Ordering::Equal)
                }
                (FieldValue::Bool(a), FieldValue::Bool(b)) => a.cmp(b),
                (FieldValue::DateTime(a), FieldValue::DateTime(b)) => a.cmp(b),
                (FieldValue::Geo(a), FieldValue::Geo(b)) => {
                    let lat_cmp = a.lat.partial_cmp(&b.lat).unwrap_or(Ordering::Equal);
                    if lat_cmp != Ordering::Equal {
                        lat_cmp
                    } else {
                        a.lon.partial_cmp(&b.lon).unwrap_or(Ordering::Equal)
                    }
                }
                (FieldValue::Bytes(a, _), FieldValue::Bytes(b, _)) => a.cmp(b),
                (FieldValue::Null, FieldValue::Null) => Ordering::Equal,
                (FieldValue::Null, _) => Ordering::Less,
                (_, FieldValue::Null) => Ordering::Greater,
                _ => Ordering::Equal,
            }
        };

        // If field values are equal, compare by doc_id for stability
        field_cmp.then_with(|| other.doc_id.cmp(&self.doc_id))
    }
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
            if let Some(worst) = self.hits.peek()
                && score > worst.score
            {
                // Replace the worst document
                self.hits.pop();
                self.hits.push(scored_doc);
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
        results.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));

        results
    }

    fn total_hits(&self) -> u64 {
        self.total_hits
    }

    fn needs_more(&self) -> bool {
        // Top-K collectors must see every candidate: a doc later in
        // the iteration with a higher score still needs to displace
        // the current heap-min. The previous
        // `self.hits.len() < self.max_docs` short-circuit returned the
        // first K matches by `doc_id` rather than the highest-scoring
        // K (#459). Early termination is now driven by
        // [`Collector::min_competitive`] + [`Scorer::max_score`] in
        // the searcher loop, which only fires when no future doc can
        // beat the current K-th score.
        true
    }

    fn min_score(&self) -> f32 {
        self.current_min_score()
    }

    fn min_competitive(&self) -> f32 {
        // The MaxScore early-termination signal: once the heap is full
        // the heap-min equals the K-th best score, and any future doc
        // whose upper-bound score is ≤ this value cannot enter the
        // top-K. While the heap has spare slots, return
        // `NEG_INFINITY` so the searcher does not attempt to skip
        // anything before the heap fills.
        //
        // Currently dead code in the default search path because
        // `needs_more()` short-circuits the loop the moment the heap
        // fills — fixing that without per-posting block-max metadata
        // would walk every candidate and regress wall time on common
        // OR workloads (the global `BM25Scorer::max_score()` upper
        // bound is too loose at `k1 + 1`). The correctness fix +
        // MaxScore activation lands together with the block-max index
        // format in the next PR (#403 PR-B onward).
        if self.hits.len() < self.max_docs {
            f32::NEG_INFINITY
        } else {
            self.hits.peek().map(|d| d.score).unwrap_or(self.min_score)
        }
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
    /// Cached sorted results to avoid repeated clone+sort.
    sorted_cache: RefCell<Option<Vec<SearchHit>>>,
}

impl AllDocsCollector {
    /// Create a new all docs collector.
    pub fn new() -> Self {
        AllDocsCollector {
            hits: Vec::new(),
            min_score: 0.0,
            sorted_cache: RefCell::new(None),
        }
    }

    /// Create a new all docs collector with minimum score threshold.
    pub fn with_min_score(min_score: f32) -> Self {
        AllDocsCollector {
            hits: Vec::new(),
            min_score,
            sorted_cache: RefCell::new(None),
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
            // Invalidate cached sorted results.
            *self.sorted_cache.borrow_mut() = None;
        }
        Ok(())
    }

    fn results(&self) -> Vec<SearchHit> {
        let mut cache = self.sorted_cache.borrow_mut();
        if let Some(ref cached) = *cache {
            return cached.clone();
        }
        let mut results = self.hits.clone();
        results.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        *cache = Some(results.clone());
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
        *self.sorted_cache.borrow_mut() = None;
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
        // `needs_more` is `true` even when the heap is full, because a
        // higher-scoring doc later in the iteration can still displace
        // the current worst. Early termination is now driven by
        // `min_competitive` + `Scorer::max_score()` in the searcher
        // loop (see #459).
        assert!(collector.needs_more());

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
    fn test_top_docs_collector_replaces_lower_after_full() {
        // Verifies the correctness contract restored by the
        // `needs_more = true` change (#459): a higher-scoring doc that
        // arrives **after** the heap fills must displace the current
        // heap-min, regardless of doc-id order. The pre-fix collector
        // short-circuited the search loop the moment the heap filled
        // and returned the first K matches by `doc_id`, dropping any
        // higher-scoring doc that came later in the iteration.
        let mut collector = TopDocsCollector::new(3);

        collector.collect(1, 0.10).unwrap();
        collector.collect(2, 0.20).unwrap();
        collector.collect(3, 0.30).unwrap();
        assert!(collector.needs_more());

        collector.collect(4, 0.99).unwrap();

        let results = collector.results();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].doc_id, 4);
        assert!((results[0].score - 0.99).abs() < 1e-6);
        // Lowest-scoring doc (1) should have been evicted.
        assert!(results.iter().all(|hit| hit.doc_id != 1));
    }

    #[test]
    fn test_top_docs_collector_min_competitive() {
        // Validates the #403 PR-A `min_competitive` signal:
        //   - empty / partially full heap → `NEG_INFINITY` (no useful
        //     bound; the searcher must keep collecting).
        //   - full heap → heap-min, i.e. the K-th best score.
        //   - replaces the worst when a higher score arrives → bound
        //     advances upward.
        let mut collector = TopDocsCollector::new(3);

        assert_eq!(collector.min_competitive(), f32::NEG_INFINITY);

        collector.collect(1, 0.5).unwrap();
        assert_eq!(collector.min_competitive(), f32::NEG_INFINITY);

        collector.collect(2, 0.8).unwrap();
        assert_eq!(collector.min_competitive(), f32::NEG_INFINITY);

        collector.collect(3, 0.3).unwrap();
        // Heap full — bound is the K-th score (lowest of the three).
        assert!((collector.min_competitive() - 0.3).abs() < 1e-6);

        // A higher-scoring doc tightens the bound to the next-lowest.
        collector.collect(4, 0.9).unwrap();
        assert!((collector.min_competitive() - 0.5).abs() < 1e-6);
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
