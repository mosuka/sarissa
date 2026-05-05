//! Searcher implementation for executing queries against an index.

use crate::lexical::core::field::FieldValue;
use std::cmp::Ordering;
use std::sync::Arc;
use std::time::Duration;

use crate::util::time::Timer;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use crate::analysis::analyzer::standard::StandardAnalyzer;
use crate::data::DataValue::{
    Bool as Boolean, Bytes, DateTime, Float64 as Float, Geo, Int64 as Integer, Null, Text,
};
use crate::error::{LaurusError, Result};
// Note: Geo and DateTime were removed from FieldValue definition implicitly by switching to DataValue.
// Only standard types remain. Logic using Geo/DateTime needs update.
use crate::lexical::index::inverted::reader::InvertedIndexReader;
use crate::lexical::query::Query;
use crate::lexical::query::boolean::{BooleanQuery, Occur};
use crate::lexical::query::collector::{
    Collector, CountCollector, TopDocsCollector, TopFieldCollector,
};
use crate::lexical::query::parser::LexicalQueryParser;
use crate::lexical::query::{LexicalSearchResults, SearchHit};
use crate::lexical::reader::LexicalIndexReader;
use crate::lexical::search::searcher::{
    LexicalSearchParams, LexicalSearchQuery, LexicalSearchRequest, SortField, SortOrder,
};

/// A searcher that executes queries against an index reader.
#[derive(Debug)]
pub struct InvertedIndexSearcher {
    /// The index reader to search against.
    reader: Arc<dyn LexicalIndexReader>,
    /// Default fields to search if none specified in query.
    default_fields: Vec<String>,
}

impl InvertedIndexSearcher {
    /// Create a new searcher with the given index reader.
    pub fn new(reader: Box<dyn LexicalIndexReader>) -> Self {
        InvertedIndexSearcher {
            reader: Arc::from(reader),
            default_fields: Vec::new(),
        }
    }

    /// Create a new searcher with an `Arc<dyn LexicalIndexReader>`.
    pub fn from_arc(reader: Arc<dyn LexicalIndexReader>) -> Self {
        InvertedIndexSearcher {
            reader,
            default_fields: Vec::new(),
        }
    }

    /// Set default fields for search.
    pub fn with_default_fields(mut self, fields: Vec<String>) -> Self {
        self.default_fields = fields;
        self
    }

    /// Get the index reader.
    pub fn reader(&self) -> &Arc<dyn LexicalIndexReader> {
        &self.reader
    }

    /// Execute a search with a custom collector.
    pub fn search_with_collector<C: Collector>(
        &self,
        query: Box<dyn Query>,
        collector: C,
    ) -> Result<C> {
        self.search_with_collector_parallel(query, collector, false)
    }

    /// Execute a search with a custom collector, with optional parallel execution.
    pub fn search_with_collector_parallel<C: Collector>(
        &self,
        query: Box<dyn Query>,
        mut collector: C,
        parallel: bool,
    ) -> Result<C> {
        // For BooleanQuery with multiple clauses, try to execute sub-queries in parallel
        if parallel && let Some(boolean_query) = query.as_any().downcast_ref::<BooleanQuery>() {
            return self.search_boolean_query_parallel(boolean_query, collector);
        }

        // Default single-threaded execution
        // Create a matcher for the query
        let mut matcher = query.matcher(self.reader.as_ref())?;

        // Create a scorer for the query
        let scorer = query.scorer(self.reader.as_ref())?;

        // Iterate through matching documents
        while !matcher.is_exhausted() {
            let doc_id = matcher.doc_id();

            if doc_id == u64::MAX {
                break;
            }

            // Block-Max skip-ahead pre-check (#403 PR-E). Before paying
            // the score / field-length cost on this doc, see whether
            // the block containing it is even competitive. The current
            // block's bound (`current_block_max_score`) is non-cumulative
            // — when it falls below the K-th score, jumping past the
            // block via `next_block_boundary` is sound (the global
            // `block_max_score_at` cumulative bound, queried right
            // below, still controls the hard `break`).
            let min_comp = collector.min_competitive();
            if scorer.current_block_max_score(doc_id) <= min_comp {
                if scorer.block_max_score_at(doc_id) <= min_comp {
                    // Cumulative suffix bound already non-competitive
                    // → no later block can produce a top-K hit.
                    break;
                }
                if let Some(target) = scorer.next_block_boundary(doc_id) {
                    if target == u64::MAX || target <= doc_id {
                        break;
                    }
                    if !matcher.skip_to(target)? || matcher.is_exhausted() {
                        break;
                    }
                    continue;
                }
                // No per-block info → fall through to existing PR-C
                // break path after scoring this doc.
            }

            // Calculate score for this document
            let term_freq = matcher.term_freq() as f32;

            // Retrieve actual field length if query targets a specific field
            let field_length = if let Some(field_name) = query.field() {
                if let Some(inverted_index_reader) =
                    self.reader.as_any().downcast_ref::<InvertedIndexReader>()
                {
                    inverted_index_reader
                        .field_length(doc_id, field_name)
                        .ok()
                        .flatten()
                        .map(|len| len as f32)
                } else {
                    None
                }
            } else {
                None
            };

            let score = scorer.score(doc_id, term_freq, field_length);

            // Collect the result
            collector.collect(doc_id, score)?;

            // Cumulative early-break (#403 PR-C). After every collect
            // the collector's `min_competitive()` may have tightened;
            // ask the scorer for the right-cumulative upper bound at
            // the matcher's current position. As soon as that bound
            // drops below the K-th score, no later doc in the posting
            // list can enter the top-K and the walk is provably done.
            if scorer.block_max_score_at(doc_id) <= collector.min_competitive() {
                break;
            }

            // Check if we need more results (count-cap collectors).
            if !collector.needs_more() {
                break;
            }

            // Move to next document
            if !matcher.next()? {
                break;
            }
        }

        Ok(collector)
    }

    /// Execute a BooleanQuery with parallel sub-query execution.
    ///
    /// Each clause is executed in parallel, then boolean logic is applied:
    /// - Must/Filter: intersection (all must match)
    /// - Should: union (adds score if matching; at least minimum_should_match required)
    /// - MustNot: exclusion (removes matching documents)
    fn search_boolean_query_parallel<C: Collector>(
        &self,
        boolean_query: &BooleanQuery,
        mut collector: C,
    ) -> Result<C> {
        use std::collections::{HashMap, HashSet};

        let clauses = boolean_query.clauses();

        if clauses.is_empty() {
            return Ok(collector);
        }

        // Single clause: no need for parallel execution
        if clauses.len() == 1 {
            return self.search_with_collector_parallel(
                clauses[0].query.clone_box(),
                collector,
                false,
            );
        }

        // Execute all clauses in parallel, collecting (doc_id, score) per clause
        #[cfg(not(target_arch = "wasm32"))]
        let iter = clauses.par_iter();
        #[cfg(target_arch = "wasm32")]
        let iter = clauses.iter();

        let clause_results: Vec<(Occur, Result<Vec<SearchHit>>)> = iter
            .map(|clause| {
                // Boolean operations (intersection/union/exclusion) require the
                // full result set from each clause, so we use an unbounded collector.
                let temp_collector = TopDocsCollector::new(usize::MAX);
                let result = self
                    .search_with_collector_parallel(clause.query.clone_box(), temp_collector, false)
                    .map(|c| c.results());
                (clause.occur, result)
            })
            .collect();

        // Separate results by Occur type
        let mut must_sets: Vec<HashMap<u64, f32>> = Vec::new();
        let mut should_map: HashMap<u64, f32> = HashMap::new();
        let mut must_not_set: HashSet<u64> = HashSet::new();
        let mut first_error: Option<LaurusError> = None;

        for (occur, result) in clause_results {
            match result {
                Ok(hits) => match occur {
                    Occur::Must | Occur::Filter => {
                        let mut m = HashMap::with_capacity(hits.len());
                        for hit in hits {
                            let score = if occur == Occur::Filter {
                                0.0
                            } else {
                                hit.score
                            };
                            m.insert(hit.doc_id, score);
                        }
                        must_sets.push(m);
                    }
                    Occur::Should => {
                        for hit in hits {
                            *should_map.entry(hit.doc_id).or_insert(0.0) += hit.score;
                        }
                    }
                    Occur::MustNot => {
                        for hit in hits {
                            must_not_set.insert(hit.doc_id);
                        }
                    }
                },
                Err(e) => {
                    if first_error.is_none() {
                        first_error = Some(e);
                    }
                }
            }
        }

        // If any clause produced an error, fail the whole query
        if let Some(e) = first_error {
            return Err(e);
        }

        // Apply boolean logic
        let minimum_should_match = boolean_query.minimum_should_match();
        let has_must = !must_sets.is_empty();

        // Build the candidate set
        let mut candidates: HashMap<u64, f32> = if has_must {
            // Sort must_sets by size ascending for faster intersection.
            must_sets.sort_unstable_by_key(|s| s.len());
            // Start with the smallest Must/Filter set, intersect with the rest
            let mut result = must_sets.swap_remove(0);
            for other in &must_sets {
                result.retain(|doc_id, score| {
                    if let Some(other_score) = other.get(doc_id) {
                        *score += other_score;
                        true
                    } else {
                        false
                    }
                });
            }
            result
        } else {
            // No Must clauses: Should clauses form the candidate set
            should_map.clone()
        };

        // Add Should scores to Must candidates (boost, not filter)
        if has_must {
            for (doc_id, score) in candidates.iter_mut() {
                if let Some(should_score) = should_map.get(doc_id) {
                    *score += should_score;
                }
            }

            // If minimum_should_match > 0, filter candidates that don't match enough Should clauses
            if minimum_should_match > 0 {
                // Count Should matches per doc
                // (should_map already contains the union; we need per-clause counts)
                // For simplicity, treat minimum_should_match as requiring the doc to appear in should_map
                candidates.retain(|doc_id, _| should_map.contains_key(doc_id));
            }
        }

        // Exclude MustNot documents
        for doc_id in &must_not_set {
            candidates.remove(doc_id);
        }

        // Feed results into the collector
        // Sort by score descending for deterministic results
        let mut sorted: Vec<(u64, f32)> = candidates.into_iter().collect();
        // Use unstable sort since stability is not needed for (doc_id, score) pairs.
        sorted.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        for (doc_id, score) in sorted {
            collector.collect(doc_id, score)?;
            if !collector.needs_more() {
                break;
            }
        }

        Ok(collector)
    }

    /// Load documents for search hits.
    fn load_documents(&self, hits: &mut [SearchHit]) -> Result<()> {
        for hit in hits {
            if let Some(doc) = self.reader.document(hit.doc_id)? {
                hit.document = Some(doc);
            }
        }
        Ok(())
    }

    /// Load documents in parallel for better performance.
    fn load_documents_parallel(&self, hits: &mut [SearchHit]) -> Result<()> {
        // Use a parallel iterator to load documents
        #[cfg(not(target_arch = "wasm32"))]
        let results: Vec<_> = hits
            .par_iter()
            .map(|hit| (hit.doc_id, self.reader.document(hit.doc_id)))
            .collect();
        #[cfg(target_arch = "wasm32")]
        let results: Vec<_> = hits
            .iter()
            .map(|hit| (hit.doc_id, self.reader.document(hit.doc_id)))
            .collect();

        // Update hits with loaded documents
        for (i, (_, doc_result)) in results.into_iter().enumerate() {
            if let Ok(Some(doc)) = doc_result {
                hits[i].document = Some(doc);
            }
        }

        Ok(())
    }

    /// Execute a search with timeout (internal implementation).
    fn search_with_timeout_internal(
        &self,
        query: Box<dyn Query>,
        params: &LexicalSearchParams,
        timeout: Duration,
    ) -> Result<LexicalSearchResults> {
        let start_time = Timer::now();

        // Create collector based on sort type
        let (mut hits, total_hits) = match &params.sort_by {
            SortField::Field { name, order } => {
                // Use TopFieldCollector for field-based sorting
                let ascending = matches!(order, SortOrder::Asc);
                let collector = TopFieldCollector::with_min_score(
                    params.limit,
                    params.min_score,
                    name.clone(),
                    ascending,
                    self.reader.as_ref(),
                );

                let result_collector = self.search_with_collector_parallel(
                    query.clone_box(),
                    collector,
                    params.parallel,
                )?;

                (result_collector.results(), result_collector.total_hits())
            }
            SortField::Score => {
                // Use TopDocsCollector for score-based sorting
                let collector = TopDocsCollector::with_min_score(params.limit, params.min_score);

                let result_collector =
                    self.search_with_collector_parallel(query, collector, params.parallel)?;

                (result_collector.results(), result_collector.total_hits())
            }
        };

        // Check if we exceeded timeout.
        // NOTE: Timeout is checked after scoring completes, not during scoring.
        // Per-document timeout checks would add overhead to every document match.
        // For very large result sets, consider using limit to bound scoring.
        if start_time.elapsed() > timeout {
            return Err(LaurusError::index("Search timeout exceeded"));
        }

        // Load documents if requested
        if params.load_documents {
            if params.parallel && hits.len() > 10 {
                self.load_documents_parallel(&mut hits)?;
            } else {
                self.load_documents(&mut hits)?;
            }
        }

        // No need to sort - already sorted during collection

        // Calculate max score
        let max_score = hits.iter().map(|hit| hit.score).fold(0.0f32, f32::max);

        Ok(LexicalSearchResults {
            hits,
            total_hits,
            max_score,
        })
    }

    /// Search with the given request.
    pub fn search(&self, request: LexicalSearchRequest) -> Result<LexicalSearchResults> {
        // Convert DSL query to Query object if necessary
        let query = match &request.query {
            LexicalSearchQuery::Dsl(dsl_string) => {
                // Get analyzer from reader
                let analyzer = if let Some(inverted_index_reader) =
                    self.reader.as_any().downcast_ref::<InvertedIndexReader>()
                {
                    inverted_index_reader.analyzer().clone()
                } else {
                    // Fallback to standard analyzer
                    Arc::new(StandardAnalyzer::new()?)
                };

                // Parse DSL string into Query object
                let mut parser = LexicalQueryParser::new(analyzer.clone());
                if !self.default_fields.is_empty() {
                    parser = parser.with_default_fields(self.default_fields.clone());
                }
                parser.parse(dsl_string)?
            }
            LexicalSearchQuery::Obj(q) => q.clone_box(),
        };

        // Check if query is empty
        if query.is_empty(self.reader.as_ref())? {
            return Ok(LexicalSearchResults {
                hits: Vec::new(),
                total_hits: 0,
                max_score: 0.0,
            });
        }

        // Execute search with timeout if specified
        if let Some(timeout_ms) = request.params.timeout_ms {
            let timeout = Duration::from_millis(timeout_ms);
            self.search_with_timeout_internal(query, &request.params, timeout)
        } else {
            // Check if we should use field-based sorting during collection
            match &request.params.sort_by {
                SortField::Field { name, order } => {
                    // Use TopFieldCollector for field-based sorting
                    let ascending = matches!(order, SortOrder::Asc);
                    let collector = TopFieldCollector::with_min_score(
                        request.params.limit,
                        request.params.min_score,
                        name.clone(),
                        ascending,
                        self.reader.as_ref(),
                    );

                    let result_collector = self.search_with_collector_parallel(
                        query.clone_box(),
                        collector,
                        request.params.parallel,
                    )?;

                    let mut hits = result_collector.results();
                    let total_hits = result_collector.total_hits();

                    // Load documents if requested
                    if request.params.load_documents {
                        self.load_documents(&mut hits)?;
                    }

                    // No need to sort - already sorted by TopFieldCollector during collection

                    // Calculate max score
                    let max_score = hits.iter().map(|hit| hit.score).fold(0.0f32, f32::max);

                    Ok(LexicalSearchResults {
                        hits,
                        total_hits,
                        max_score,
                    })
                }
                SortField::Score => {
                    // Use TopDocsCollector for score-based sorting
                    let collector = TopDocsCollector::with_min_score(
                        request.params.limit,
                        request.params.min_score,
                    );
                    let result_collector = self.search_with_collector_parallel(
                        query,
                        collector,
                        request.params.parallel,
                    )?;

                    let mut hits = result_collector.results();
                    let total_hits = result_collector.total_hits();

                    // Load documents if requested
                    if request.params.load_documents {
                        self.load_documents(&mut hits)?;
                    }

                    // No need to sort - already sorted by score in TopDocsCollector

                    // Calculate max score
                    let max_score = hits.iter().map(|hit| hit.score).fold(0.0f32, f32::max);

                    Ok(LexicalSearchResults {
                        hits,
                        total_hits,
                        max_score,
                    })
                }
            }
        }
    }

    /// Sort search hits according to the specified sort field.
    /// This is the old post-collection sorting approach, kept for compatibility.
    #[allow(dead_code)]
    fn sort_hits(&self, hits: &mut [SearchHit], sort_by: &SortField) -> Result<()> {
        match sort_by {
            SortField::Score => {
                // Default behavior: already sorted by score from collector
                // Re-sort to ensure descending order
                hits.sort_unstable_by(|a, b| {
                    b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal)
                });
            }
            SortField::Field { name, order } => {
                // Sort by field value
                hits.sort_unstable_by(|a, b| {
                    let cmp = self.compare_field_values(a, b, name);
                    match order {
                        SortOrder::Asc => cmp,
                        SortOrder::Desc => cmp.reverse(),
                    }
                });
            }
        }
        Ok(())
    }

    /// Compare two search hits by a specific field value.
    #[allow(dead_code)]
    fn compare_field_values(&self, a: &SearchHit, b: &SearchHit, field_name: &str) -> Ordering {
        let val_a = a.document.as_ref().and_then(|doc| doc.get(field_name));
        let val_b = b.document.as_ref().and_then(|doc| doc.get(field_name));

        match (val_a, val_b) {
            (Some(a_val), Some(b_val)) => self.compare_values(a_val, b_val),
            (Some(_), None) => Ordering::Less, // Documents with value come first
            (None, Some(_)) => Ordering::Greater, // Documents without value come last
            (None, None) => Ordering::Equal,
        }
    }

    /// Compare two field values.
    #[allow(dead_code)]
    fn compare_values(&self, a: &FieldValue, b: &FieldValue) -> Ordering {
        match (a, b) {
            // Same type comparisons
            (Text(a_str), Text(b_str)) => a_str.cmp(b_str),
            (Integer(a_int), Integer(b_int)) => a_int.cmp(b_int),
            (Float(a_float), Float(b_float)) => {
                a_float.partial_cmp(b_float).unwrap_or(Ordering::Equal)
            }
            (Boolean(a_bool), Boolean(b_bool)) => a_bool.cmp(b_bool),
            (DateTime(a_dt), DateTime(b_dt)) => a_dt.cmp(b_dt),
            (Geo(a), Geo(b)) => a
                .lat
                .partial_cmp(&b.lat)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.lon.partial_cmp(&b.lon).unwrap_or(Ordering::Equal)),
            (Bytes(_, a_bytes), Bytes(_, b_bytes)) => a_bytes.cmp(b_bytes),
            (Null, Null) => Ordering::Equal,

            // Mixed types ordering precedence
            // Null < Bool < Int < Float < Text < Bytes
            (Null, _) => Ordering::Less,
            (_, Null) => Ordering::Greater,

            (Boolean(_), _) => Ordering::Less,
            (_, Boolean(_)) => Ordering::Greater,

            (Integer(_), _) => Ordering::Less,
            (_, Integer(_)) => Ordering::Greater,

            (Float(_), _) => Ordering::Less,
            (_, Float(_)) => Ordering::Greater,

            (Text(_), _) => Ordering::Less,
            (_, Text(_)) => Ordering::Greater,

            (Bytes(_, _), _) => Ordering::Less,
            (_, Bytes(_, _)) => Ordering::Greater,

            _ => Ordering::Equal, // Fallback
        }
    }

    /// Count documents matching the request.
    ///
    /// If `min_score` is specified in the request parameters, only documents
    /// with a score equal to or greater than the threshold are counted.
    pub fn count(&self, request: LexicalSearchRequest) -> Result<u64> {
        let lexical_query = request.query;

        // Parse DSL string if needed
        let query = if let LexicalSearchQuery::Dsl(_) = &lexical_query {
            // Get analyzer from reader
            let analyzer = if let Some(inverted_index_reader) =
                self.reader.as_any().downcast_ref::<InvertedIndexReader>()
            {
                inverted_index_reader.analyzer().clone()
            } else {
                // Fallback to standard analyzer
                Arc::new(StandardAnalyzer::new()?)
            };

            // Parse DSL string into Query object
            lexical_query.into_query(&analyzer)?
        } else {
            match lexical_query {
                LexicalSearchQuery::Obj(q) => q,
                _ => unreachable!(),
            }
        };

        // Check if query is empty
        if query.is_empty(self.reader.as_ref())? {
            return Ok(0);
        }

        // Use count collector with min_score if specified
        let collector = if request.params.min_score > 0.0 {
            CountCollector::with_min_score(request.params.min_score)
        } else {
            CountCollector::new()
        };

        let result_collector = self.search_with_collector(query, collector)?;
        Ok(result_collector.total_hits())
    }
}

// Implement LexicalSearcher trait for InvertedIndexSearcher
impl crate::lexical::search::searcher::LexicalSearcher for InvertedIndexSearcher {
    fn search(&self, request: LexicalSearchRequest) -> Result<LexicalSearchResults> {
        InvertedIndexSearcher::search(self, request)
    }

    fn count(
        &self,
        request: crate::lexical::search::searcher::LexicalSearchRequest,
    ) -> Result<u64> {
        InvertedIndexSearcher::count(self, request)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexical::index::inverted::reader::{InvertedIndexReader, InvertedIndexReaderConfig};
    use crate::lexical::query::boolean::{BooleanQuery, BooleanQueryBuilder};
    use crate::lexical::query::term::TermQuery;

    use crate::storage::memory::MemoryStorage;
    use crate::storage::memory::MemoryStorageConfig;
    use std::sync::Arc;

    #[allow(dead_code)]
    fn create_test_searcher() -> InvertedIndexSearcher {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let reader = Box::new(
            InvertedIndexReader::new(vec![], storage, InvertedIndexReaderConfig::default())
                .unwrap(),
        );
        InvertedIndexSearcher::new(reader)
    }

    #[test]
    fn test_searcher_creation() {
        let searcher = create_test_searcher();

        // Verify searcher has a valid reader
        let reader = searcher.reader();
        assert!(Arc::strong_count(reader) >= 1, "Reader should be valid");

        // Verify reader has expected initial state
        assert_eq!(
            reader.doc_count(),
            0,
            "New searcher should have 0 documents"
        );
    }

    #[test]
    fn test_search_term_query() {
        let searcher = create_test_searcher();
        let query = Box::new(TermQuery::new("title", "hello")) as Box<dyn Query>;

        let request = LexicalSearchRequest::new(query);
        let results = searcher.search(request).unwrap();

        // Should return empty results for non-existent terms
        assert_eq!(results.hits.len(), 0);
        assert_eq!(results.total_hits, 0);
        assert_eq!(results.max_score, 0.0);
    }

    #[test]
    fn test_search_boolean_query() {
        let searcher = create_test_searcher();

        let query = Box::new(
            BooleanQueryBuilder::new()
                .must(Box::new(TermQuery::new("title", "hello")))
                .should(Box::new(TermQuery::new("body", "world")))
                .build(),
        ) as Box<dyn Query>;

        let request = LexicalSearchRequest::new(query);
        let results = searcher.search(request).unwrap();

        // Should return empty results for non-existent terms
        assert_eq!(results.hits.len(), 0);
        assert_eq!(results.total_hits, 0);
        assert_eq!(results.max_score, 0.0);
    }

    #[test]
    fn test_search_with_config() {
        let searcher = create_test_searcher();
        let query = Box::new(TermQuery::new("title", "hello")) as Box<dyn Query>;

        let request = LexicalSearchRequest::new(query)
            .limit(5)
            .min_score(0.5)
            .load_documents(false);

        let results = searcher.search(request).unwrap();

        // Should respect configuration
        assert_eq!(results.hits.len(), 0);
        assert_eq!(results.total_hits, 0);
    }

    #[test]
    fn test_count_query() {
        let searcher = create_test_searcher();
        let query = Box::new(TermQuery::new("title", "hello")) as Box<dyn Query>;

        let count = searcher.count(LexicalSearchRequest::new(query)).unwrap();

        // Should return 0 for non-existent terms
        assert_eq!(count, 0);
    }

    #[test]
    fn test_search_with_timeout() {
        let searcher = create_test_searcher();
        let query = Box::new(TermQuery::new("title", "hello")) as Box<dyn Query>;

        let request = LexicalSearchRequest::new(query).timeout_ms(1000); // 1 second timeout

        let results = searcher.search(request).unwrap();

        // Should complete within timeout
        assert_eq!(results.hits.len(), 0);
        assert_eq!(results.total_hits, 0);
    }

    #[test]
    fn test_search_with_collector() {
        let searcher = create_test_searcher();
        let query = Box::new(TermQuery::new("title", "hello"));
        let collector = TopDocsCollector::new(10);

        let result_collector = searcher.search_with_collector(query, collector).unwrap();

        assert_eq!(result_collector.total_hits(), 0);
        assert_eq!(result_collector.results().len(), 0);
    }

    #[test]
    fn test_search_empty_query() {
        let searcher = create_test_searcher();
        // Create a boolean query with no clauses (empty query)
        let query = Box::new(BooleanQuery::new()) as Box<dyn Query>;

        let request = LexicalSearchRequest::new(query);
        let results = searcher.search(request).unwrap();

        // Should return empty results for empty query
        assert_eq!(results.hits.len(), 0);
        assert_eq!(results.total_hits, 0);
        assert_eq!(results.max_score, 0.0);
    }

    #[test]
    fn test_count_empty_query() {
        let searcher = create_test_searcher();
        let query = Box::new(BooleanQuery::new()) as Box<dyn Query>;

        let count = searcher.count(LexicalSearchRequest::new(query)).unwrap();

        // Should return 0 for empty query
        assert_eq!(count, 0);
    }

    #[test]
    fn test_search_request_builder() {
        let query = Box::new(TermQuery::new("title", "hello")) as Box<dyn Query>;

        let request = LexicalSearchRequest::new(query)
            .limit(20)
            .min_score(0.1)
            .load_documents(false)
            .timeout_ms(5000);

        assert_eq!(request.params.limit, 20);
        assert_eq!(request.params.min_score, 0.1);
        assert!(!request.params.load_documents);
        assert_eq!(request.params.timeout_ms, Some(5000));
    }
}
