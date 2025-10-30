//! Searcher implementation for executing queries against an index.

use std::cmp::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

use rayon::prelude::*;

use crate::analysis::analyzer::standard::StandardAnalyzer;
use crate::document::field_value::FieldValue;
use crate::error::{Result, SageError};
use crate::lexical::index::reader::inverted::InvertedIndexReader;
use crate::lexical::reader::IndexReader;
use crate::lexical::types::{
    LexicalSearchParams, LexicalSearchQuery, LexicalSearchRequest, SortField, SortOrder,
};
use crate::query::boolean::BooleanQuery;
use crate::query::collector::{Collector, CountCollector, TopDocsCollector, TopFieldCollector};
use crate::query::parser::QueryParser;
use crate::query::query::Query;
use crate::query::{SearchHit, SearchResults};

/// A searcher that executes queries against an index reader.
#[derive(Debug)]
pub struct InvertedIndexSearcher {
    /// The index reader to search against.
    reader: Arc<dyn IndexReader>,
}

impl InvertedIndexSearcher {
    /// Create a new searcher with the given index reader.
    pub fn new(reader: Box<dyn IndexReader>) -> Self {
        InvertedIndexSearcher {
            reader: Arc::from(reader),
        }
    }

    /// Create a new searcher with an `Arc<dyn IndexReader>`.
    pub fn from_arc(reader: Arc<dyn IndexReader>) -> Self {
        InvertedIndexSearcher { reader }
    }

    /// Get the index reader.
    pub fn reader(&self) -> &Arc<dyn IndexReader> {
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

            // Check if we need more results
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
    fn search_boolean_query_parallel<C: Collector>(
        &self,
        boolean_query: &BooleanQuery,
        collector: C,
    ) -> Result<C> {
        let clauses = boolean_query.clauses();

        // If we have multiple clauses, execute them in parallel and merge results
        if clauses.len() > 1 {
            use std::sync::{Arc, Mutex};

            let collector_arc = Arc::new(Mutex::new(collector));
            let results: Vec<_> = clauses
                .par_iter()
                .map(|clause| {
                    // Create a temporary collector for this clause
                    let temp_collector = TopDocsCollector::new(1000); // Reasonable limit
                    match self.search_with_collector_parallel(
                        clause.query.clone_box(),
                        temp_collector,
                        false,
                    ) {
                        Ok(result_collector) => Ok(result_collector.results()),
                        Err(e) => Err(e),
                    }
                })
                .collect();

            // Merge results back into the main collector
            let mut collector = Arc::try_unwrap(collector_arc)
                .unwrap()
                .into_inner()
                .unwrap();
            for result in results {
                match result {
                    Ok(hits) => {
                        for hit in hits {
                            collector.collect(hit.doc_id, hit.score)?;
                            if !collector.needs_more() {
                                break;
                            }
                        }
                    }
                    Err(_) => {
                        // Continue with other results if one fails
                        continue;
                    }
                }
            }

            Ok(collector)
        } else {
            // Single clause, no need for parallel execution
            if let Some(clause) = clauses.first() {
                self.search_with_collector_parallel(clause.query.clone_box(), collector, false)
            } else {
                Ok(collector)
            }
        }
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
        let results: Vec<_> = hits
            .par_iter()
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
    ) -> Result<SearchResults> {
        let start_time = Instant::now();

        // Create collector based on sort type
        let (mut hits, total_hits) = match &params.sort_by {
            SortField::Field { name, order } => {
                // Use TopFieldCollector for field-based sorting
                let ascending = matches!(order, SortOrder::Asc);
                let collector = TopFieldCollector::with_min_score(
                    params.max_docs,
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
                let collector = TopDocsCollector::with_min_score(params.max_docs, params.min_score);

                let result_collector =
                    self.search_with_collector_parallel(query, collector, params.parallel)?;

                (result_collector.results(), result_collector.total_hits())
            }
        };

        // Check if we exceeded timeout
        if start_time.elapsed() > timeout {
            return Err(SageError::index("Search timeout exceeded"));
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

        Ok(SearchResults {
            hits,
            total_hits,
            max_score,
        })
    }

    /// Search with the given request.
    pub fn search(&self, request: LexicalSearchRequest) -> Result<SearchResults> {
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
                let parser = QueryParser::new(analyzer.clone());
                parser.parse(dsl_string)?
            }
            LexicalSearchQuery::Obj(q) => q.clone_box(),
        };

        // Check if query is empty
        if query.is_empty(self.reader.as_ref())? {
            return Ok(SearchResults {
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
                        request.params.max_docs,
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

                    Ok(SearchResults {
                        hits,
                        total_hits,
                        max_score,
                    })
                }
                SortField::Score => {
                    // Use TopDocsCollector for score-based sorting
                    let collector = TopDocsCollector::with_min_score(
                        request.params.max_docs,
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

                    Ok(SearchResults {
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
                hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
            }
            SortField::Field { name, order } => {
                // Sort by field value
                hits.sort_by(|a, b| {
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
        let val_a = a
            .document
            .as_ref()
            .and_then(|doc| doc.get_field(field_name));
        let val_b = b
            .document
            .as_ref()
            .and_then(|doc| doc.get_field(field_name));

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
        use FieldValue::*;

        match (a, b) {
            // Same type comparisons
            (Text(a_str), Text(b_str)) => a_str.cmp(b_str),
            (Integer(a_int), Integer(b_int)) => a_int.cmp(b_int),
            (Float(a_float), Float(b_float)) => {
                a_float.partial_cmp(b_float).unwrap_or(Ordering::Equal)
            }
            (Boolean(a_bool), Boolean(b_bool)) => a_bool.cmp(b_bool),
            (Geo(a_geo), Geo(b_geo)) => {
                // Compare by latitude first, then longitude
                match a_geo.lat.partial_cmp(&b_geo.lat) {
                    Some(Ordering::Equal) | None => {
                        a_geo.lon.partial_cmp(&b_geo.lon).unwrap_or(Ordering::Equal)
                    }
                    Some(ord) => ord,
                }
            }
            (DateTime(a_dt), DateTime(b_dt)) => a_dt.cmp(b_dt),
            (Binary(a_bin), Binary(b_bin)) => a_bin.cmp(b_bin),
            (Null, Null) => Ordering::Equal,

            // For different types, use a consistent ordering based on type precedence
            // Text < Integer < Float < Boolean < Geo < DateTime < Binary < Null
            (Text(_), _) => Ordering::Less,
            (_, Text(_)) => Ordering::Greater,
            (Integer(_), _) => Ordering::Less,
            (_, Integer(_)) => Ordering::Greater,
            (Float(_), _) => Ordering::Less,
            (_, Float(_)) => Ordering::Greater,
            (Boolean(_), _) => Ordering::Less,
            (_, Boolean(_)) => Ordering::Greater,
            (Geo(_), _) => Ordering::Less,
            (_, Geo(_)) => Ordering::Greater,
            (DateTime(_), _) => Ordering::Less,
            (_, DateTime(_)) => Ordering::Greater,
            (Binary(_), _) => Ordering::Less,
            (_, Binary(_)) => Ordering::Greater,
        }
    }

    /// Count documents matching the query.
    pub fn count(&self, query: LexicalSearchQuery) -> Result<u64> {
        // Use the provided LexicalSearchQuery
        let lexical_query = query;

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

        // Use count collector
        let collector = CountCollector::new();
        let result_collector = self.search_with_collector(query, collector)?;

        Ok(result_collector.total_hits())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexical::index::reader::inverted::{InvertedIndexReader, InvertedIndexReaderConfig};
    use crate::query::boolean::{BooleanQuery, BooleanQueryBuilder};
    use crate::query::term::TermQuery;

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
            .max_docs(5)
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

        let count = searcher.count(query.into()).unwrap();

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

        let count = searcher.count(query.into()).unwrap();

        // Should return 0 for empty query
        assert_eq!(count, 0);
    }

    #[test]
    fn test_search_request_builder() {
        let query = Box::new(TermQuery::new("title", "hello")) as Box<dyn Query>;

        let request = LexicalSearchRequest::new(query)
            .max_docs(20)
            .min_score(0.1)
            .load_documents(false)
            .timeout_ms(5000);

        assert_eq!(request.params.max_docs, 20);
        assert_eq!(request.params.min_score, 0.1);
        assert!(!request.params.load_documents);
        assert_eq!(request.params.timeout_ms, Some(5000));
    }
}

// Implement LexicalSearcher trait for InvertedIndexSearcher
impl crate::lexical::search::searcher::LexicalSearcher for InvertedIndexSearcher {
    fn search(&self, request: LexicalSearchRequest) -> Result<SearchResults> {
        InvertedIndexSearcher::search(self, request)
    }

    fn count(&self, query: crate::lexical::types::LexicalSearchQuery) -> Result<u64> {
        InvertedIndexSearcher::count(self, query)
    }
}
