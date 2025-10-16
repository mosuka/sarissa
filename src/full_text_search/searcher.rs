//! Searcher implementation for executing queries against an index.

use std::sync::Arc;
use std::time::{Duration, Instant};

use rayon::prelude::*;

use crate::error::{Result, SageError};
use crate::full_text::reader::IndexReader;
use crate::full_text_search::SearchRequest;
use crate::query::collector::{Collector, CountCollector, TopDocsCollector};
use crate::query::query::Query;
use crate::query::{SearchHit, SearchResults};

/// A searcher that executes queries against an index reader.
#[derive(Debug)]
pub struct Searcher {
    /// The index reader to search against.
    reader: Arc<dyn IndexReader>,
}

impl Searcher {
    /// Create a new searcher with the given index reader.
    pub fn new(reader: Box<dyn IndexReader>) -> Self {
        Searcher {
            reader: Arc::from(reader),
        }
    }

    /// Create a new searcher with an Arc<dyn IndexReader>.
    pub fn from_arc(reader: Arc<dyn IndexReader>) -> Self {
        Searcher { reader }
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
        if parallel
            && let Some(boolean_query) = query
                .as_any()
                .downcast_ref::<crate::query::boolean::BooleanQuery>()
        {
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
            let score = scorer.score(doc_id, term_freq);

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
        boolean_query: &crate::query::boolean::BooleanQuery,
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

    /// Execute a search with timeout.
    fn search_with_timeout(
        &self,
        request: SearchRequest,
        timeout: Duration,
    ) -> Result<SearchResults> {
        let start_time = Instant::now();

        // Create collector
        let collector =
            TopDocsCollector::with_min_score(request.config.max_docs, request.config.min_score);

        // Execute search
        let result_collector =
            self.search_with_collector_parallel(request.query, collector, request.config.parallel)?;

        // Check if we exceeded timeout
        if start_time.elapsed() > timeout {
            return Err(SageError::index("Search timeout exceeded"));
        }

        // Get results
        let mut hits = result_collector.results();
        let total_hits = result_collector.total_hits();

        // Load documents if requested
        if request.config.load_documents {
            if request.config.parallel && hits.len() > 10 {
                self.load_documents_parallel(&mut hits)?;
            } else {
                self.load_documents(&mut hits)?;
            }
        }

        // Calculate max score
        let max_score = hits.iter().map(|hit| hit.score).fold(0.0f32, f32::max);

        Ok(SearchResults {
            hits,
            total_hits,
            max_score,
        })
    }

    /// Search with the given request.
    pub fn search(&self, request: SearchRequest) -> Result<SearchResults> {
        // Check if query is empty
        if request.query.is_empty(self.reader.as_ref())? {
            return Ok(SearchResults {
                hits: Vec::new(),
                total_hits: 0,
                max_score: 0.0,
            });
        }

        // Execute search with timeout if specified
        if let Some(timeout_ms) = request.config.timeout_ms {
            let timeout = Duration::from_millis(timeout_ms);
            self.search_with_timeout(request, timeout)
        } else {
            // Execute search without timeout
            let collector =
                TopDocsCollector::with_min_score(request.config.max_docs, request.config.min_score);
            let result_collector = self.search_with_collector_parallel(
                request.query,
                collector,
                request.config.parallel,
            )?;

            let mut hits = result_collector.results();
            let total_hits = result_collector.total_hits();

            // Load documents if requested
            if request.config.load_documents {
                self.load_documents(&mut hits)?;
            }

            // Calculate max score
            let max_score = hits.iter().map(|hit| hit.score).fold(0.0f32, f32::max);

            Ok(SearchResults {
                hits,
                total_hits,
                max_score,
            })
        }
    }

    /// Count documents matching the query.
    pub fn count(&self, query: Box<dyn Query>) -> Result<u64> {
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
    use crate::full_text_search::advanced_reader::{AdvancedIndexReader, AdvancedReaderConfig};
    use crate::query::boolean::{BooleanQuery, BooleanQueryBuilder};
    use crate::query::term::TermQuery;
    use crate::storage::memory::MemoryStorage;
    use crate::storage::traits::StorageConfig;
    use std::sync::Arc;

    #[allow(dead_code)]
    fn create_test_searcher() -> Searcher {
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let reader = Box::new(
            AdvancedIndexReader::new(vec![], storage, AdvancedReaderConfig::default()).unwrap(),
        );
        Searcher::new(reader)
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
        let query = Box::new(TermQuery::new("title", "hello"));

        let request = SearchRequest::new(query);
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
        );

        let request = SearchRequest::new(query);
        let results = searcher.search(request).unwrap();

        // Should return empty results for non-existent terms
        assert_eq!(results.hits.len(), 0);
        assert_eq!(results.total_hits, 0);
        assert_eq!(results.max_score, 0.0);
    }

    #[test]
    fn test_search_with_config() {
        let searcher = create_test_searcher();
        let query = Box::new(TermQuery::new("title", "hello"));

        let request = SearchRequest::new(query)
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
        let query = Box::new(TermQuery::new("title", "hello"));

        let count = searcher.count(query).unwrap();

        // Should return 0 for non-existent terms
        assert_eq!(count, 0);
    }

    #[test]
    fn test_search_with_timeout() {
        let searcher = create_test_searcher();
        let query = Box::new(TermQuery::new("title", "hello"));

        let request = SearchRequest::new(query).timeout_ms(1000); // 1 second timeout

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
        let query = Box::new(BooleanQuery::new());

        let request = SearchRequest::new(query);
        let results = searcher.search(request).unwrap();

        // Should return empty results for empty query
        assert_eq!(results.hits.len(), 0);
        assert_eq!(results.total_hits, 0);
        assert_eq!(results.max_score, 0.0);
    }

    #[test]
    fn test_count_empty_query() {
        let searcher = create_test_searcher();
        let query = Box::new(BooleanQuery::new());

        let count = searcher.count(query).unwrap();

        // Should return 0 for empty query
        assert_eq!(count, 0);
    }

    #[test]
    fn test_search_request_builder() {
        let query = Box::new(TermQuery::new("title", "hello"));

        let request = SearchRequest::new(query)
            .max_docs(20)
            .min_score(0.1)
            .load_documents(false)
            .timeout_ms(5000);

        assert_eq!(request.config.max_docs, 20);
        assert_eq!(request.config.min_score, 0.1);
        assert!(!request.config.load_documents);
        assert_eq!(request.config.timeout_ms, Some(5000));
    }
}
