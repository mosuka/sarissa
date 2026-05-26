//! Integration tests for `VectorIndexSearcher::search_batch` trait method
//! (issue [#712](https://github.com/mosuka/laurus/issues/712) Phase 2 of
//! [#648](https://github.com/mosuka/laurus/issues/648)).
//!
//! These tests verify the trait method's default impl directly, independent
//! of `VectorStore::search`:
//!
//! - The default impl produces one result per input query in order.
//! - The threshold parameter controls parallel vs serial without changing
//!   the result.
//! - An empty input returns an empty result vector without invoking
//!   `search` at all.

use laurus::Result;
use laurus::vector::Vector;
use laurus::vector::search::searcher::{VectorIndexQueryResult, VectorIndexQueryResults};
use laurus::vector::{VectorIndexQuery, VectorIndexSearcher};
use std::sync::atomic::{AtomicUsize, Ordering};

/// A minimal `VectorIndexSearcher` implementation that records how many
/// times its `search` method was called and returns one synthetic hit per
/// call.
#[derive(Debug, Default)]
struct CountingSearcher {
    call_count: AtomicUsize,
}

impl VectorIndexSearcher for CountingSearcher {
    fn search(&self, request: &VectorIndexQuery) -> Result<VectorIndexQueryResults> {
        let idx = self.call_count.fetch_add(1, Ordering::Relaxed);
        // Encode the call index into the synthetic hit so the test can
        // verify ordering of returned results.
        Ok(VectorIndexQueryResults {
            results: vec![VectorIndexQueryResult {
                doc_id: idx as u64,
                field_name: request
                    .field_name
                    .clone()
                    .unwrap_or_else(|| "test".to_string()),
                similarity: 1.0 - (idx as f32) * 0.01,
                distance: (idx as f32) * 0.01,
                vector: None,
            }],
            candidates_examined: 1,
            search_time_ms: 0.0,
            query_metadata: std::collections::HashMap::new(),
        })
    }

    fn count(&self, _request: VectorIndexQuery) -> Result<u64> {
        Ok(0)
    }
}

fn dummy_query(seed: u64) -> VectorIndexQuery {
    VectorIndexQuery::new(Vector::new(vec![seed as f32, 0.0, 0.0, 0.0])).top_k(1)
}

#[test]
fn test_search_batch_default_impl_preserves_order() {
    let searcher = CountingSearcher::default();
    let queries: Vec<VectorIndexQuery> = (0..5).map(dummy_query).collect();

    // Force serial path so ordering is straightforward to assert.
    let results = searcher
        .search_batch_with_threshold(&queries, usize::MAX)
        .expect("search_batch_with_threshold");

    assert_eq!(results.len(), queries.len());
    assert_eq!(searcher.call_count.load(Ordering::Relaxed), 5);
    // Serial path: doc_ids should be 0..5 in order.
    for (i, r) in results.iter().enumerate() {
        assert_eq!(r.results.len(), 1, "each query should produce one hit");
        assert_eq!(
            r.results[0].doc_id, i as u64,
            "serial dispatch order should match input order"
        );
    }
}

#[test]
fn test_search_batch_threshold_zero_runs_parallel_but_results_match() {
    let queries: Vec<VectorIndexQuery> = (0..8).map(dummy_query).collect();

    let serial_searcher = CountingSearcher::default();
    let serial = serial_searcher
        .search_batch_with_threshold(&queries, usize::MAX)
        .expect("serial");

    let parallel_searcher = CountingSearcher::default();
    let parallel = parallel_searcher
        .search_batch_with_threshold(&queries, 0)
        .expect("parallel");

    // Both paths must invoke `search` exactly B times.
    assert_eq!(serial_searcher.call_count.load(Ordering::Relaxed), 8);
    assert_eq!(parallel_searcher.call_count.load(Ordering::Relaxed), 8);

    assert_eq!(serial.len(), parallel.len());
    // The parallel path may interleave doc_id assignment (because
    // `call_count.fetch_add` races between threads), but the returned
    // `Vec` must still preserve the input order — i.e., results[i] is
    // the answer for queries[i] regardless of which thread serviced it.
    // Verify that each parallel result contains exactly one hit and that
    // the set of doc_ids returned matches the set produced by the serial
    // path (a permutation, not necessarily the same order of doc_ids).
    let mut serial_ids: Vec<u64> = serial.iter().map(|r| r.results[0].doc_id).collect();
    let mut parallel_ids: Vec<u64> = parallel.iter().map(|r| r.results[0].doc_id).collect();
    serial_ids.sort_unstable();
    parallel_ids.sort_unstable();
    assert_eq!(
        serial_ids, parallel_ids,
        "parallel and serial must produce the same doc_id set"
    );
}

#[test]
fn test_search_batch_empty_queries() {
    let searcher = CountingSearcher::default();
    let queries: Vec<VectorIndexQuery> = Vec::new();

    let results = searcher.search_batch(&queries).expect("empty batch");
    assert!(results.is_empty());
    assert_eq!(
        searcher.call_count.load(Ordering::Relaxed),
        0,
        "empty input must not invoke search"
    );

    // Also verify the threshold variant short-circuits identically.
    let results = searcher
        .search_batch_with_threshold(&queries, 0)
        .expect("empty batch with threshold=0");
    assert!(results.is_empty());
    assert_eq!(searcher.call_count.load(Ordering::Relaxed), 0);
}

#[test]
fn test_search_batch_default_threshold_value() {
    let searcher = CountingSearcher::default();
    // The trait's default `parallel_threshold` is 4 (matching Phase 1).
    assert_eq!(searcher.parallel_threshold(), 4);
}
