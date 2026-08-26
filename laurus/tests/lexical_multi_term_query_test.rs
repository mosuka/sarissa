//! Index-backed end-to-end tests for the multi-term query types
//! (Prefix / Wildcard / Fuzzy / Regexp) — Issue #613.
//!
//! These are the first hit-level tests for these query types (prior
//! coverage was structural only). The multi-segment cases pin the
//! Issue #613 bug: before the searcher-level rewrite, the per-segment
//! fanout handed each segment a `PerSegmentReaderView`, which cannot
//! enumerate the term dictionary, so a raw multi-term query against a
//! ≥2-segment index silently returned 0 hits.

use std::sync::Arc;

use laurus::Document;
use laurus::lexical::query::Query;
use laurus::lexical::query::fuzzy::FuzzyQuery;
use laurus::lexical::query::prefix::PrefixQuery;
use laurus::lexical::query::regexp::RegexpQuery;
use laurus::lexical::query::wildcard::WildcardQuery;
use laurus::lexical::{LexicalIndexConfig, LexicalSearchRequest, LexicalStore};
use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};

fn doc(body: &str) -> Document {
    Document::builder().add_text("body", body).build()
}

/// Corpus with distinctive stems. Expected matches:
/// - prefix/wildcard/regexp `program…`: docs 1, 2, 3
/// - fuzzy "programing" (1 edit): doc 1 ("programming")
const CORPUS: [(u64, &str); 4] = [
    (1, "programming in rust"),
    (2, "a programmer at work"),
    (3, "program notes for the concert"),
    (4, "python coding session"),
];

/// Build a store with the corpus split across `n_segments` commits
/// (one commit per chunk; auto-merge disabled via a high threshold).
fn store_with_segments(n_segments: usize) -> (LexicalStore, Arc<dyn Storage>) {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let config = LexicalIndexConfig::builder().max_segments(1000).build();
    let store = LexicalStore::new(storage.clone(), config).unwrap();

    let chunk = CORPUS.len().div_ceil(n_segments);
    for group in CORPUS.chunks(chunk) {
        for (id, body) in group {
            store.upsert_document(*id, doc(body)).unwrap();
        }
        store.commit().unwrap();
    }
    (store, storage)
}

fn search_ids(store: &LexicalStore, query: Box<dyn Query>) -> Vec<u64> {
    let mut ids: Vec<u64> = store
        .search(LexicalSearchRequest::new(query).limit(10))
        .unwrap()
        .hits
        .iter()
        .map(|h| h.doc_id)
        .collect();
    ids.sort_unstable();
    ids
}

/// Run the four query types against a store and assert hit sets.
fn assert_multi_term_hits(store: &LexicalStore, label: &str) {
    assert_eq!(
        search_ids(store, Box::new(PrefixQuery::new("body", "program"))),
        vec![1, 2, 3],
        "{label}: PrefixQuery(body, program)"
    );
    assert_eq!(
        search_ids(
            store,
            Box::new(WildcardQuery::new("body", "program*").unwrap())
        ),
        vec![1, 2, 3],
        "{label}: WildcardQuery(body, program*)"
    );
    assert_eq!(
        search_ids(
            store,
            Box::new(FuzzyQuery::new("body", "programing").max_edits(1))
        ),
        vec![1],
        "{label}: FuzzyQuery(body, programing, 1 edit)"
    );
    assert_eq!(
        search_ids(
            store,
            Box::new(RegexpQuery::new("body", "program.*").unwrap())
        ),
        vec![1, 2, 3],
        "{label}: RegexpQuery(body, program.*)"
    );
}

/// Single segment: all four multi-term query types return the expected
/// hits (this passed before #613's fix as well).
#[test]
fn multi_term_queries_hit_on_single_segment() {
    let (store, storage) = store_with_segments(1);
    let segs = storage
        .list_files()
        .unwrap()
        .iter()
        .filter(|f| f.starts_with("segment_") && f.ends_with(".cfs"))
        .count();
    assert_eq!(segs, 1, "harness must produce exactly 1 segment");
    assert_multi_term_hits(&store, "single-segment");
}

/// Multi-segment: the same queries over a 2-segment index. Before the
/// Issue #613 searcher-level rewrite this returned 0 hits (the fanout's
/// `PerSegmentReaderView` cannot enumerate terms); it must return the
/// same hits as the single-segment case.
#[test]
fn multi_term_queries_hit_on_multi_segment() {
    let (store, storage) = store_with_segments(2);
    let segs = storage
        .list_files()
        .unwrap()
        .iter()
        .filter(|f| f.starts_with("segment_") && f.ends_with(".cfs"))
        .count();
    assert_eq!(segs, 2, "harness must produce exactly 2 segments");
    assert_multi_term_hits(&store, "multi-segment");
}
