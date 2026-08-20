//! Index-backed end-to-end tests for `SortField::Field` — Issue #608.
//!
//! `TopFieldCollector` had two stacked bugs: (1) `needs_more()`
//! short-circuited the searcher loop the instant the heap filled to
//! `limit`, so later-iterated documents with a better field value were
//! never seen; (2) `FieldScoredDoc::Ord` was inverted for both sort
//! directions, so once eviction *did* run (bug (1) hid this in
//! production) it discarded the best candidate instead of the worst.
//! These tests build a corpus where the true top-K by field value is
//! NOT the first-K-by-doc_id, so they fail under either bug alone.

use std::sync::Arc;

use laurus::Document;
use laurus::lexical::query::Query;
use laurus::lexical::{
    BooleanQueryBuilder, LexicalIndexConfig, LexicalSearchRequest, LexicalStore, TermQuery,
};
use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};

fn doc(popularity: i64) -> Document {
    Document::builder()
        .add_text("body", "alpha")
        .add_integer("popularity", popularity)
        .build()
}

/// 12 docs, `popularity` grouped so the true top-3 (either direction)
/// is never the first 3 or last 3 doc_ids encountered in iteration
/// order: doc_id 1..12 -> popularity 50,51,52, 10,11,12, 90,91,92,
/// 30,31,32.
const POPULARITY: [i64; 12] = [50, 51, 52, 10, 11, 12, 90, 91, 92, 30, 31, 32];

/// Build a store with one doc per `popularity` entry, split across
/// `n_segments` commits (one commit per chunk; auto-merge disabled via
/// a high `max_segments` threshold), mirroring
/// `lexical_multi_term_query_test.rs::store_with_segments`.
fn store_with_segments(n_segments: usize) -> (LexicalStore, Arc<dyn Storage>) {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let config = LexicalIndexConfig::builder().max_segments(1000).build();
    let store = LexicalStore::new(storage.clone(), config).unwrap();

    let chunk = POPULARITY.len().div_ceil(n_segments);
    for (group_idx, group) in POPULARITY.chunks(chunk).enumerate() {
        for (offset, popularity) in group.iter().enumerate() {
            let doc_id = (group_idx * chunk + offset) as u64 + 1;
            store.upsert_document(doc_id, doc(*popularity)).unwrap();
        }
        store.commit().unwrap();
    }
    (store, storage)
}

/// Run a field-sorted search and return hit doc_ids **in result order**
/// (not re-sorted — result order is exactly what's under test).
fn field_sorted_ids(store: &LexicalStore, request: LexicalSearchRequest) -> Vec<u64> {
    store
        .search(request)
        .unwrap()
        .hits
        .iter()
        .map(|h| h.doc_id)
        .collect()
}

#[test]
fn field_sort_desc_single_segment() {
    let (store, _storage) = store_with_segments(1);
    let query: Box<dyn Query> = Box::new(TermQuery::new("body", "alpha"));
    let results = store
        .search(
            LexicalSearchRequest::new(query)
                .limit(3)
                .sort_by_field_desc("popularity"),
        )
        .unwrap();

    // True top-3 by popularity: docs 7,8,9 (90,91,92), descending.
    assert_eq!(
        results.hits.iter().map(|h| h.doc_id).collect::<Vec<_>>(),
        vec![9, 8, 7]
    );
    assert_eq!(
        results.total_hits, 12,
        "total_hits must be the true match count"
    );
}

#[test]
fn field_sort_asc_single_segment() {
    let (store, _storage) = store_with_segments(1);
    let query: Box<dyn Query> = Box::new(TermQuery::new("body", "alpha"));
    let ids = field_sorted_ids(
        &store,
        LexicalSearchRequest::new(query)
            .limit(3)
            .sort_by_field_asc("popularity"),
    );

    // True top-3 by popularity: docs 4,5,6 (10,11,12), ascending.
    // (The naive `needs_more`-only fix returns the largest 3 instead —
    // the strongest detector of the inverted-comparator bug.)
    assert_eq!(ids, vec![4, 5, 6]);
}

#[test]
fn field_sort_multi_segment() {
    let (store, _storage) = store_with_segments(4);
    let query: Box<dyn Query> = Box::new(TermQuery::new("body", "alpha"));
    let results = store
        .search(
            LexicalSearchRequest::new(query)
                .limit(3)
                .sort_by_field_desc("popularity"),
        )
        .unwrap();

    assert_eq!(
        results.hits.iter().map(|h| h.doc_id).collect::<Vec<_>>(),
        vec![9, 8, 7]
    );
    // #944 Phase A: the per-segment fanout must preserve the documented
    // true-match-count contract by summing per-segment totals.
    assert_eq!(
        results.total_hits, 12,
        "total_hits must remain the true match count under the fanout"
    );
}

/// #944 Phase A: `min_score` must be honored on the multi-segment
/// fanout path too (per-segment collectors apply it).
#[test]
fn field_sort_multi_segment_with_min_score() {
    let (store, _storage) = store_with_segments(4);
    let query: Box<dyn Query> = Box::new(TermQuery::new("body", "alpha"));

    let all = store
        .search(
            LexicalSearchRequest::new(query.clone_box())
                .limit(3)
                .sort_by_field_desc("popularity"),
        )
        .unwrap();
    let uniform_score = all.hits[0].score;

    let excluded = field_sorted_ids(
        &store,
        LexicalSearchRequest::new(query.clone_box())
            .limit(3)
            .min_score(uniform_score + 1.0)
            .sort_by_field_desc("popularity"),
    );
    assert!(excluded.is_empty());

    let included = field_sorted_ids(
        &store,
        LexicalSearchRequest::new(query)
            .limit(3)
            .min_score(0.0)
            .sort_by_field_desc("popularity"),
    );
    assert_eq!(included, vec![9, 8, 7]);
}

#[test]
fn field_sort_parallel_boolean() {
    // Two Should clauses (both match every doc) so the search routes
    // through `search_boolean_query_parallel`'s survivor loop rather
    // than the default single-clause loop.
    let (store, _storage) = store_with_segments(1);
    let query: Box<dyn Query> = Box::new(
        BooleanQueryBuilder::new()
            .should(Box::new(TermQuery::new("body", "alpha")))
            .should(Box::new(TermQuery::new("body", "alpha")))
            .build(),
    );
    let ids = field_sorted_ids(
        &store,
        LexicalSearchRequest::new(query)
            .limit(3)
            .sort_by_field_desc("popularity")
            .parallel(true),
    );

    assert_eq!(ids, vec![9, 8, 7]);
}

#[test]
fn field_sort_with_min_score() {
    let (store, _storage) = store_with_segments(1);
    let query: Box<dyn Query> = Box::new(TermQuery::new("body", "alpha"));

    // A term query scores every match identically, so a `min_score`
    // above that score should exclude everything without disturbing
    // the sort logic for whatever remains.
    let all = store
        .search(LexicalSearchRequest::new(query.clone_box()).limit(1))
        .unwrap();
    let uniform_score = all.hits[0].score;

    let excluded = field_sorted_ids(
        &store,
        LexicalSearchRequest::new(query.clone_box())
            .limit(3)
            .min_score(uniform_score + 1.0)
            .sort_by_field_desc("popularity"),
    );
    assert!(excluded.is_empty());

    let included = field_sorted_ids(
        &store,
        LexicalSearchRequest::new(query)
            .limit(3)
            .min_score(0.0)
            .sort_by_field_desc("popularity"),
    );
    assert_eq!(included, vec![9, 8, 7]);
}

/// #943: a doc upserted across a segment boundary must sort by its NEW
/// field value. The tombstoned pre-upsert copy's DocValues entry in the
/// older segment used to shadow it (segments are scanned oldest-first
/// with no `is_deleted` check).
#[test]
fn field_sort_reflects_upserted_value_across_segments() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let config = LexicalIndexConfig::builder().max_segments(1000).build();
    let store = LexicalStore::new(storage, config).unwrap();

    // Segment 1: doc 1 with the soon-to-be-stale value + doc 2.
    store.upsert_document(1, doc(10)).unwrap();
    store.upsert_document(2, doc(50)).unwrap();
    store.commit().unwrap();
    // Segment 2: doc 1 upserted with a new value + doc 3.
    store.upsert_document(1, doc(99)).unwrap();
    store.upsert_document(3, doc(20)).unwrap();
    store.commit().unwrap();

    let query: Box<dyn Query> = Box::new(TermQuery::new("body", "alpha"));
    let ids = field_sorted_ids(
        &store,
        LexicalSearchRequest::new(query)
            .limit(3)
            .sort_by_field_desc("popularity"),
    );

    assert_eq!(
        ids,
        vec![1, 2, 3],
        "doc 1 must sort by its upserted value (99), not the stale 10"
    );
}

/// Storage decorator counting `open_input` calls on `.dv` files, so a
/// test can assert DocValues are loaded once per segment rather than
/// once per `get_doc_value` call (#943; same pattern as #995's
/// stored-documents gate).
#[derive(Debug)]
struct DvOpenCountingStorage {
    inner: Arc<dyn Storage>,
    dv_opens: Arc<std::sync::atomic::AtomicU64>,
}

impl Storage for DvOpenCountingStorage {
    fn open_input(&self, name: &str) -> laurus::Result<Box<dyn laurus::storage::StorageInput>> {
        if name.ends_with(".dv") {
            self.dv_opens
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        self.inner.open_input(name)
    }
    fn create_output(&self, name: &str) -> laurus::Result<Box<dyn laurus::storage::StorageOutput>> {
        self.inner.create_output(name)
    }
    fn create_output_append(
        &self,
        name: &str,
    ) -> laurus::Result<Box<dyn laurus::storage::StorageOutput>> {
        self.inner.create_output_append(name)
    }
    fn delete_file(&self, name: &str) -> laurus::Result<()> {
        self.inner.delete_file(name)
    }
    fn file_exists(&self, name: &str) -> bool {
        self.inner.file_exists(name)
    }
    fn list_files(&self) -> laurus::Result<Vec<String>> {
        self.inner.list_files()
    }
    fn file_size(&self, name: &str) -> laurus::Result<u64> {
        self.inner.file_size(name)
    }
    fn rename_file(&self, from: &str, to: &str) -> laurus::Result<()> {
        self.inner.rename_file(from, to)
    }
    fn metadata(&self, name: &str) -> laurus::Result<laurus::storage::FileMetadata> {
        self.inner.metadata(name)
    }
    fn create_temp_output(
        &self,
        prefix: &str,
    ) -> laurus::Result<(String, Box<dyn laurus::storage::StorageOutput>)> {
        self.inner.create_temp_output(prefix)
    }
    fn sync(&self) -> laurus::Result<()> {
        self.inner.sync()
    }
    fn close(&mut self) -> laurus::Result<()> {
        Ok(())
    }
}

/// #943: repeated field-sorted searches must load each segment's `.dv`
/// file once — not re-parse it on every per-hit `get_doc_value` call.
#[test]
fn doc_values_load_once_per_segment() {
    let dv_opens = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let storage: Arc<dyn Storage> = Arc::new(DvOpenCountingStorage {
        inner: Arc::new(MemoryStorage::new(MemoryStorageConfig::default())),
        dv_opens: dv_opens.clone(),
    });
    let config = LexicalIndexConfig::builder().max_segments(1000).build();
    let store = LexicalStore::new(storage, config).unwrap();

    store.upsert_document(1, doc(10)).unwrap();
    store.upsert_document(2, doc(50)).unwrap();
    store.commit().unwrap();
    store.upsert_document(3, doc(99)).unwrap();
    store.upsert_document(4, doc(20)).unwrap();
    store.commit().unwrap();

    for _ in 0..2 {
        let query: Box<dyn Query> = Box::new(TermQuery::new("body", "alpha"));
        let ids = field_sorted_ids(
            &store,
            LexicalSearchRequest::new(query)
                .limit(4)
                .sort_by_field_desc("popularity"),
        );
        assert_eq!(ids, vec![3, 2, 4, 1]);
    }

    assert_eq!(
        dv_opens.load(std::sync::atomic::Ordering::Relaxed),
        2,
        ".dv must be loaded once per segment, not per get_doc_value call"
    );
}
