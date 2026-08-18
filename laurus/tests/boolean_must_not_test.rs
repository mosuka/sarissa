//! Integration tests for issue #997 — MustNot-only boolean queries on
//! multi-segment indices.
//!
//! Before #997 these queries walked the dense `0..max_doc()` universe,
//! which under the per-segment fan-out is the *global* doc count: each
//! segment could only exclude its own negative matches, so the merged
//! results contained cross-segment duplicates, docs that should have
//! been excluded, phantom ids, and soft-deleted docs — and ids above
//! `max_doc()` (sparse post-merge id spaces) were unreachable.

use std::sync::Arc;

use laurus::Document;
use laurus::lexical::core::field::{FieldOption, TextOption};
use laurus::lexical::query::{BooleanQueryBuilder, Query, TermQuery};
use laurus::lexical::{LexicalIndexConfig, LexicalSearchRequest, LexicalStore};
use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};

fn body_doc(text: &str) -> Document {
    Document::builder().add_text("body", text).build()
}

fn new_store() -> LexicalStore {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let config = LexicalIndexConfig::builder()
        .add_field("body", FieldOption::Text(TextOption::default()))
        .build();
    LexicalStore::new(storage, config).unwrap()
}

fn must_not_alpha() -> Box<dyn Query> {
    Box::new(
        BooleanQueryBuilder::new()
            .must_not(Box::new(TermQuery::new("body", "alpha")))
            .build(),
    )
}

fn search_hits(store: &LexicalStore, query: Box<dyn Query>) -> Vec<u64> {
    let mut ids: Vec<u64> = store
        .search(LexicalSearchRequest::new(query))
        .unwrap()
        .hits
        .iter()
        .map(|hit| hit.doc_id)
        .collect();
    ids.sort_unstable();
    ids
}

/// #997: a MustNot-only query on a multi-segment index must return each
/// non-matching doc exactly once — no cross-segment duplicates, no docs
/// that match the negated term in another segment, no phantom ids.
#[test]
fn must_not_only_multi_segment_returns_each_doc_once() {
    let store = new_store();

    // Segment 1: doc 2 matches the negated term.
    store.upsert_document(1, body_doc("bravo charlie")).unwrap();
    store.upsert_document(2, body_doc("alpha bravo")).unwrap();
    store.commit().unwrap();
    // Segment 2: doc 5 matches the negated term.
    store.upsert_document(3, body_doc("delta echo")).unwrap();
    store.upsert_document(4, body_doc("foxtrot")).unwrap();
    store.upsert_document(5, body_doc("alpha golf")).unwrap();
    store.commit().unwrap();
    // Segment 3: no matches for the negated term.
    store.upsert_document(6, body_doc("hotel india")).unwrap();
    store.commit().unwrap();

    assert_eq!(
        search_hits(&store, must_not_alpha()),
        vec![1, 3, 4, 6],
        "every non-alpha doc exactly once; no duplicates, exclusions honored, no phantoms"
    );
}

/// #997: the parallel execution path must honor MustNot-only semantics
/// too. Before the fix, a single-clause MustNot-only query run with
/// `parallel = true` returned the *negated* docs (semantic inversion),
/// and a multi-clause one returned nothing.
#[test]
fn must_not_only_parallel_matches_serial() {
    let store = new_store();

    store.upsert_document(1, body_doc("bravo charlie")).unwrap();
    store.upsert_document(2, body_doc("alpha bravo")).unwrap();
    store.commit().unwrap();
    store.upsert_document(3, body_doc("delta echo")).unwrap();
    store.commit().unwrap();

    // Single MustNot clause, parallel.
    let mut request = LexicalSearchRequest::new(must_not_alpha());
    request.params.parallel = true;
    let mut ids: Vec<u64> = store
        .search(request)
        .unwrap()
        .hits
        .iter()
        .map(|hit| hit.doc_id)
        .collect();
    ids.sort_unstable();
    assert_eq!(ids, vec![1, 3], "parallel single-clause MustNot-only");

    // Two MustNot clauses, parallel.
    let query = Box::new(
        BooleanQueryBuilder::new()
            .must_not(Box::new(TermQuery::new("body", "alpha")))
            .must_not(Box::new(TermQuery::new("body", "echo")))
            .build(),
    );
    let mut request = LexicalSearchRequest::new(query);
    request.params.parallel = true;
    let mut ids: Vec<u64> = store
        .search(request)
        .unwrap()
        .hits
        .iter()
        .map(|hit| hit.doc_id)
        .collect();
    ids.sort_unstable();
    assert_eq!(ids, vec![1], "parallel multi-clause MustNot-only");
}

/// #997: docs whose ids lie above `max_doc()` (= Σ doc_count; sparse
/// post-merge id spaces) must still appear in MustNot-only results.
#[test]
fn must_not_only_includes_ids_above_max_doc() {
    let store = new_store();

    // Segment 1: low ids, doc 2 matches the negated term.
    store.upsert_document(1, body_doc("bravo")).unwrap();
    store.upsert_document(2, body_doc("alpha")).unwrap();
    store.commit().unwrap();
    // Segment 2: ids far above Σ doc_count (= 4).
    store.upsert_document(100, body_doc("charlie")).unwrap();
    store.upsert_document(101, body_doc("delta")).unwrap();
    store.commit().unwrap();

    assert_eq!(
        search_hits(&store, must_not_alpha()),
        vec![1, 100, 101],
        "ids above max_doc() must not be missed"
    );
}

/// #997: negating a term that matches nothing must return every live
/// doc — an exhausted negative excludes nothing. The old
/// `BooleanQuery::is_empty` treated an all-empty-clause boolean as
/// unmatchable, so the searcher's early exit returned zero hits.
#[test]
fn must_not_only_with_absent_term_matches_everything() {
    let store = new_store();

    store.upsert_document(1, body_doc("bravo")).unwrap();
    store.upsert_document(2, body_doc("charlie")).unwrap();
    store.commit().unwrap();
    store.upsert_document(3, body_doc("delta")).unwrap();
    store.commit().unwrap();

    let query = Box::new(
        BooleanQueryBuilder::new()
            .must_not(Box::new(TermQuery::new("body", "zzz")))
            .build(),
    );
    assert_eq!(
        search_hits(&store, query),
        vec![1, 2, 3],
        "an exhausted negative excludes nothing — the query matches everything"
    );
}

/// #997: a doc superseded by an upsert (its old copy is soft-deleted in
/// the earlier segment) must appear exactly once, and soft-deleted
/// copies must not surface as hits.
#[test]
fn must_not_only_excludes_soft_deleted_copies() {
    let store = new_store();

    // Segment 1: original version of doc 1.
    store.upsert_document(1, body_doc("bravo")).unwrap();
    store.commit().unwrap();
    // Segment 2: doc 1 upserted (segment-1 copy tombstoned) + one alpha doc.
    store.upsert_document(1, body_doc("charlie")).unwrap();
    store.upsert_document(2, body_doc("alpha")).unwrap();
    store.commit().unwrap();

    assert_eq!(
        search_hits(&store, must_not_alpha()),
        vec![1],
        "doc 1 exactly once; the tombstoned copy must not add a duplicate hit"
    );
}
