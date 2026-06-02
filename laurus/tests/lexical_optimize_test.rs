//! Integration test for `LexicalStore::optimize()` force-merging segments
//! (Issue #754 — wiring the merge engine into production).
//!
//! Before #754 `optimize()` was a no-op, so committing repeatedly grew the
//! segment count without bound. This verifies that optimize now compacts every
//! segment into one, deletes the source segments, and leaves search results
//! unchanged — and is idempotent.

use std::sync::Arc;

use laurus::Document;
use laurus::lexical::{LexicalIndexConfig, LexicalSearchRequest, LexicalStore, TermQuery};
use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};

fn doc(title: &str) -> Document {
    Document::builder()
        .add_text("title", title)
        .add_text("body", "lorem ipsum")
        .build()
}

/// Count discovered segment metadata files (`segment_*` flushed + `merged_*`).
fn segment_count(storage: &Arc<dyn Storage>) -> usize {
    storage
        .list_files()
        .unwrap()
        .iter()
        .filter(|f| (f.starts_with("segment_") || f.starts_with("merged_")) && f.ends_with(".meta"))
        .count()
}

fn hits(store: &LexicalStore, field: &str, term: &str) -> usize {
    let query = Box::new(TermQuery::new(field, term));
    store
        .search(LexicalSearchRequest::new(query))
        .unwrap()
        .hits
        .len()
}

#[test]
fn optimize_force_merges_segments_into_one() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let store = LexicalStore::new(storage.clone(), LexicalIndexConfig::default()).unwrap();

    // Three commits -> three segments.
    store.upsert_document(1, doc("alpha")).unwrap();
    store.upsert_document(2, doc("bravo")).unwrap();
    store.commit().unwrap();
    store.upsert_document(3, doc("charlie")).unwrap();
    store.commit().unwrap();
    store.upsert_document(4, doc("delta")).unwrap();
    store.commit().unwrap();

    assert_eq!(
        segment_count(&storage),
        3,
        "three commits => three segments"
    );
    let before = hits(&store, "body", "lorem");
    assert_eq!(before, 4, "all four docs match body:lorem before optimize");

    store.optimize().unwrap();

    assert_eq!(
        segment_count(&storage),
        1,
        "optimize must force-merge into a single segment"
    );
    let leftover_sources = storage
        .list_files()
        .unwrap()
        .into_iter()
        .filter(|f| f.starts_with("segment_"))
        .count();
    assert_eq!(leftover_sources, 0, "source segment files must be deleted");

    // Search results are unchanged after the merge.
    assert_eq!(hits(&store, "body", "lorem"), before);
    assert_eq!(hits(&store, "title", "charlie"), 1, "per-doc term survives");

    // Idempotent: re-optimizing a single-segment index is a no-op.
    store.optimize().unwrap();
    assert_eq!(segment_count(&storage), 1, "re-optimize is a no-op");
    assert_eq!(hits(&store, "body", "lorem"), before);
}
