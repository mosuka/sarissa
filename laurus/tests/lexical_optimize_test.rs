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
    // Count via the manifest (#1024): `.meta` files are gone, and
    // `segments.json` is the sole record of the committed segment set.
    let mut input = storage.open_input("segments.json").unwrap();
    let mut bytes = Vec::new();
    std::io::Read::read_to_end(&mut input, &mut bytes).unwrap();
    let payload: serde_json::Value = match serde_json::from_slice(&bytes) {
        Ok(v) => v,
        Err(_) => {
            let mut len: u64 = 0;
            let mut shift = 0;
            let mut cursor = 0usize;
            loop {
                let byte = bytes[cursor];
                cursor += 1;
                len |= u64::from(byte & 0x7F) << shift;
                if byte & 0x80 == 0 {
                    break;
                }
                shift += 7;
            }
            serde_json::from_slice(&bytes[cursor..cursor + len as usize]).unwrap()
        }
    };
    payload["segments"].as_array().unwrap().len()
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

/// #1017 — `optimize()` with a flushed-but-uncommitted segment in flight.
///
/// Segment discovery now skips unpublished segments, so a force-merge sees
/// only committed ones. That would have left an unpublished segment outside
/// the merge while `invalidate_segment_cache` still dropped the writer's NRT
/// readers for it — reintroducing #1016, where `get` and `delete` silently
/// stop finding those documents and an upsert leaves a duplicate.
///
/// `optimize()` therefore commits the live writer first. This pins that: the
/// in-flight documents survive the optimize, resolve by `_id`, and end up in
/// the merged segment rather than being stranded.
#[test]
fn optimize_commits_uncommitted_documents_instead_of_stranding_them() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let store = LexicalStore::new(storage.clone(), LexicalIndexConfig::default()).unwrap();

    // Two committed segments, so the force-merge has something to do.
    for i in 0..2u64 {
        store
            .upsert_document(
                i + 1,
                Document::builder()
                    .add_text("_id", format!("committed{i}"))
                    .add_text("title", "committed")
                    .add_text("body", "lorem ipsum")
                    .build(),
            )
            .unwrap();
        store.commit().unwrap();
    }

    // A third document left uncommitted in the live writer.
    store
        .upsert_document(
            99,
            Document::builder()
                .add_text("_id", "inflight")
                .add_text("title", "inflight")
                .add_text("body", "lorem ipsum")
                .build(),
        )
        .unwrap();

    store.optimize().unwrap();

    // The in-flight document must not have been stranded by the merge:
    // optimize commits first, so it is published and searchable afterwards.
    assert_eq!(
        hits(&store, "title", "inflight"),
        1,
        "an uncommitted document must survive optimize, not be stranded by it"
    );
    assert_eq!(
        hits(&store, "_id", "inflight"),
        1,
        "and it must still resolve by `_id`, exactly once"
    );

    // And everything ended up in one merged segment.
    assert_eq!(
        segment_count(&storage),
        1,
        "optimize must leave one segment"
    );
    assert_eq!(hits(&store, "body", "lorem"), 3);
}
