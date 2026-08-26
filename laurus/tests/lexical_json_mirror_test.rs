//! Integration test for dropping the `.json` stored-field mirror (Issue #756).
//!
//! Stored fields are now written only to the typed binary `.docs` file (no
//! `.json` mirror), and read from `.docs`. `.docs` records the real per-document
//! id, so stored-field lookup is correct even for **non-contiguous** doc ids
//! (e.g. merged segments) — unlike the old `.json` path, which assigned ids
//! positionally (`min_doc_id + index`).

use std::collections::BTreeMap;
use std::sync::Arc;

use laurus::lexical::{LexicalIndexConfig, LexicalSearchRequest, LexicalStore, TermQuery};
use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::{DataValue, Document};

fn doc(title: &str) -> Document {
    Document::builder()
        .add_text("title", title)
        .add_text("body", "lorem ipsum")
        .build()
}

/// Any segment `.json` stored-field mirror present?
fn has_json_mirror(storage: &Arc<dyn Storage>) -> bool {
    storage
        .list_files()
        .unwrap()
        .iter()
        .any(|f| (f.starts_with("segment_") || f.starts_with("merged_")) && f.ends_with(".json"))
}

fn count_files(storage: &Arc<dyn Storage>, suffix: &str) -> usize {
    storage
        .list_files()
        .unwrap()
        .iter()
        .filter(|f| f.ends_with(suffix))
        .count()
}

/// Map each hit's doc_id to its stored `title` (via `load_documents`, which
/// retrieves stored fields by doc_id — the path that differs between `.docs`
/// and the legacy `.json` mirror).
fn titles_by_id(store: &LexicalStore) -> BTreeMap<u64, String> {
    let query = Box::new(TermQuery::new("body", "lorem"));
    let request = LexicalSearchRequest::new(query).load_documents(true);
    store
        .search(request)
        .unwrap()
        .hits
        .into_iter()
        .map(|h| {
            let title = h
                .document
                .as_ref()
                .and_then(|d| d.fields.get("title"))
                .and_then(|v| match v {
                    DataValue::Text(t) => Some(t.clone()),
                    _ => None,
                })
                .unwrap_or_default();
            (h.doc_id, title)
        })
        .collect()
}

#[test]
fn drops_json_mirror_and_reads_stored_fields_from_docs() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    // Loose layout, explicitly: this test pins the #756 loose-file
    // contract (a `.docs` file and no `.json` mirror). Under the compound
    // default there are no loose files at all.
    let store = LexicalStore::new(
        storage.clone(),
        LexicalIndexConfig::Inverted(laurus::lexical::InvertedIndexConfig {
            use_compound: false,
            ..Default::default()
        }),
    )
    .unwrap();

    // Non-contiguous ids: the old positional `.json` read would map these wrong.
    store.upsert_document(10, doc("ten")).unwrap();
    store.upsert_document(20, doc("twenty")).unwrap();
    store.upsert_document(30, doc("thirty")).unwrap();
    store.commit().unwrap();

    let expected: BTreeMap<u64, String> = [
        (10, "ten".to_string()),
        (20, "twenty".to_string()),
        (30, "thirty".to_string()),
    ]
    .into_iter()
    .collect();

    // No `.json` mirror is written; `.docs` is.
    assert!(
        !has_json_mirror(&storage),
        "no .json mirror should be written"
    );
    assert!(count_files(&storage, ".docs") >= 1, ".docs must be written");

    // Stored fields read back correctly by their real (non-contiguous) ids.
    assert_eq!(titles_by_id(&store), expected, "stored fields via .docs");

    // After a merge the ids stay non-contiguous in the merged segment's `.docs`;
    // still no `.json`, still correct.
    store.optimize().unwrap();
    assert!(!has_json_mirror(&storage), "merged segment writes no .json");
    assert_eq!(
        titles_by_id(&store),
        expected,
        "stored fields correct after merge"
    );
}
