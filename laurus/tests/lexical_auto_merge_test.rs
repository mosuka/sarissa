//! Integration test for the post-commit auto-merge hook (Issue #755).
//!
//! After each commit, `LexicalStore` invokes `maybe_merge`, which merges the
//! smallest `merge_factor` segments once the segment count exceeds
//! `max_segments`. This keeps the segment count bounded without a manual
//! `optimize()`, and is a no-op below the threshold.

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

/// With a low `max_segments`, repeated commits stay bounded: each commit adds a
/// segment, and `maybe_merge` compacts the smallest ones back down once the
/// threshold is crossed — without any manual `optimize()`.
#[test]
fn auto_merge_keeps_segment_count_bounded() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let config = LexicalIndexConfig::builder()
        .max_segments(2)
        .merge_factor(2)
        .build();
    let store = LexicalStore::new(storage.clone(), config).unwrap();

    // One doc per commit => one segment per commit, but auto-merge keeps the
    // count from growing past the threshold.
    let titles = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"];
    for (i, title) in titles.iter().enumerate() {
        store.upsert_document((i + 1) as u64, doc(title)).unwrap();
        store.commit().unwrap();
        assert!(
            segment_count(&storage) <= 2,
            "after commit {}: segment count {} must stay <= max_segments (2)",
            i + 1,
            segment_count(&storage),
        );
    }

    // Steady state: exactly `max_segments` segments after enough commits.
    assert_eq!(segment_count(&storage), 2);
    // Every document is still searchable, and a per-doc term survives.
    assert_eq!(hits(&store, "body", "lorem"), titles.len());
    assert_eq!(hits(&store, "title", "echo"), 1);
}

/// A high `max_segments` disables auto-merge: commits accumulate segments (the
/// `maybe_merge` no-op path), so users can opt out by raising the threshold.
#[test]
fn auto_merge_noop_above_threshold() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let config = LexicalIndexConfig::builder().max_segments(1000).build();
    let store = LexicalStore::new(storage.clone(), config).unwrap();

    for i in 1..=4u64 {
        store.upsert_document(i, doc("doc")).unwrap();
        store.commit().unwrap();
    }

    assert_eq!(
        segment_count(&storage),
        4,
        "no merge below threshold => one segment per commit"
    );
    assert_eq!(hits(&store, "body", "lorem"), 4);
}
