//! #554 — compound segment containers, end to end.
//!
//! A segment flushed with `use_compound` writes ONE `.cfs` container
//! instead of loose per-part files; readers detect the layout per
//! segment, so loose and compound segments coexist in one index. The
//! round-trip matrix runs the four read paths with distinct access
//! patterns — postings (offset seeks), BKD trees (absolute seeks +
//! re-open per query), stored documents and doc values (sequential) —
//! over every storage backend, because the windowed views translate
//! seeks differently per backend.

use std::sync::Arc;

use laurus::lexical::index::LexicalIndex;
use laurus::lexical::index::inverted::InvertedIndex;
use laurus::lexical::{
    InvertedIndexConfig, LexicalSearchRequest, NumericRangeQuery, NumericType, TermQuery,
};
use laurus::storage::Storage;
use laurus::storage::file::{FileStorage, FileStorageConfig};
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::{DataValue, Document};
use tempfile::TempDir;

fn compound_config() -> InvertedIndexConfig {
    InvertedIndexConfig {
        use_compound: true,
        ..Default::default()
    }
}

fn doc(body: &str, rank: i64) -> Document {
    Document::builder()
        .add_field("body", DataValue::Text(body.into()))
        .add_field("rank", DataValue::Int64(rank))
        .build()
}

/// Commit two segments (3 + 1 docs) through an index and return it.
fn build_index(storage: Arc<dyn Storage>) -> InvertedIndex {
    let index = InvertedIndex::create(storage, compound_config()).unwrap();
    let mut writer = index.writer().unwrap();
    for rank in 1..=3i64 {
        writer
            .upsert_document(rank as u64, doc(&format!("alpha doc{rank}"), rank))
            .unwrap();
    }
    writer.commit().unwrap();
    writer.upsert_document(4, doc("alpha doc4", 40)).unwrap();
    writer.commit().unwrap();
    index
}

/// The four read paths against a compound index.
fn assert_round_trip(index: &InvertedIndex, storage: &Arc<dyn Storage>) {
    let searcher = index.searcher().unwrap();

    // Postings (offset-seeking reads inside the `.post` window).
    let hits = searcher
        .search(LexicalSearchRequest::new(Box::new(TermQuery::new(
            "body", "alpha",
        ))))
        .unwrap();
    assert_eq!(hits.hits.len(), 4, "postings through the container");

    // BKD (absolute seeks, re-opened per query through the facade).
    let hits = searcher
        .search(LexicalSearchRequest::new(Box::new(NumericRangeQuery::new(
            "rank",
            NumericType::Integer,
            Some(2.0),
            Some(3.0),
            true,
            true,
        ))))
        .unwrap();
    assert_eq!(hits.hits.len(), 2, "BKD range through the container");

    // Stored documents (sequential window read).
    let one = searcher
        .search(LexicalSearchRequest::new(Box::new(TermQuery::new(
            "body", "doc4",
        ))))
        .unwrap();
    assert_eq!(one.hits.len(), 1);
    let stored = one.hits[0]
        .document
        .as_ref()
        .expect("stored document must be readable through the container");
    assert_eq!(stored.fields.get("rank"), Some(&DataValue::Int64(40)));

    // Layout: one container per segment, zero loose part files.
    let files = storage.list_files().unwrap();
    let containers = files.iter().filter(|f| f.ends_with(".cfs")).count();
    assert_eq!(containers, 2, "one container per committed segment");
    for suffix in [".post", ".dict", ".docs", ".lens", ".fstats", ".dv", ".bkd"] {
        assert!(
            !files.iter().any(|f| f.ends_with(suffix)),
            "no loose {suffix} may exist next to the containers: {files:?}"
        );
    }
}

#[test]
fn round_trip_on_memory_storage() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let index = build_index(storage.clone());
    assert_round_trip(&index, &storage);
}

#[test]
fn round_trip_on_mmap_file_storage() {
    let dir = TempDir::new().unwrap();
    let config = FileStorageConfig::new(dir.path());
    assert!(config.use_mmap, "mmap must be the default under test");
    let storage: Arc<dyn Storage> = Arc::new(FileStorage::new(dir.path(), config).unwrap());
    let index = build_index(storage.clone());
    assert_round_trip(&index, &storage);
}

#[test]
fn round_trip_on_buffered_file_storage() {
    let dir = TempDir::new().unwrap();
    let config = FileStorageConfig {
        use_mmap: false,
        ..FileStorageConfig::new(dir.path())
    };
    let storage: Arc<dyn Storage> = Arc::new(FileStorage::new(dir.path(), config).unwrap());
    let index = build_index(storage.clone());
    assert_round_trip(&index, &storage);
}

/// Reopening reads compound segments cold (fresh facades, fresh windows).
#[test]
fn compound_segments_survive_a_reopen() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let index = build_index(storage.clone());
    drop(index);

    let reopened = InvertedIndex::open(storage.clone(), compound_config()).unwrap();
    assert_round_trip(&reopened, &storage);
}

/// BLOCKER-1 (#554 review): a merge of compound sources must preserve
/// every numeric point. The merge engine used to enumerate `.bkd` files
/// by listing raw storage — inside a container it finds none, and the
/// merged segment silently drops all points (`verify_after_merge` only
/// checks doc_count).
#[test]
fn merge_of_compound_sources_preserves_points() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let index = build_index(storage.clone());

    index.optimize().unwrap();

    let searcher = index.searcher().unwrap();
    let hits = searcher
        .search(LexicalSearchRequest::new(Box::new(NumericRangeQuery::new(
            "rank",
            NumericType::Integer,
            Some(1.0),
            Some(40.0),
            true,
            true,
        ))))
        .unwrap();
    assert_eq!(
        hits.hits.len(),
        4,
        "every numeric point must survive a merge of compound sources"
    );

    // The merged output is itself a container…
    let merged_cfs = storage
        .list_files()
        .unwrap()
        .into_iter()
        .find(|f| f.starts_with("merged_") && f.ends_with(".cfs"))
        .expect("the merged segment must use the compound layout too");

    // …and it must physically CARRY the BKD part. The query assertion
    // above is not sufficient on its own: numeric range queries fall back
    // to a scan when a segment has no BKD tree, so a merge that silently
    // dropped the points would still answer correctly (just slowly) — the
    // part table is the ground truth.
    let mut input = storage.open_input(&merged_cfs).unwrap();
    let mut bytes = Vec::new();
    std::io::Read::read_to_end(&mut input, &mut bytes).unwrap();
    assert!(
        bytes.windows(b"rank.bkd".len()).any(|w| w == b"rank.bkd"),
        "the merged container must carry the rank BKD part"
    );
}

/// Loose (pre-#554) and compound segments coexist in one index.
#[test]
fn loose_and_compound_segments_coexist() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

    // Segment 1: loose layout (the pre-#554 default).
    let index = InvertedIndex::create(storage.clone(), InvertedIndexConfig::default()).unwrap();
    let mut writer = index.writer().unwrap();
    writer.upsert_document(1, doc("alpha loose", 1)).unwrap();
    writer.commit().unwrap();
    drop(writer);
    drop(index);

    // Segment 2: compound layout, same directory.
    let index = InvertedIndex::open(storage.clone(), compound_config()).unwrap();
    let mut writer = index.writer().unwrap();
    writer.upsert_document(2, doc("alpha packed", 2)).unwrap();
    writer.commit().unwrap();

    let searcher = index.searcher().unwrap();
    let hits = searcher
        .search(LexicalSearchRequest::new(Box::new(TermQuery::new(
            "body", "alpha",
        ))))
        .unwrap();
    assert_eq!(hits.hits.len(), 2, "both layouts must serve the same query");

    let files = storage.list_files().unwrap();
    assert!(files.iter().any(|f| f.ends_with(".post")), "loose segment");
    assert!(
        files.iter().any(|f| f.ends_with(".cfs")),
        "compound segment"
    );
}

/// Deletions on a compound segment: the loose `.delmap` next to the
/// container must be found and applied — the facade's passthrough half.
#[test]
fn deletion_on_a_compound_segment_is_applied() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let index = InvertedIndex::create(storage.clone(), compound_config()).unwrap();
    let mut writer = index.writer().unwrap();
    writer.upsert_document(1, doc("alpha original", 1)).unwrap();
    writer.commit().unwrap();

    // Overwrite: marks a deletion in the sealed compound segment.
    writer.upsert_document(1, doc("alpha replaced", 1)).unwrap();
    writer.commit().unwrap();

    let searcher = index.searcher().unwrap();
    let hits = searcher
        .search(LexicalSearchRequest::new(Box::new(TermQuery::new(
            "body", "original",
        ))))
        .unwrap();
    assert!(
        hits.hits.is_empty(),
        "the superseded version must be filtered via the loose .delmap"
    );
    let hits = searcher
        .search(LexicalSearchRequest::new(Box::new(TermQuery::new(
            "body", "replaced",
        ))))
        .unwrap();
    assert_eq!(hits.hits.len(), 1);
    assert!(
        storage
            .list_files()
            .unwrap()
            .iter()
            .any(|f| f.ends_with(".delmap")),
        "the deletion bitmap stays a loose file next to the container"
    );
}
