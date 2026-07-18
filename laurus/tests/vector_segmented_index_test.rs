//! Gates for the segment-per-commit HNSW index (#634 PR-3 / #881).
//!
//! The deterministic core gate: committing 1 new document on an N-document
//! base writes O(delta) bytes — a new small segment file plus the manifest —
//! never a rewrite of the existing segments. Plus: config-OFF invariance
//! (the monolithic layout stays byte-identical), legacy-index rejection
//! (zero-copy migration is #882), multi-segment recall parity, and
//! upsert/delete semantics across commits.

use std::sync::Arc;

use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::vector::index::VectorIndex;
use laurus::vector::index::config::HnswIndexConfig;
use laurus::vector::index::hnsw::HnswIndex;
use laurus::vector::index::hnsw::segmented::SegmentedHnswIndex;
use laurus::vector::search::searcher::{VectorIndexQuery, VectorIndexQueryParams};
use laurus::vector::{DistanceMetric, Vector};

fn doc_vec(i: u64) -> Vector {
    let mut v = vec![0.0f32; 16];
    let t = i as f32 * 0.001;
    v[0] = t.cos();
    v[1] = t.sin();
    v[2] = (t * 2.0).cos();
    v[3] = (t * 3.0).sin();
    Vector::new(v)
}

fn config(segmented: bool) -> HnswIndexConfig {
    HnswIndexConfig {
        dimension: 16,
        m: 16,
        ef_construction: 100,
        normalize_vectors: false,
        distance_metric: DistanceMetric::Cosine,
        segmented,
        ..Default::default()
    }
}

fn storage() -> Arc<MemoryStorage> {
    Arc::new(MemoryStorage::new(MemoryStorageConfig::default()))
}

/// Add `ids` to a fresh writer and commit (one sealed segment).
fn commit_batch(index: &dyn VectorIndex, ids: std::ops::Range<u64>) {
    let mut writer = index.writer().unwrap();
    let vectors: Vec<_> = ids.map(|i| (i, "v".to_string(), doc_vec(i))).collect();
    writer.add_vectors(vectors).unwrap();
    writer.commit().unwrap();
}

fn query(id: u64, top_k: usize) -> VectorIndexQuery {
    VectorIndexQuery {
        query: doc_vec(id),
        params: VectorIndexQueryParams {
            top_k,
            ..Default::default()
        },
        field_name: Some("v".to_string()),
        filter: None,
    }
}

/// #881 core gate: a 1-doc commit on a 1000-doc base writes a new segment
/// that is a tiny fraction of the base segment, and never rewrites the base.
#[test]
fn one_doc_commit_writes_o_delta_bytes() {
    let storage = storage();
    let index = SegmentedHnswIndex::open_or_create(
        storage.clone() as Arc<dyn Storage>,
        "vector_index",
        config(true),
    )
    .unwrap();

    commit_batch(&index, 0..1000);
    let base_file = "segment_000000.hnsw";
    let base_size = storage.file_size(base_file).unwrap();

    commit_batch(&index, 1000..1001);
    let delta_file = "segment_000001.hnsw";
    let delta_size = storage.file_size(delta_file).unwrap();

    assert!(
        delta_size * 20 < base_size,
        "a 1-doc commit must write O(delta) bytes, got delta={delta_size} vs base={base_size}"
    );
    assert_eq!(
        storage.file_size(base_file).unwrap(),
        base_size,
        "the base segment must never be rewritten by a later commit (#634)"
    );

    // Exactly the two sealed segments exist — no other .hnsw files.
    let hnsw_files: Vec<String> = storage
        .list_files()
        .unwrap()
        .into_iter()
        .filter(|f| f.ends_with(".hnsw"))
        .collect();
    assert_eq!(
        hnsw_files.len(),
        2,
        "exactly one new segment per non-empty commit, got {hnsw_files:?}"
    );
}

/// #881: with `segmented: false` (the default) the factory path stays
/// monolithic — no manifest, single `.hnsw` file.
#[test]
fn config_off_keeps_monolithic_layout() {
    let storage = storage();
    let index = HnswIndex::create(
        storage.clone() as Arc<dyn Storage>,
        "vector_index",
        config(false),
    )
    .unwrap();

    commit_batch(&index, 0..100);
    assert!(
        !storage.file_exists("segments.json"),
        "config OFF must not create a segment manifest"
    );
    assert!(storage.file_exists("vector_index.hnsw"));
}

/// #881: opening a legacy monolithic index with the flag ON is rejected —
/// the zero-copy migration lands with #882.
#[test]
fn flag_on_rejects_legacy_monolithic_index() {
    let storage = storage();
    {
        let index = HnswIndex::create(
            storage.clone() as Arc<dyn Storage>,
            "vector_index",
            config(false),
        )
        .unwrap();
        commit_batch(&index, 0..10);
    }
    let err = SegmentedHnswIndex::open_or_create(
        storage as Arc<dyn Storage>,
        "vector_index",
        config(true),
    );
    assert!(
        err.is_err(),
        "a legacy monolithic index must be rejected until the #882 migration"
    );
}

/// #881: self-recall@10 of a 5-segment index matches a monolithic build of
/// the same corpus within tolerance (#872-style gate).
#[test]
fn multi_segment_self_recall_matches_monolithic() {
    let n = 2000u64;
    let per_commit = 400u64;

    let seg_storage = storage();
    let seg_index = SegmentedHnswIndex::open_or_create(
        seg_storage.clone() as Arc<dyn Storage>,
        "vector_index",
        config(true),
    )
    .unwrap();
    let mut lo = 0u64;
    while lo < n {
        commit_batch(&seg_index, lo..(lo + per_commit).min(n));
        lo += per_commit;
    }

    let mono_storage = storage();
    let mono_index = HnswIndex::create(
        mono_storage.clone() as Arc<dyn Storage>,
        "vector_index",
        config(false),
    )
    .unwrap();
    commit_batch(&mono_index, 0..n);

    let recall = |index: &dyn VectorIndex| -> f32 {
        let searcher = index.searcher().unwrap();
        let mut hits = 0u64;
        for id in 0..n {
            let results = searcher.search(&query(id, 10)).unwrap();
            if results.results.iter().any(|r| r.doc_id == id) {
                hits += 1;
            }
        }
        hits as f32 / n as f32
    };

    let mono = recall(&mono_index);
    let seg = recall(&seg_index);
    assert!(mono > 0.9, "monolithic self-recall sanity, got {mono:.4}");
    assert!(
        seg >= mono - 0.05,
        "multi-segment self-recall ({seg:.4}) must match the monolithic build \
         ({mono:.4}) within tolerance (#881)"
    );
}

/// #881: a same-id upsert across commits resolves to the newest copy exactly
/// once, and a soft delete of a sealed doc is search-invisible and physically
/// reclaimed by optimize().
#[test]
fn upsert_and_soft_delete_across_commits() {
    let storage = storage();
    let index = SegmentedHnswIndex::open_or_create(
        storage.clone() as Arc<dyn Storage>,
        "vector_index",
        config(true),
    )
    .unwrap();

    // Commit 1: docs 1..50 (doc 1's OLD embedding = doc_vec(1)).
    commit_batch(&index, 1..50);

    // Commit 2: upsert doc 1 to doc_vec(9000) via delete-first + re-add.
    {
        let mut writer = index.writer().unwrap();
        writer.delete_document(1).unwrap();
        writer
            .add_vectors(vec![(1, "v".to_string(), doc_vec(9000))])
            .unwrap();
        writer.commit().unwrap();
    }

    // The doc resolves via its NEW embedding...
    let searcher = index.searcher().unwrap();
    let results = searcher.search(&query(9000, 1)).unwrap();
    assert_eq!(results.results[0].doc_id, 1, "newest copy must win");
    // ...and its stale OLD copy must not outrank the true neighbours of the
    // old embedding.
    let results = searcher.search(&query(1, 1)).unwrap();
    assert_ne!(
        results.results[0].doc_id, 1,
        "the stale copy in the older segment must be masked (#880/#881)"
    );

    // Soft delete a sealed doc: search-invisible immediately.
    index.soft_delete_document(10).unwrap();
    index.persist_deletions().unwrap();
    let searcher = index.searcher().unwrap();
    let results = searcher.search(&query(10, 5)).unwrap();
    assert!(
        results.results.iter().all(|r| r.doc_id != 10),
        "a soft-deleted sealed doc must be search-invisible"
    );

    // Optimize: one merged segment, deletion physically reclaimed, bitmap
    // cleared.
    index.optimize().unwrap();
    let hnsw_files: Vec<String> = storage
        .list_files()
        .unwrap()
        .into_iter()
        .filter(|f| f.ends_with(".hnsw"))
        .collect();
    assert_eq!(
        hnsw_files.len(),
        1,
        "optimize must force-merge to one segment"
    );
    assert!(
        !storage.file_exists("vector_index.delmap"),
        "optimize must clear the persisted deletion bitmap"
    );
    let searcher = index.searcher().unwrap();
    let results = searcher.search(&query(10, 5)).unwrap();
    assert!(results.results.iter().all(|r| r.doc_id != 10));
    let results = searcher.search(&query(9000, 1)).unwrap();
    assert_eq!(
        results.results[0].doc_id, 1,
        "upserted copy survives the merge"
    );

    let stats = index.stats().unwrap();
    // ids 1..=49 (49 docs; the upsert replaces doc 1 in place) minus the
    // physically reclaimed doc 10.
    assert_eq!(stats.vector_count, 48);
}

/// #881 review (HIGH): the upsert dance can undelete every mark; a previously
/// persisted delmap must then be REMOVED — a stale file would mask the
/// committed upsert in every segment after reopen.
#[test]
fn undelete_to_zero_removes_stale_delmap_and_survives_reopen() {
    let storage = storage();
    {
        let index = SegmentedHnswIndex::open_or_create(
            storage.clone() as Arc<dyn Storage>,
            "vector_index",
            config(true),
        )
        .unwrap();
        commit_batch(&index, 1..10);

        // Soft-delete doc 3 and persist: the delmap file exists.
        index.soft_delete_document(3).unwrap();
        index.persist_deletions().unwrap();
        assert!(storage.file_exists("vector_index.delmap"));

        // Same-id upsert of doc 3: the re-add clears the only mark.
        let mut writer = index.writer().unwrap();
        writer.delete_document(3).unwrap();
        writer
            .add_vectors(vec![(3, "v".to_string(), doc_vec(9000))])
            .unwrap();
        writer.commit().unwrap();
        index.persist_deletions().unwrap();
        assert!(
            !storage.file_exists("vector_index.delmap"),
            "undelete-to-zero must remove the stale delmap (#881)"
        );
    }

    // Reopen: the committed upsert must be visible (a stale delmap would
    // have masked doc 3 in every segment).
    let index = SegmentedHnswIndex::open_or_create(
        storage as Arc<dyn Storage>,
        "vector_index",
        config(true),
    )
    .unwrap();
    let searcher = index.searcher().unwrap();
    let results = searcher.search(&query(9000, 1)).unwrap();
    assert_eq!(
        results.results[0].doc_id, 3,
        "the committed upsert must survive a reopen (#881)"
    );
}

/// #881 review (HIGH): a sealed writer must reject further commits (loud
/// error instead of silently resetting the segment's generation), while a
/// post-commit close() stays a clean no-op.
#[test]
fn sealed_writer_rejects_second_commit_and_close_is_noop() {
    let storage = storage();
    let index = SegmentedHnswIndex::open_or_create(
        storage.clone() as Arc<dyn Storage>,
        "vector_index",
        config(true),
    )
    .unwrap();

    let mut writer = index.writer().unwrap();
    writer
        .add_vectors(vec![(1, "v".to_string(), doc_vec(1))])
        .unwrap();
    writer.commit().unwrap();

    // Post-commit close: no pending changes, no error, no metadata churn.
    assert!(!writer.has_pending_changes());
    writer.close().unwrap();

    // A sealed writer with NEW changes must refuse to commit again.
    let mut writer = index.writer().unwrap();
    writer
        .add_vectors(vec![(2, "v".to_string(), doc_vec(2))])
        .unwrap();
    writer.commit().unwrap();
    writer
        .add_vectors(vec![(3, "v".to_string(), doc_vec(3))])
        .unwrap();
    let err = writer.commit();
    assert!(
        err.is_err(),
        "a sealed writer must reject a second commit with new changes (#881)"
    );
}

/// #881 review: count() excludes soft-deleted docs.
#[test]
fn count_excludes_soft_deleted_docs() {
    let storage = storage();
    let index = SegmentedHnswIndex::open_or_create(
        storage as Arc<dyn Storage>,
        "vector_index",
        config(true),
    )
    .unwrap();
    commit_batch(&index, 0..10);
    index.soft_delete_document(4).unwrap();

    let searcher = index.searcher().unwrap();
    let count = searcher.count(query(0, 1)).unwrap();
    assert_eq!(count, 9, "count must exclude soft-deleted docs (#881)");
}
