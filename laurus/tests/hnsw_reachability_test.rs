//! Regression test for #868 — HNSW incremental-append can leave layer-0 nodes
//! unreachable from the entry point (silent recall loss across commit cycles).
//!
//! The connectivity invariant: after building an HNSW index (here across
//! several append+write cycles, the common multi-commit ingest pattern), every
//! node in the written graph must be reachable from the entry point by a BFS
//! over the layer-0 adjacency. A node with no in-edge is forward-connected but
//! never *discovered* by a search from the entry point, so it silently drops
//! out of results even though `document_count` still counts it.
//!
//! This asserts the property directly on the on-disk adjacency (via the
//! reader's `OrdinalHnswGraph`), so it is deterministic and free of search
//! recall / benchmark noise.

use std::collections::VecDeque;
use std::sync::Arc;

use laurus::storage::file::{FileStorage, FileStorageConfig};
use laurus::vector::{
    DistanceMetric, HnswIndexConfig, HnswIndexReader, HnswIndexWriter, Vector, VectorIndexWriter,
    VectorIndexWriterConfig,
};
use tempfile::tempdir;

/// Deterministic unit-ish vector at a small angle step, so the M nearest
/// neighbours are genuinely competitive (the regime where back-edge pruning
/// bites).
fn doc_vec(i: u64) -> Vector {
    let theta = i as f32 * 0.01;
    Vector::new(vec![theta.cos(), theta.sin()])
}

fn hnsw_config() -> HnswIndexConfig {
    HnswIndexConfig {
        dimension: 2,
        m: 16,
        ef_construction: 100,
        normalize_vectors: false,
        distance_metric: DistanceMetric::Cosine,
        ..Default::default()
    }
}

/// BFS from the entry point over layer-0 adjacency; returns the number of
/// nodes reachable (including the entry point).
fn layer0_reachable_count(graph: &laurus::vector::index::hnsw::graph::OrdinalHnswGraph) -> usize {
    let Some(entry) = graph.entry_point() else {
        return 0;
    };
    let n = graph.node_count();
    let mut seen = vec![false; n];
    let mut queue = VecDeque::new();
    seen[entry as usize] = true;
    queue.push_back(entry);
    let mut count = 1;
    while let Some(ord) = queue.pop_front() {
        if let Some(neighbors) = graph.neighbors(ord, 0) {
            for &nb in neighbors {
                if !seen[nb as usize] {
                    seen[nb as usize] = true;
                    count += 1;
                    queue.push_back(nb);
                }
            }
        }
    }
    count
}

/// Build `total` docs across `cycles` append+write cycles, then assert every
/// node in the written graph is reachable from the entry point on layer 0.
fn assert_all_reachable_after_multi_commit(cycles: u64, per_cycle: u64) {
    let dir = tempdir().unwrap();
    let path = dir.path();
    let name = "reach";
    let config = hnsw_config();
    let writer_config = VectorIndexWriterConfig {
        parallel_build: true,
        ..Default::default()
    };

    for cycle in 0..cycles {
        let storage_config = FileStorageConfig::new(path);
        let storage = Arc::new(FileStorage::new(path, storage_config).unwrap());

        let mut writer = if cycle == 0 {
            HnswIndexWriter::with_storage(config.clone(), writer_config.clone(), name, storage)
                .unwrap()
        } else {
            HnswIndexWriter::load(config.clone(), writer_config.clone(), storage, name).unwrap()
        };

        let vectors: Vec<(u64, String, Vector)> = (0..per_cycle)
            .map(|i| {
                let id = cycle * per_cycle + i;
                (id, format!("doc{id}"), doc_vec(id))
            })
            .collect();
        writer.add_vectors(vectors).unwrap();
        writer.finalize().unwrap();
        writer.write().unwrap();
    }

    let total = cycles * per_cycle;
    let storage_config = FileStorageConfig::new(path);
    let storage = Arc::new(FileStorage::new(path, storage_config).unwrap());
    let reader = HnswIndexReader::load(storage, name, DistanceMetric::Cosine).unwrap();
    let graph = reader
        .graph
        .as_ref()
        .expect("a graph must have been written");

    assert_eq!(
        graph.node_count(),
        total as usize,
        "all {total} vectors must be present in the written graph",
    );
    let reachable = layer0_reachable_count(graph);
    assert_eq!(
        reachable, total as usize,
        "every node must be reachable from the entry point on layer 0 \
         (got {reachable}/{total}); unreachable nodes are silently dropped \
         from search results (#868)",
    );
}

/// Build `total` docs in a SINGLE fresh build (the full-build path), then
/// assert every node is reachable from the entry point on layer 0.
fn assert_all_reachable_after_fresh_build(total: u64) {
    let dir = tempdir().unwrap();
    let path = dir.path();
    let name = "reach_fresh";
    let config = hnsw_config();
    let writer_config = VectorIndexWriterConfig {
        parallel_build: true,
        ..Default::default()
    };

    {
        let storage_config = FileStorageConfig::new(path);
        let storage = Arc::new(FileStorage::new(path, storage_config).unwrap());
        let mut writer =
            HnswIndexWriter::with_storage(config.clone(), writer_config.clone(), name, storage)
                .unwrap();
        let vectors: Vec<(u64, String, Vector)> = (0..total)
            .map(|id| (id, format!("doc{id}"), doc_vec(id)))
            .collect();
        writer.add_vectors(vectors).unwrap();
        writer.finalize().unwrap();
        writer.write().unwrap();
    }

    let storage_config = FileStorageConfig::new(path);
    let storage = Arc::new(FileStorage::new(path, storage_config).unwrap());
    let reader = HnswIndexReader::load(storage, name, DistanceMetric::Cosine).unwrap();
    let graph = reader.graph.as_ref().expect("a graph must be written");
    assert_eq!(graph.node_count(), total as usize);
    let reachable = layer0_reachable_count(graph);
    assert_eq!(
        reachable, total as usize,
        "every node in a fresh {total}-vector build must be reachable on layer 0 \
         (got {reachable}/{total}) (#868)",
    );
}

/// #868: 30 docs across 3 commit cycles — the multi-commit ingest pattern that
/// surfaced the bug during the #864 campaign.
#[test]
fn multi_commit_graph_is_fully_reachable() {
    assert_all_reachable_after_multi_commit(3, 10);
}

/// #868: a larger multi-commit build, run a few times to catch the
/// nondeterministic disconnection (the missing set varies per run).
#[test]
fn multi_commit_graph_is_fully_reachable_repeated() {
    for _ in 0..5 {
        assert_all_reachable_after_multi_commit(5, 20);
    }
}

/// #868: the catastrophic case — a fresh parallel build at scale (the concrete
/// pre-fix failure was ~186/5000 reachable). Repeated to catch nondeterministic
/// disconnection.
#[test]
fn fresh_build_5k_is_fully_reachable_repeated() {
    for _ in 0..5 {
        assert_all_reachable_after_fresh_build(5_000);
    }
}

/// #868: a larger fresh parallel build (pre-fix ~4933/20000 reachable).
#[test]
fn fresh_build_20k_is_fully_reachable() {
    assert_all_reachable_after_fresh_build(20_000);
}
