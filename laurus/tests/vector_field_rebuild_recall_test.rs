//! Issue #1080 acceptance test: rebuilding a live HNSW field's `m` /
//! `ef_construction` via [`MultiFieldVectorIndex::rebuild_field`] must
//! actually change search behavior, not just accept the call.
//!
//! Mirrors `vector_recall_test.rs`'s ground-truth-via-brute-force approach
//! (independent of any laurus index code path) but drives the index
//! through [`MultiFieldVectorIndex`] end to end: build with deliberately
//! poor HNSW parameters, measure recall, `rebuild_field` to much better
//! parameters, and confirm recall improves.

use std::any::Any;
use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;

use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::vector::Vector;
use laurus::vector::core::distance::DistanceMetric;
use laurus::vector::index::VectorIndex;
use laurus::vector::index::config::{HnswIndexConfig, VectorIndexTypeConfig};
use laurus::vector::index::multi_field::MultiFieldVectorIndex;
use laurus::vector::search::searcher::{VectorIndexQuery, VectorIndexQueryParams};
use laurus::{EmbedInput, EmbedInputType, Embedder, Result};

/// Never actually invoked: every vector in this test is inserted directly
/// via the writer, not embedded from raw input.
#[derive(Debug)]
struct UnusedEmbedder;

#[async_trait]
impl Embedder for UnusedEmbedder {
    async fn embed(&self, _input: &EmbedInput<'_>) -> Result<Vector> {
        Err(laurus::LaurusError::invalid_argument(
            "embedding not used by this test",
        ))
    }
    fn supported_input_types(&self) -> Vec<EmbedInputType> {
        vec![EmbedInputType::Text]
    }
    fn name(&self) -> &str {
        "unused"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

const DIM: usize = 32;
const N_VECTORS: usize = 400;
const N_QUERIES: usize = 30;
const TOP_K: usize = 10;
const FIELD: &str = "embedding";

fn pseudo_random_f32(seed: u32, len: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9).wrapping_add(0xDEAD_BEEF);
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            let bits = (state >> 16) as u16;
            (bits as f32 / u16::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

fn exact_cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0_f32;
    let mut na = 0.0_f32;
    let mut nb = 0.0_f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom == 0.0 { 1.0 } else { 1.0 - dot / denom }
}

fn exact_top_k(corpus: &[Vec<f32>], query: &[f32], k: usize) -> HashSet<u64> {
    let mut scored: Vec<(u64, f32)> = corpus
        .iter()
        .enumerate()
        .map(|(idx, v)| (idx as u64, exact_cosine_distance(query, v)))
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().take(k).map(|(id, _)| id).collect()
}

fn recall_at_k(exact: &HashSet<u64>, approx: &HashSet<u64>, k: usize) -> f32 {
    exact.intersection(approx).count() as f32 / k as f32
}

fn hnsw_config(m: usize, ef_construction: usize) -> VectorIndexTypeConfig {
    VectorIndexTypeConfig::HNSW(HnswIndexConfig {
        dimension: DIM,
        m,
        ef_construction,
        distance_metric: DistanceMetric::Cosine,
        normalize_vectors: true,
        ..Default::default()
    })
}

fn measure_recall(index: &MultiFieldVectorIndex, corpus: &[Vec<f32>], queries: &[Vec<f32>]) -> f32 {
    let searcher = index.searcher().expect("searcher");
    let mut total = 0.0_f32;
    for query in queries {
        let exact = exact_top_k(corpus, query, TOP_K);
        let request = VectorIndexQuery {
            query: Vector::new(query.clone()),
            params: VectorIndexQueryParams {
                top_k: TOP_K,
                ..Default::default()
            },
            field_name: Some(FIELD.to_string()),
            filter: None,
        };
        let results = searcher.search(&request).expect("search");
        let approx: HashSet<u64> = results.results.iter().map(|r| r.doc_id).collect();
        total += recall_at_k(&exact, &approx, TOP_K);
    }
    total / queries.len() as f32
}

/// Issue #1080 acceptance criterion: "Search results reflect the new
/// parameters after the rebuild (e.g. recall changes as expected)."
///
/// Builds an HNSW field with a deliberately weak graph (`m: 2`,
/// `ef_construction: 4` -- poor recall by construction, matching the
/// documented recall-vs-`ef_search`/`m` relationship also exercised in
/// `vector_recall_test.rs`), measures recall, `rebuild_field`s to a much
/// stronger graph (`m: 32`, `ef_construction: 200`) with the SAME vectors
/// already committed, and confirms recall improves substantially -- proof
/// the rebuild actually re-ran graph construction under the new config,
/// not just accepted the call as a no-op.
#[test]
fn rebuild_field_improves_recall_with_stronger_hnsw_params() {
    let corpus: Vec<Vec<f32>> = (0..N_VECTORS)
        .map(|i| pseudo_random_f32(i as u32, DIM))
        .collect();
    let queries: Vec<Vec<f32>> = (0..N_QUERIES)
        .map(|i| pseudo_random_f32(10_000 + i as u32, DIM))
        .collect();

    let mut fields = BTreeMap::new();
    fields.insert(FIELD.to_string(), hnsw_config(2, 4));
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let index =
        MultiFieldVectorIndex::open_or_create(storage, &fields, Arc::new(UnusedEmbedder)).unwrap();

    let mut writer = index.writer().unwrap();
    let docs: Vec<(u64, String, Vector)> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u64, FIELD.to_string(), Vector::new(v.clone())))
        .collect();
    writer.build(docs).unwrap();
    writer.finalize().unwrap();
    writer.commit().unwrap();

    let weak_recall = measure_recall(&index, &corpus, &queries);

    index.rebuild_field(FIELD, hnsw_config(32, 200)).unwrap();

    let strong_recall = measure_recall(&index, &corpus, &queries);

    assert!(
        strong_recall > weak_recall + 0.1,
        "rebuild_field with much stronger HNSW params must measurably improve \
         recall: weak={weak_recall}, strong={strong_recall}"
    );
}
