//! Issue #673 (#650 PR-3): PQ → SQ → f32 three-stage rerank chain.
//!
//! The existing Issue #481 Stage 3 rerank widens the PQ ADC candidate set
//! to `top_k * rerank_factor` and rescores those against the exact f32
//! sidecar. When `ef_search` is set wider than that budget, the graph
//! traversal computes extra candidates that Stage 3 alone simply
//! discarded — #673 inserts a cheap int8 (SQ) stage ahead of the exact
//! stage so that surplus gets a chance to compete instead.
//!
//! `pq_three_stage_rerank_end_to_end` exercises the mechanism end-to-end
//! through the public `HnswIndex` / `HnswSearcher` API against one shared
//! index build (the PQ training / graph construction cost dominates this
//! fixture's runtime, so splitting these into independently-built tests
//! would triple it for no added isolation value), asserting:
//!
//! 1. Widening `ef_search` past the exact-stage budget must not hurt
//!    Recall@K relative to the narrow (budget-exact) case — the new SQ
//!    stage narrows the surplus by a much better proxy (int8) than the
//!    PQ ADC order it arrived in.
//! 2. The exact stage still runs last, so the `score_basis` contract
//!    (Issue #927) is unaffected regardless of whether the SQ stage
//!    activated.
//! 3. At the narrow (budget-exact) `ef_search`, the SQ-stage gate
//!    (`effective_ef > top_k * rerank_factor`) is false by construction,
//!    so results are deterministic and reproducible across repeated
//!    queries (pinning the "SQ stage off" branch stays inert).
//!
//! Causal attribution — i.e. that the SQ stage specifically (not just a
//! wider graph search) is responsible for the recall floor — is proven
//! separately via a RED proof (temporarily disabling the SQ-stage
//! insertion in `hnsw/searcher.rs` and observing the wide-`ef_search`
//! recall drop to the narrow-`ef_search` level); see the implementation
//! report for #673.

use std::collections::HashSet;

use laurus::storage::StorageConfig;
use laurus::storage::StorageFactory;
use laurus::storage::memory::MemoryStorageConfig;
use laurus::vector::core::distance::DistanceMetric;
use laurus::vector::core::quantization::QuantizationMethod;
use laurus::vector::core::rerank::RerankStorageKind;
use laurus::vector::core::vector::Vector;
use laurus::vector::index::VectorIndex;
use laurus::vector::index::config::HnswIndexConfig;
use laurus::vector::index::hnsw::HnswIndex;
use laurus::vector::index::hnsw::searcher::HnswSearcher;
use laurus::vector::search::searcher::{VectorIndexQuery, VectorIndexSearcher};

/// Dimension / subvector-count pair matches the campaign's known-lossy
/// PQ regime (Issue #481 Stage 3 recall gate: `sub_dim = 4` clears the
/// ~0.78 PQ-only recall floor measured on this synthetic distribution,
/// leaving real ADC-vs-exact ranking error for the SQ stage to recover).
/// A near-lossless PQ config would make the narrow- and wide-`ef_search`
/// recall converge to ~1.0 regardless of the SQ stage, defeating the
/// purpose of these tests.
const DIM: usize = 128;
const SUBVECTOR_COUNT: usize = 32;
const TOP_K: usize = 10;
const N_CORPUS: usize = 3_000;
const N_QUERIES: usize = 30;
/// Small on purpose: a narrow exact-stage budget maximizes the surplus
/// (`ef_search - budget`) the SQ stage has to work with at the same
/// `ef_search`, making its effect (and the RED proof's regression)
/// clearly observable without a large fixture.
const RERANK_FACTOR: usize = 3;
/// Exactly `top_k * rerank_factor`: the SQ-stage gate
/// (`effective_ef > top_k * rerank_factor`) is false here by
/// construction, so this `ef_search` always takes the pre-#673 (2-stage)
/// code path regardless of segment kind.
const EF_NARROW: usize = TOP_K * RERANK_FACTOR;
/// Well past the exact-stage budget, activating the SQ stage on a PQ
/// segment.
const EF_WIDE: usize = 200;

/// `query_metadata` key/value the multi-segment fan-out (#927) keys on
/// to decide whether a segment's scores are on the exact f32 basis.
/// Mirrors the crate-private constants in
/// `laurus::vector::search::searcher` (`SCORE_BASIS_METADATA_KEY` /
/// `SCORE_BASIS_F32_RERANK`), which integration tests cannot import.
const SCORE_BASIS_KEY: &str = "score_basis";
const SCORE_BASIS_F32_RERANK: &str = "f32-rerank";

/// Deterministic pseudo-random f32 in `[lo, hi)` (same LCG as the other
/// vector test fixtures in this crate, duplicated per-file by
/// convention — see `vector_recall_test.rs`).
fn pseudo_random_f32(seed: u32, len: usize, lo: f32, hi: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9).wrapping_add(0xDEAD_BEEF);
    let range = hi - lo;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            let bits = (state >> 16) as u16;
            lo + (bits as f32 / u16::MAX as f32) * range
        })
        .collect()
}

/// Deterministic pseudo-random unit-norm vector.
fn pseudo_random_unit_norm(seed: u32, len: usize) -> Vec<f32> {
    let mut v = pseudo_random_f32(seed, len, -1.0, 1.0);
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

fn exact_cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        1.0
    } else {
        1.0 - dot / (na * nb)
    }
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

/// Build a PQ + rerank-sidecar HNSW index over `corpus` and return its
/// searcher, ready for repeated `.search()` calls at any `ef_search`.
fn build_pq_rerank_searcher(corpus: &[Vec<f32>]) -> HnswSearcher {
    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))
        .expect("memory storage");
    let config = HnswIndexConfig {
        dimension: DIM,
        m: 16,
        ef_construction: 200,
        distance_metric: DistanceMetric::Cosine,
        quantization_method: QuantizationMethod::ProductQuantization {
            subvector_count: SUBVECTOR_COUNT,
        },
        rerank_storage: Some(RerankStorageKind::F32),
        ..Default::default()
    };
    let index = HnswIndex::create(storage, "pq_three_stage_index", config).expect("create index");
    let mut writer = index.writer().expect("writer");
    let docs: Vec<(u64, String, Vector)> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u64, "embedding".to_string(), Vector::new(v.clone())))
        .collect();
    writer.build(docs).expect("build");
    writer.finalize().expect("finalize");
    writer.commit().expect("commit");

    let reader = index.reader().expect("reader");
    HnswSearcher::new(reader).expect("searcher")
}

fn fixture() -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let corpus: Vec<Vec<f32>> = (0..N_CORPUS)
        .map(|i| pseudo_random_unit_norm(0xCAFE_0000 + i as u32, DIM))
        .collect();
    let queries: Vec<Vec<f32>> = (0..N_QUERIES)
        .map(|i| pseudo_random_unit_norm(0xBEEF_0000 + i as u32, DIM))
        .collect();
    (corpus, queries)
}

/// Requirement (1): widening `ef_search` past the exact-stage budget
/// must not hurt Recall@K.
/// Single scenario test covering requirements (1)-(3) against one shared
/// PQ + rerank index build — the corpus/PQ-training/graph-construction
/// cost dominates this fixture's runtime, so splitting these checks
/// across independently-built indexes would triple it for no added
/// isolation value (all three assertions are about the same pipeline
/// behavior, not independent concerns).
#[test]
fn pq_three_stage_rerank_end_to_end() {
    let (corpus, queries) = fixture();
    let searcher = build_pq_rerank_searcher(&corpus);

    // Requirement (1): widening `ef_search` past the exact-stage budget
    // must not hurt Recall@K — the SQ stage narrows the surplus by a
    // much better proxy (int8) than the PQ ADC order it arrived in.
    let mut narrow_recall = 0.0_f32;
    let mut wide_recall = 0.0_f32;
    let mut wide_results_first: Option<laurus::vector::search::searcher::VectorIndexQueryResults> =
        None;
    for (i, query) in queries.iter().enumerate() {
        let exact = exact_top_k(&corpus, query, TOP_K);

        let narrow_request = VectorIndexQuery::new(Vector::new(query.clone()))
            .top_k(TOP_K)
            .field_name("embedding".to_string())
            .rerank_factor(RERANK_FACTOR)
            .ef_search(EF_NARROW);
        let narrow_results = searcher.search(&narrow_request).expect("narrow search");
        let narrow_approx: HashSet<u64> = narrow_results.results.iter().map(|r| r.doc_id).collect();
        narrow_recall += recall_at_k(&exact, &narrow_approx, TOP_K);

        // Requirement (3): at the narrow (budget-exact) `ef_search` the
        // SQ-stage gate is false, so the (unaffected) pre-#673 code path
        // runs; repeated identical queries must be bit-for-bit
        // reproducible.
        let narrow_repeat = searcher
            .search(&narrow_request)
            .expect("narrow search repeat");
        assert_eq!(narrow_results.results.len(), narrow_repeat.results.len());
        for (a, b) in narrow_results.results.iter().zip(&narrow_repeat.results) {
            assert_eq!(a.doc_id, b.doc_id);
            assert_eq!(a.distance.to_bits(), b.distance.to_bits());
        }
        assert_eq!(
            narrow_results
                .query_metadata
                .get(SCORE_BASIS_KEY)
                .map(String::as_str),
            Some(SCORE_BASIS_F32_RERANK),
            "the exact stage still runs at narrow ef_search, just without an SQ stage ahead of it"
        );

        let wide_request = VectorIndexQuery::new(Vector::new(query.clone()))
            .top_k(TOP_K)
            .field_name("embedding".to_string())
            .rerank_factor(RERANK_FACTOR)
            .ef_search(EF_WIDE);
        let wide_results = searcher.search(&wide_request).expect("wide search");
        let wide_approx: HashSet<u64> = wide_results.results.iter().map(|r| r.doc_id).collect();
        wide_recall += recall_at_k(&exact, &wide_approx, TOP_K);
        if i == 0 {
            wide_results_first = Some(wide_results);
        }
    }
    narrow_recall /= queries.len() as f32;
    wide_recall /= queries.len() as f32;

    eprintln!(
        "PQ 3-stage rerank: narrow ef={EF_NARROW} recall={narrow_recall:.4}, \
         wide ef={EF_WIDE} recall={wide_recall:.4} (rerank_factor={RERANK_FACTOR})"
    );
    assert!(
        wide_recall >= narrow_recall,
        "widening ef_search past the exact-stage budget must not hurt recall: \
         narrow={narrow_recall:.4} (ef={EF_NARROW}), wide={wide_recall:.4} (ef={EF_WIDE})"
    );

    // Requirement (2): the `score_basis` contract is unaffected by the
    // SQ stage — the pipeline still ends in the exact f32 stage
    // regardless of whether the SQ stage ran ahead of it.
    let wide_results = wide_results_first.expect("at least one query");
    assert_eq!(
        wide_results
            .query_metadata
            .get(SCORE_BASIS_KEY)
            .map(String::as_str),
        Some(SCORE_BASIS_F32_RERANK),
        "3-stage pipeline must still report the exact f32 basis: {:?}",
        wide_results.query_metadata
    );
}
