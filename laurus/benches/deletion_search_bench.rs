//! Deletion-aware HNSW search benchmark (Issue #625).
//!
//! Measures `VectorStore::search` (HNSW graph path) on a committed
//! in-memory index in two states:
//!
//! - `del0pct` — no deletions: the pristine, bookkeeping-free traversal;
//!   the no-regression guard for #625.
//! - `del10pct` — 10% of docs soft-deleted: every admitted neighbour is
//!   probed against the deletion set inside the bookkeeping loop. This
//!   is the gate metric for #625 (per-query snapshot + lock-free
//!   `contains` vs the previous per-probe bitmap read-lock).
//!
//! In-memory storage loads Eager, so distances run the int8 SIMD kernel
//! and the traversal + deletion bookkeeping dominate. Setup is
//! deterministic via `common::DEFAULT_SEED`. Uses only public APIs that
//! also exist on `main`, so the same bench runs on the src-only
//! `git stash` baseline.

mod common;

use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use criterion::{Criterion, criterion_group, criterion_main};
use tokio::runtime::Runtime;

use common::{DEFAULT_SEED, SAMPLE_SIZE_FAST, lcg_vec_unit};

use laurus::lexical::LexicalIndexConfig;
use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::vector::core::distance::DistanceMetric;
use laurus::vector::core::field::HnswOption;
use laurus::vector::search::searcher::{VectorSearchParams, VectorSearchRequest};
use laurus::vector::store::config::VectorFieldConfig;
use laurus::vector::store::request::{QueryVector, VectorScoreMode};
use laurus::vector::{FieldOption, Vector, VectorIndexConfig, VectorSearchQuery, VectorStore};
use laurus::{DataValue, Document, EmbedInput, EmbedInputType, Embedder, LaurusError, Result};

const DIM: usize = 128;
/// Corpus size: large enough that ef-search traversal probes hundreds
/// of neighbours per query.
const N: u64 = 10_000;

#[derive(Debug)]
struct MockEmbedder;

#[async_trait]
impl Embedder for MockEmbedder {
    async fn embed(&self, _input: &EmbedInput<'_>) -> Result<Vector> {
        Err(LaurusError::invalid_argument(
            "vectors are supplied directly",
        ))
    }
    fn supported_input_types(&self) -> Vec<EmbedInputType> {
        vec![EmbedInputType::Text]
    }
    fn name(&self) -> &str {
        "mock"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn make_config() -> VectorIndexConfig {
    let mut fields = HashMap::new();
    fields.insert(
        "vec".to_string(),
        VectorFieldConfig {
            vector: Some(FieldOption::Hnsw(HnswOption {
                dimension: DIM,
                distance: DistanceMetric::Cosine,
                m: 16,
                ef_construction: 200,
                default_ef_search: None,
                base_weight: 1.0,
                quantizer: Default::default(),
                rerank_storage: None,
                embedder: None,
            })),
            lexical: None,
        },
    );
    VectorIndexConfig {
        fields,
        embedder: Arc::new(MockEmbedder),
        default_fields: vec!["vec".to_string()],
        metadata: HashMap::new(),
        deletion_config: laurus::DeletionConfig::default(),
        shard_id: 0,
        metadata_config: LexicalIndexConfig::default(),
    }
}

/// Build a committed `VectorStore` holding `N` deterministic vectors.
fn build_committed_store(rt: &Runtime) -> VectorStore {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let store = VectorStore::new(storage, make_config()).unwrap();
    let mut state = DEFAULT_SEED;
    rt.block_on(async {
        for id in 0..N {
            let vec = lcg_vec_unit(&mut state, DIM);
            let doc = Document::builder()
                .add_field("vec", DataValue::Vector(vec))
                .build();
            store.upsert_document_by_internal_id(id, doc).await.unwrap();
        }
        store.commit().await.unwrap();
    });
    store
}

/// Top-10 request against the `vec` field (graph path).
fn request() -> VectorSearchRequest {
    let mut state = DEFAULT_SEED.wrapping_add(1);
    VectorSearchRequest {
        query: VectorSearchQuery::Vectors(vec![QueryVector {
            vector: Vector::new(lcg_vec_unit(&mut state, DIM)),
            weight: 1.0,
            fields: Some(vec!["vec".to_string()]),
        }]),
        params: VectorSearchParams {
            limit: 10,
            score_mode: VectorScoreMode::WeightedSum,
            ..Default::default()
        },
    }
}

fn bench_deletion_search(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("Deletion Search");
    group.sample_size(SAMPLE_SIZE_FAST);

    let store = build_committed_store(&rt);

    // Sanity: the graph path must return hits before timing.
    let probe = store.search(request()).unwrap();
    assert!(!probe.hits.is_empty(), "probe must return hits");

    // 0% deletions: pristine traversal — the no-regression guard.
    group.bench_function("top10/del0pct", |b| {
        b.iter(|| store.search(request()).unwrap())
    });

    // Soft-delete 10% of docs (every 10th id), then measure the
    // deletion-bookkeeping traversal. No commit needed: the bitmap is
    // live and attached to freshly built readers.
    rt.block_on(async {
        for id in (0..N).step_by(10) {
            store.delete_document_by_internal_id(id).await.unwrap();
        }
    });
    let probe = store.search(request()).unwrap();
    assert!(
        !probe.hits.is_empty(),
        "post-delete probe must still return hits"
    );

    group.bench_function("top10/del10pct", |b| {
        b.iter(|| store.search(request()).unwrap())
    });

    group.finish();
}

criterion_group!(benches, bench_deletion_search);
criterion_main!(benches);
