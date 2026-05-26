//! Multi-vector search throughput benchmark for
//! [`laurus::vector::VectorStore::search`] (issue
//! [#710](https://github.com/mosuka/laurus/issues/710) Phase 1 of
//! [#648](https://github.com/mosuka/laurus/issues/648)).
//!
//! Measures end-to-end search throughput when the request carries B query
//! vectors. The bench compares the rayon-parallelised path
//! (`parallel_threshold = 0`) against the serial-only path
//! (`parallel_threshold = usize::MAX`) for the same input.
//!
//! # Acceptance thresholds (PR is held back if any line fails)
//!
//! - `B=1`: parallel regression ≤ +2 % vs serial. The B=1 case actually
//!   exercises [`laurus::vector::VectorStore::search`]'s single-vector fast
//!   path, but is included as a sanity guard against accidental regressions
//!   in unrelated paths.
//! - `B=4`: parallel speedup ≥ 1.8× vs serial (4-core baseline).
//! - `B=16`: parallel speedup ≥ 5×.
//! - `B=64`: parallel speedup ≥ 8× (or up to host CPU core count).
//!
//! # Run
//!
//! ```sh
//! cargo bench --bench vector_multi_query_bench
//! cargo bench --bench vector_multi_query_bench -- "multi_query_b/parallel/4"
//! ```

use async_trait::async_trait;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::any::Any;
use std::hint::black_box;
use std::sync::Arc;

use laurus::lexical::LexicalIndexConfig;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::vector::Vector;
use laurus::vector::core::distance::DistanceMetric;
use laurus::vector::core::field::HnswOption;
use laurus::vector::store::config::VectorFieldConfig;
use laurus::vector::store::request::{
    QueryVector, VectorScoreMode, VectorSearchParams, VectorSearchRequest,
};
use laurus::vector::{FieldOption, VectorIndexConfig, VectorSearchQuery};
use laurus::{DataValue, Document};
use laurus::{EmbedInput, EmbedInputType, Embedder};
use laurus::{LaurusError, Result};

const DIMENSION: usize = 128;
const CORPUS_SIZE: usize = 5_000;

#[derive(Debug)]
struct MockEmbedder {
    dimension: usize,
}

#[async_trait]
impl Embedder for MockEmbedder {
    async fn embed(&self, input: &EmbedInput<'_>) -> Result<Vector> {
        match input {
            EmbedInput::Text(_) => Ok(Vector::new(vec![0.0; self.dimension])),
            _ => Err(LaurusError::invalid_argument(
                "this embedder only supports text input",
            )),
        }
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

/// Build a deterministic vector with a single hot dimension.
fn make_vector(seed: u64, dimension: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; dimension];
    let hot = (seed as usize) % dimension;
    v[hot] = 1.0;
    v[(hot + 1) % dimension] = 0.5;
    v[(hot + 7) % dimension] = 0.25;
    v
}

fn build_store_with_dataset(dimension: usize, n: usize) -> laurus::vector::VectorStore {
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    rt.block_on(async {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let mut field_configs = std::collections::HashMap::new();
        field_configs.insert(
            "vector_field".to_string(),
            VectorFieldConfig {
                vector: Some(FieldOption::Hnsw(HnswOption {
                    dimension,
                    distance: DistanceMetric::Cosine,
                    m: 16,
                    ef_construction: 100,
                    default_ef_search: None,
                    base_weight: 1.0,
                    quantizer: Default::default(),
                    rerank_storage: None,
                    embedder: None,
                })),
                lexical: None,
            },
        );
        let config = VectorIndexConfig {
            fields: field_configs,
            embedder: Arc::new(MockEmbedder { dimension }),
            default_fields: vec!["vector_field".to_string()],
            metadata: std::collections::HashMap::new(),
            deletion_config: laurus::DeletionConfig::default(),
            shard_id: 0,
            metadata_config: LexicalIndexConfig::default(),
        };
        let store = laurus::vector::VectorStore::new(storage, config).unwrap();
        for i in 0..n {
            let v = make_vector(i as u64, dimension);
            let doc = Document::builder()
                .add_field("vector_field", DataValue::Vector(v))
                .build();
            store
                .upsert_document_by_internal_id((i + 1) as u64, doc)
                .await
                .unwrap();
        }
        store.commit().await.unwrap();
        store
    })
}

fn build_queries(b: usize, dimension: usize) -> Vec<QueryVector> {
    // Use a different seed offset from the corpus so queries are
    // deterministic but not identical to indexed vectors.
    (0..b)
        .map(|i| QueryVector {
            vector: Vector::new(make_vector((i as u64).wrapping_mul(31) + 17, dimension)),
            weight: 1.0,
            fields: None,
        })
        .collect()
}

fn build_request(queries: Vec<QueryVector>) -> VectorSearchRequest {
    VectorSearchRequest {
        query: VectorSearchQuery::Vectors(queries),
        params: VectorSearchParams {
            limit: 10,
            score_mode: VectorScoreMode::WeightedSum,
            ..Default::default()
        },
    }
}

fn bench_multi_query(c: &mut Criterion) {
    let store = build_store_with_dataset(DIMENSION, CORPUS_SIZE);

    let mut group = c.benchmark_group("multi_query_b");
    for b in [1_usize, 4, 16, 64] {
        let queries = build_queries(b, DIMENSION);

        group.bench_with_input(BenchmarkId::new("serial", b), &b, |bench, _| {
            bench.iter(|| {
                let req = build_request(queries.clone());
                let res = store
                    .search_with_threshold(req, usize::MAX)
                    .expect("search");
                black_box(res);
            });
        });

        group.bench_with_input(BenchmarkId::new("parallel", b), &b, |bench, _| {
            bench.iter(|| {
                let req = build_request(queries.clone());
                let res = store.search_with_threshold(req, 0).expect("search");
                black_box(res);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_multi_query);
criterion_main!(benches);
