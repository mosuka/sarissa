//! HNSW deletion + commit latency benchmark (Issue #624).
//!
//! Measures the production delete path: `VectorStore::delete_document_by_internal_id`
//! followed by `commit()` on an HNSW index that already has `N` committed
//! vectors on disk.
//!
//! This is the path #624 targets. Before #624 a delete loaded the existing
//! graph, nulled it, and the following commit rebuilt the whole graph
//! (`O(N log N)` distance evaluations). After #624 a delete is a logical mark
//! on a deletion bitmap, so commit cost is independent of `N`. Run the same
//! benchmark with the change stashed (`git stash push -- laurus/src`) to get
//! the before/after comparison.
//!
//! # Run
//!
//! ```sh
//! cargo bench --bench vector_delete_bench
//! LAURUS_BENCH_LARGE=1 cargo bench --bench vector_delete_bench   # +10k
//! ```

mod common;

use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use tokio::runtime::Runtime;

use common::{DEFAULT_SEED, lcg_vec_unit};

use laurus::lexical::LexicalIndexConfig;
use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::vector::core::distance::DistanceMetric;
use laurus::vector::core::field::HnswOption;
use laurus::vector::store::config::VectorFieldConfig;
use laurus::vector::{FieldOption, Vector, VectorIndexConfig, VectorStore};
use laurus::{DataValue, Document, EmbedInput, EmbedInputType, Embedder, LaurusError, Result};

const DIM: usize = 128;
/// Number of documents deleted before the timed commit.
const DELETE_BATCH: u64 = 10;

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

fn corpus_sizes() -> Vec<u64> {
    let mut sizes = vec![1000u64, 3000];
    if std::env::var("LAURUS_BENCH_LARGE").is_ok() {
        sizes.push(10_000);
    }
    sizes
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
                pq_codebook_path: None,
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

/// Build a committed `VectorStore` holding `n` deterministic HNSW vectors.
fn build_committed_store(rt: &Runtime, n: u64) -> VectorStore {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let store = VectorStore::new(storage, make_config()).unwrap();
    let mut state = DEFAULT_SEED;
    rt.block_on(async {
        for id in 0..n {
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

fn bench_delete_then_commit(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("vector_mutation/delete_then_commit");
    // Each iteration rebuilds an N-vector committed store, so keep the sample
    // count low; the before/after ratio is the signal, not the sample density.
    group.sample_size(10);

    for &n in &corpus_sizes() {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter_batched(
                || build_committed_store(&rt, n),
                |store| {
                    rt.block_on(async {
                        for id in 0..DELETE_BATCH {
                            store.delete_document_by_internal_id(id).await.unwrap();
                        }
                        store.commit().await.unwrap();
                    });
                },
                BatchSize::PerIteration,
            );
        });
    }

    group.finish();
}

criterion_group!(benches, bench_delete_then_commit);
criterion_main!(benches);
