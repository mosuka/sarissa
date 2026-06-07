//! Integration tests for crash-safe atomic writes of the production HNSW
//! index files (issue #784).
//!
//! `HnswIndexWriter::write` / `HnswIndex::write_metadata` /
//! `HnswIndex::persist_deletions` now write to a `.tmp` file and atomically
//! `rename_file` it into place. A crash between writing the temp file and the
//! rename therefore leaves the previously committed file intact, and a
//! successful commit leaves no temp file behind.

use async_trait::async_trait;
use std::any::Any;
use std::collections::HashMap;
use std::collections::HashSet;
use std::io::Write;
use std::sync::Arc;

use laurus::lexical::LexicalIndexConfig;
use laurus::storage::Storage;
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

const DIM: usize = 16;
const N: u64 = 50;
const STEP: f32 = 0.01;
const HNSW_FILE: &str = "vector_index.hnsw";
const HNSW_TMP: &str = "vector_index.hnsw.tmp";

#[derive(Debug)]
struct MockEmbedder {
    dimension: usize,
}

#[async_trait]
impl Embedder for MockEmbedder {
    async fn embed(&self, input: &EmbedInput<'_>) -> Result<Vector> {
        match input {
            EmbedInput::Text(_) => Ok(Vector::new(vec![0.0; self.dimension])),
            _ => Err(LaurusError::invalid_argument("text only")),
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

fn doc_vec(i: u64) -> Vec<f32> {
    let theta = i as f32 * STEP;
    let mut v = vec![0.0; DIM];
    v[0] = theta.cos();
    v[1] = theta.sin();
    v
}

fn query_vec() -> Vec<f32> {
    let mut v = vec![0.0; DIM];
    v[0] = 1.0;
    v
}

fn hnsw() -> FieldOption {
    FieldOption::Hnsw(HnswOption {
        dimension: DIM,
        distance: DistanceMetric::Cosine,
        m: 16,
        ef_construction: 100,
        default_ef_search: None,
        base_weight: 1.0,
        quantizer: Default::default(),
        rerank_storage: None,
        embedder: None,
    })
}

fn make_config() -> VectorIndexConfig {
    let mut field_configs = HashMap::new();
    field_configs.insert(
        "vec".to_string(),
        VectorFieldConfig {
            vector: Some(hnsw()),
            lexical: None,
        },
    );
    VectorIndexConfig {
        fields: field_configs,
        embedder: Arc::new(MockEmbedder { dimension: DIM }),
        default_fields: vec!["vec".to_string()],
        metadata: HashMap::new(),
        deletion_config: laurus::DeletionConfig {
            auto_compaction: false,
            ..Default::default()
        },
        shard_id: 0,
        metadata_config: LexicalIndexConfig::default(),
    }
}

async fn build_committed(storage: Arc<dyn Storage>) -> laurus::vector::VectorStore {
    let store = laurus::vector::VectorStore::new(storage, make_config()).unwrap();
    for id in 0..N {
        let doc = Document::builder()
            .add_field("vec", DataValue::Vector(doc_vec(id)))
            .build();
        store.upsert_document_by_internal_id(id, doc).await.unwrap();
    }
    store.commit().await.unwrap();
    store
}

fn request() -> VectorSearchRequest {
    VectorSearchRequest {
        query: VectorSearchQuery::Vectors(vec![QueryVector {
            vector: Vector::new(query_vec()),
            weight: 1.0,
            fields: Some(vec!["vec".into()]),
        }]),
        params: VectorSearchParams {
            limit: 10,
            score_mode: VectorScoreMode::WeightedSum,
            fields: None,
            allowed_ids: None,
            ..Default::default()
        },
    }
}

fn hit_ids(results: &laurus::vector::VectorSearchResults) -> HashSet<u64> {
    results.hits.iter().map(|h| h.doc_id).collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn commit_leaves_no_temp_file() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let store = build_committed(storage.clone()).await;

    // After a successful commit the segment is in place and the temp file has
    // been renamed away.
    assert!(storage.file_exists(HNSW_FILE), "committed .hnsw must exist");
    assert!(
        !storage.file_exists(HNSW_TMP),
        "a successful commit must not leave a .hnsw.tmp behind"
    );
    assert_eq!(hit_ids(&store.search(request()).unwrap()).len(), 10);
}

#[tokio::test(flavor = "multi_thread")]
async fn orphaned_temp_from_crashed_write_is_ignored() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let store = build_committed(storage.clone()).await;
    let before = hit_ids(&store.search(request()).unwrap());
    assert_eq!(before.len(), 10);
    drop(store);

    // Simulate a crash *during* a later write: the temp file was created but
    // the atomic rename never happened. The committed `.hnsw` must be untouched.
    {
        let mut out = storage.create_output(HNSW_TMP).unwrap();
        out.write_all(b"partially-written-garbage-from-a-crashed-commit")
            .unwrap();
        out.close().unwrap();
    }
    assert!(
        storage.file_exists(HNSW_FILE),
        "committed .hnsw still present"
    );

    // Reopening reads the valid committed segment and ignores the orphaned
    // temp file — same results as before the simulated crash.
    let reopened = laurus::vector::VectorStore::new(storage.clone(), make_config()).unwrap();
    let after = hit_ids(&reopened.search(request()).unwrap());
    assert_eq!(
        after, before,
        "the committed index must survive an orphaned temp file from a crashed write"
    );
}
