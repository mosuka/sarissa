//! End-to-end engine tests for `Engine::train_pq_codebook` (Issue #631 PR-2).
//!
//! Complements `pq_shared_codebook_test.rs` (which drives the HNSW index
//! layer directly) by exercising the schema-level wiring: a
//! `pq_codebook_path` on the schema's `HnswOption` must reach the HNSW
//! writer through `Engine` → `VectorStore` → `from_hnsw_option` →
//! `resolve_pq_codebook`, and `Engine::train_pq_codebook` must produce a
//! codebook that a subsequently opened engine encodes with.

use std::sync::Arc;

use laurus::storage::file::FileStorageConfig;
use laurus::storage::prefixed::PrefixedStorage;
use laurus::storage::{Storage, StorageConfig, StorageFactory};
use laurus::vector::HnswOption;
use laurus::vector::Vector;
use laurus::vector::core::distance::DistanceMetric;
use laurus::vector::core::quantization::QuantizationMethod;
use laurus::vector::index::hnsw::reader::HnswIndexReader;
use laurus::vector::index::pq_codebook::read_pq_codebook;
use laurus::vector::index::storage::VectorStorage;
use laurus::{DataValue, Document, Engine, FieldOption, Schema};
use tempfile::TempDir;

const DIM: usize = 32;
const M: usize = 4;

/// Deterministic pseudo-random vectors (same LCG as
/// `pq_shared_codebook_test.rs`); distinct seeds give distinct corpora.
fn make_vectors(count: usize, seed: u64) -> Vec<Vector> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            let data: Vec<f32> = (0..DIM)
                .map(|_| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
                })
                .collect();
            Vector::new(data)
        })
        .collect()
}

/// Schema with one HNSW + PQ field pointing at a shared codebook file.
fn pq_schema(pq_codebook_path: Option<&str>) -> Schema {
    Schema::builder()
        .add_field(
            "embedding",
            FieldOption::Hnsw(HnswOption {
                dimension: DIM,
                distance: DistanceMetric::Euclidean,
                m: 8,
                ef_construction: 32,
                quantizer: QuantizationMethod::ProductQuantization { subvector_count: M },
                pq_codebook_path: pq_codebook_path.map(str::to_string),
                ..HnswOption::default()
            }),
        )
        .build()
}

async fn file_engine(dir: &TempDir, schema: Schema) -> laurus::Result<Engine> {
    let storage = StorageFactory::create(StorageConfig::File(FileStorageConfig::new(dir.path())))?;
    Engine::new(storage, schema).await
}

/// `train_pq_codebook` must reject fields it cannot train for, naming the
/// problem: unknown field, non-HNSW field, HNSW field without PQ.
#[tokio::test(flavor = "multi_thread")]
async fn train_pq_codebook_validates_the_field() -> laurus::Result<()> {
    let dir = TempDir::new().unwrap();
    let schema = Schema::builder()
        .add_field(
            "embedding",
            FieldOption::Hnsw(HnswOption {
                dimension: DIM,
                distance: DistanceMetric::Euclidean,
                // Scalar8Bit (the default): no PQ, so training must be refused.
                ..HnswOption::default()
            }),
        )
        .add_field(
            "title",
            FieldOption::Text(laurus::TextOption {
                indexed: true,
                stored: true,
                ..Default::default()
            }),
        )
        .build();
    let engine = file_engine(&dir, schema).await?;
    let vectors = make_vectors(10, 1);

    let err = engine
        .train_pq_codebook("missing", &vectors, None)
        .unwrap_err();
    assert!(
        err.to_string().contains("not defined in the schema"),
        "unknown field must be named: {err}"
    );

    let err = engine
        .train_pq_codebook("title", &vectors, None)
        .unwrap_err();
    assert!(
        err.to_string().contains("not an HNSW vector field"),
        "non-vector field must be refused: {err}"
    );

    let err = engine
        .train_pq_codebook("embedding", &vectors, None)
        .unwrap_err();
    assert!(
        err.to_string().contains("ProductQuantization"),
        "a non-PQ quantizer must be refused: {err}"
    );

    Ok(())
}

/// The full schema-level loop: with `pq_codebook_path` configured,
/// (1) a commit before training hard-errors with the training command,
/// (2) `train_pq_codebook` writes the codebook into the vector namespace,
/// (3) a reopened engine commits successfully and the sealed segment's
/// embedded codebook is byte-identical to the trained file — proving the
/// writer encoded with the shared codebook instead of retraining (the
/// 20-doc commit is far below `PQ_MIN_TRAIN_VECTORS`, so a silent
/// fresh-training fallback would have produced a Scalar8Bit segment).
#[tokio::test(flavor = "multi_thread")]
async fn train_then_reopen_commits_with_the_shared_codebook() -> laurus::Result<()> {
    let dir = TempDir::new().unwrap();
    let storage = StorageFactory::create(StorageConfig::File(FileStorageConfig::new(dir.path())))?;

    // (1) Before training: the configured-but-missing codebook must fail the
    // commit loudly (no silent per-segment retrain), pointing at the fix.
    {
        let engine = Engine::new(storage.clone(), pq_schema(Some("embedding.pqcb"))).await?;
        let doc = Document::builder()
            .add_field(
                "embedding",
                DataValue::Vector(make_vectors(1, 7)[0].data.to_vec()),
            )
            .build();
        engine.put_document("doc-early", doc).await?;
        let err = engine.commit().await.unwrap_err();
        assert!(
            err.to_string().contains("train"),
            "an untrained configured codebook must point at the training step: {err}"
        );
    }

    // (2) Train on a corpus DISTINCT from what gets indexed below, so an
    // accidental inline retrain could not reproduce the same centroids.
    let fresh_dir = TempDir::new().unwrap();
    let fresh_storage = StorageFactory::create(StorageConfig::File(FileStorageConfig::new(
        fresh_dir.path(),
    )))?;
    let trainer = Engine::new(fresh_storage.clone(), pq_schema(Some("embedding.pqcb"))).await?;
    let info = trainer.train_pq_codebook("embedding", &make_vectors(400, 0x0F0F), None)?;
    assert_eq!(info.path, "embedding.pqcb");
    assert_eq!(info.subvector_count, M);
    assert_eq!(info.centroids, 256);
    assert_eq!(info.sub_dimension, DIM / M);
    assert_eq!(info.dimension, DIM);
    assert_eq!(info.training_vectors, 400);
    assert!(
        fresh_storage.file_exists("vector/embedding.pqcb"),
        "the codebook must land inside the engine's vector storage namespace"
    );
    drop(trainer);

    // (3) A reopened engine picks the codebook up at open and commits a
    // tiny (< PQ_MIN_TRAIN_VECTORS) batch on PQ.
    let engine = Engine::new(fresh_storage.clone(), pq_schema(Some("embedding.pqcb"))).await?;
    for (i, v) in make_vectors(20, 0xA5A5).iter().enumerate() {
        let doc = Document::builder()
            .add_field("embedding", DataValue::Vector(v.data.to_vec()))
            .build();
        engine.put_document(&format!("doc{i}"), doc).await?;
    }
    engine.commit().await?;

    // The sealed segment's embedded codebook must equal the trained file.
    let vector_storage: Arc<dyn Storage> =
        Arc::new(PrefixedStorage::new("vector", fresh_storage.clone()));
    let trained = read_pq_codebook(vector_storage.as_ref(), "embedding.pqcb")?;
    let reader =
        HnswIndexReader::load(vector_storage, "segment_000000", DistanceMetric::Euclidean)?;
    let pool = match reader.vectors() {
        VectorStorage::OwnedPq(pool) => pool.clone(),
        other => panic!("the committed segment must stay on PQ, got {other:?}"),
    };
    assert_eq!(pool.params, trained.params);
    assert_eq!(
        pool.codebook, trained.codebook,
        "the segment must embed the shared codebook byte-for-byte (no retrain)"
    );

    Ok(())
}

/// `output: Some(..)` must override both the configured path and the
/// default `{field}.pqcb` naming, so a v2 codebook can be trained next to
/// the live one.
#[tokio::test(flavor = "multi_thread")]
async fn train_pq_codebook_honors_the_output_override() -> laurus::Result<()> {
    let dir = TempDir::new().unwrap();
    let storage = StorageFactory::create(StorageConfig::File(FileStorageConfig::new(dir.path())))?;
    let engine = Engine::new(storage.clone(), pq_schema(None)).await?;

    // No configured path, no override: default name.
    let info = engine.train_pq_codebook("embedding", &make_vectors(300, 3), None)?;
    assert_eq!(info.path, "embedding.pqcb");
    assert!(storage.file_exists("vector/embedding.pqcb"));

    // Override wins.
    let info = engine.train_pq_codebook(
        "embedding",
        &make_vectors(300, 4),
        Some("embedding.v2.pqcb"),
    )?;
    assert_eq!(info.path, "embedding.v2.pqcb");
    assert!(storage.file_exists("vector/embedding.v2.pqcb"));

    Ok(())
}
