//! End-to-end engine wiring test for Issue #481 Stage 2.
//!
//! Verifies that `SearchRequestBuilder::vector_rerank_factor` flows
//! through `Engine::search` to the HNSW searcher when the field has
//! `rerank_storage` enabled, and that the request succeeds (no
//! NotImplemented) for both Stage 2 (sidecar) and Stage 1 (no
//! sidecar) configurations.

use tempfile::TempDir;

use laurus::DistanceMetric;
use laurus::Engine;
use laurus::SearchRequestBuilder;
use laurus::storage::file::FileStorageConfig;
use laurus::storage::{StorageConfig, StorageFactory};
use laurus::vector::HnswOption;
use laurus::vector::Vector;
use laurus::vector::core::rerank::RerankStorageKind;
use laurus::{DataValue, Document};
use laurus::{FieldOption, QueryVector, Schema, VectorSearchQuery};

#[tokio::test(flavor = "multi_thread")]
async fn engine_search_with_rerank_factor_succeeds_on_stage2_field() -> laurus::Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let storage =
        StorageFactory::create(StorageConfig::File(FileStorageConfig::new(temp_dir.path())))?;

    let hnsw_opt = HnswOption {
        dimension: 4,
        distance: DistanceMetric::Cosine,
        m: 4,
        ef_construction: 16,
        rerank_storage: Some(RerankStorageKind::F32),
        ..HnswOption::default()
    };

    let schema = Schema::builder()
        .add_field("embedding", FieldOption::Hnsw(hnsw_opt))
        .build();

    let engine = Engine::new(storage.clone(), schema).await?;

    let doc1 = Document::builder()
        .add_field("embedding", DataValue::Vector(vec![1.0, 0.0, 0.0, 0.0]))
        .build();
    let doc2 = Document::builder()
        .add_field("embedding", DataValue::Vector(vec![0.0, 1.0, 0.0, 0.0]))
        .build();
    let doc3 = Document::builder()
        .add_field("embedding", DataValue::Vector(vec![0.0, 0.0, 1.0, 0.0]))
        .build();

    engine.put_document("doc1", doc1).await?;
    engine.put_document("doc2", doc2).await?;
    engine.put_document("doc3", doc3).await?;
    engine.commit().await?;

    // Query close to doc1, asking for rerank_factor = 3. The wiring
    // must let the request succeed and return doc1 as the top hit.
    let query_vec = Vector::new(vec![0.99, 0.01, 0.0, 0.0]);
    let request = SearchRequestBuilder::new()
        .vector_query(VectorSearchQuery::Vectors(vec![QueryVector {
            vector: query_vec,
            weight: 1.0,
            fields: Some(vec!["embedding".to_string()]),
        }]))
        .vector_rerank_factor(3)
        .limit(1)
        .build();

    let results = engine.search(request).await?;
    assert_eq!(results.len(), 1, "expected exactly 1 hit");
    assert_eq!(
        results[0].id, "doc1",
        "doc1 should be the closest match to the rerank-augmented query"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn engine_search_with_rerank_factor_silently_falls_back_on_stage1_field() -> laurus::Result<()>
{
    let temp_dir = TempDir::new().unwrap();
    let storage =
        StorageFactory::create(StorageConfig::File(FileStorageConfig::new(temp_dir.path())))?;

    // No rerank_storage on the schema -> Stage 1 segment.
    let hnsw_opt = HnswOption {
        dimension: 4,
        distance: DistanceMetric::Cosine,
        m: 4,
        ef_construction: 16,
        rerank_storage: None,
        ..HnswOption::default()
    };

    let schema = Schema::builder()
        .add_field("embedding", FieldOption::Hnsw(hnsw_opt))
        .build();

    let engine = Engine::new(storage.clone(), schema).await?;
    let doc1 = Document::builder()
        .add_field("embedding", DataValue::Vector(vec![1.0, 0.0, 0.0, 0.0]))
        .build();
    let doc2 = Document::builder()
        .add_field("embedding", DataValue::Vector(vec![0.0, 1.0, 0.0, 0.0]))
        .build();
    engine.put_document("doc1", doc1).await?;
    engine.put_document("doc2", doc2).await?;
    engine.commit().await?;

    // Stage 1 segment + rerank_factor must succeed (no NotImplemented)
    // because the searcher silently degrades to int8 ranking.
    let request = SearchRequestBuilder::new()
        .vector_query(VectorSearchQuery::Vectors(vec![QueryVector {
            vector: Vector::new(vec![0.95, 0.05, 0.0, 0.0]),
            weight: 1.0,
            fields: Some(vec!["embedding".to_string()]),
        }]))
        .vector_rerank_factor(5)
        .limit(1)
        .build();

    let results = engine.search(request).await?;
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "doc1");
    Ok(())
}
