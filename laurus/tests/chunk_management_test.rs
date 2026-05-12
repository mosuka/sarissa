use laurus::Engine;
use laurus::Result;
use laurus::storage::memory::MemoryStorageConfig;
use laurus::storage::{StorageConfig, StorageFactory};
use laurus::vector::DistanceMetric;
use laurus::vector::FlatOption;
use laurus::{DataValue, Document};
use laurus::{FieldOption, Schema};

async fn build_test_engine() -> Result<Engine> {
    let storage_config = StorageConfig::Memory(MemoryStorageConfig::default());
    let storage = StorageFactory::create(storage_config)?;

    let field_option = FieldOption::Flat(FlatOption {
        dimension: 3,
        distance: DistanceMetric::Cosine,
        base_weight: 1.0,
        quantizer: Default::default(),
        rerank_storage: None,
        embedder: None,
    });

    let config = Schema::builder().add_field("body", field_option).build();

    Engine::new(storage, config).await
}

fn create_payload(vector: Vec<f32>) -> Document {
    Document::builder()
        .add_field("body", DataValue::Vector(vector))
        .build()
}

#[tokio::test(flavor = "multi_thread")]
async fn test_chunk_addition() -> Result<()> {
    let engine = build_test_engine().await?;

    // 1. Add first chunk for "doc_A"
    let p1 = create_payload(vec![1.0, 0.0, 0.0]);
    engine.add_document("doc_A", p1).await?;

    // 2. Add second chunk for "doc_A"
    let p2 = create_payload(vec![0.0, 1.0, 0.0]);
    engine.add_document("doc_A", p2).await?;

    engine.commit().await?;

    let stats = engine.stats()?;
    assert_eq!(stats.document_count, 2, "Should have 2 documents total");

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_chunk_deletion() -> Result<()> {
    let engine = build_test_engine().await?;

    // Add 2 chunks
    let p1 = create_payload(vec![1.0, 0.0, 0.0]);
    engine.add_document("doc_A", p1).await?;

    let p2 = create_payload(vec![0.0, 1.0, 0.0]);
    engine.add_document("doc_A", p2).await?;

    engine.commit().await?;

    let stats_before = engine.stats()?;
    assert_eq!(stats_before.document_count, 2);

    // Delete "doc_A"
    engine.delete_documents("doc_A").await?;
    engine.commit().await?;

    // Verify deletion
    let stats_after = engine.stats()?;
    assert_eq!(
        stats_after.document_count, 0,
        "All chunks should be deleted"
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_mixed_mode_behavior() -> Result<()> {
    let engine = build_test_engine().await?;

    // Add chunk 1
    engine
        .add_document("doc_B", create_payload(vec![1.0, 0.0, 0.0]))
        .await?;

    // Add chunk 2
    engine
        .add_document("doc_B", create_payload(vec![0.0, 1.0, 0.0]))
        .await?;

    engine.commit().await?;
    assert_eq!(engine.stats()?.document_count, 2);

    // Now put_document (upsert) "doc_B" (should overwrite ALL of them)
    engine
        .put_document("doc_B", create_payload(vec![0.0, 0.0, 1.0]))
        .await?;
    engine.commit().await?;

    // All chunks replaced by a single doc. Total should be 1.
    assert_eq!(engine.stats()?.document_count, 1);

    // Delete "doc_B" -> Should delete the remaining doc.
    engine.delete_documents("doc_B").await?;
    engine.commit().await?;
    assert_eq!(engine.stats()?.document_count, 0);

    Ok(())
}
