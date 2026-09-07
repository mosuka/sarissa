//! Regression tests for `MultiFieldVectorIndex`'s WAL checkpoint
//! aggregation (Issue #948 Phase 6).
//!
//! `MultiFieldVectorIndex::last_wal_seq()` reports the MINIMUM checkpoint
//! across every field's independent sub-index, not the maximum: a `max`
//! would let a lagging field's WAL records be skipped on recovery forever.
//! These tests exercise that aggregation directly (low-level `VectorIndex`
//! API), plus an end-to-end `Engine` recovery scenario that is the whole
//! point of getting it right: a document with vectors in two fields must
//! come back in BOTH fields after a crash-and-reopen, not just one (the
//! bug this entire index type exists to prevent -- see the `multi_field`
//! module docs).

use std::any::Any;
use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;

use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::vector::DistanceMetric;
use laurus::vector::core::field::HnswOption;
use laurus::vector::index::VectorIndex;
use laurus::vector::index::config::{HnswIndexConfig, VectorIndexTypeConfig};
use laurus::vector::index::multi_field::MultiFieldVectorIndex;
use laurus::{
    DataValue, Document, EmbedInput, EmbedInputType, Embedder, Engine, FieldOption, Result, Schema,
};

#[derive(Debug)]
struct MockEmbedder;

#[async_trait]
impl Embedder for MockEmbedder {
    async fn embed(&self, _input: &EmbedInput<'_>) -> Result<laurus::vector::Vector> {
        Err(laurus::LaurusError::invalid_argument(
            "embedding not used by this test",
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

fn storage() -> Arc<dyn Storage> {
    Arc::new(MemoryStorage::new(MemoryStorageConfig::default()))
}

fn hnsw_config(dimension: usize) -> VectorIndexTypeConfig {
    VectorIndexTypeConfig::HNSW(HnswIndexConfig {
        dimension,
        distance_metric: DistanceMetric::Cosine,
        normalize_vectors: true,
        ..Default::default()
    })
}

/// The aggregate `last_wal_seq()` must be the MINIMUM across every field,
/// not a value that only reflects some of them -- reported here via the
/// natural window between `add_field` (which seeds the new field's
/// checkpoint as PENDING) and the next `persist_deletions()` (which
/// publishes it), the same mechanism a real commit goes through.
#[test]
fn last_wal_seq_is_the_minimum_across_all_fields() {
    let mut fields = BTreeMap::new();
    fields.insert("a".to_string(), hnsw_config(3));
    fields.insert("b".to_string(), hnsw_config(3));
    let index =
        MultiFieldVectorIndex::open_or_create(storage(), &fields, Arc::new(MockEmbedder)).unwrap();

    index.set_last_wal_seq(100).unwrap();
    index.persist_deletions().unwrap();
    assert_eq!(
        index.last_wal_seq(),
        100,
        "both fields start in sync at 100"
    );

    // Adding "c" seeds its PENDING checkpoint from the current minimum
    // (100) but does not publish it -- so until the next
    // `persist_deletions()`, "c"'s published checkpoint is still 0 and
    // the aggregate must reflect that worst case, not the two fields
    // that are already at 100.
    index.add_field("c", hnsw_config(3)).unwrap();
    assert_eq!(
        index.last_wal_seq(),
        0,
        "the minimum must reflect the just-added field's still-unpublished checkpoint"
    );

    index.persist_deletions().unwrap();
    assert_eq!(
        index.last_wal_seq(),
        100,
        "once every field's checkpoint is published, the minimum is the common value"
    );
}

/// Removing a field must correctly recompute the aggregate from the
/// REMAINING fields -- neither keeping the removed field's low checkpoint
/// forever (that would be as wrong as never removing it) nor going stale.
#[test]
fn removing_the_lagging_field_lifts_the_aggregate_to_the_remaining_fields_minimum() {
    let mut fields = BTreeMap::new();
    fields.insert("a".to_string(), hnsw_config(3));
    fields.insert("b".to_string(), hnsw_config(3));
    let index =
        MultiFieldVectorIndex::open_or_create(storage(), &fields, Arc::new(MockEmbedder)).unwrap();
    index.set_last_wal_seq(100).unwrap();
    index.persist_deletions().unwrap();

    // "c" is added but never published -- the classic lagging-field case.
    index.add_field("c", hnsw_config(3)).unwrap();
    assert_eq!(index.last_wal_seq(), 0);

    index.remove_field("c", false).unwrap();
    assert_eq!(
        index.last_wal_seq(),
        100,
        "removing the lagging field must lift the aggregate back to the \
         remaining fields' minimum, not leave it stuck at the removed \
         field's value"
    );
}

fn two_field_schema() -> Schema {
    Schema::builder()
        .add_field(
            "title_vec",
            FieldOption::Hnsw(HnswOption {
                dimension: 3,
                distance: DistanceMetric::Cosine,
                ..HnswOption::default()
            }),
        )
        .add_field(
            "body_vec",
            FieldOption::Hnsw(HnswOption {
                dimension: 3,
                distance: DistanceMetric::Cosine,
                ..HnswOption::default()
            }),
        )
        .build()
}

fn doc_with_both_fields(i: usize) -> Document {
    let t = i as f32 * 0.01;
    Document::builder()
        .add_field("title_vec", DataValue::Vector(vec![t.cos(), t.sin(), 0.0]))
        .add_field("body_vec", DataValue::Vector(vec![0.0, t.cos(), t.sin()]))
        .build()
}

/// End-to-end `Engine` recovery: documents with vectors in BOTH fields,
/// acknowledged via WAL but never committed, then a crash (drop without
/// commit) and reopen. Recovery must replay every record into BOTH
/// fields -- the WAL min-aggregation this whole file is about is what
/// makes that guarantee hold when fields are independent sub-indexes
/// (Issue #948).
#[tokio::test(flavor = "multi_thread")]
async fn engine_recovery_replays_both_fields_after_uncommitted_crash() -> Result<()> {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let schema = two_field_schema();

    // Round 1: index docs (each touching BOTH vector fields) without ever
    // calling commit(), then drop the engine -- simulating a crash. Durability
    // must come from the WAL alone.
    {
        let engine = Engine::new(storage.clone(), schema.clone()).await?;
        for i in 0..20 {
            engine
                .put_document(&format!("doc{i}"), doc_with_both_fields(i))
                .await?;
        }
    }

    // Round 2: reopen on the SAME storage and commit; recovery replays the
    // WAL into both fields independently.
    {
        let engine = Engine::new(storage.clone(), schema.clone()).await?;
        engine.commit().await?;

        let stats = engine.stats()?;
        assert_eq!(stats.document_count, 20);

        let title_stats = stats
            .vector_fields
            .get("title_vec")
            .expect("title_vec must be present in stats");
        let body_stats = stats
            .vector_fields
            .get("body_vec")
            .expect("body_vec must be present in stats");
        assert_eq!(
            title_stats.vector_count, 20,
            "title_vec must recover all 20 vectors, not be shadowed by body_vec"
        );
        assert_eq!(
            body_stats.vector_count, 20,
            "body_vec must recover all 20 vectors, not be shadowed by title_vec"
        );
    }

    Ok(())
}
