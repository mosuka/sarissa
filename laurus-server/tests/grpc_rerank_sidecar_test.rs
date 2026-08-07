//! Server-level end-to-end test for Issue #793.
//!
//! Before #793 the proto `HnswOption` had no `rerank_storage` field, so
//! `laurus_server::convert::schema::from_proto` hard-coded
//! `rerank_storage: None` and a gRPC-created HNSW field could never emit
//! the Stage-2 rerank sidecar (`.hnsw.f32`).
//!
//! This test drives the real server conversion path: a laurus `Schema`
//! is serialized to the proto representation (`to_proto`) and parsed
//! back (`from_proto`) — exactly what the `CreateIndex` RPC does with a
//! client-supplied proto schema — then an `Engine` built from the
//! round-tripped schema must emit the sidecar on commit.

use std::sync::Arc;

use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::vector::core::rerank::RerankStorageKind;
use laurus::{DataValue, DistanceMetric, Document, Engine, FieldOption, HnswOption, Schema};

use laurus_server::convert::schema::{from_proto, to_proto};

/// Sidecar file name as it lands under the engine's vector storage
/// prefix (`PrefixedStorage("vector", ..)`), then one level deeper under
/// the `embedding` field's own sub-namespace (Issue #948:
/// `MultiFieldVectorIndex` gives every vector field its own
/// `PrefixedStorage` directory rather than sharing the storage root): the
/// first sealed segment of the (default, #882) segmented layout is
/// `segment_000000.hnsw`.
const SIDECAR_NAME: &str = "vector/embedding/segment_000000.hnsw.f32";

#[tokio::test(flavor = "multi_thread")]
async fn grpc_schema_with_rerank_storage_emits_sidecar_on_commit() -> laurus::Result<()> {
    // A schema as a gRPC client would configure it: HNSW + Stage-2 rerank.
    let schema = Schema::builder()
        .add_field(
            "embedding",
            FieldOption::Hnsw(HnswOption {
                dimension: 4,
                distance: DistanceMetric::Cosine,
                m: 4,
                ef_construction: 16,
                rerank_storage: Some(RerankStorageKind::F32),
                ..HnswOption::default()
            }),
        )
        .build();

    // Round-trip through the proto representation, the way CreateIndex
    // does (client sends proto -> server from_proto). This is where
    // #793's gap lived.
    let proto = to_proto(&schema);
    let server_schema = from_proto(&proto).expect("from_proto must succeed");

    // The proto layer must have preserved rerank_storage end to end.
    match server_schema.fields.get("embedding") {
        Some(FieldOption::Hnsw(h)) => {
            assert_eq!(
                h.rerank_storage,
                Some(RerankStorageKind::F32),
                "the proto round-trip must preserve rerank_storage (Issue #793)"
            );
        }
        other => panic!("expected FieldOption::Hnsw, got {other:?}"),
    }

    // Build an engine from the round-tripped schema and commit a doc.
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let engine = Engine::new(storage.clone(), server_schema).await?;
    let doc = Document::builder()
        .add_field("embedding", DataValue::Vector(vec![0.92, 0.31, 0.17, 0.05]))
        .build();
    engine.put_document("doc1", doc).await?;
    engine.commit().await?;

    // The gRPC-configured field must now actually emit the sidecar.
    assert!(
        storage.file_exists(SIDECAR_NAME),
        "a gRPC-created HNSW field with rerank_storage must emit {SIDECAR_NAME} on commit (Issue #793)"
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn grpc_schema_without_rerank_storage_emits_no_sidecar() -> laurus::Result<()> {
    // Pins the no-behavior-change guarantee: a Stage-1 HNSW field
    // (rerank_storage unset) must not emit a sidecar after #793.
    let schema = Schema::builder()
        .add_field(
            "embedding",
            FieldOption::Hnsw(HnswOption {
                dimension: 4,
                distance: DistanceMetric::Cosine,
                m: 4,
                ef_construction: 16,
                rerank_storage: None,
                ..HnswOption::default()
            }),
        )
        .build();

    let server_schema = from_proto(&to_proto(&schema)).expect("from_proto must succeed");
    match server_schema.fields.get("embedding") {
        Some(FieldOption::Hnsw(h)) => assert_eq!(h.rerank_storage, None),
        other => panic!("expected FieldOption::Hnsw, got {other:?}"),
    }

    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let engine = Engine::new(storage.clone(), server_schema).await?;
    let doc = Document::builder()
        .add_field("embedding", DataValue::Vector(vec![1.0, 0.0, 0.0, 0.0]))
        .build();
    engine.put_document("doc1", doc).await?;
    engine.commit().await?;

    assert!(
        !storage.file_exists(SIDECAR_NAME),
        "a field without rerank_storage must not emit {SIDECAR_NAME}"
    );

    Ok(())
}
