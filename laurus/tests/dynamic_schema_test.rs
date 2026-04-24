//! End-to-end tests for the dynamic schema feature.
//!
//! These tests drive the full `Engine` surface (put → commit → search /
//! get_documents) for each [`DynamicFieldPolicy`] variant, and also cover
//! the type-conflict coercion rules exercised during document ingestion.

use laurus::lexical::TextOption;
use laurus::lexical::core::field::IntegerOption;
use laurus::storage::memory::MemoryStorageConfig;
use laurus::storage::{StorageConfig, StorageFactory};
use laurus::{
    DataValue, Document, DynamicFieldPolicy, Engine, FieldOption, LaurusError, Result, Schema,
};

async fn engine_with_policy(policy: DynamicFieldPolicy) -> Result<Engine> {
    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;
    let schema = Schema::builder().dynamic_field_policy(policy).build();
    Engine::new(storage, schema).await
}

/// Strict: an undeclared field must cause the ingest to fail.
#[tokio::test(flavor = "multi_thread")]
async fn strict_rejects_undeclared_fields() -> Result<()> {
    let engine = engine_with_policy(DynamicFieldPolicy::Strict).await?;

    let doc = Document::builder().add_field("title", "hello").build();

    let err = engine
        .put_document("doc1", doc)
        .await
        .expect_err("Strict policy must reject undeclared field 'title'");
    let msg = err.to_string();
    assert!(
        msg.contains("title"),
        "error should mention the field: {msg}"
    );
    assert!(
        msg.contains("Strict") || msg.contains("undeclared"),
        "error should explain the policy: {msg}"
    );
    Ok(())
}

/// Dynamic: undeclared text/integer/float/bool are auto-added.
#[tokio::test(flavor = "multi_thread")]
async fn dynamic_auto_adds_primitive_fields() -> Result<()> {
    let engine = engine_with_policy(DynamicFieldPolicy::Dynamic).await?;

    let doc = Document::builder()
        .add_field("title", "hello world")
        .add_field("count", 42i64)
        .add_field("rating", 4.5f64)
        .add_field("published", true)
        .build();
    engine.put_document("doc1", doc).await?;
    engine.commit().await?;

    let schema = engine.schema();
    assert!(
        matches!(schema.fields.get("title"), Some(FieldOption::Text(_))),
        "title should be auto-added as Text"
    );
    assert!(
        matches!(schema.fields.get("count"), Some(FieldOption::Integer(_))),
        "count should be auto-added as Integer"
    );
    assert!(
        matches!(schema.fields.get("rating"), Some(FieldOption::Float(_))),
        "rating should be auto-added as Float"
    );
    assert!(
        matches!(
            schema.fields.get("published"),
            Some(FieldOption::Boolean(_))
        ),
        "published should be auto-added as Boolean"
    );

    // Values should be retrievable.
    let docs = engine.get_documents("doc1").await?;
    assert_eq!(docs.len(), 1);
    let d = &docs[0];
    assert_eq!(
        d.get("title").and_then(|v| v.as_text()),
        Some("hello world")
    );
    assert_eq!(d.get("count").and_then(|v| v.as_integer()), Some(42));
    assert_eq!(d.get("rating").and_then(|v| v.as_float()), Some(4.5));
    assert_eq!(d.get("published").and_then(|v| v.as_boolean()), Some(true));
    Ok(())
}

/// Dynamic: a geo value (DataValue::Geo) is auto-added as a Geo field.
#[tokio::test(flavor = "multi_thread")]
async fn dynamic_auto_adds_geo_field() -> Result<()> {
    let engine = engine_with_policy(DynamicFieldPolicy::Dynamic).await?;

    let doc = Document::builder().add_geo("location", 35.1, 139.0).build();
    engine.put_document("doc1", doc).await?;
    engine.commit().await?;

    let schema = engine.schema();
    assert!(matches!(
        schema.fields.get("location"),
        Some(FieldOption::Geo(_))
    ));

    let docs = engine.get_documents("doc1").await?;
    assert_eq!(
        docs[0].get("location").and_then(|v| v.as_geo()),
        Some((35.1, 139.0))
    );
    Ok(())
}

/// Dynamic: a raw vector without a declared schema is rejected.
#[tokio::test(flavor = "multi_thread")]
async fn dynamic_rejects_undeclared_vector() -> Result<()> {
    let engine = engine_with_policy(DynamicFieldPolicy::Dynamic).await?;

    let doc = Document::builder()
        .add_vector("embedding", vec![0.1, 0.2, 0.3])
        .build();
    let err = engine
        .put_document("doc1", doc)
        .await
        .expect_err("vector fields must be declared explicitly");
    assert!(
        err.to_string().contains("vector"),
        "unexpected error: {err}"
    );
    Ok(())
}

/// Ignore: undeclared fields are silently dropped.
#[tokio::test(flavor = "multi_thread")]
async fn ignore_drops_undeclared_fields() -> Result<()> {
    let engine = engine_with_policy(DynamicFieldPolicy::Ignore).await?;

    // Declare one field so we can verify declared fields still ingest.
    engine
        .add_field("title", FieldOption::Text(TextOption::default()))
        .await?;

    let doc = Document::builder()
        .add_field("title", "hello")
        .add_field("drop_me", "gone")
        .build();
    engine.put_document("doc1", doc).await?;
    engine.commit().await?;

    let schema = engine.schema();
    assert!(schema.fields.contains_key("title"));
    assert!(
        !schema.fields.contains_key("drop_me"),
        "Ignore should not add the field to the schema"
    );

    let docs = engine.get_documents("doc1").await?;
    assert_eq!(
        docs[0].get("title").and_then(|v| v.as_text()),
        Some("hello")
    );
    assert!(docs[0].get("drop_me").is_none());
    Ok(())
}

/// Integer fields truncate incoming float values (documented data loss).
#[tokio::test(flavor = "multi_thread")]
async fn integer_field_truncates_float() -> Result<()> {
    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;
    let schema = Schema::builder()
        .add_field("count", FieldOption::Integer(IntegerOption::default()))
        .build();
    let engine = Engine::new(storage, schema).await?;

    let doc = Document::builder().add_field("count", 4.7f64).build();
    engine.put_document("doc1", doc).await?;
    engine.commit().await?;

    let docs = engine.get_documents("doc1").await?;
    assert_eq!(
        docs[0].get("count").and_then(|v| v.as_integer()),
        Some(4),
        "integer field must truncate incoming float"
    );
    Ok(())
}

/// Integer field parses numeric strings, and rejects non-numeric strings.
#[tokio::test(flavor = "multi_thread")]
async fn integer_field_parses_numeric_string() -> Result<()> {
    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;
    let schema = Schema::builder()
        .add_field("count", FieldOption::Integer(IntegerOption::default()))
        .build();
    let engine = Engine::new(storage, schema).await?;

    let doc = Document::builder()
        .add_field("count", DataValue::Text("42".to_string()))
        .build();
    engine.put_document("doc1", doc).await?;
    engine.commit().await?;

    let docs = engine.get_documents("doc1").await?;
    assert_eq!(docs[0].get("count").and_then(|v| v.as_integer()), Some(42));

    let doc_bad = Document::builder()
        .add_field("count", DataValue::Text("abc".to_string()))
        .build();
    let err = engine.put_document("doc2", doc_bad).await.unwrap_err();
    assert!(
        err.to_string().contains("parse") || err.to_string().contains("integer"),
        "{err}"
    );
    Ok(())
}

/// Parsing a query that names a field outside the schema must fail.
#[tokio::test(flavor = "multi_thread")]
async fn query_dsl_rejects_unknown_field() -> Result<()> {
    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;
    let schema = Schema::builder()
        .add_field("title", FieldOption::Text(TextOption::default()))
        .build();
    let engine = Engine::new(storage, schema).await?;

    let parser = engine.unified_query_parser()?;

    // Declared field parses fine.
    parser.parse("title:hello").await?;

    // Typo'd / undeclared field is rejected with a message that names it.
    let result = parser.parse("titl:hello").await;
    let err = match result {
        Ok(_) => panic!("expected error for unknown field 'titl'"),
        Err(e) => e,
    };
    assert!(err.to_string().contains("titl"), "{err}");

    Ok(())
}

/// User-supplied `_`-prefixed field names are rejected under any policy.
#[tokio::test(flavor = "multi_thread")]
async fn reserved_prefix_rejected() -> Result<()> {
    for policy in [
        DynamicFieldPolicy::Strict,
        DynamicFieldPolicy::Dynamic,
        DynamicFieldPolicy::Ignore,
    ] {
        let engine = engine_with_policy(policy).await?;
        let doc = Document::builder().add_field("_secret", "nope").build();
        let err = engine.put_document("doc1", doc).await.unwrap_err();
        assert!(
            matches!(err, LaurusError::Other(_)) || err.to_string().contains("reserved"),
            "policy {:?}: unexpected error {err}",
            policy
        );
    }
    Ok(())
}
