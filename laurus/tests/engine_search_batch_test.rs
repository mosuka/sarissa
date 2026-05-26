//! Integration tests for [`laurus::Engine::search_batch`]
//! (issue [#715](https://github.com/mosuka/laurus/issues/715) — Phase 3
//! prerequisite of [#648](https://github.com/mosuka/laurus/issues/648)).
//!
//! These tests verify the engine-level batched search API:
//!
//! - Empty input returns an empty `Vec` without invoking `search`.
//! - Single-request batch produces the same result as
//!   `Engine::search` invoked directly.
//! - Multi-request batch preserves input order and returns one result
//!   list per query.
//! - Existing `Engine::search` behaviour is unchanged.

use tempfile::TempDir;

use laurus::Engine;
use laurus::SearchRequestBuilder;
use laurus::lexical::Query;
use laurus::lexical::TermQuery;
use laurus::storage::file::FileStorageConfig;
use laurus::storage::{StorageConfig, StorageFactory};
use laurus::{DataValue, Document};
use laurus::{FieldOption, LexicalSearchQuery, Schema};

async fn build_engine_with_corpus() -> laurus::Result<Engine> {
    let temp_dir = TempDir::new().unwrap();
    let storage_config = StorageConfig::File(FileStorageConfig::new(temp_dir.path()));
    let storage = StorageFactory::create(storage_config)?;

    let config = Schema::builder()
        .add_field("title", FieldOption::Text(Default::default()))
        .build();

    let engine = Engine::new(storage, config).await?;

    engine
        .put_document(
            "doc1",
            Document::builder()
                .add_field("title", DataValue::Text("Rust Programming".into()))
                .build(),
        )
        .await?;
    engine
        .put_document(
            "doc2",
            Document::builder()
                .add_field("title", DataValue::Text("Vector Search".into()))
                .build(),
        )
        .await?;
    engine
        .put_document(
            "doc3",
            Document::builder()
                .add_field("title", DataValue::Text("Distributed Systems".into()))
                .build(),
        )
        .await?;
    engine.commit().await?;

    // Persist `_temp_dir` for the engine lifetime by leaking it intentionally
    // in this test helper. The OS reclaims the directory on process exit.
    std::mem::forget(temp_dir);

    Ok(engine)
}

fn term_request(term: &str) -> laurus::SearchRequest {
    let q = Box::new(TermQuery::new("title", term)) as Box<dyn Query>;
    SearchRequestBuilder::new()
        .lexical_query(LexicalSearchQuery::Obj(q))
        .build()
}

#[tokio::test(flavor = "multi_thread")]
async fn test_search_batch_empty_input_returns_empty() -> laurus::Result<()> {
    let engine = build_engine_with_corpus().await?;
    let results = engine.search_batch(Vec::new()).await?;
    assert!(results.is_empty(), "empty input must return empty output");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_search_batch_single_request_matches_search() -> laurus::Result<()> {
    let engine = build_engine_with_corpus().await?;

    let req = term_request("rust");
    let serial = engine.search(req).await?;

    let req = term_request("rust");
    let batch = engine.search_batch(vec![req]).await?;

    assert_eq!(
        batch.len(),
        1,
        "single-request batch should yield exactly one result list"
    );
    assert_eq!(
        batch[0].len(),
        serial.len(),
        "single-request batch result count must match Engine::search",
    );
    for (b, s) in batch[0].iter().zip(serial.iter()) {
        assert_eq!(b.id, s.id, "doc_id must match");
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_search_batch_multi_request_preserves_order() -> laurus::Result<()> {
    let engine = build_engine_with_corpus().await?;

    let queries = ["rust", "vector", "distributed"];
    let requests: Vec<_> = queries.iter().map(|t| term_request(t)).collect();

    let batch = engine.search_batch(requests).await?;
    assert_eq!(batch.len(), queries.len());

    // Each query targets exactly one document; verify position-preserving result.
    for (i, term) in queries.iter().enumerate() {
        let serial = engine.search(term_request(term)).await?;
        assert_eq!(
            batch[i].len(),
            serial.len(),
            "result list length must match for query[{i}] = {term}",
        );
        for (b, s) in batch[i].iter().zip(serial.iter()) {
            assert_eq!(b.id, s.id, "doc_id must match for query[{i}] = {term}");
        }
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_search_batch_existing_search_unchanged() -> laurus::Result<()> {
    let engine = build_engine_with_corpus().await?;

    // Sanity check that the existing single-request `Engine::search` still
    // returns the expected document for an ad-hoc query that the batch tests
    // also exercise.
    let req = term_request("rust");
    let results = engine.search(req).await?;
    assert!(
        results.iter().any(|r| r.id == "doc1"),
        "Engine::search must still find doc1 for term 'rust' — batch refactor must not regress"
    );
    Ok(())
}
