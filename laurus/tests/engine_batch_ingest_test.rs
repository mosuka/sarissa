//! Integration tests for the batch ingestion API (#551):
//! [`Engine::put_documents`] / [`Engine::add_documents`] durability across an
//! uncommitted reopen.
//!
//! The batch call defers the per-record WAL fsync and flushes once at batch
//! end, so an acknowledged batch — even without a commit — must survive a
//! process restart exactly like the equivalent singular puts, and a
//! fail-fast batch must recover exactly its applied prefix.

use std::sync::Arc;

use laurus::lexical::TextOption;
use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::{DataValue, Document, Engine, FieldOption, LaurusError, Schema};

/// Build a `(id, doc)` batch entry with a single `title` text field.
fn batch_entry(id: &str, title: &str) -> (String, Document) {
    let doc = Document::builder()
        .add_field("title", DataValue::Text(title.into()))
        .build();
    (id.to_string(), doc)
}

/// A schema with a single `title` text field.
fn title_schema() -> Schema {
    Schema::builder()
        .add_field("title", FieldOption::Text(TextOption::default()))
        .build()
}

/// An acknowledged-but-uncommitted batch must be fully durable: reopening the
/// engine on the same storage replays every batched doc from the WAL.
#[tokio::test(flavor = "multi_thread")]
async fn test_put_documents_uncommitted_batch_recovers_after_reopen() -> laurus::Result<()> {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let schema = title_schema();

    // Round 1: batch-ingest WITHOUT commit, then drop the engine.
    {
        let engine = Engine::new(storage.clone(), schema.clone()).await?;
        let docs: Vec<_> = (0..25)
            .map(|i| batch_entry(&format!("id{i}"), &format!("title-{i}")))
            .collect();
        engine.put_documents(docs).await?;
        // Drop without commit — durability must come from the batch-end
        // WAL flush alone.
    }

    // Round 2: reopen on the SAME storage; recovery replays the WAL.
    {
        let engine = Engine::new(storage.clone(), schema.clone()).await?;
        engine.commit().await?;

        let stats = engine.stats()?;
        assert_eq!(
            stats.document_count, 25,
            "every batched doc must be replayed from the WAL after reopen"
        );
        for i in 0..25 {
            let docs = engine.get_documents(&format!("id{i}")).await?;
            assert_eq!(docs.len(), 1, "id{i} must survive the reopen");
        }
    }

    Ok(())
}

/// A fail-fast batch recovers exactly its applied prefix: the docs before the
/// failing one are durable (batch-end flush runs on the error path too), the
/// failing doc and its successors never existed.
#[tokio::test(flavor = "multi_thread")]
async fn test_put_documents_failed_batch_recovers_applied_prefix() -> laurus::Result<()> {
    use laurus::DynamicFieldPolicy;

    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let schema = Schema::builder()
        .add_field("title", FieldOption::Text(TextOption::default()))
        .dynamic_field_policy(DynamicFieldPolicy::Strict)
        .build();

    // Round 1: batch fails at position 3 (undeclared field under Strict);
    // drop the engine without commit.
    {
        let engine = Engine::new(storage.clone(), schema.clone()).await?;
        let mut docs: Vec<_> = (0..3)
            .map(|i| batch_entry(&format!("ok{i}"), &format!("title-{i}")))
            .collect();
        docs.push((
            "bad".to_string(),
            Document::builder()
                .add_field("undeclared", DataValue::Text("boom".into()))
                .build(),
        ));
        docs.push(batch_entry("never", "never-applied"));

        let err = engine
            .put_documents(docs)
            .await
            .expect_err("Strict policy must fail the batch");
        assert!(
            matches!(
                err,
                LaurusError::BatchIngest {
                    failed_index: 3,
                    applied: 3,
                    ..
                }
            ),
            "expected BatchIngest at index 3, got: {err}"
        );
    }

    // Round 2: reopen — exactly the applied prefix must be recovered.
    {
        let engine = Engine::new(storage.clone(), schema.clone()).await?;
        engine.commit().await?;

        assert_eq!(
            engine.stats()?.document_count,
            3,
            "exactly the applied prefix must survive the reopen"
        );
        for i in 0..3 {
            let docs = engine.get_documents(&format!("ok{i}")).await?;
            assert_eq!(docs.len(), 1, "applied doc ok{i} must be recovered");
        }
        assert!(
            engine.get_documents("bad").await?.is_empty(),
            "the failing doc must not exist after recovery"
        );
        assert!(
            engine.get_documents("never").await?.is_empty(),
            "docs after the failing one must not exist after recovery"
        );
    }

    Ok(())
}

/// `add_documents` chunks sharing one external id survive an uncommitted
/// reopen as distinct chunks (no delete-first on replay).
#[tokio::test(flavor = "multi_thread")]
async fn test_add_documents_chunks_recover_after_reopen() -> laurus::Result<()> {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let schema = title_schema();

    {
        let engine = Engine::new(storage.clone(), schema.clone()).await?;
        let docs: Vec<_> = (0..4)
            .map(|i| batch_entry("doc", &format!("chunk-{i}")))
            .collect();
        engine.add_documents(docs).await?;
    }

    {
        let engine = Engine::new(storage.clone(), schema.clone()).await?;
        engine.commit().await?;
        let chunks = engine.get_documents("doc").await?;
        assert_eq!(
            chunks.len(),
            4,
            "all four chunks must be replayed as chunks, not deduped"
        );
    }

    Ok(())
}
