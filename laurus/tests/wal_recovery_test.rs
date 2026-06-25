use tempfile::TempDir;

use laurus::Engine;
use laurus::lexical::TermQuery;
use laurus::lexical::TextOption;
use laurus::storage::file::FileStorageConfig;
use laurus::storage::{StorageConfig, StorageFactory};
use laurus::vector::FlatOption;
use laurus::{DataValue, Document};
use laurus::{FieldOption, LexicalSearchQuery, Schema, WalSyncPolicy};

#[tokio::test(flavor = "multi_thread")]
async fn test_wal_recovery_uncommitted() -> laurus::Result<()> {
    // 1. Setup Storage
    let temp_dir = TempDir::new().unwrap();
    let storage_config = StorageConfig::File(FileStorageConfig::new(temp_dir.path()));
    let storage = StorageFactory::create(storage_config)?;

    // 2. Configure Config
    let vector_opt = FlatOption::default();
    let lexical_opt = TextOption::default();

    let config = Schema::builder()
        .add_field("title", FieldOption::Text(lexical_opt))
        .add_field("embedding", FieldOption::Flat(vector_opt))
        .build();

    // 3. Round 1: Index but DO NOT commit
    {
        let engine = Engine::new(storage.clone(), config.clone()).await?;

        // Initial state
        let query = Box::new(TermQuery::new("title", "rust"));
        let search_request = laurus::SearchRequestBuilder::new()
            .lexical_query(LexicalSearchQuery::Obj(query))
            .build();
        let search_results = engine.search(search_request).await?;
        assert_eq!(search_results.len(), 0);

        let doc1 = Document::builder()
            .add_field("title", DataValue::Text("Rust Programming".into()))
            .add_field("embedding", DataValue::Vector(vec![0.1; 128]))
            .build();

        engine.put_document("doc1", doc1).await?;

        // Drop engine WITHOUT commit
    }

    // 4. Round 2: Recover from WAL
    {
        // Re-open engine on SAME storage
        let engine = Engine::new(storage.clone(), config.clone()).await?;

        // Commit to ensure flushed to searchable index
        engine.commit().await?;

        // Should have recovered doc1 from WAL and now committed
        let query = Box::new(TermQuery::new("title", "rust"));
        let search_request = laurus::SearchRequestBuilder::new()
            .lexical_query(LexicalSearchQuery::Obj(query))
            .build();
        let search_results = engine.search(search_request).await?;
        assert_eq!(
            search_results.len(),
            1,
            "Document should be recovered from WAL"
        );
    }

    Ok(())
}

/// Under the group-commit policy (Issue #542, Phase 4) appends defer their
/// fsync, so durability is reached via [`laurus::Engine::flush_wal`] rather than
/// per-record. A record made durable by `flush_wal` (without a full commit) must
/// still recover after an uncommitted reopen, just like the per-record case.
#[tokio::test(flavor = "multi_thread")]
async fn test_wal_recovery_group_commit_flush_wal() -> laurus::Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let storage_config = StorageConfig::File(FileStorageConfig::new(temp_dir.path()));
    let storage = StorageFactory::create(storage_config)?;

    let config = Schema::builder()
        .add_field("title", FieldOption::Text(TextOption::default()))
        .add_field("embedding", FieldOption::Flat(FlatOption::default()))
        .build();

    // Round 1: index under the group-commit policy and force the WAL durable
    // with flush_wal(), but DO NOT commit.
    {
        let engine = Engine::builder(storage.clone(), config.clone())
            .wal_sync_policy(WalSyncPolicy::group_with_defaults())
            .build()
            .await?;

        let doc1 = Document::builder()
            .add_field("title", DataValue::Text("Rust Programming".into()))
            .add_field("embedding", DataValue::Vector(vec![0.1; 128]))
            .build();
        engine.put_document("doc1", doc1).await?;

        // The group policy defers the fsync; flush_wal makes the partial batch
        // durable. Drop the engine WITHOUT commit.
        engine.flush_wal()?;
    }

    // Round 2: recover from the WAL.
    {
        let engine = Engine::new(storage.clone(), config.clone()).await?;
        engine.commit().await?;

        let query = Box::new(TermQuery::new("title", "rust"));
        let search_request = laurus::SearchRequestBuilder::new()
            .lexical_query(LexicalSearchQuery::Obj(query))
            .build();
        let search_results = engine.search(search_request).await?;
        assert_eq!(
            search_results.len(),
            1,
            "flush_wal'd group-commit record should recover from WAL"
        );
    }

    Ok(())
}
