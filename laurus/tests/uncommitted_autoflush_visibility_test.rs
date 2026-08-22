//! End-to-end tests for #1016: documents written before the writer's
//! automatic `flush_segment` must stay reachable by `_id` before a commit.
//!
//! `InvertedIndexWriter` flushes a segment automatically once
//! `max_buffered_docs` (10 000) documents are buffered, which empties the
//! in-memory buffer the NRT `_id` lookup used to be limited to. Under the
//! default `CommitPolicy::Manual` that made every document written earlier
//! in an uncommitted batch silently unreachable: `get_documents` returned
//! nothing and `delete_documents` did nothing, with no error either way.
//!
//! These tests drive the public `Engine` API, so they pin the symptom a
//! user would actually hit rather than the internals that cause it.

use std::sync::Arc;

use laurus::lexical::TextOption;
use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::{DataValue, Document, Engine, FieldOption, Schema};

/// `InvertedIndexWriterConfig::default().max_buffered_docs`.
const MAX_BUFFERED_DOCS: usize = 10_000;

/// A schema with a single `title` text field.
fn title_schema() -> Schema {
    Schema::builder()
        .add_field("title", FieldOption::Text(TextOption::default()))
        .build()
}

/// Build a `(id, doc)` batch entry.
fn entry(i: usize, title: &str) -> (String, Document) {
    let doc = Document::builder()
        .add_field("title", DataValue::Text(title.into()))
        .build();
    (format!("id{i:06}"), doc)
}

/// Write past the automatic flush boundary and confirm it actually fired,
/// returning the engine and its storage.
async fn engine_past_auto_flush() -> laurus::Result<(Engine, Arc<dyn Storage>)> {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let engine = Engine::new(storage.clone(), title_schema()).await?;

    let docs: Vec<_> = (0..MAX_BUFFERED_DOCS + 50)
        .map(|i| entry(i, "alpha"))
        .collect();
    engine.put_documents(docs).await?;

    // The flush must really have happened, or these tests prove nothing:
    // a segment `.meta` exists even though nothing has been committed.
    let metas: Vec<String> = storage
        .list_files()?
        .into_iter()
        .filter(|f| f.contains("segment_") && f.ends_with(".meta"))
        .collect();
    assert!(
        !metas.is_empty(),
        "automatic flush_segment must fire at max_buffered_docs"
    );

    // And it must be written unpublished (#1017): the segment exists on
    // storage, but nothing has committed it, so search must not see it. That
    // is the invariant these NRT tests share a fixture with — they prove
    // `_id` resolution still works precisely *because* the segment is
    // invisible to the searcher.
    for meta in &metas {
        let mut input = storage.open_input(meta)?;
        let mut json = String::new();
        std::io::Read::read_to_string(&mut input, &mut json)?;
        assert!(
            json.contains("\"committed\": false"),
            "a flushed-but-uncommitted segment must be unpublished; {meta} says: {json}"
        );
    }

    Ok((engine, storage))
}

/// A document written before the automatic flush must still be readable
/// before commit — the core symptom of #1016.
#[tokio::test(flavor = "multi_thread")]
async fn get_document_written_before_auto_flush_is_readable() -> laurus::Result<()> {
    let (engine, _storage) = engine_past_auto_flush().await?;

    // Control: a document still sitting in the in-memory buffer.
    let late = engine
        .get_documents(&format!("id{:06}", MAX_BUFFERED_DOCS + 10))
        .await?;
    assert_eq!(late.len(), 1, "a buffered document must be readable");

    // Subject: a document written before the flush emptied the buffer.
    let early = engine.get_documents("id000000").await?;
    assert_eq!(
        early.len(),
        1,
        "a document written before the automatic flush must be readable before commit"
    );

    Ok(())
}

/// Deleting such a document must actually remove it rather than silently
/// doing nothing.
#[tokio::test(flavor = "multi_thread")]
async fn delete_document_written_before_auto_flush_removes_it() -> laurus::Result<()> {
    let (engine, _storage) = engine_past_auto_flush().await?;

    engine.delete_documents("id000000").await?;
    engine.commit().await?;

    let after_delete = engine.get_documents("id000000").await?;
    assert!(
        after_delete.is_empty(),
        "deleting a pre-flush document must remove it, got {} hit(s)",
        after_delete.len()
    );
    // Counts are only comparable on the same side of a commit, so assert
    // the absolute figure: everything written, less the one deleted.
    assert_eq!(
        engine.stats()?.document_count,
        (MAX_BUFFERED_DOCS + 50 - 1) as u64,
        "the delete must be reflected in the document count, not silently dropped"
    );

    Ok(())
}

/// Re-putting an `_id` written before the automatic flush must supersede
/// the earlier version instead of leaving a duplicate behind.
///
/// `docs/src/laurus/deletions.md` states that "upsert deduplication within
/// an uncommitted batch is handled separately and is always correct". When
/// `_id` resolution came back empty, the upsert found no previous version
/// to delete and the batch ended with two live copies.
#[tokio::test(flavor = "multi_thread")]
async fn upsert_across_auto_flush_leaves_no_duplicate() -> laurus::Result<()> {
    let (engine, _storage) = engine_past_auto_flush().await?;

    // Overwrite a document from before the flush, still without committing.
    let (id, doc) = entry(0, "beta");
    engine.put_document(&id, doc).await?;
    engine.commit().await?;

    let hits = engine.get_documents("id000000").await?;
    assert_eq!(
        hits.len(),
        1,
        "the re-put must supersede the earlier version, not duplicate it"
    );
    assert_eq!(
        engine.stats()?.document_count,
        (MAX_BUFFERED_DOCS + 50) as u64,
        "an overwrite must not increase the document count"
    );

    Ok(())
}
