//! Implementations for the `put` subcommand.
//!
//! Handles upserting resources into an existing index:
//!
//! - [`run_doc`] - Put (upsert) a document into the index.

use std::path::Path;

use anyhow::{Context, Result};

use crate::context;
use crate::json_doc::parse_document_json;

/// Execute the `put doc` command.
///
/// Opens the index at `index_dir`, parses the document JSON, and upserts it
/// into the index. If a document with the same external ID already exists,
/// all its chunks are deleted before the new document is indexed.
/// Changes are not committed automatically.
///
/// # Arguments
///
/// * `id` - External document ID.
/// * `data_json` - A JSON string of the shape `{"fields": {...}}`.
/// * `index_dir` - Path to the index directory holding the index.
///
/// # Errors
///
/// Returns an error if:
/// - The index cannot be opened.
/// - The JSON string cannot be parsed into a [`laurus::Document`].
/// - The engine rejects the document.
pub async fn run_doc(id: &str, data_json: &str, index_dir: &Path) -> Result<()> {
    let engine = context::open_index(index_dir).await?;
    let doc = parse_document_json(data_json).context("Failed to parse document JSON")?;
    engine.put_document(id, doc).await?;
    println!("Document '{id}' put (upserted). Run 'commit' to persist changes.");
    Ok(())
}
