//! Implementations for the `put docs` / `add docs` bulk-ingest subcommands.
//!
//! Streams a JSONL file — one `{"id": "...", "document": {"fields": {...}}}`
//! entry per line — and applies it through the engine's batch API
//! ([`Engine::put_documents`](laurus::Engine::put_documents) /
//! [`Engine::add_documents`](laurus::Engine::add_documents)), which pays one
//! WAL fsync per batch instead of one per record. Unlike the singular
//! `put doc` / `add doc` commands, bulk ingestion commits automatically:
//! every `--commit-every` applied documents and once at the end.

use std::io::BufRead;
use std::path::Path;

use anyhow::{Context, Result, bail};
use laurus::Document;

use crate::context;

/// Whether a bulk run upserts (`put docs`) or appends chunks (`add docs`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BulkMode {
    /// Upsert: duplicate ids replace earlier versions (last occurrence wins).
    Put,
    /// Chunk append: repeated ids accumulate as chunks.
    Add,
}

/// One parsed JSONL entry.
#[derive(serde::Deserialize)]
struct BulkEntry {
    /// External document ID.
    id: String,
    /// The document, in the same serde JSON shape as `put doc --data`
    /// (`{"fields": {"title": {"Text": "..."}}}`).
    document: Document,
}

/// Parse one JSONL line into an `(id, document)` pair.
///
/// # Arguments
///
/// * `line` - The raw line content (must not be blank).
/// * `line_no` - 1-based line number, used in error messages.
///
/// # Errors
///
/// Returns an error naming the line when the JSON does not parse into a
/// `{"id", "document"}` entry.
fn parse_entry(line: &str, line_no: usize) -> Result<(String, Document)> {
    let entry: BulkEntry = serde_json::from_str(line)
        .with_context(|| format!("line {line_no}: failed to parse JSONL entry"))?;
    Ok((entry.id, entry.document))
}

/// Execute the `put docs` / `add docs` command.
///
/// Reads `file` line by line (blank lines are skipped), groups entries into
/// batches of `batch_size`, and applies each batch via the engine's bulk API.
/// Commits after every `commit_every` applied documents (`0` disables the
/// periodic commits) and once at the end, then prints the applied count.
///
/// On a mid-batch failure the engine's fail-fast semantics apply: the error
/// message names the offending **line** of the input file, the applied prefix
/// is committed (so the work is not lost), and re-running with the remaining
/// suffix of the file is idempotent under `put` mode.
///
/// # Arguments
///
/// * `file` - Path to the JSONL file to ingest.
/// * `mode` - [`BulkMode::Put`] (upsert) or [`BulkMode::Add`] (chunk append).
/// * `batch_size` - Documents per engine batch call (must be > 0).
/// * `commit_every` - Commit every N applied documents; `0` = final only.
/// * `index_dir` - Path to the index directory holding the index.
///
/// # Errors
///
/// Returns an error if the index cannot be opened, the file cannot be read,
/// a line fails to parse, or the engine rejects a batch (the message carries
/// the failing line number and the count applied before the failure).
pub async fn run(
    file: &Path,
    mode: BulkMode,
    batch_size: usize,
    commit_every: usize,
    index_dir: &Path,
) -> Result<()> {
    if batch_size == 0 {
        bail!("--batch-size must be greater than 0");
    }

    let engine = context::open_index(index_dir).await?;
    let reader = std::io::BufReader::new(
        std::fs::File::open(file)
            .with_context(|| format!("failed to open JSONL file '{}'", file.display()))?,
    );

    let mut batch: Vec<(String, Document)> = Vec::with_capacity(batch_size);
    // 1-based input line number of each entry in `batch`, for error reports.
    let mut batch_lines: Vec<usize> = Vec::with_capacity(batch_size);
    let mut applied: usize = 0;
    let mut since_commit: usize = 0;

    let flush = async |batch: Vec<(String, Document)>,
                       lines: Vec<usize>,
                       already_applied: usize|
           -> Result<usize> {
        if batch.is_empty() {
            return Ok(0);
        }
        let size = batch.len();
        let result = match mode {
            BulkMode::Put => engine.put_documents(batch).await,
            BulkMode::Add => engine.add_documents(batch).await,
        };
        if let Err(e) = result {
            // Commit the applied prefix so completed work survives, then
            // surface the failing input line.
            engine.commit().await.ok();
            if let laurus::LaurusError::BatchIngest {
                failed_index,
                failed_id,
                applied: batch_applied,
                ..
            } = &e
            {
                let line = lines.get(*failed_index).copied().unwrap_or(0);
                bail!(
                    "line {line} (id '{failed_id}'): {e} — {} documents were applied and \
                     committed before the failure",
                    already_applied + batch_applied
                );
            }
            return Err(e.into());
        }
        Ok(size)
    };

    for (index, line) in reader.lines().enumerate() {
        let line_no = index + 1;
        let line = line.with_context(|| format!("line {line_no}: failed to read"))?;
        if line.trim().is_empty() {
            continue;
        }
        let (id, doc) = parse_entry(&line, line_no)?;
        batch.push((id, doc));
        batch_lines.push(line_no);

        if batch.len() >= batch_size {
            let docs = std::mem::take(&mut batch);
            let lines = std::mem::take(&mut batch_lines);
            let flushed = flush(docs, lines, applied).await?;
            applied += flushed;
            since_commit += flushed;
            if commit_every > 0 && since_commit >= commit_every {
                engine.commit().await?;
                since_commit = 0;
                println!("... {applied} documents applied (committed)");
            }
        }
    }
    applied += flush(batch, batch_lines, applied).await?;
    engine.commit().await?;

    let verb = match mode {
        BulkMode::Put => "put (upserted)",
        BulkMode::Add => "added as chunks",
    };
    println!("{applied} documents {verb} and committed.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_entry_accepts_the_put_doc_data_shape() {
        let line = r#"{"id": "doc1", "document": {"fields": {"title": {"Text": "Hello"}}}}"#;
        let (id, doc) = parse_entry(line, 1).unwrap();
        assert_eq!(id, "doc1");
        assert!(doc.fields.contains_key("title"));
    }

    #[test]
    fn parse_entry_names_the_failing_line() {
        let err = parse_entry("{not json", 42).unwrap_err();
        assert!(
            err.to_string().contains("line 42"),
            "error must carry the line number: {err}"
        );

        let err = parse_entry(r#"{"document": {"fields": {}}}"#, 7).unwrap_err();
        assert!(
            err.to_string().contains("line 7"),
            "a missing id must also name the line: {err}"
        );
    }

    /// End-to-end: `put docs` ingests a JSONL file (small batches + periodic
    /// commits + in-batch dedup) and the docs are retrievable afterwards.
    #[tokio::test]
    async fn bulk_put_ingests_jsonl_end_to_end() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("schema.toml"),
            "[fields.title.Text]\nstored = true\nindexed = true\n",
        )
        .unwrap();

        let jsonl_path = dir.path().join("docs.jsonl");
        let mut jsonl = String::new();
        for i in 0..7 {
            jsonl.push_str(&format!(
                "{{\"id\": \"doc{i}\", \"document\": {{\"fields\": {{\"title\": {{\"Text\": \"t{i}\"}}}}}}}}\n",
            ));
        }
        // Blank line + a duplicate id (must dedup, last wins under put mode).
        jsonl.push('\n');
        jsonl.push_str(
            "{\"id\": \"doc0\", \"document\": {\"fields\": {\"title\": {\"Text\": \"t0v2\"}}}}\n",
        );
        std::fs::write(&jsonl_path, jsonl).unwrap();

        // batch_size 3 exercises multiple flushes; commit_every 5 exercises
        // the periodic commit path.
        run(&jsonl_path, BulkMode::Put, 3, 5, dir.path())
            .await
            .unwrap();

        let engine = context::open_index(dir.path()).await.unwrap();
        let docs = engine.get_documents("doc0").await.unwrap();
        assert_eq!(docs.len(), 1, "duplicate id must dedup under put mode");
        let title = docs[0]
            .fields
            .get("title")
            .and_then(|v| v.as_text())
            .unwrap();
        assert_eq!(title, "t0v2", "the last occurrence must win");
        for i in 1..7 {
            assert_eq!(
                engine
                    .get_documents(&format!("doc{i}"))
                    .await
                    .unwrap()
                    .len(),
                1,
                "doc{i} must be ingested"
            );
        }
    }
}
