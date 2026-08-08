//! Implementation for the `train pq-codebook` subcommand (Issue #631).
//!
//! Trains a shared PQ codebook for one HNSW vector field — from a JSONL
//! file (the same `{"id": "...", "fields": {...}}` shape the bulk-ingest
//! commands read), or with `--from-index` (Issue #920)
//! from the vectors already committed to the index itself — and persists
//! it into the index's vector storage namespace via
//! [`Engine::train_pq_codebook`](laurus::Engine::train_pq_codebook).
//! Subsequent engine opens (the next `add` / `put` / `commit` CLI
//! invocation) pick the codebook up through the field's
//! `pq_codebook_path` and encode segments against it instead of
//! re-training k-means on every commit and merge.

use std::io::BufRead;
use std::path::Path;

use anyhow::{Context, Result, bail};
use laurus::FieldOption;
use laurus::vector::Vector;

use crate::commands::bulk::parse_entry;
use crate::context;

/// Collect pre-computed training vectors for `field` from a JSONL file
/// (bulk-ingest shape: `{"id": "...", "fields": {...}}`).
///
/// Blank lines are skipped; every non-blank entry must carry a
/// pre-computed vector value for `field` as a plain numeric JSON array
/// (`[0.1, 0.2, ...]`). Parsing here goes through
/// [`laurus::json_to_document`] directly — the same as [`parse_entry`], with
/// no schema access and no engine-side coercion — so a numeric array always
/// arrives as `Float64Array`/`Int64Array`, never `DataValue::Vector`
/// (`type_inference::infer_from_json` never produces `Vector` from JSON).
/// A `DataValue::Vector` value is accepted too, as a pass-through, in case
/// entries are constructed some other way. Collection stops after the
/// first `sample_size` vectors when set (deterministic file order — no
/// random sampling).
///
/// Shared between `train pq-codebook --input` and
/// `create index --train-pq-codebook` (Issue #920).
///
/// # Arguments
///
/// * `field` - The vector field whose values to extract.
/// * `input` - Path to the JSONL training file.
/// * `sample_size` - Optional cap: only the first N vectors are collected.
///
/// # Errors
///
/// Returns an error if the file cannot be opened or read, an entry fails
/// to parse, or an entry lacks a pre-computed vector for `field` (the
/// message names the line).
pub(crate) fn collect_vectors_from_jsonl(
    field: &str,
    input: &Path,
    sample_size: Option<usize>,
) -> Result<Vec<Vector>> {
    let reader = std::io::BufReader::new(
        std::fs::File::open(input)
            .with_context(|| format!("failed to open JSONL file '{}'", input.display()))?,
    );

    let mut vectors: Vec<Vector> = Vec::new();
    for (index, line) in reader.lines().enumerate() {
        let line_no = index + 1;
        let line = line.with_context(|| format!("line {line_no}: failed to read"))?;
        if line.trim().is_empty() {
            continue;
        }
        let (_, doc) = parse_entry(&line, line_no)?;
        let Some(value) = doc.fields.get(field) else {
            bail!("line {line_no}: entry has no '{field}' field");
        };
        // A JSON numeric array is inferred as Float64Array/Int64Array (see
        // the doc comment above), never Vector directly — accept all three
        // shapes and cast element-wise to f32.
        let data: Vec<f32> = if let Some(v) = value.as_vector() {
            v.clone()
        } else if let Some(arr) = value.as_float64_array() {
            arr.iter().map(|f| *f as f32).collect()
        } else if let Some(arr) = value.as_int64_array() {
            arr.iter().map(|i| *i as f32).collect()
        } else {
            bail!(
                "line {line_no}: field '{field}' is not a pre-computed vector \
                 (embedder-generated training input is not supported; provide \
                 a numeric array, e.g. `\"{field}\": [0.1, 0.2, ...]`)"
            );
        };
        vectors.push(Vector::new(data));
        if let Some(cap) = sample_size
            && vectors.len() >= cap
        {
            break;
        }
    }
    Ok(vectors)
}

/// Execute the `train pq-codebook` command.
///
/// Collects training vectors from exactly one of two sources — a JSONL
/// file (`input`: read line by line, blank lines skipped, each entry's
/// pre-computed vector value for `field` extracted) or the index itself
/// (`from_index`: the vectors already committed for `field`, in
/// ascending doc_id order, Issue #920) — and trains the codebook on the
/// collected vectors (all of them, or the first `sample_size` when set
/// — deterministic, no random sampling). With `update_schema`, the
/// index's `schema.toml` is rewritten afterwards so the field's
/// `pq_codebook_path` names the trained file.
///
/// # Arguments
///
/// * `field` - The HNSW vector field to train for. Must be configured
///   with `ProductQuantization`.
/// * `input` - Path to the JSONL training file (bulk-ingest shape).
///   Mutually exclusive with `from_index`.
/// * `from_index` - Sample the vectors already committed to the index
///   instead of reading a file. Mutually exclusive with `input`.
/// * `sample_size` - Optional cap: only the first N vectors are used.
/// * `output` - Optional storage-relative codebook file name override
///   (defaults to the field's configured `pq_codebook_path`, else
///   `{field}.pqcb`).
/// * `update_schema` - Persist the trained file name into `schema.toml`'s
///   `pq_codebook_path` for the field.
/// * `index_dir` - Path to the index directory holding the index.
///
/// # Errors
///
/// Returns an error if neither or both of `input` / `from_index` are
/// given, the index cannot be opened, the file cannot be read, an entry
/// fails to parse, an entry lacks a pre-computed vector for `field` (the
/// message names the line; embedder-generated training input is not
/// supported), no vectors are found, training fails (wrong field type,
/// empty sample, dimension mismatch), or the schema rewrite fails.
pub async fn run_pq_codebook(
    field: &str,
    input: Option<&Path>,
    from_index: bool,
    sample_size: Option<usize>,
    output: Option<&str>,
    update_schema: bool,
    index_dir: &Path,
) -> Result<()> {
    if sample_size == Some(0) {
        bail!("--sample-size must be greater than 0");
    }
    match (input.is_some(), from_index) {
        (true, true) => bail!("--input and --from-index are mutually exclusive"),
        (false, false) => bail!("either --input or --from-index must be given"),
        _ => {}
    }

    let engine = context::open_index(index_dir).await?;

    let vectors: Vec<Vector> = if from_index {
        engine.sample_committed_vectors(field, sample_size)?
    } else {
        // Checked above: when `from_index` is false, `input` is Some.
        let input = input.expect("validated: --input given when --from-index is not");
        collect_vectors_from_jsonl(field, input, sample_size)?
    };
    if vectors.is_empty() {
        if from_index {
            bail!(
                "no committed vectors found in the index at '{}' for field '{field}' \
                 (commit some documents first, or train from a JSONL file via --input)",
                index_dir.display()
            );
        }
        bail!(
            "no training vectors found in '{}' for field '{field}'",
            input
                .expect("validated: --input given when --from-index is not")
                .display()
        );
    }

    println!(
        "Training PQ codebook for field '{field}' on {} vectors...",
        vectors.len()
    );
    let info = engine.train_pq_codebook(field, &vectors, output)?;
    println!(
        "Trained codebook '{}' (m = {}, k = {}, sub_dim = {}, dimension = {}) \
         from {} vectors.",
        info.path,
        info.subvector_count,
        info.centroids,
        info.sub_dimension,
        info.dimension,
        info.training_vectors
    );

    if update_schema {
        let mut schema = context::read_schema(index_dir)?;
        let Some(FieldOption::Hnsw(opt)) = schema.fields.get_mut(field) else {
            bail!("schema.toml has no HNSW field '{field}' to update");
        };
        opt.pq_codebook_path = Some(info.path.clone());
        context::save_schema(index_dir, &schema)?;
        println!(
            "Updated schema.toml: {field}.pq_codebook_path = \"{}\".",
            info.path
        );
    } else {
        println!(
            "Note: commits use the codebook only when the schema's \
             pq_codebook_path names it (re-run with --update-schema, or set \
             it manually)."
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use laurus::vector::core::quantization::QuantizationMethod;
    use laurus::{DataValue, Document};

    const DIM: usize = 32;

    /// Write a minimal index dir (schema.toml with one HNSW + PQ field) and
    /// a JSONL training file of `count` deterministic vectors.
    fn setup(dir: &Path, count: usize, pq_codebook_path: Option<&str>) -> std::path::PathBuf {
        let mut schema = String::from("[fields.embedding.Hnsw]\ndimension = 32\n");
        schema.push_str("distance = \"Euclidean\"\nm = 8\nef_construction = 32\n\n");
        schema.push_str("[fields.embedding.Hnsw.quantizer.ProductQuantization]\n");
        schema.push_str("subvector_count = 4\n");
        if let Some(path) = pq_codebook_path {
            // pq_codebook_path belongs to the Hnsw table, before the
            // quantizer sub-table — rebuild in valid TOML order.
            schema = format!(
                "[fields.embedding.Hnsw]\ndimension = 32\ndistance = \"Euclidean\"\n\
                 m = 8\nef_construction = 32\npq_codebook_path = \"{path}\"\n\n\
                 [fields.embedding.Hnsw.quantizer.ProductQuantization]\nsubvector_count = 4\n"
            );
        }
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("schema.toml"), schema).unwrap();

        let mut state = 0x1234_5678_u64;
        let mut jsonl = String::new();
        for i in 0..count {
            let data: Vec<String> = (0..DIM)
                .map(|_| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    format!(
                        "{:.4}",
                        ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
                    )
                })
                .collect();
            jsonl.push_str(&format!(
                "{{\"id\": \"doc{i}\", \"fields\": {{\"embedding\": [{}]}}}}\n",
                data.join(", ")
            ));
        }
        let jsonl_path = dir.join("train.jsonl");
        std::fs::write(&jsonl_path, jsonl).unwrap();
        jsonl_path
    }

    /// End-to-end: train from JSONL with `--update-schema`, then reopen the
    /// index, ingest a small batch, and commit — the sealed segment must
    /// encode against the shared codebook (a commit this small would
    /// hard-error or degrade without it).
    #[tokio::test]
    async fn train_updates_schema_and_subsequent_commit_uses_the_codebook() {
        let dir = tempfile::tempdir().unwrap();
        let jsonl = setup(dir.path(), 300, None);

        run_pq_codebook(
            "embedding",
            Some(&jsonl),
            false,
            None,
            None,
            true,
            dir.path(),
        )
        .await
        .unwrap();

        // schema.toml now names the codebook.
        let schema = context::read_schema(dir.path()).unwrap();
        let Some(FieldOption::Hnsw(opt)) = schema.fields.get("embedding") else {
            panic!("embedding field must survive the schema rewrite");
        };
        assert_eq!(opt.pq_codebook_path.as_deref(), Some("embedding.pqcb"));
        assert_eq!(
            opt.quantizer,
            QuantizationMethod::ProductQuantization { subvector_count: 4 },
            "the rewrite must not clobber sibling options"
        );
        // The codebook file landed inside the vector storage namespace.
        assert!(dir.path().join("store/vector/embedding.pqcb").exists());

        // A reopened engine commits a tiny batch on the shared codebook.
        let engine = context::open_index(dir.path()).await.unwrap();
        for i in 0..5 {
            let doc = Document::builder()
                .add_field(
                    "embedding",
                    DataValue::Vector((0..DIM).map(|j| (i * DIM + j) as f32 * 0.01).collect()),
                )
                .build();
            engine.put_document(&format!("d{i}"), doc).await.unwrap();
        }
        engine.commit().await.unwrap();
    }

    /// `--sample-size` caps the vectors used; `--output` overrides the file
    /// name; without `--update-schema` the schema file is left untouched.
    #[tokio::test]
    async fn train_honors_sample_size_and_output_without_schema_update() {
        let dir = tempfile::tempdir().unwrap();
        let jsonl = setup(dir.path(), 300, None);
        let before = std::fs::read_to_string(dir.path().join("schema.toml")).unwrap();

        run_pq_codebook(
            "embedding",
            Some(&jsonl),
            false,
            Some(280),
            Some("embedding.v2.pqcb"),
            false,
            dir.path(),
        )
        .await
        .unwrap();

        assert!(dir.path().join("store/vector/embedding.v2.pqcb").exists());
        let after = std::fs::read_to_string(dir.path().join("schema.toml")).unwrap();
        assert_eq!(before, after, "schema.toml must stay untouched");
    }

    /// Write the schema and ingest+commit `count` deterministic vectors so
    /// `--from-index` has committed data to sample (Issue #920). Returns
    /// after the commit; the engine is dropped so the training run reopens
    /// the index fresh.
    async fn setup_committed_index(dir: &Path, count: usize) {
        setup(dir, 0, None); // schema.toml only; the JSONL is empty/unused.
        let engine = context::open_index(dir).await.unwrap();
        let mut state = 0x9876_5432_u64;
        for i in 0..count {
            let data: Vec<f32> = (0..DIM)
                .map(|_| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
                })
                .collect();
            let doc = Document::builder()
                .add_field("embedding", DataValue::Vector(data))
                .build();
            engine.put_document(&format!("doc{i}"), doc).await.unwrap();
        }
        engine.commit().await.unwrap();
    }

    /// Issue #920: `--from-index` trains from the committed vectors —
    /// no JSONL file involved — and `--update-schema` still rewrites
    /// `pq_codebook_path`. A subsequent tiny commit must encode against
    /// the shared codebook (it would hard-error without one).
    #[tokio::test]
    async fn train_from_index_uses_committed_vectors_and_updates_schema() {
        let dir = tempfile::tempdir().unwrap();
        setup_committed_index(dir.path(), 300).await;

        run_pq_codebook("embedding", None, true, None, None, true, dir.path())
            .await
            .unwrap();

        let schema = context::read_schema(dir.path()).unwrap();
        let Some(FieldOption::Hnsw(opt)) = schema.fields.get("embedding") else {
            panic!("embedding field must survive the schema rewrite");
        };
        assert_eq!(opt.pq_codebook_path.as_deref(), Some("embedding.pqcb"));
        assert!(dir.path().join("store/vector/embedding.pqcb").exists());

        // A reopened engine commits a tiny batch on the shared codebook.
        let engine = context::open_index(dir.path()).await.unwrap();
        let doc = Document::builder()
            .add_field(
                "embedding",
                DataValue::Vector((0..DIM).map(|j| j as f32 * 0.01).collect()),
            )
            .build();
        engine.put_document("extra", doc).await.unwrap();
        engine.commit().await.unwrap();
    }

    /// Issue #920: `--sample-size` caps the committed vectors used under
    /// `--from-index` too (first N by ascending doc_id).
    #[tokio::test]
    async fn train_from_index_honors_sample_size() {
        let dir = tempfile::tempdir().unwrap();
        setup_committed_index(dir.path(), 300).await;

        run_pq_codebook(
            "embedding",
            None,
            true,
            Some(280),
            Some("embedding.v2.pqcb"),
            false,
            dir.path(),
        )
        .await
        .unwrap();

        assert!(dir.path().join("store/vector/embedding.v2.pqcb").exists());
    }

    /// Issue #920: exactly one of `--input` / `--from-index` must be
    /// given — both and neither are rejected up front.
    #[tokio::test]
    async fn train_rejects_conflicting_or_missing_sources() {
        let dir = tempfile::tempdir().unwrap();
        let jsonl = setup(dir.path(), 1, None);

        let err = run_pq_codebook(
            "embedding",
            Some(&jsonl),
            true,
            None,
            None,
            false,
            dir.path(),
        )
        .await
        .unwrap_err();
        assert!(
            err.to_string().contains("mutually exclusive"),
            "both sources must be rejected: {err}"
        );

        let err = run_pq_codebook("embedding", None, false, None, None, false, dir.path())
            .await
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("either --input or --from-index must be given"),
            "no source must be rejected: {err}"
        );
    }

    /// Issue #920: `--from-index` on an index with no committed vectors
    /// for the field errors with a message pointing at both remedies.
    #[tokio::test]
    async fn train_from_index_errors_on_empty_index() {
        let dir = tempfile::tempdir().unwrap();
        setup(dir.path(), 0, None); // schema only, nothing committed.

        let err = run_pq_codebook("embedding", None, true, None, None, false, dir.path())
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("no committed vectors"),
            "empty index must be named: {err}"
        );
    }

    /// A non-vector value for the field names the failing line.
    #[tokio::test]
    async fn train_rejects_non_vector_values_naming_the_line() {
        let dir = tempfile::tempdir().unwrap();
        setup(dir.path(), 1, None);
        let jsonl = dir.path().join("bad.jsonl");
        std::fs::write(
            &jsonl,
            "{\"id\": \"doc0\", \"fields\": {\"embedding\": \"oops\"}}\n",
        )
        .unwrap();

        let err = run_pq_codebook(
            "embedding",
            Some(&jsonl),
            false,
            None,
            None,
            false,
            dir.path(),
        )
        .await
        .unwrap_err();
        assert!(
            err.to_string().contains("line 1"),
            "error must name the line: {err}"
        );
        assert!(
            err.to_string().contains("not a pre-computed vector"),
            "error must explain the problem: {err}"
        );
    }
}
