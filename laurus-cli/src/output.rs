//! Output formatting utilities for CLI results.
//!
//! This module provides functions to render search results, documents, and
//! index statistics in either a human-readable table or JSON format. The
//! desired format is selected via the [`OutputFormat`] enum.

use std::collections::HashMap;

use clap::ValueEnum;
use laurus::{DataValue, Document, EngineStats, SearchResult};
use serde_json::json;
use tabled::settings::Style;
use tabled::{Table, Tabled};

/// Output format for CLI results.
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum OutputFormat {
    /// Human-readable table.
    Table,
    /// JSON output.
    Json,
}

/// Print search results to stdout.
///
/// # Arguments
///
/// * `results` - Slice of [`SearchResult`] entries returned by the engine.
/// * `format` - The desired output format (table or JSON).
pub fn print_search_results(results: &[SearchResult], format: OutputFormat) {
    match format {
        OutputFormat::Json => {
            let json_results: Vec<serde_json::Value> = results
                .iter()
                .map(|r| {
                    let mut obj = json!({
                        "id": r.id,
                        "score": r.score,
                    });
                    if let Some(ref doc) = r.document {
                        obj["document"] = fields_to_json(&doc.fields);
                    }
                    obj
                })
                .collect();
            println!("{}", serde_json::to_string_pretty(&json_results).unwrap());
        }
        OutputFormat::Table => {
            if results.is_empty() {
                println!("No results found.");
                return;
            }

            let rows: Vec<SearchResultRow> = results
                .iter()
                .map(|r| {
                    let fields = r
                        .document
                        .as_ref()
                        .map(|doc| format_fields_compact(&doc.fields))
                        .unwrap_or_default();
                    SearchResultRow {
                        id: r.id.clone(),
                        score: format!("{:.4}", r.score),
                        fields,
                    }
                })
                .collect();

            let table = Table::new(&rows).with(Style::rounded()).to_string();
            println!("{table}");
        }
    }
}

/// Print documents to stdout.
///
/// # Arguments
///
/// * `id` - The external document ID used for display.
/// * `documents` - Slice of [`Document`] entries (may contain multiple chunks).
/// * `format` - The desired output format (table or JSON).
pub fn print_documents(id: &str, documents: &[Document], format: OutputFormat) {
    match format {
        OutputFormat::Json => {
            let json_docs: Vec<serde_json::Value> = documents
                .iter()
                .map(|doc| {
                    json!({
                        "id": id,
                        "document": fields_to_json(&doc.fields),
                    })
                })
                .collect();
            println!("{}", serde_json::to_string_pretty(&json_docs).unwrap());
        }
        OutputFormat::Table => {
            if documents.is_empty() {
                println!("No documents found for id '{id}'.");
                return;
            }

            let rows: Vec<DocumentRow> = documents
                .iter()
                .enumerate()
                .map(|(i, doc)| DocumentRow {
                    id: if i == 0 {
                        id.to_string()
                    } else {
                        format!("{id} (chunk {i})")
                    },
                    fields: format_fields_compact(&doc.fields),
                })
                .collect();

            let table = Table::new(&rows).with(Style::rounded()).to_string();
            println!("{table}");
        }
    }
}

/// Print index statistics to stdout.
///
/// # Arguments
///
/// * `stats` - [`EngineStats`] containing document count and per-field vector
///   statistics.
/// * `format` - The desired output format (table or JSON).
pub fn print_stats(stats: &EngineStats, format: OutputFormat) {
    match format {
        OutputFormat::Json => {
            let vector_fields_json: serde_json::Value = stats
                .vector_fields
                .iter()
                .map(|(name, fs)| {
                    (
                        name.clone(),
                        json!({
                            "vector_count": fs.vector_count,
                            "dimension": fs.dimension,
                        }),
                    )
                })
                .collect::<serde_json::Map<String, serde_json::Value>>()
                .into();
            let output = json!({
                "document_count": stats.document_count,
                "vector_fields": vector_fields_json,
            });
            println!("{}", serde_json::to_string_pretty(&output).unwrap());
        }
        OutputFormat::Table => {
            println!("Document count: {}", stats.document_count);

            if !stats.vector_fields.is_empty() {
                let rows: Vec<FieldStatsRow> = stats
                    .vector_fields
                    .iter()
                    .map(|(name, fs)| FieldStatsRow {
                        field: name.clone(),
                        vector_count: fs.vector_count,
                        dimension: fs.dimension,
                    })
                    .collect();

                let table = Table::new(&rows).with(Style::rounded()).to_string();
                println!("\nVector fields:\n{table}");
            }
        }
    }
}

// --- Helper types and functions ---

#[derive(Tabled)]
struct SearchResultRow {
    #[tabled(rename = "ID")]
    id: String,
    #[tabled(rename = "Score")]
    score: String,
    #[tabled(rename = "Fields")]
    fields: String,
}

#[derive(Tabled)]
struct DocumentRow {
    #[tabled(rename = "ID")]
    id: String,
    #[tabled(rename = "Fields")]
    fields: String,
}

#[derive(Tabled)]
struct FieldStatsRow {
    #[tabled(rename = "Field")]
    field: String,
    #[tabled(rename = "Vectors")]
    vector_count: usize,
    #[tabled(rename = "Dimension")]
    dimension: usize,
}

/// Convert field data to a compact display string.
fn format_fields_compact(fields: &HashMap<String, DataValue>) -> String {
    let mut parts: Vec<String> = fields
        .iter()
        .filter(|(k, _)| k.as_str() != "_id")
        .map(|(k, v)| format!("{k}: {}", format_data_value(v)))
        .collect();
    parts.sort();
    parts.join(", ")
}

/// Convert fields to JSON value.
fn fields_to_json(fields: &HashMap<String, DataValue>) -> serde_json::Value {
    let map: serde_json::Map<String, serde_json::Value> = fields
        .iter()
        .filter(|(k, _)| k.as_str() != "_id")
        .map(|(k, v)| (k.clone(), data_value_to_json(v)))
        .collect();
    serde_json::Value::Object(map)
}

/// Format a DataValue for compact display.
fn format_data_value(value: &DataValue) -> String {
    match value {
        DataValue::Null => "null".to_string(),
        DataValue::Bool(b) => b.to_string(),
        DataValue::Int64(i) => i.to_string(),
        DataValue::Float64(f) => f.to_string(),
        DataValue::Text(s) => {
            if s.len() > 80 {
                format!("{}...", &s[..77])
            } else {
                s.clone()
            }
        }
        DataValue::Bytes(b, _) => format!("<{} bytes>", b.len()),
        DataValue::Vector(v) => format!("<vector dim={}>", v.len()),
        DataValue::DateTime(dt) => dt.to_rfc3339(),
        DataValue::Geo(lat, lon) => format!("({lat}, {lon})"),
        DataValue::Int64Array(arr) => format!(
            "[{}]",
            arr.iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
        DataValue::Float64Array(arr) => format!(
            "[{}]",
            arr.iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// Convert DataValue to serde_json::Value.
fn data_value_to_json(value: &DataValue) -> serde_json::Value {
    match value {
        DataValue::Null => serde_json::Value::Null,
        DataValue::Bool(b) => json!(b),
        DataValue::Int64(i) => json!(i),
        DataValue::Float64(f) => json!(f),
        DataValue::Text(s) => json!(s),
        DataValue::Bytes(b, mime) => json!({"bytes_len": b.len(), "mime": mime}),
        DataValue::Vector(v) => json!(v),
        DataValue::DateTime(dt) => json!(dt.to_rfc3339()),
        DataValue::Geo(lat, lon) => json!({"lat": lat, "lon": lon}),
        DataValue::Int64Array(arr) => json!(arr),
        DataValue::Float64Array(arr) => json!(arr),
    }
}
