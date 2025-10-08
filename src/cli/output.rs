//! Output formatting for CLI commands.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::cli::args::{OutputFormat, SarissaArgs};
use crate::error::Result;
use crate::query::Hit;

/// Result structure for index creation.
#[derive(Debug, Serialize, Deserialize)]
pub struct IndexCreationResult {
    pub path: String,
    pub schema_fields: usize,
}

/// Result structure for document addition.
#[derive(Debug, Serialize, Deserialize)]
pub struct DocumentAdditionResult {
    pub documents_added: usize,
    pub duration_ms: u64,
    pub docs_per_second: f64,
}

/// Result structure for search operations.
#[derive(Debug, Serialize, Deserialize)]
pub struct SearchResults {
    pub hits: Vec<Hit>,
    pub total_hits: u64,
    pub duration_ms: u64,
    pub facets: Option<HashMap<String, Vec<(String, u64)>>>,
    pub highlights: Option<HashMap<u32, HashMap<String, Vec<String>>>>,
}

/// Result structure for index optimization.
#[derive(Debug, Serialize, Deserialize)]
pub struct OptimizationResult {
    pub segments_before: usize,
    pub segments_after: usize,
    pub duration_ms: u64,
    pub size_reduction_bytes: u64,
}

/// Index statistics.
#[derive(Debug, Serialize, Deserialize)]
pub struct IndexStats {
    pub total_documents: u64,
    pub total_terms: u64,
    pub index_size_bytes: u64,
    pub number_of_segments: usize,
    pub field_stats: Option<HashMap<String, FieldStats>>,
}

/// Field-specific statistics.
#[derive(Debug, Serialize, Deserialize)]
pub struct FieldStats {
    pub total_terms: u64,
    pub unique_terms: u64,
    pub average_length: f64,
    pub max_length: usize,
}

/// Benchmark results.
#[derive(Debug, Serialize, Deserialize)]
pub struct BenchmarkResults {
    pub search_results: Option<SearchBenchmarkResults>,
    pub indexing_results: Option<IndexingBenchmarkResults>,
    pub total_duration_ms: u64,
}

/// Search benchmark results.
#[derive(Debug, Serialize, Deserialize)]
pub struct SearchBenchmarkResults {
    pub queries_per_second: f64,
    pub average_latency_ms: f64,
    pub min_latency_ms: f64,
    pub max_latency_ms: f64,
    pub total_queries: usize,
}

/// Indexing benchmark results.
#[derive(Debug, Serialize, Deserialize)]
pub struct IndexingBenchmarkResults {
    pub documents_per_second: f64,
    pub average_indexing_time_ms: f64,
    pub total_documents: usize,
    pub throughput_mb_per_second: f64,
}

/// Output a result in the specified format.
pub fn output_result<T: Serialize>(message: &str, result: &T, args: &SarissaArgs) -> Result<()> {
    match args.output_format {
        OutputFormat::Human => output_human(message, result, args),
        OutputFormat::Json => output_json(result, args),
        OutputFormat::Csv => output_csv(result, args),
        OutputFormat::Table => output_table(message, result, args),
        OutputFormat::Yaml => output_yaml(result, args),
    }
}

/// Output in human-readable format.
fn output_human<T: Serialize>(message: &str, result: &T, args: &SarissaArgs) -> Result<()> {
    if args.verbosity() > 0 {
        println!("{message}");
        println!();
    }

    // Convert to JSON value for easier manipulation
    let value = serde_json::to_value(result)?;

    match result {
        _ if std::any::type_name::<T>().contains("SearchResults") => {
            output_search_results_human(&value, args)
        }
        _ if std::any::type_name::<T>().contains("IndexStats") => {
            output_index_stats_human(&value, args)
        }
        _ if std::any::type_name::<T>().contains("BenchmarkResults") => {
            output_benchmark_results_human(&value, args)
        }
        _ => {
            // Generic output for other types
            output_generic_human(&value, args)
        }
    }
}

/// Output search results in human format.
fn output_search_results_human(value: &serde_json::Value, _args: &SarissaArgs) -> Result<()> {
    if let Some(obj) = value.as_object()
        && let Some(hits) = obj.get("hits").and_then(|h| h.as_array())
    {
        println!("Search Results:");
        println!("═══════════════");

        for (i, hit) in hits.iter().enumerate() {
            println!();
            println!(
                "Result {}: (Score: {:.3})",
                i + 1,
                hit.get("score").and_then(|s| s.as_f64()).unwrap_or(0.0)
            );
            println!("─────────────");

            if let Some(fields) = hit.get("fields").and_then(|f| f.as_object()) {
                for (field_name, field_value) in fields {
                    if let Some(text) = field_value.as_str() {
                        println!("{field_name}: {text}");
                    }
                }
            }
        }

        println!();

        if let Some(total) = obj.get("total_hits").and_then(|t| t.as_u64()) {
            println!("Total hits: {total}");
        }

        if let Some(duration) = obj.get("duration_ms").and_then(|d| d.as_u64()) {
            println!("Search time: {duration}ms");
        }

        // Show facets if available
        if let Some(facets) = obj.get("facets").and_then(|f| f.as_object()) {
            println!();
            println!("Facets:");
            println!("───────");
            for (field_name, facet_values) in facets {
                println!("{field_name}:");
                if let Some(values) = facet_values.as_array() {
                    for value in values {
                        if let Some(arr) = value.as_array()
                            && arr.len() >= 2
                        {
                            let label = arr[0].as_str().unwrap_or("unknown");
                            let count = arr[1].as_u64().unwrap_or(0);
                            println!("  {label} ({count})");
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// Output index statistics in human format.
fn output_index_stats_human(value: &serde_json::Value, _args: &SarissaArgs) -> Result<()> {
    if let Some(obj) = value.as_object() {
        println!("Index Statistics:");
        println!("════════════════");

        if let Some(docs) = obj.get("total_documents").and_then(|d| d.as_u64()) {
            println!("Total documents: {docs}");
        }

        if let Some(terms) = obj.get("total_terms").and_then(|t| t.as_u64()) {
            println!("Total terms: {terms}");
        }

        if let Some(size) = obj.get("index_size_bytes").and_then(|s| s.as_u64()) {
            let formatted_size = format_bytes(size);
            println!("Index size: {formatted_size}");
        }

        if let Some(segments) = obj.get("number_of_segments").and_then(|s| s.as_u64()) {
            println!("Number of segments: {segments}");
        }

        // Show field statistics if available and detailed
        if let Some(field_stats) = obj.get("field_stats").and_then(|f| f.as_object()) {
            println!();
            println!("Field Statistics:");
            println!("────────────────");

            for (field_name, stats) in field_stats {
                println!();
                println!("Field: {field_name}");

                if let Some(stats_obj) = stats.as_object() {
                    if let Some(total) = stats_obj.get("total_terms").and_then(|t| t.as_u64()) {
                        println!("  Total terms: {total}");
                    }
                    if let Some(unique) = stats_obj.get("unique_terms").and_then(|u| u.as_u64()) {
                        println!("  Unique terms: {unique}");
                    }
                    if let Some(avg_len) = stats_obj.get("average_length").and_then(|a| a.as_f64())
                    {
                        println!("  Average length: {avg_len:.1}");
                    }
                    if let Some(max_len) = stats_obj.get("max_length").and_then(|m| m.as_u64()) {
                        println!("  Max length: {max_len}");
                    }
                }
            }
        }
    }
    Ok(())
}

/// Output benchmark results in human format.
fn output_benchmark_results_human(value: &serde_json::Value, _args: &SarissaArgs) -> Result<()> {
    if let Some(obj) = value.as_object() {
        println!("Benchmark Results:");
        println!("═════════════════");

        if let Some(search_results) = obj.get("search_results").and_then(|s| s.as_object()) {
            println!();
            println!("Search Performance:");
            println!("──────────────────");

            if let Some(qps) = search_results
                .get("queries_per_second")
                .and_then(|q| q.as_f64())
            {
                println!("Queries per second: {qps:.1}");
            }
            if let Some(avg) = search_results
                .get("average_latency_ms")
                .and_then(|a| a.as_f64())
            {
                println!("Average latency: {avg:.2}ms");
            }
            if let Some(min) = search_results
                .get("min_latency_ms")
                .and_then(|m| m.as_f64())
            {
                println!("Min latency: {min:.2}ms");
            }
            if let Some(max) = search_results
                .get("max_latency_ms")
                .and_then(|m| m.as_f64())
            {
                println!("Max latency: {max:.2}ms");
            }
            if let Some(total) = search_results.get("total_queries").and_then(|t| t.as_u64()) {
                println!("Total queries: {total}");
            }
        }

        if let Some(indexing_results) = obj.get("indexing_results").and_then(|i| i.as_object()) {
            println!();
            println!("Indexing Performance:");
            println!("────────────────────");

            if let Some(dps) = indexing_results
                .get("documents_per_second")
                .and_then(|d| d.as_f64())
            {
                println!("Documents per second: {dps:.1}");
            }
            if let Some(avg) = indexing_results
                .get("average_indexing_time_ms")
                .and_then(|a| a.as_f64())
            {
                println!("Average indexing time: {avg:.2}ms");
            }
            if let Some(total) = indexing_results
                .get("total_documents")
                .and_then(|t| t.as_u64())
            {
                println!("Total documents: {total}");
            }
            if let Some(throughput) = indexing_results
                .get("throughput_mb_per_second")
                .and_then(|t| t.as_f64())
            {
                println!("Throughput: {throughput:.1} MB/s");
            }
        }

        if let Some(total_duration) = obj.get("total_duration_ms").and_then(|t| t.as_u64()) {
            println!();
            println!("Total benchmark time: {total_duration}ms");
        }
    }
    Ok(())
}

/// Output generic data in human format.
fn output_generic_human(value: &serde_json::Value, _args: &SarissaArgs) -> Result<()> {
    match value {
        serde_json::Value::Object(obj) => {
            for (key, val) in obj {
                let formatted_val = format_value(val);
                println!("{key}: {formatted_val}");
            }
        }
        _ => {
            let formatted_value = format_value(value);
            println!("{formatted_value}");
        }
    }
    Ok(())
}

/// Output in JSON format.
fn output_json<T: Serialize>(result: &T, args: &SarissaArgs) -> Result<()> {
    let json = if args.pretty {
        serde_json::to_string_pretty(result)?
    } else {
        serde_json::to_string(result)?
    };

    println!("{json}");
    Ok(())
}

/// Output in CSV format.
fn output_csv<T: Serialize>(result: &T, _args: &SarissaArgs) -> Result<()> {
    // For CSV output, we'll convert to JSON first and then try to flatten
    let value = serde_json::to_value(result)?;

    match value {
        serde_json::Value::Array(arr) => {
            // Output array as CSV rows
            for (i, item) in arr.iter().enumerate() {
                if i == 0 {
                    // Output header
                    if let Some(obj) = item.as_object() {
                        let headers: Vec<String> = obj.keys().cloned().collect();
                        let header_line = headers.join(",");
                        println!("{header_line}");
                    }
                }

                // Output row
                if let Some(obj) = item.as_object() {
                    let values: Vec<String> = obj.values().map(format_csv_value).collect();
                    let value_line = values.join(",");
                    println!("{value_line}");
                }
            }
        }
        serde_json::Value::Object(obj) => {
            // Output single object as key-value pairs
            println!("key,value");
            for (key, value) in obj {
                let formatted_csv_value = format_csv_value(&value);
                println!("{key},{formatted_csv_value}");
            }
        }
        _ => {
            println!("value");
            let formatted_csv_value = format_csv_value(&value);
            println!("{formatted_csv_value}");
        }
    }

    Ok(())
}

/// Output in table format.
fn output_table<T: Serialize>(message: &str, result: &T, args: &SarissaArgs) -> Result<()> {
    // For simplicity, table format will be similar to human format but with better alignment
    output_human(message, result, args)
}

/// Output in YAML format.
fn output_yaml<T: Serialize>(result: &T, _args: &SarissaArgs) -> Result<()> {
    // Convert to JSON value first, then format as simple YAML
    let value = serde_json::to_value(result)?;
    print_yaml_value(&value, 0);
    Ok(())
}

/// Print YAML value with indentation.
fn print_yaml_value(value: &serde_json::Value, indent: usize) {
    let spaces = "  ".repeat(indent);

    match value {
        serde_json::Value::Object(obj) => {
            for (key, val) in obj {
                match val {
                    serde_json::Value::Object(_) | serde_json::Value::Array(_) => {
                        println!("{spaces}{key}:");
                        print_yaml_value(val, indent + 1);
                    }
                    _ => {
                        let formatted_yaml_val = format_yaml_value(val);
                        println!("{spaces}{key}: {formatted_yaml_val}");
                    }
                }
            }
        }
        serde_json::Value::Array(arr) => {
            for item in arr {
                match item {
                    serde_json::Value::Object(_) | serde_json::Value::Array(_) => {
                        println!("{spaces}  -");
                        print_yaml_value(item, indent + 2);
                    }
                    _ => {
                        let formatted_yaml_item = format_yaml_value(item);
                        println!("{spaces}  - {formatted_yaml_item}");
                    }
                }
            }
        }
        _ => {
            let formatted_yaml_value = format_yaml_value(value);
            println!("{formatted_yaml_value}");
        }
    }
}

/// Format a JSON value for YAML output.
fn format_yaml_value(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => {
            if s.contains('\n') || s.contains('"') || s.contains('\\') {
                let escaped = s.replace('"', "\\\"");
                format!("\"{escaped}\"")
            } else {
                s.clone()
            }
        }
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Null => "null".to_string(),
        _ => "~".to_string(), // For complex types
    }
}

/// Format a JSON value for display.
fn format_value(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Array(arr) => {
            let formatted_values = arr.iter().map(format_value).collect::<Vec<_>>().join(", ");
            format!("[{formatted_values}]")
        }
        serde_json::Value::Object(_) => "[object]".to_string(),
        serde_json::Value::Null => "null".to_string(),
    }
}

/// Format a JSON value for CSV output.
fn format_csv_value(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => {
            if s.contains(',') || s.contains('"') || s.contains('\n') {
                let escaped = s.replace('"', "\"\"");
                format!("\"{escaped}\"")
            } else {
                s.clone()
            }
        }
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Array(arr) => {
            let formatted_values = arr.iter().map(format_value).collect::<Vec<_>>().join("; ");
            format!("\"[{formatted_values}]\"")
        }
        serde_json::Value::Object(_) => "\"[object]\"".to_string(),
        serde_json::Value::Null => "".to_string(),
    }
}

/// Format bytes into human-readable format.
fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_index = 0;

    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }

    if unit_index == 0 {
        let unit = UNITS[unit_index];
        format!("{bytes} {unit}")
    } else {
        let unit = UNITS[unit_index];
        format!("{size:.1} {unit}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(1536), "1.5 KB");
        assert_eq!(format_bytes(1048576), "1.0 MB");
        assert_eq!(format_bytes(1073741824), "1.0 GB");
    }

    #[test]
    fn test_format_csv_value() {
        assert_eq!(
            format_csv_value(&serde_json::Value::String("test".to_string())),
            "test"
        );
        assert_eq!(
            format_csv_value(&serde_json::Value::String("test,with,commas".to_string())),
            "\"test,with,commas\""
        );
        assert_eq!(
            format_csv_value(&serde_json::Value::Number(serde_json::Number::from(42))),
            "42"
        );
        assert_eq!(format_csv_value(&serde_json::Value::Bool(true)), "true");
        assert_eq!(format_csv_value(&serde_json::Value::Null), "");
    }

    #[test]
    fn test_format_value() {
        assert_eq!(
            format_value(&serde_json::Value::String("test".to_string())),
            "test"
        );
        assert_eq!(
            format_value(&serde_json::Value::Number(serde_json::Number::from(42))),
            "42"
        );
        assert_eq!(format_value(&serde_json::Value::Bool(false)), "false");
        assert_eq!(format_value(&serde_json::Value::Null), "null");
    }
}

/// Result structure for index validation.
#[derive(Debug, Serialize, Deserialize)]
pub struct ValidationResult {
    pub path: String,
    pub issues_found: usize,
    pub issues_fixed: usize,
}

/// Information about an index.
#[derive(Debug, Serialize, Deserialize)]
pub struct IndexInfo {
    pub name: String,
    pub path: String,
    pub document_count: u64,
    pub size: u64,
    pub last_modified: Option<std::time::SystemTime>,
}

/// Result structure for listing indices.
#[derive(Debug, Serialize, Deserialize)]
pub struct IndicesListResult {
    pub directory: String,
    pub indices: Vec<IndexInfo>,
    pub detailed: bool,
}
