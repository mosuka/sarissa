//! Implementations for the `create` subcommand.
//!
//! Handles creating new resources:
//!
//! - [`run_index`] - Create a new index from a schema TOML file.
//! - [`run_schema`] - Interactive schema TOML generation wizard.
//!
//! The schema wizard guides the user through an interactive terminal session
//! to define index fields and their options, then writes the resulting schema
//! as a TOML file. Supports all field types provided by the laurus engine,
//! including lexical fields (Text, Integer, Float, etc.) and vector index
//! fields (HNSW, Flat, IVF).

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use dialoguer::{Confirm, Input, MultiSelect, Select};

use crate::context;

/// Execute the `create index` command.
///
/// If `schema_path` is `Some`, creates a new index from the given schema TOML
/// file. If `None`, launches the interactive schema wizard to build a schema
/// in-memory and then creates the index directly without writing a separate
/// schema file (the schema is persisted inside the index directory as
/// `schema.toml`).
///
/// # Arguments
///
/// * `schema_path` - Optional path to a schema TOML file. When `None`, the
///   interactive wizard is used instead.
/// * `index_dir` - Path to the index directory for the new index.
///
/// # Errors
///
/// Returns an error if:
/// - The schema file cannot be read or parsed (when `schema_path` is given).
/// - The interactive wizard fails (when `schema_path` is `None`).
/// - The index cannot be created.
pub async fn run_index(schema_path: Option<&Path>, index_dir: &Path) -> Result<()> {
    match schema_path {
        Some(path) => {
            context::create_index(index_dir, path).await?;
        }
        None => {
            // If schema.toml already exists, use it directly instead of
            // launching the wizard (recovery path for missing store/).
            if index_dir.join("schema.toml").exists() {
                let schema = context::read_schema(index_dir)?;
                context::create_index_from_schema(index_dir, schema).await?;
            } else {
                let schema = build_schema_interactive()?;
                context::create_index_from_schema(index_dir, schema).await?;
            }
        }
    }
    println!("Index created at {}.", index_dir.display());
    Ok(())
}
use laurus::lexical::core::field::{
    BooleanOption, BytesOption, DateTimeOption, FloatOption, Geo3dOption, GeoOption, IntegerOption,
    TextOption,
};
use laurus::vector::DistanceMetric;
use laurus::vector::core::field::{FlatOption, HnswOption, IvfOption};
use laurus::vector::core::rerank::RerankStorageKind;
use laurus::{AnalyzerSpec, BuiltinAnalyzerSpec, FieldOption, Schema};

/// Field type names shown in the interactive prompt.
const FIELD_TYPES: &[&str] = &[
    "Text", "Integer", "Float", "Boolean", "DateTime", "Geo", "Geo3d", "Bytes", "Hnsw", "Flat",
    "Ivf",
];

/// Distance metric names shown in the interactive prompt.
const DISTANCE_METRICS: &[&str] = &["Cosine", "Euclidean", "Manhattan", "DotProduct", "Angular"];

/// Run the interactive schema generation wizard (`create schema`).
///
/// Prompts the user to define fields one by one, asks for default search
/// fields among the lexical fields, previews the resulting TOML, and writes
/// it to `output` upon confirmation.
///
/// # Arguments
///
/// * `output` - Destination file path for the generated schema TOML.
///
/// # Returns
///
/// Returns `Ok(())` on success, or if the user cancels before writing.
///
/// # Errors
///
/// Returns an error if:
/// - An interactive prompt fails (e.g. terminal I/O error).
/// - The schema cannot be serialised to TOML.
/// - The output file cannot be written.
pub fn run_schema(output: &Path) -> Result<()> {
    let schema = build_schema_interactive()?;

    // Show preview.
    let toml_str = toml::to_string_pretty(&schema).context("Failed to serialize schema to TOML")?;
    println!("\n--- Preview ---");
    println!("{toml_str}");
    println!("---------------\n");

    if !Confirm::new()
        .with_prompt(format!("Write to {}?", output.display()))
        .default(true)
        .interact()?
    {
        println!("Cancelled.");
        return Ok(());
    }

    std::fs::write(output, &toml_str).context("Failed to write schema file")?;
    println!("Schema written to {}.", output.display());

    Ok(())
}

/// Run the interactive schema wizard and return the resulting [`Schema`].
///
/// Prompts the user to define fields one by one, then asks for default
/// search fields among the lexical fields. Returns the built schema without
/// writing it to disk.
///
/// # Errors
///
/// Returns an error if an interactive prompt fails (e.g. terminal I/O error).
pub fn build_schema_interactive() -> Result<Schema> {
    println!("\n=== Laurus Schema Generator ===\n");

    let mut fields: HashMap<String, FieldOption> = HashMap::new();
    let mut field_order: Vec<String> = Vec::new();

    loop {
        let name = prompt_field_name(&fields)?;
        let field_option = prompt_field_type_and_options()?;

        println!(
            "\nField \"{}\" ({}) added.\n",
            name,
            field_type_label(&field_option)
        );

        field_order.push(name.clone());
        fields.insert(name, field_option);

        if !Confirm::new()
            .with_prompt("Add another field?")
            .default(true)
            .interact()?
        {
            break;
        }
        println!();
    }

    // Collect lexical field names for default field selection.
    let lexical_fields: Vec<&str> = field_order
        .iter()
        .filter(|name| fields.get(*name).map(is_lexical_field).unwrap_or(false))
        .map(|s| s.as_str())
        .collect();

    let default_fields = if lexical_fields.is_empty() {
        Vec::new()
    } else {
        prompt_default_fields(&lexical_fields)?
    };

    Ok(Schema {
        analyzers: std::collections::HashMap::new(),
        embedders: std::collections::HashMap::new(),
        fields,
        default_fields,
        dynamic_field_policy: Default::default(),
    })
}

/// Prompt for a unique field name.
fn prompt_field_name(existing: &HashMap<String, FieldOption>) -> Result<String> {
    loop {
        let name: String = Input::new().with_prompt("Field name").interact_text()?;

        if name.is_empty() {
            println!("Field name cannot be empty.");
            continue;
        }

        if existing.contains_key(&name) {
            println!(
                "Field \"{}\" already exists. Please choose a different name.",
                name
            );
            continue;
        }

        return Ok(name);
    }
}

/// Prompt for field type selection and then type-specific options.
fn prompt_field_type_and_options() -> Result<FieldOption> {
    let type_index = Select::new()
        .with_prompt("Field type")
        .items(FIELD_TYPES)
        .default(0)
        .interact()?;

    match FIELD_TYPES[type_index] {
        "Text" => prompt_text_option(),
        "Integer" => prompt_indexed_stored_option("Integer"),
        "Float" => prompt_indexed_stored_option("Float"),
        "Boolean" => prompt_indexed_stored_option("Boolean"),
        "DateTime" => prompt_indexed_stored_option("DateTime"),
        "Geo" => prompt_indexed_stored_option("Geo"),
        "Geo3d" => prompt_indexed_stored_option("Geo3d"),
        "Bytes" => prompt_bytes_option(),
        "Hnsw" => prompt_hnsw_option(),
        "Flat" => prompt_flat_option(),
        "Ivf" => prompt_ivf_option(),
        _ => unreachable!(),
    }
}

/// Prompt for TextOption (indexed, stored, term_vectors, analyzer).
fn prompt_text_option() -> Result<FieldOption> {
    let indexed = Confirm::new()
        .with_prompt("Indexed?")
        .default(true)
        .interact()?;
    let stored = Confirm::new()
        .with_prompt("Stored?")
        .default(true)
        .interact()?;
    let term_vectors = Confirm::new()
        .with_prompt("Term vectors?")
        .default(false)
        .interact()?;

    let analyzer_choices = [
        "standard", "keyword", "english", "japanese", "simple", "noop",
    ];
    let analyzer_idx = dialoguer::Select::new()
        .with_prompt("Analyzer")
        .items(analyzer_choices)
        .default(0)
        .interact()?;
    let analyzer = if analyzer_choices[analyzer_idx] == "japanese" {
        // The Japanese preset requires a Lindera dictionary path.
        let dict: String = dialoguer::Input::new()
            .with_prompt("Lindera dictionary path (e.g. /var/lib/lindera/ipadic)")
            .interact_text()?;
        let mode_choices = ["normal", "search", "decompose"];
        let mode_idx = dialoguer::Select::new()
            .with_prompt("Lindera segmentation mode")
            .items(mode_choices)
            .default(0)
            .interact()?;
        let user_dict: String = dialoguer::Input::new()
            .with_prompt("User dictionary path (leave empty for none)")
            .allow_empty(true)
            .interact_text()?;
        Some(AnalyzerSpec::Builtin(BuiltinAnalyzerSpec::Japanese {
            mode: mode_choices[mode_idx].to_string(),
            dict,
            user_dict: if user_dict.is_empty() {
                None
            } else {
                Some(user_dict)
            },
        }))
    } else {
        Some(AnalyzerSpec::Named(
            analyzer_choices[analyzer_idx].to_string(),
        ))
    };

    Ok(FieldOption::Text(TextOption {
        indexed,
        stored,
        term_vectors,
        analyzer,
    }))
}

/// Prompt for field types that have indexed + stored options
/// (Integer, Float, Boolean, DateTime, Geo, Geo3d).
fn prompt_indexed_stored_option(type_name: &str) -> Result<FieldOption> {
    let indexed = Confirm::new()
        .with_prompt("Indexed?")
        .default(true)
        .interact()?;
    let stored = Confirm::new()
        .with_prompt("Stored?")
        .default(true)
        .interact()?;

    let multi_valued = if matches!(type_name, "Integer" | "Float") {
        Confirm::new()
            .with_prompt("Multi-valued? (accepts arrays of values)")
            .default(false)
            .interact()?
    } else {
        false
    };

    Ok(match type_name {
        "Integer" => FieldOption::Integer(IntegerOption {
            indexed,
            stored,
            multi_valued,
        }),
        "Float" => FieldOption::Float(FloatOption {
            indexed,
            stored,
            multi_valued,
        }),
        "Boolean" => FieldOption::Boolean(BooleanOption { indexed, stored }),
        "DateTime" => FieldOption::DateTime(DateTimeOption { indexed, stored }),
        "Geo" => FieldOption::Geo(GeoOption { indexed, stored }),
        "Geo3d" => FieldOption::Geo3d(Geo3dOption { indexed, stored }),
        _ => unreachable!(),
    })
}

/// Prompt for BytesOption (stored only).
fn prompt_bytes_option() -> Result<FieldOption> {
    let stored = Confirm::new()
        .with_prompt("Stored?")
        .default(true)
        .interact()?;
    Ok(FieldOption::Bytes(BytesOption { stored }))
}

/// Prompt for a distance metric selection.
fn prompt_distance_metric() -> Result<DistanceMetric> {
    let idx = Select::new()
        .with_prompt("Distance metric")
        .items(DISTANCE_METRICS)
        .default(0)
        .interact()?;

    Ok(match DISTANCE_METRICS[idx] {
        "Cosine" => DistanceMetric::Cosine,
        "Euclidean" => DistanceMetric::Euclidean,
        "Manhattan" => DistanceMetric::Manhattan,
        "DotProduct" => DistanceMetric::DotProduct,
        "Angular" => DistanceMetric::Angular,
        _ => unreachable!(),
    })
}

/// Prompt for a positive usize value with a default.
fn prompt_usize(prompt: &str, default: usize) -> Result<usize> {
    let val: usize = Input::new()
        .with_prompt(prompt)
        .default(default)
        .interact_text()?;
    Ok(val)
}

/// Prompt for HnswOption.
fn prompt_hnsw_option() -> Result<FieldOption> {
    let dimension = prompt_usize("Dimension", 128)?;
    let distance = prompt_distance_metric()?;
    let m = prompt_usize("M (max connections per node)", 16)?;
    let ef_construction = prompt_usize("ef_construction", 200)?;
    let rerank_storage = prompt_rerank_storage()?;

    Ok(FieldOption::Hnsw(HnswOption {
        dimension,
        distance,
        m,
        ef_construction,
        base_weight: 1.0,
        quantizer: Default::default(),
        rerank_storage,
        embedder: None,
    }))
}

/// Prompt the user to optionally enable Stage 2 rerank storage
/// (Issue #481). Defaults to disabled (Stage 1 int8-only behavior).
///
/// When enabled, the writer emits a per-field f32 sidecar so the
/// HNSW searcher can run two-stage rerank: a wide int8 candidate
/// fetch followed by an exact f32 rescoring of the top
/// `top_k * rerank_factor` candidates.
fn prompt_rerank_storage() -> Result<Option<RerankStorageKind>> {
    let enable = Confirm::new()
        .with_prompt("Enable Stage 2 rerank storage (extra f32 sidecar, +4 bytes/dim)?")
        .default(false)
        .interact()?;
    Ok(if enable {
        Some(RerankStorageKind::F32)
    } else {
        None
    })
}

/// Prompt for FlatOption.
fn prompt_flat_option() -> Result<FieldOption> {
    let dimension = prompt_usize("Dimension", 128)?;
    let distance = prompt_distance_metric()?;

    Ok(FieldOption::Flat(FlatOption {
        dimension,
        distance,
        base_weight: 1.0,
        quantizer: Default::default(),
        rerank_storage: None,
        embedder: None,
    }))
}

/// Prompt for IvfOption.
fn prompt_ivf_option() -> Result<FieldOption> {
    let dimension = prompt_usize("Dimension", 128)?;
    let distance = prompt_distance_metric()?;
    let n_clusters = prompt_usize("Number of clusters", 100)?;
    let n_probe = prompt_usize("Number of probes", 1)?;

    Ok(FieldOption::Ivf(IvfOption {
        dimension,
        distance,
        n_clusters,
        n_probe,
        base_weight: 1.0,
        quantizer: Default::default(),
        rerank_storage: None,
        embedder: None,
    }))
}

/// Prompt for default search fields from lexical fields.
fn prompt_default_fields(lexical_fields: &[&str]) -> Result<Vec<String>> {
    if lexical_fields.is_empty() {
        return Ok(Vec::new());
    }

    let selections = MultiSelect::new()
        .with_prompt("Select default search fields")
        .items(lexical_fields)
        .interact()?;

    Ok(selections
        .into_iter()
        .map(|i| lexical_fields[i].to_string())
        .collect())
}

/// Check if a field option is a lexical (non-vector) field type.
fn is_lexical_field(option: &FieldOption) -> bool {
    matches!(
        option,
        FieldOption::Text(_)
            | FieldOption::Integer(_)
            | FieldOption::Float(_)
            | FieldOption::Boolean(_)
            | FieldOption::DateTime(_)
            | FieldOption::Geo(_)
            | FieldOption::Geo3d(_)
            | FieldOption::Bytes(_)
    )
}

/// Return a human-readable label for a field option variant.
fn field_type_label(option: &FieldOption) -> &'static str {
    match option {
        FieldOption::Text(_) => "Text",
        FieldOption::Integer(_) => "Integer",
        FieldOption::Float(_) => "Float",
        FieldOption::Boolean(_) => "Boolean",
        FieldOption::DateTime(_) => "DateTime",
        FieldOption::Geo(_) => "Geo",
        FieldOption::Geo3d(_) => "Geo3d",
        FieldOption::Bytes(_) => "Bytes",
        FieldOption::Hnsw(_) => "Hnsw",
        FieldOption::Flat(_) => "Flat",
        FieldOption::Ivf(_) => "Ivf",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The interactive wizard must list `Geo3d` so users can declare a 3D
    /// ECEF field without dropping down to TOML hand-editing. Tracked
    /// originally in #337.
    #[test]
    fn field_types_includes_geo3d() {
        assert!(
            FIELD_TYPES.contains(&"Geo3d"),
            "FIELD_TYPES must include \"Geo3d\""
        );
    }

    /// Sanity-check the existing display path: a `FieldOption::Geo3d` value
    /// must render as `"Geo3d"` and be classified as a lexical field.
    #[test]
    fn field_type_label_handles_geo3d() {
        let opt = FieldOption::Geo3d(Geo3dOption {
            indexed: true,
            stored: true,
        });
        assert_eq!(field_type_label(&opt), "Geo3d");
        assert!(is_lexical_field(&opt));
    }

    /// Stage 2 (Issue #481): the schema TOML accepts `rerank_storage =
    /// "F32"` on an HNSW field and the field round-trips through the
    /// same toml::from_str path the cli uses to load schema.toml.
    /// Omitting the key keeps the Stage 1 default (None).
    #[test]
    fn schema_toml_round_trips_rerank_storage_on_hnsw_field() {
        let toml_with = r#"
[fields.embedding.Hnsw]
dimension = 4
distance = "Cosine"
m = 4
ef_construction = 16
rerank_storage = "F32"
"#;
        let schema: Schema =
            toml::from_str(toml_with).expect("schema TOML with rerank_storage must parse");
        let opt = schema
            .fields
            .get("embedding")
            .expect("embedding field must be present");
        match opt {
            FieldOption::Hnsw(h) => {
                assert_eq!(h.rerank_storage, Some(RerankStorageKind::F32));
            }
            other => panic!("expected Hnsw field, got {:?}", field_type_label(other)),
        }

        let toml_without = r#"
[fields.embedding.Hnsw]
dimension = 4
distance = "Cosine"
m = 4
ef_construction = 16
"#;
        let schema: Schema = toml::from_str(toml_without)
            .expect("schema TOML without rerank_storage must still parse");
        let opt = schema.fields.get("embedding").unwrap();
        match opt {
            FieldOption::Hnsw(h) => assert!(h.rerank_storage.is_none()),
            other => panic!("expected Hnsw field, got {:?}", field_type_label(other)),
        }
    }
}
