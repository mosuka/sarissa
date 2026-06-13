//! Conversion helpers between JSON (`serde_json::Value`) and proto messages.

use std::collections::HashMap;

use laurus::{InferredValue, infer_from_json};
use serde_json::{Map, Value, json};

use crate::convert::document::data_value_to_proto;
use crate::proto::laurus::v1;

// ---------------------------------------------------------------------------
// Value conversion
// ---------------------------------------------------------------------------

/// Convert a JSON value to a proto `Value` using engine type inference.
///
/// Delegates to [`laurus::engine::type_inference::infer_from_json`] so the
/// gateway path produces the same [`laurus::DataValue`] shapes as the
/// engine's schema-less ingestion path. The resulting `DataValue` is then
/// lowered to a proto `Value` via
/// [`crate::convert::document::data_value_to_proto`].
///
/// Mapping summary:
///
/// | JSON | proto `Value` |
/// | --- | --- |
/// | `null` | `NullValue` |
/// | `bool` | `BoolValue` |
/// | integer (fits in i64) | `Int64Value` |
/// | non-integer / large number | `Float64Value` |
/// | `string` | `TextValue` |
/// | numeric array (all i64) | `Int64ArrayValue` |
/// | numeric array (any non-i64 number) | `Float64ArrayValue` |
/// | empty array | `NullValue` (treated as skip) |
/// | object with `lat\|latitude` + `lon\|lng\|longitude` | `GeoValue` |
///
/// # Arguments
///
/// * `json` - The JSON value to convert.
///
/// # Errors
///
/// Returns the engine error message (as a `String`) when the JSON value
/// cannot be inferred — for example, mixed-type arrays, non-geo objects,
/// or out-of-range geo coordinates.
pub fn json_value_to_proto(json: &Value) -> Result<v1::Value, String> {
    use v1::value::Kind;
    match infer_from_json(json).map_err(|e| e.to_string())? {
        InferredValue::Skip => Ok(v1::Value {
            kind: Some(Kind::NullValue(true)),
        }),
        InferredValue::Inferred { value, .. } => Ok(data_value_to_proto(&value)),
    }
}

/// Converts a proto `Value` to a JSON value.
pub fn proto_value_to_json(val: &v1::Value) -> Value {
    use v1::value::Kind;
    match &val.kind {
        Some(Kind::NullValue(_)) => Value::Null,
        Some(Kind::BoolValue(b)) => Value::Bool(*b),
        Some(Kind::Int64Value(i)) => json!(*i),
        Some(Kind::Float64Value(f)) => json!(*f),
        Some(Kind::TextValue(s)) => Value::String(s.clone()),
        Some(Kind::BytesValue(b)) => {
            use std::io::Write;
            let mut buf = Vec::new();
            // Base64 encode
            let engine = base64_engine();
            write!(buf, "{}", base64_encode(&engine, b)).ok();
            Value::String(String::from_utf8_lossy(&buf).into_owned())
        }
        Some(Kind::VectorValue(v)) => Value::Array(v.values.iter().map(|f| json!(*f)).collect()),
        Some(Kind::DatetimeValue(us)) => {
            let secs = us / 1_000_000;
            let nanos = ((us % 1_000_000) * 1_000) as u32;
            if let Some(dt) = chrono::DateTime::from_timestamp(secs, nanos) {
                Value::String(dt.to_rfc3339())
            } else {
                json!(*us)
            }
        }
        Some(Kind::GeoValue(g)) => {
            json!({"latitude": g.latitude, "longitude": g.longitude})
        }
        Some(Kind::Geo3dValue(p)) => {
            json!({"x": p.x, "y": p.y, "z": p.z})
        }
        Some(Kind::Int64ArrayValue(arr)) => {
            Value::Array(arr.values.iter().map(|v| json!(*v)).collect())
        }
        Some(Kind::Float64ArrayValue(arr)) => {
            Value::Array(arr.values.iter().map(|v| json!(*v)).collect())
        }
        None => Value::Null,
    }
}

// Base64 helper (standard encoding without external crates)
fn base64_engine() -> Base64Engine {
    Base64Engine
}

struct Base64Engine;

fn base64_encode(_engine: &Base64Engine, data: &[u8]) -> String {
    // Standard base64 encoding implementation
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut result = String::new();
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;
        result.push(CHARS[((triple >> 18) & 0x3F) as usize] as char);
        result.push(CHARS[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            result.push(CHARS[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
        if chunk.len() > 2 {
            result.push(CHARS[(triple & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Document conversion
// ---------------------------------------------------------------------------

/// Convert a JSON object to a proto `Document`.
///
/// Input format: `{"fields": {"field_name": value, ...}}`. Each field value
/// is run through [`json_value_to_proto`], so the same type-inference rules
/// as the engine apply (geo aliases, range checks, multi-valued numerics,
/// mixed-array rejection).
///
/// # Arguments
///
/// * `json` - The JSON document to convert.
///
/// # Errors
///
/// Returns a `String` error message when the input is structurally invalid
/// (missing `fields` key, non-object `fields`) or when any field value
/// cannot be inferred. The error includes the offending field name.
pub fn json_to_proto_document(json: &Value) -> Result<v1::Document, String> {
    let fields_val = json
        .get("fields")
        .ok_or_else(|| "missing \"fields\" key".to_string())?;
    let fields_obj = fields_val
        .as_object()
        .ok_or_else(|| "\"fields\" must be an object".to_string())?;

    let mut fields: HashMap<String, v1::Value> = HashMap::with_capacity(fields_obj.len());
    for (k, v) in fields_obj {
        let proto_val = json_value_to_proto(v).map_err(|e| format!("field \"{k}\": {e}"))?;
        fields.insert(k.clone(), proto_val);
    }

    Ok(v1::Document { fields })
}

/// Converts a proto `Document` to a JSON value.
///
/// The internal `_id` system field is excluded from the output since it
/// is already available as the top-level `id` in search results and
/// document responses.
pub fn proto_document_to_json(doc: &v1::Document) -> Value {
    let fields: Map<String, Value> = doc
        .fields
        .iter()
        .filter(|(k, _)| k.as_str() != "_id")
        .map(|(k, v)| (k.clone(), proto_value_to_json(v)))
        .collect();
    json!({ "fields": fields })
}

// ---------------------------------------------------------------------------
// Schema conversion
// ---------------------------------------------------------------------------

/// Converts a JSON object to a proto `Schema`.
pub fn json_to_proto_schema(json: &Value) -> Result<v1::Schema, String> {
    let fields_val = json
        .get("fields")
        .ok_or_else(|| "missing \"fields\" key in schema".to_string())?;
    let fields_obj = fields_val
        .as_object()
        .ok_or_else(|| "\"fields\" must be an object".to_string())?;

    let mut fields = HashMap::new();
    for (name, opt_json) in fields_obj {
        let field_option =
            json_to_proto_field_option(opt_json).map_err(|e| format!("field \"{name}\": {e}"))?;
        fields.insert(name.clone(), field_option);
    }

    let default_fields = json
        .get("default_fields")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();

    let analyzers = json
        .get("analyzers")
        .and_then(|v| v.as_object())
        .map(|obj| {
            obj.iter()
                .map(|(name, def_json)| {
                    json_to_proto_analyzer_definition(def_json)
                        .map(|def| (name.clone(), def))
                        .map_err(|e| format!("analyzer \"{name}\": {e}"))
                })
                .collect::<Result<HashMap<_, _>, String>>()
        })
        .transpose()?
        .unwrap_or_default();

    let embedders = json
        .get("embedders")
        .and_then(|v| v.as_object())
        .map(|obj| {
            obj.iter()
                .map(|(name, def_json)| {
                    json_to_proto_embedder_definition(def_json)
                        .map(|def| (name.clone(), def))
                        .map_err(|e| format!("embedder \"{name}\": {e}"))
                })
                .collect::<Result<HashMap<_, _>, String>>()
        })
        .transpose()?
        .unwrap_or_default();

    let dynamic_field_policy = json
        .get("dynamic_field_policy")
        .and_then(|v| v.as_str())
        .map(json_to_proto_dynamic_field_policy)
        .transpose()?
        .unwrap_or(v1::DynamicFieldPolicy::Unspecified) as i32;

    Ok(v1::Schema {
        fields,
        default_fields,
        analyzers,
        embedders,
        dynamic_field_policy,
    })
}

/// Converts a JSON string to a proto `DynamicFieldPolicy`.
///
/// Accepts the lowercase names `"strict"`, `"dynamic"`, and `"ignore"`.
///
/// # Arguments
///
/// * `s` - The policy name.
///
/// # Errors
///
/// Returns an error if the string does not match a known policy name.
fn json_to_proto_dynamic_field_policy(s: &str) -> Result<v1::DynamicFieldPolicy, String> {
    match s {
        "strict" => Ok(v1::DynamicFieldPolicy::Strict),
        "dynamic" => Ok(v1::DynamicFieldPolicy::Dynamic),
        "ignore" => Ok(v1::DynamicFieldPolicy::Ignore),
        other => Err(format!(
            "unknown dynamic_field_policy \"{other}\" (expected \"strict\", \"dynamic\", or \"ignore\")"
        )),
    }
}

/// Converts a proto `DynamicFieldPolicy` enum value to its JSON string name.
///
/// The proto value `UNSPECIFIED` maps to `"dynamic"` (the default). Returns
/// `None` if the value is unrecognized.
///
/// # Arguments
///
/// * `value` - The proto enum value (i32).
fn proto_dynamic_field_policy_to_json(value: i32) -> Option<&'static str> {
    match v1::DynamicFieldPolicy::try_from(value).ok()? {
        v1::DynamicFieldPolicy::Strict => Some("strict"),
        v1::DynamicFieldPolicy::Dynamic | v1::DynamicFieldPolicy::Unspecified => Some("dynamic"),
        v1::DynamicFieldPolicy::Ignore => Some("ignore"),
    }
}

/// Converts a proto `Schema` to a JSON value.
pub fn proto_schema_to_json(schema: &v1::Schema) -> Value {
    let fields: Map<String, Value> = schema
        .fields
        .iter()
        .map(|(k, v)| (k.clone(), proto_field_option_to_json(v)))
        .collect();
    let mut result = json!({
        "fields": fields,
        "default_fields": schema.default_fields,
    });
    if !schema.analyzers.is_empty() {
        let analyzers: Map<String, Value> = schema
            .analyzers
            .iter()
            .map(|(k, v)| (k.clone(), proto_analyzer_definition_to_json(v)))
            .collect();
        result["analyzers"] = Value::Object(analyzers);
    }
    if !schema.embedders.is_empty() {
        let embedders: Map<String, Value> = schema
            .embedders
            .iter()
            .map(|(k, v)| (k.clone(), proto_embedder_definition_to_json(v)))
            .collect();
        result["embedders"] = Value::Object(embedders);
    }
    if let Some(name) = proto_dynamic_field_policy_to_json(schema.dynamic_field_policy) {
        result["dynamic_field_policy"] = Value::String(name.to_string());
    }
    result
}

// ---------------------------------------------------------------------------
// FieldOption conversion
// ---------------------------------------------------------------------------

/// Converts a JSON object to a proto `FieldOption`.
///
/// The JSON object must contain exactly one key identifying the field type
/// (e.g. `"text"`, `"integer"`, `"hnsw"`) with its configuration as the value.
///
/// # Arguments
///
/// * `json` - The JSON value representing a single field option.
///
/// # Errors
///
/// Returns an error string if the JSON is not an object or contains an
/// unknown field option type.
/// Convert a JSON analyzer reference into the proto wire form.
///
/// Accepts either a bare string (`"standard"`) or a structured object
/// (`{"language": "japanese", "dict": "..."}`). The structured form
/// currently supports only the Japanese preset.
fn json_to_analyzer_spec(value: &Value) -> Result<v1::AnalyzerSpec, String> {
    use v1::analyzer_spec::Spec;
    let inner = if let Some(name) = value.as_str() {
        Spec::Named(name.to_string())
    } else if let Some(obj) = value.as_object() {
        let language = obj
            .get("language")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "analyzer object must include a \"language\" field".to_string())?;
        match language {
            "japanese" => {
                let dict = obj
                    .get("dict")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| "japanese analyzer requires a \"dict\" path".to_string())?;
                let mode = obj
                    .get("mode")
                    .and_then(|v| v.as_str())
                    .unwrap_or("normal")
                    .to_string();
                let user_dict = obj
                    .get("user_dict")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                Spec::Builtin(v1::BuiltinAnalyzerSpec {
                    preset: Some(v1::builtin_analyzer_spec::Preset::Japanese(
                        v1::JapaneseAnalyzerSpec {
                            mode,
                            dict: dict.to_string(),
                            user_dict,
                        },
                    )),
                })
            }
            other => return Err(format!("unsupported analyzer language: {other}")),
        }
    } else {
        return Err("analyzer must be a string or object".to_string());
    };
    Ok(v1::AnalyzerSpec { spec: Some(inner) })
}

/// Convert a proto [`v1::AnalyzerSpec`] into its JSON wire form.
///
/// Returns `None` when the proto message has no spec set or the named
/// variant carries an empty string (which means "use the default").
fn analyzer_spec_to_json(spec: &v1::AnalyzerSpec) -> Option<Value> {
    use v1::analyzer_spec::Spec;
    match spec.spec.as_ref()? {
        Spec::Named(name) if name.is_empty() => None,
        Spec::Named(name) => Some(Value::String(name.clone())),
        Spec::Builtin(builtin) => match builtin.preset.as_ref()? {
            v1::builtin_analyzer_spec::Preset::Japanese(jp) => {
                let mut obj = Map::new();
                obj.insert(
                    "language".to_string(),
                    Value::String("japanese".to_string()),
                );
                let mode = if jp.mode.is_empty() {
                    "normal".to_string()
                } else {
                    jp.mode.clone()
                };
                obj.insert("mode".to_string(), Value::String(mode));
                obj.insert("dict".to_string(), Value::String(jp.dict.clone()));
                if !jp.user_dict.is_empty() {
                    obj.insert("user_dict".to_string(), Value::String(jp.user_dict.clone()));
                }
                Some(Value::Object(obj))
            }
        },
    }
}

pub fn json_to_proto_field_option(json: &Value) -> Result<v1::FieldOption, String> {
    let obj = json
        .as_object()
        .ok_or_else(|| "field option must be an object".to_string())?;

    use v1::field_option::Option as Opt;

    let option = if let Some(v) = obj.get("text") {
        Opt::Text(v1::TextOption {
            indexed: v.get("indexed").and_then(|v| v.as_bool()).unwrap_or(false),
            stored: v.get("stored").and_then(|v| v.as_bool()).unwrap_or(false),
            term_vectors: v
                .get("term_vectors")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            analyzer: v.get("analyzer").map(json_to_analyzer_spec).transpose()?,
        })
    } else if let Some(v) = obj.get("integer") {
        Opt::Integer(v1::IntegerOption {
            indexed: v.get("indexed").and_then(|v| v.as_bool()).unwrap_or(false),
            stored: v.get("stored").and_then(|v| v.as_bool()).unwrap_or(false),
            multi_valued: v
                .get("multi_valued")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
        })
    } else if let Some(v) = obj.get("float") {
        Opt::Float(v1::FloatOption {
            indexed: v.get("indexed").and_then(|v| v.as_bool()).unwrap_or(false),
            stored: v.get("stored").and_then(|v| v.as_bool()).unwrap_or(false),
            multi_valued: v
                .get("multi_valued")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
        })
    } else if let Some(v) = obj.get("boolean") {
        Opt::Boolean(v1::BooleanOption {
            indexed: v.get("indexed").and_then(|v| v.as_bool()).unwrap_or(false),
            stored: v.get("stored").and_then(|v| v.as_bool()).unwrap_or(false),
        })
    } else if let Some(v) = obj.get("date_time") {
        Opt::DateTime(v1::DateTimeOption {
            indexed: v.get("indexed").and_then(|v| v.as_bool()).unwrap_or(false),
            stored: v.get("stored").and_then(|v| v.as_bool()).unwrap_or(false),
        })
    } else if let Some(v) = obj.get("geo") {
        Opt::Geo(v1::GeoOption {
            indexed: v.get("indexed").and_then(|v| v.as_bool()).unwrap_or(false),
            stored: v.get("stored").and_then(|v| v.as_bool()).unwrap_or(false),
        })
    } else if let Some(v) = obj.get("geo3d") {
        Opt::Geo3d(v1::Geo3dOption {
            indexed: v.get("indexed").and_then(|v| v.as_bool()).unwrap_or(false),
            stored: v.get("stored").and_then(|v| v.as_bool()).unwrap_or(false),
        })
    } else if let Some(v) = obj.get("bytes") {
        Opt::Bytes(v1::BytesOption {
            stored: v.get("stored").and_then(|v| v.as_bool()).unwrap_or(false),
        })
    } else if let Some(v) = obj.get("hnsw") {
        Opt::Hnsw(json_to_hnsw_option(v)?)
    } else if let Some(v) = obj.get("flat") {
        Opt::Flat(json_to_flat_option(v)?)
    } else if let Some(v) = obj.get("ivf") {
        Opt::Ivf(json_to_ivf_option(v)?)
    } else {
        return Err("unknown field option type".to_string());
    };

    Ok(v1::FieldOption {
        option: Some(option),
    })
}

fn proto_field_option_to_json(opt: &v1::FieldOption) -> Value {
    use v1::field_option::Option as Opt;
    match &opt.option {
        Some(Opt::Text(v)) => {
            let mut text_obj = json!({
                "indexed": v.indexed,
                "stored": v.stored,
                "term_vectors": v.term_vectors,
            });
            if let Some(spec) = v.analyzer.as_ref().and_then(analyzer_spec_to_json) {
                text_obj["analyzer"] = spec;
            }
            json!({ "text": text_obj })
        }
        Some(Opt::Integer(v)) => json!({
            "integer": { "indexed": v.indexed, "stored": v.stored }
        }),
        Some(Opt::Float(v)) => json!({
            "float": { "indexed": v.indexed, "stored": v.stored }
        }),
        Some(Opt::Boolean(v)) => json!({
            "boolean": { "indexed": v.indexed, "stored": v.stored }
        }),
        Some(Opt::DateTime(v)) => json!({
            "date_time": { "indexed": v.indexed, "stored": v.stored }
        }),
        Some(Opt::Geo(v)) => json!({
            "geo": { "indexed": v.indexed, "stored": v.stored }
        }),
        Some(Opt::Geo3d(v)) => json!({
            "geo3d": { "indexed": v.indexed, "stored": v.stored }
        }),
        Some(Opt::Bytes(v)) => json!({
            "bytes": { "stored": v.stored }
        }),
        Some(Opt::Hnsw(v)) => json!({ "hnsw": hnsw_option_to_json(v) }),
        Some(Opt::Flat(v)) => json!({ "flat": flat_option_to_json(v) }),
        Some(Opt::Ivf(v)) => json!({ "ivf": ivf_option_to_json(v) }),
        None => Value::Null,
    }
}

// ---------------------------------------------------------------------------
// Vector field option conversion
// ---------------------------------------------------------------------------

fn parse_distance_metric(s: &str) -> i32 {
    match s.to_lowercase().as_str() {
        "cosine" => v1::DistanceMetric::Cosine as i32,
        "euclidean" => v1::DistanceMetric::Euclidean as i32,
        "manhattan" => v1::DistanceMetric::Manhattan as i32,
        "dot_product" => v1::DistanceMetric::DotProduct as i32,
        "angular" => v1::DistanceMetric::Angular as i32,
        _ => v1::DistanceMetric::Cosine as i32,
    }
}

fn distance_metric_to_string(val: i32) -> &'static str {
    match v1::DistanceMetric::try_from(val) {
        Ok(v1::DistanceMetric::Cosine) => "cosine",
        Ok(v1::DistanceMetric::Euclidean) => "euclidean",
        Ok(v1::DistanceMetric::Manhattan) => "manhattan",
        Ok(v1::DistanceMetric::DotProduct) => "dot_product",
        Ok(v1::DistanceMetric::Angular) => "angular",
        Err(_) => "cosine",
    }
}

fn json_to_quantizer(json: &Value) -> Option<v1::QuantizationConfig> {
    let obj = json.as_object()?;
    let method_str = obj.get("method")?.as_str()?;
    let method = match method_str.to_lowercase().as_str() {
        "scalar_8bit" => v1::QuantizationMethod::Scalar8bit as i32,
        "product_quantization" => v1::QuantizationMethod::ProductQuantization as i32,
        "product_quantization_fastscan" => {
            v1::QuantizationMethod::ProductQuantizationFastscan as i32
        }
        _ => v1::QuantizationMethod::None as i32,
    };
    let subvector_count = obj
        .get("subvector_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    Some(v1::QuantizationConfig {
        method,
        subvector_count,
    })
}

fn quantizer_to_json(q: &v1::QuantizationConfig) -> Value {
    let method = match v1::QuantizationMethod::try_from(q.method) {
        Ok(v1::QuantizationMethod::None) => "none",
        Ok(v1::QuantizationMethod::Scalar8bit) => "scalar_8bit",
        Ok(v1::QuantizationMethod::ProductQuantization) => "product_quantization",
        Ok(v1::QuantizationMethod::ProductQuantizationFastscan) => "product_quantization_fastscan",
        Err(_) => "none",
    };
    json!({
        "method": method,
        "subvector_count": q.subvector_count,
    })
}

/// Parse the JSON `rerank_storage` field (string, e.g. `"F32"`) into the
/// proto enum value (Issue #793). Returns `None` (no sidecar) when the
/// field is absent or unrecognized.
fn json_to_rerank_storage(json: &Value) -> Option<i32> {
    match json.get("rerank_storage").and_then(|v| v.as_str()) {
        Some(s) if s.eq_ignore_ascii_case("f32") => Some(v1::RerankStorageKind::F32 as i32),
        _ => None,
    }
}

/// Render a proto `rerank_storage` enum value as a JSON string (Issue
/// #793). Returns `None` for `UNSPECIFIED`/unknown so the field is
/// omitted from the JSON object.
fn rerank_storage_to_json(value: i32) -> Option<&'static str> {
    match v1::RerankStorageKind::try_from(value) {
        Ok(v1::RerankStorageKind::F32) => Some("F32"),
        _ => None,
    }
}

fn json_to_hnsw_option(json: &Value) -> Result<v1::HnswOption, String> {
    Ok(v1::HnswOption {
        dimension: json.get("dimension").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
        distance: json
            .get("distance")
            .and_then(|v| v.as_str())
            .map(parse_distance_metric)
            .unwrap_or(v1::DistanceMetric::Cosine as i32),
        m: json.get("m").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
        ef_construction: json
            .get("ef_construction")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32,
        base_weight: json
            .get("base_weight")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32,
        quantizer: json.get("quantizer").and_then(json_to_quantizer),
        embedder: json
            .get("embedder")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        // Issue #644: optional schema-level `ef_search` default.
        default_ef_search: json
            .get("default_ef_search")
            .and_then(|v| v.as_u64())
            .map(|n| n as u32),
        // Issue #793: optional Stage-2 rerank sidecar storage.
        rerank_storage: json_to_rerank_storage(json),
    })
}

fn json_to_flat_option(json: &Value) -> Result<v1::FlatOption, String> {
    Ok(v1::FlatOption {
        dimension: json.get("dimension").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
        distance: json
            .get("distance")
            .and_then(|v| v.as_str())
            .map(parse_distance_metric)
            .unwrap_or(v1::DistanceMetric::Cosine as i32),
        base_weight: json
            .get("base_weight")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32,
        quantizer: json.get("quantizer").and_then(json_to_quantizer),
        embedder: json
            .get("embedder")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        // Issue #793: optional Stage-2 rerank sidecar storage.
        rerank_storage: json_to_rerank_storage(json),
    })
}

fn json_to_ivf_option(json: &Value) -> Result<v1::IvfOption, String> {
    Ok(v1::IvfOption {
        dimension: json.get("dimension").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
        distance: json
            .get("distance")
            .and_then(|v| v.as_str())
            .map(parse_distance_metric)
            .unwrap_or(v1::DistanceMetric::Cosine as i32),
        n_clusters: json.get("n_clusters").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
        n_probe: json.get("n_probe").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
        base_weight: json
            .get("base_weight")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32,
        quantizer: json.get("quantizer").and_then(json_to_quantizer),
        embedder: json
            .get("embedder")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        // Issue #793: optional Stage-2 rerank sidecar storage.
        rerank_storage: json_to_rerank_storage(json),
    })
}

fn hnsw_option_to_json(opt: &v1::HnswOption) -> Value {
    let mut obj = json!({
        "dimension": opt.dimension,
        "distance": distance_metric_to_string(opt.distance),
        "m": opt.m,
        "ef_construction": opt.ef_construction,
        "base_weight": opt.base_weight,
    });
    if let Some(q) = &opt.quantizer {
        obj["quantizer"] = quantizer_to_json(q);
    }
    if !opt.embedder.is_empty() {
        obj["embedder"] = json!(opt.embedder);
    }
    if let Some(s) = opt.rerank_storage.and_then(rerank_storage_to_json) {
        obj["rerank_storage"] = json!(s);
    }
    obj
}

fn flat_option_to_json(opt: &v1::FlatOption) -> Value {
    let mut obj = json!({
        "dimension": opt.dimension,
        "distance": distance_metric_to_string(opt.distance),
        "base_weight": opt.base_weight,
    });
    if let Some(q) = &opt.quantizer {
        obj["quantizer"] = quantizer_to_json(q);
    }
    if !opt.embedder.is_empty() {
        obj["embedder"] = json!(opt.embedder);
    }
    if let Some(s) = opt.rerank_storage.and_then(rerank_storage_to_json) {
        obj["rerank_storage"] = json!(s);
    }
    obj
}

fn ivf_option_to_json(opt: &v1::IvfOption) -> Value {
    let mut obj = json!({
        "dimension": opt.dimension,
        "distance": distance_metric_to_string(opt.distance),
        "n_clusters": opt.n_clusters,
        "n_probe": opt.n_probe,
        "base_weight": opt.base_weight,
    });
    if let Some(q) = &opt.quantizer {
        obj["quantizer"] = quantizer_to_json(q);
    }
    if !opt.embedder.is_empty() {
        obj["embedder"] = json!(opt.embedder);
    }
    if let Some(s) = opt.rerank_storage.and_then(rerank_storage_to_json) {
        obj["rerank_storage"] = json!(s);
    }
    obj
}

// ---------------------------------------------------------------------------
// SearchRequest conversion
// ---------------------------------------------------------------------------

/// Converts a JSON object to a proto `SearchRequest`.
pub fn json_to_proto_search_request(json: &Value) -> Result<v1::SearchRequest, String> {
    let query = json
        .get("query")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let query_vectors = json
        .get("query_vectors")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(json_to_query_vector).collect())
        .unwrap_or_default();

    let limit = json.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as u32;
    let offset = json.get("offset").and_then(|v| v.as_u64()).unwrap_or(0) as u32;

    let fusion = json.get("fusion").and_then(json_to_fusion_algorithm);
    let lexical_params = json.get("lexical_params").and_then(json_to_lexical_params);
    let vector_params = json.get("vector_params").and_then(json_to_vector_params);

    let field_boosts = json
        .get("field_boosts")
        .and_then(|v| v.as_object())
        .map(|obj| {
            obj.iter()
                .filter_map(|(k, v)| v.as_f64().map(|f| (k.clone(), f as f32)))
                .collect()
        })
        .unwrap_or_default();

    Ok(v1::SearchRequest {
        query,
        query_vectors,
        limit,
        offset,
        fusion,
        lexical_params,
        vector_params,
        field_boosts,
    })
}

fn json_to_query_vector(json: &Value) -> Option<v1::QueryVector> {
    let vector = json
        .get("vector")?
        .as_array()?
        .iter()
        .filter_map(|v| v.as_f64().map(|f| f as f32))
        .collect();
    let weight = json.get("weight").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
    let fields = json
        .get("fields")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();
    Some(v1::QueryVector {
        vector,
        weight,
        fields,
    })
}

fn json_to_fusion_algorithm(json: &Value) -> Option<v1::FusionAlgorithm> {
    use v1::fusion_algorithm::Algorithm;
    if let Some(rrf) = json.get("rrf") {
        let k = rrf.get("k").and_then(|v| v.as_f64()).unwrap_or(60.0);
        Some(v1::FusionAlgorithm {
            algorithm: Some(Algorithm::Rrf(v1::Rrf { k })),
        })
    } else if let Some(ws) = json.get("weighted_sum") {
        let lexical_weight = ws
            .get("lexical_weight")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5) as f32;
        let vector_weight = ws
            .get("vector_weight")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5) as f32;
        Some(v1::FusionAlgorithm {
            algorithm: Some(Algorithm::WeightedSum(v1::WeightedSum {
                lexical_weight,
                vector_weight,
            })),
        })
    } else {
        None
    }
}

fn json_to_lexical_params(json: &Value) -> Option<v1::LexicalParams> {
    let obj = json.as_object()?;
    Some(v1::LexicalParams {
        min_score: obj.get("min_score").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
        timeout_ms: obj.get("timeout_ms").and_then(|v| v.as_u64()),
        parallel: obj
            .get("parallel")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        sort_by: obj.get("sort_by").and_then(|v| {
            let field = v.get("field")?.as_str()?.to_string();
            let order = match v.get("order").and_then(|o| o.as_str()).unwrap_or("asc") {
                "desc" => v1::SortOrder::Desc as i32,
                _ => v1::SortOrder::Asc as i32,
            };
            Some(v1::SortSpec { field, order })
        }),
    })
}

fn json_to_vector_params(json: &Value) -> Option<v1::VectorParams> {
    let obj = json.as_object()?;
    Some(v1::VectorParams {
        fields: obj
            .get("fields")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default(),
        score_mode: obj
            .get("score_mode")
            .and_then(|v| v.as_str())
            .map(|s| match s.to_lowercase().as_str() {
                "max_sim" => v1::VectorScoreMode::MaxSim as i32,
                "late_interaction" => v1::VectorScoreMode::LateInteraction as i32,
                _ => v1::VectorScoreMode::WeightedSum as i32,
            })
            .unwrap_or(v1::VectorScoreMode::WeightedSum as i32),
        overfetch: obj.get("overfetch").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
        min_score: obj.get("min_score").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
        // Issue #481 Stage 2 (rerank). Optional in proto so older
        // gateway clients can omit the field; the engine forwards it
        // to the HNSW searcher, which honors it when the queried
        // field has rerank_storage enabled and otherwise silently
        // falls back to Stage 1 ranking.
        rerank_factor: obj
            .get("rerank_factor")
            .and_then(|v| v.as_u64())
            .map(|n| n as u32),
        // Issue #644: per-query override for HNSW `ef_search`.
        ef_search: obj
            .get("ef_search")
            .and_then(|v| v.as_u64())
            .map(|n| n as u32),
    })
}

/// Converts a proto `SearchResult` to a JSON value.
pub fn proto_search_result_to_json(result: &v1::SearchResult) -> Value {
    let mut obj = json!({
        "id": result.id,
        "score": result.score,
    });
    if let Some(doc) = &result.document {
        obj["document"] = proto_document_to_json(doc);
    }
    obj
}

// ---------------------------------------------------------------------------
// Analyzer definition conversion
// ---------------------------------------------------------------------------

fn json_to_proto_analyzer_definition(json: &Value) -> Result<v1::AnalyzerDefinition, String> {
    let obj = json
        .as_object()
        .ok_or_else(|| "analyzer definition must be an object".to_string())?;

    let tokenizer = obj
        .get("tokenizer")
        .ok_or("analyzer definition missing 'tokenizer'")?;
    let tokenizer = json_to_proto_component(tokenizer)?;

    let char_filters = obj
        .get("char_filters")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .map(json_to_proto_component)
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?
        .unwrap_or_default();

    let token_filters = obj
        .get("token_filters")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .map(json_to_proto_component)
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?
        .unwrap_or_default();

    Ok(v1::AnalyzerDefinition {
        char_filters,
        tokenizer: Some(tokenizer),
        token_filters,
    })
}

fn json_to_proto_component(json: &Value) -> Result<v1::ComponentConfig, String> {
    let obj = json
        .as_object()
        .ok_or_else(|| "component config must be an object".to_string())?;
    let type_name = obj
        .get("type")
        .and_then(|v| v.as_str())
        .ok_or("component config missing 'type'")?
        .to_string();

    let mut params = HashMap::new();
    for (k, v) in obj {
        if k == "type" {
            continue;
        }
        match v {
            Value::String(s) => {
                params.insert(k.clone(), s.clone());
            }
            Value::Bool(b) => {
                params.insert(k.clone(), b.to_string());
            }
            Value::Number(n) => {
                params.insert(k.clone(), n.to_string());
            }
            Value::Array(arr) => {
                // Encode arrays as comma-separated strings.
                let items: Vec<String> = arr
                    .iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect();
                params.insert(k.clone(), items.join(","));
            }
            Value::Object(map) => {
                // Encode nested objects as key=value entries with prefixed keys.
                for (mk, mv) in map {
                    if let Some(s) = mv.as_str() {
                        params.insert(mk.clone(), s.to_string());
                    }
                }
            }
            _ => {}
        }
    }

    Ok(v1::ComponentConfig {
        r#type: type_name,
        params,
    })
}

fn proto_analyzer_definition_to_json(def: &v1::AnalyzerDefinition) -> Value {
    let mut obj = json!({});
    if !def.char_filters.is_empty() {
        let cfs: Vec<Value> = def
            .char_filters
            .iter()
            .map(proto_component_to_json)
            .collect();
        obj["char_filters"] = json!(cfs);
    }
    if let Some(tok) = &def.tokenizer {
        obj["tokenizer"] = proto_component_to_json(tok);
    }
    if !def.token_filters.is_empty() {
        let tfs: Vec<Value> = def
            .token_filters
            .iter()
            .map(proto_component_to_json)
            .collect();
        obj["token_filters"] = json!(tfs);
    }
    obj
}

fn proto_component_to_json(comp: &v1::ComponentConfig) -> Value {
    let mut obj = json!({"type": comp.r#type});
    for (k, v) in &comp.params {
        obj[k] = json!(v);
    }
    obj
}

// ---------------------------------------------------------------------------
// Embedder definition conversion
// ---------------------------------------------------------------------------

fn json_to_proto_embedder_definition(json: &Value) -> Result<v1::EmbedderConfig, String> {
    let obj = json
        .as_object()
        .ok_or_else(|| "embedder definition must be an object".to_string())?;

    let type_name = obj
        .get("type")
        .and_then(|v| v.as_str())
        .ok_or("embedder definition missing 'type'")?
        .to_string();

    let mut params = HashMap::new();
    for (k, v) in obj {
        if k == "type" {
            continue;
        }
        if let Some(s) = v.as_str() {
            params.insert(k.clone(), s.to_string());
        }
    }

    Ok(v1::EmbedderConfig {
        r#type: type_name,
        params,
    })
}

fn proto_embedder_definition_to_json(def: &v1::EmbedderConfig) -> Value {
    let mut obj = json!({"type": def.r#type});
    for (k, v) in &def.params {
        obj[k] = json!(v);
    }
    obj
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_value_roundtrip_null() {
        let json = Value::Null;
        let proto = json_value_to_proto(&json).unwrap();
        let back = proto_value_to_json(&proto);
        assert_eq!(back, Value::Null);
    }

    #[test]
    fn test_json_value_roundtrip_bool() {
        let json = json!(true);
        let proto = json_value_to_proto(&json).unwrap();
        let back = proto_value_to_json(&proto);
        assert_eq!(back, json!(true));
    }

    #[test]
    fn test_json_value_roundtrip_int() {
        let json = json!(42);
        let proto = json_value_to_proto(&json).unwrap();
        let back = proto_value_to_json(&proto);
        assert_eq!(back, json!(42));
    }

    #[test]
    fn test_json_value_roundtrip_float() {
        let json = json!(1.23);
        let proto = json_value_to_proto(&json).unwrap();
        let back = proto_value_to_json(&proto);
        assert_eq!(back, json!(1.23));
    }

    #[test]
    fn test_json_value_roundtrip_text() {
        let json = json!("hello");
        let proto = json_value_to_proto(&json).unwrap();
        let back = proto_value_to_json(&proto);
        assert_eq!(back, json!("hello"));
    }

    #[test]
    fn test_json_value_integer_array_roundtrip() {
        // Integer-only arrays now roundtrip as Int64ArrayValue (multi-valued integer field).
        let json = json!([10, 20, 30]);
        let proto = json_value_to_proto(&json).unwrap();
        let back = proto_value_to_json(&proto);
        assert_eq!(back, json!([10, 20, 30]));
    }

    #[test]
    fn test_json_value_float_array_roundtrip() {
        // Mixed-precision numeric arrays roundtrip as Float64ArrayValue.
        let json = json!([1.0, 2.5, 3.0]);
        let proto = json_value_to_proto(&json).unwrap();
        let back = proto_value_to_json(&proto);
        assert_eq!(back, json!([1.0, 2.5, 3.0]));
    }

    #[test]
    fn test_json_value_empty_array_skips_to_null() {
        // Empty arrays are inferred as Skip → emitted as NullValue.
        let json = json!([]);
        let proto = json_value_to_proto(&json).unwrap();
        let back = proto_value_to_json(&proto);
        assert_eq!(back, Value::Null);
    }

    #[test]
    fn test_json_value_mixed_array_errors() {
        let json = json!([1, "x"]);
        let err = json_value_to_proto(&json).unwrap_err();
        assert!(err.contains("array"), "unexpected error: {err}");
    }

    #[test]
    fn test_json_value_geo_canonical_keys() {
        let json = json!({"latitude": 35.6762, "longitude": 139.6503});
        let proto = json_value_to_proto(&json).unwrap();
        let back = proto_value_to_json(&proto);
        assert_eq!(back["latitude"], json!(35.6762));
        assert_eq!(back["longitude"], json!(139.6503));
    }

    #[test]
    fn test_json_value_geo_short_aliases() {
        // The engine accepts `lat` / `lon` aliases.
        let json = json!({"lat": 35.6762, "lon": 139.6503});
        let proto = json_value_to_proto(&json).unwrap();
        let back = proto_value_to_json(&proto);
        assert_eq!(back["latitude"], json!(35.6762));
        assert_eq!(back["longitude"], json!(139.6503));
    }

    #[test]
    fn test_json_value_geo_lng_alias() {
        let json = json!({"latitude": 35.6762, "lng": 139.6503});
        let proto = json_value_to_proto(&json).unwrap();
        let back = proto_value_to_json(&proto);
        assert_eq!(back["latitude"], json!(35.6762));
        assert_eq!(back["longitude"], json!(139.6503));
    }

    #[test]
    fn test_json_value_geo_out_of_range_errors() {
        let json = json!({"latitude": 200.0, "longitude": 0.0});
        let err = json_value_to_proto(&json).unwrap_err();
        assert!(err.contains("latitude"), "unexpected error: {err}");
    }

    #[test]
    fn test_json_value_non_geo_object_errors() {
        let json = json!({"foo": "bar"});
        let err = json_value_to_proto(&json).unwrap_err();
        assert!(err.contains("geographic"), "unexpected error: {err}");
    }

    #[test]
    fn test_json_value_geo3d_roundtrip() {
        // {x, y, z} JSON inputs are inferred as a Geo3d (ECEF) point and
        // round-trip back through proto without loss.
        let json = json!({"x": -3_955_182.0, "y": 3_350_553.0, "z": 3_700_276.0});
        let proto = json_value_to_proto(&json).unwrap();
        let back = proto_value_to_json(&proto);
        assert_eq!(back["x"], json!(-3_955_182.0));
        assert_eq!(back["y"], json!(3_350_553.0));
        assert_eq!(back["z"], json!(3_700_276.0));
    }

    #[test]
    fn test_json_value_geo_geo3d_mix_errors() {
        // Mixing 2D (lat/lon) and 3D (x/y/z) markers in the same object is
        // ambiguous and must be rejected.
        let json = json!({"lat": 35.0, "x": 1.0, "y": 2.0, "z": 3.0});
        let err = json_value_to_proto(&json).unwrap_err();
        assert!(err.contains("mix"), "unexpected error: {err}");
    }

    #[test]
    fn test_json_to_proto_document() {
        let json = json!({
            "fields": {
                "title": "hello",
                "count": 42
            }
        });
        let doc = json_to_proto_document(&json).unwrap();
        assert_eq!(doc.fields.len(), 2);
    }

    #[test]
    fn test_json_to_proto_document_field_error_includes_name() {
        let json = json!({
            "fields": {
                "good": "ok",
                "bad": [1, "x"]
            }
        });
        let err = json_to_proto_document(&json).unwrap_err();
        assert!(err.starts_with("field \"bad\":"), "unexpected error: {err}");
    }

    #[test]
    fn test_json_to_proto_schema() {
        let json = json!({
            "fields": {
                "title": { "text": { "indexed": true, "stored": true } },
                "embedding": { "hnsw": { "dimension": 768, "distance": "cosine" } }
            },
            "default_fields": ["title"]
        });
        let schema = json_to_proto_schema(&json).unwrap();
        assert_eq!(schema.fields.len(), 2);
        assert_eq!(schema.default_fields, vec!["title"]);
    }

    #[test]
    fn test_json_to_proto_search_request() {
        let json = json!({
            "query": "body:test",
            "limit": 10,
            "offset": 0,
            "field_boosts": { "title": 2.0 }
        });
        let req = json_to_proto_search_request(&json).unwrap();
        assert_eq!(req.query, "body:test");
        assert_eq!(req.limit, 10);
        assert_eq!(req.offset, 0);
        assert_eq!(*req.field_boosts.get("title").unwrap(), 2.0);
    }
}
