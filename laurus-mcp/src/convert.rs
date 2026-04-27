//! Conversions between protobuf types and JSON values.
//!
//! These helpers translate between the laurus-server proto `Document` / `Value`
//! types and `serde_json::Value` for the MCP tool input/output.

use laurus_server::proto::laurus::v1;
use serde_json::{Value, json};

/// Convert a proto [`v1::Document`] to a [`serde_json::Value`] (JSON object).
///
/// # Arguments
///
/// * `doc` - The proto document to convert.
pub fn document_to_json(doc: &v1::Document) -> Value {
    let fields: serde_json::Map<String, Value> = doc
        .fields
        .iter()
        .map(|(k, v)| (k.clone(), proto_value_to_json(v)))
        .collect();
    Value::Object(fields)
}

/// Convert a [`serde_json::Value`] (JSON object) to a proto [`v1::Document`].
///
/// JSON types are mapped to proto `Value` kinds as follows:
///
/// | JSON type | Proto Value kind |
/// |-----------|-----------------|
/// | null | `null_value` |
/// | boolean | `bool_value` |
/// | integer number | `int64_value` |
/// | float number | `float64_value` |
/// | string | `text_value` |
/// | array of numbers | `vector_value` (f32 elements) |
/// | other | `null_value` |
///
/// # Arguments
///
/// * `value` - A JSON object whose keys become document field names.
///
/// # Errors
///
/// Returns an error if `value` is not a JSON object.
pub fn json_to_document(value: Value) -> anyhow::Result<v1::Document> {
    let obj = value
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("document must be a JSON object"))?;

    let fields = obj
        .iter()
        .map(|(k, v)| (k.clone(), json_to_proto_value(v)))
        .collect();

    Ok(v1::Document { fields })
}

fn proto_value_to_json(val: &v1::Value) -> Value {
    use v1::value::Kind;
    match &val.kind {
        None | Some(Kind::NullValue(_)) => Value::Null,
        Some(Kind::BoolValue(b)) => Value::Bool(*b),
        Some(Kind::Int64Value(i)) => json!(i),
        Some(Kind::Float64Value(f)) => json!(f),
        Some(Kind::TextValue(s)) => Value::String(s.clone()),
        Some(Kind::BytesValue(b)) => {
            Value::String(b.iter().map(|byte| format!("{byte:02x}")).collect())
        }
        Some(Kind::VectorValue(v)) => json!(v.values),
        Some(Kind::DatetimeValue(us)) => {
            // Convert Unix microseconds to ISO 8601.
            let secs = us / 1_000_000;
            let nanos = ((us % 1_000_000) * 1_000) as u32;
            if let Some(dt) = chrono::DateTime::from_timestamp(secs, nanos) {
                Value::String(dt.to_rfc3339())
            } else {
                json!(us)
            }
        }
        Some(Kind::GeoValue(g)) => json!({ "lat": g.latitude, "lon": g.longitude }),
        Some(Kind::Geo3dValue(p)) => json!({ "x": p.x, "y": p.y, "z": p.z }),
        Some(Kind::Int64ArrayValue(arr)) => json!(arr.values),
        Some(Kind::Float64ArrayValue(arr)) => json!(arr.values),
    }
}

fn json_to_proto_value(val: &Value) -> v1::Value {
    use v1::value::Kind;
    let kind = match val {
        Value::Null => Some(Kind::NullValue(true)),
        Value::Bool(b) => Some(Kind::BoolValue(*b)),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Some(Kind::Int64Value(i))
            } else if let Some(f) = n.as_f64() {
                Some(Kind::Float64Value(f))
            } else {
                Some(Kind::NullValue(true))
            }
        }
        Value::String(s) => Some(Kind::TextValue(s.clone())),
        Value::Array(arr) => {
            let floats: Option<Vec<f32>> =
                arr.iter().map(|v| v.as_f64().map(|f| f as f32)).collect();
            floats.map(|values| Kind::VectorValue(v1::VectorValue { values }))
        }
        Value::Object(obj) => {
            // 3D ECEF point: { x, y, z } — must be checked before 2D geo
            // since both are objects.
            if let (Some(x), Some(y), Some(z)) = (
                obj.get("x").and_then(|v| v.as_f64()),
                obj.get("y").and_then(|v| v.as_f64()),
                obj.get("z").and_then(|v| v.as_f64()),
            ) {
                Some(Kind::Geo3dValue(v1::Geo3dPoint { x, y, z }))
            } else if let (Some(lat), Some(lon)) = (
                // 2D geo point: { lat, lon } (also accepts the full names
                // "latitude" / "longitude" for symmetry with the gateway
                // converter that infers `DataValue::Geo` from JSON).
                obj.get("lat")
                    .or_else(|| obj.get("latitude"))
                    .and_then(|v| v.as_f64()),
                obj.get("lon")
                    .or_else(|| obj.get("longitude"))
                    .and_then(|v| v.as_f64()),
            ) {
                Some(Kind::GeoValue(v1::GeoPoint {
                    latitude: lat,
                    longitude: lon,
                }))
            } else {
                Some(Kind::NullValue(true))
            }
        }
    };
    v1::Value { kind }
}

/// Parse a JSON string into a proto [`v1::FusionAlgorithm`].
///
/// Accepts two formats:
/// - `{"rrf": {"k": 60.0}}`
/// - `{"weighted_sum": {"lexical_weight": 0.7, "vector_weight": 0.3}}`
///
/// # Arguments
///
/// * `json_str` - JSON string representing the fusion algorithm.
///
/// # Errors
///
/// Returns an error if the JSON is malformed or does not match either format.
pub fn json_to_fusion_algorithm(json_str: &str) -> anyhow::Result<v1::FusionAlgorithm> {
    let val: Value = serde_json::from_str(json_str)?;

    if let Some(rrf) = val.get("rrf") {
        let k = rrf.get("k").and_then(|v| v.as_f64()).unwrap_or(60.0);
        Ok(v1::FusionAlgorithm {
            algorithm: Some(v1::fusion_algorithm::Algorithm::Rrf(v1::Rrf { k })),
        })
    } else if let Some(ws) = val.get("weighted_sum") {
        let lexical_weight = ws
            .get("lexical_weight")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5) as f32;
        let vector_weight = ws
            .get("vector_weight")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5) as f32;
        Ok(v1::FusionAlgorithm {
            algorithm: Some(v1::fusion_algorithm::Algorithm::WeightedSum(
                v1::WeightedSum {
                    lexical_weight,
                    vector_weight,
                },
            )),
        })
    } else {
        Err(anyhow::anyhow!(
            "fusion must contain either \"rrf\" or \"weighted_sum\" key"
        ))
    }
}

/// Parse a JSON string into a field boost map for the proto
/// [`SearchRequest`](v1::SearchRequest).
///
/// Expects a JSON object mapping field names to numeric boost values.
///
/// # Arguments
///
/// * `json_str` - JSON string like `{"title": 2.0, "body": 1.0}`.
///
/// # Errors
///
/// Returns an error if the JSON is malformed or not an object.
pub fn json_to_field_boosts(
    json_str: &str,
) -> anyhow::Result<std::collections::HashMap<String, f32>> {
    let val: Value = serde_json::from_str(json_str)?;
    let obj = val
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("field_boosts must be a JSON object"))?;
    Ok(obj
        .iter()
        .filter_map(|(k, v)| v.as_f64().map(|f| (k.clone(), f as f32)))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn text_value(s: &str) -> v1::Value {
        v1::Value {
            kind: Some(v1::value::Kind::TextValue(s.to_string())),
        }
    }

    fn int_value(i: i64) -> v1::Value {
        v1::Value {
            kind: Some(v1::value::Kind::Int64Value(i)),
        }
    }

    fn float_value(f: f64) -> v1::Value {
        v1::Value {
            kind: Some(v1::value::Kind::Float64Value(f)),
        }
    }

    #[test]
    fn test_document_to_json() {
        let mut fields = HashMap::new();
        fields.insert("title".to_string(), text_value("hello"));
        fields.insert("score".to_string(), float_value(1.5));
        fields.insert("count".to_string(), int_value(42));
        let doc = v1::Document { fields };

        let json = document_to_json(&doc);
        assert_eq!(json["title"], "hello");
        assert_eq!(json["score"], 1.5);
        assert_eq!(json["count"], 42);
    }

    #[test]
    fn test_json_to_fusion_algorithm_rrf() {
        let json = r#"{"rrf": {"k": 30.0}}"#;
        let fusion = json_to_fusion_algorithm(json).unwrap();
        match fusion.algorithm {
            Some(v1::fusion_algorithm::Algorithm::Rrf(rrf)) => {
                assert!((rrf.k - 30.0).abs() < f64::EPSILON);
            }
            _ => panic!("Expected RRF"),
        }
    }

    #[test]
    fn test_json_to_fusion_algorithm_rrf_default_k() {
        let json = r#"{"rrf": {}}"#;
        let fusion = json_to_fusion_algorithm(json).unwrap();
        match fusion.algorithm {
            Some(v1::fusion_algorithm::Algorithm::Rrf(rrf)) => {
                assert!((rrf.k - 60.0).abs() < f64::EPSILON);
            }
            _ => panic!("Expected RRF with default k"),
        }
    }

    #[test]
    fn test_json_to_fusion_algorithm_weighted_sum() {
        let json = r#"{"weighted_sum": {"lexical_weight": 0.7, "vector_weight": 0.3}}"#;
        let fusion = json_to_fusion_algorithm(json).unwrap();
        match fusion.algorithm {
            Some(v1::fusion_algorithm::Algorithm::WeightedSum(ws)) => {
                assert!((ws.lexical_weight - 0.7).abs() < f32::EPSILON);
                assert!((ws.vector_weight - 0.3).abs() < f32::EPSILON);
            }
            _ => panic!("Expected WeightedSum"),
        }
    }

    #[test]
    fn test_json_to_fusion_algorithm_invalid() {
        let json = r#"{"unknown": {}}"#;
        assert!(json_to_fusion_algorithm(json).is_err());
    }

    #[test]
    fn test_json_to_field_boosts() {
        let json = r#"{"title": 2.0, "body": 1.0}"#;
        let boosts = json_to_field_boosts(json).unwrap();
        assert_eq!(boosts.len(), 2);
        assert!((boosts["title"] - 2.0).abs() < f32::EPSILON);
        assert!((boosts["body"] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_json_to_field_boosts_empty() {
        let json = r#"{}"#;
        let boosts = json_to_field_boosts(json).unwrap();
        assert!(boosts.is_empty());
    }

    #[test]
    fn test_json_to_field_boosts_invalid() {
        let json = r#"[1, 2, 3]"#;
        assert!(json_to_field_boosts(json).is_err());
    }

    #[test]
    fn test_json_to_document() {
        let json_val = json!({
            "text_field": "hello",
            "int_field": 10,
            "float_field": 2.78,
            "bool_field": true,
            "null_field": null,
            "vec_field": [0.1_f32, 0.2_f32, 0.3_f32]
        });

        let doc = json_to_document(json_val).unwrap();
        assert!(matches!(
            doc.fields["text_field"].kind,
            Some(v1::value::Kind::TextValue(_))
        ));
        assert!(matches!(
            doc.fields["int_field"].kind,
            Some(v1::value::Kind::Int64Value(10))
        ));
        assert!(matches!(
            doc.fields["bool_field"].kind,
            Some(v1::value::Kind::BoolValue(true))
        ));
        assert!(matches!(
            doc.fields["null_field"].kind,
            Some(v1::value::Kind::NullValue(_))
        ));
        assert!(matches!(
            doc.fields["vec_field"].kind,
            Some(v1::value::Kind::VectorValue(_))
        ));
    }

    #[test]
    fn json_to_proto_geo_3d_object() {
        // `{ x, y, z }` JSON input must be encoded as a Geo3dValue proto
        // kind so MCP `put_document` can pass ECEF coordinates through
        // to laurus-server.
        let json_val =
            json!({ "position": { "x": 1_000_000.0, "y": 2_000_000.0, "z": 3_000_000.0 } });
        let doc = json_to_document(json_val).unwrap();
        match &doc.fields["position"].kind {
            Some(v1::value::Kind::Geo3dValue(p)) => {
                assert_eq!(p.x, 1_000_000.0);
                assert_eq!(p.y, 2_000_000.0);
                assert_eq!(p.z, 3_000_000.0);
            }
            other => panic!("expected Geo3dValue, got {other:?}"),
        }
    }

    #[test]
    fn json_to_proto_geo_2d_object() {
        // `{ lat, lon }` (and the long-form `latitude` / `longitude`)
        // produces a 2D GeoValue. Confirms #305 wiring did not break
        // the historical 2D shape.
        let json_val = json!({
            "spot_a": { "lat": 35.6, "lon": 139.7 },
            "spot_b": { "latitude": -33.86, "longitude": 151.21 },
        });
        let doc = json_to_document(json_val).unwrap();
        for field in ["spot_a", "spot_b"] {
            assert!(
                matches!(doc.fields[field].kind, Some(v1::value::Kind::GeoValue(_))),
                "{field} must be GeoValue"
            );
        }
    }

    #[test]
    fn proto_to_json_geo_3d_round_trip() {
        // Reverse direction: a Geo3dValue from the server is surfaced
        // to MCP clients as `{ x, y, z }`.
        let mut fields = HashMap::new();
        fields.insert(
            "position".to_string(),
            v1::Value {
                kind: Some(v1::value::Kind::Geo3dValue(v1::Geo3dPoint {
                    x: 4.0,
                    y: 5.0,
                    z: 6.0,
                })),
            },
        );
        let doc = v1::Document { fields };
        let json = document_to_json(&doc);
        assert_eq!(json["position"], json!({"x": 4.0, "y": 5.0, "z": 6.0}));
    }
}
