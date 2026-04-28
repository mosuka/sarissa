//! Conversions between serde_json values and Laurus types.

use chrono::DateTime;
use laurus::{DataValue, Document};
use serde_json::Value;
use wasm_bindgen::JsValue;

/// Convert a `serde_json::Value` (object) to a [`Document`].
///
/// The input must be a JSON object whose keys are field names.
///
/// # Arguments
///
/// * `value` - A JSON object value.
///
/// # Returns
///
/// A [`Document`] with fields populated from the JSON object.
pub fn json_to_document(value: &Value) -> Result<Document, JsValue> {
    let obj = value
        .as_object()
        .ok_or_else(|| JsValue::from_str("Document must be a JSON object"))?;

    let mut builder = Document::builder();
    for (field, val) in obj {
        let dv = json_to_data_value(val)?;
        builder = builder.add_field(field, dv);
    }
    Ok(builder.build())
}

/// Convert a `serde_json::Value` to a [`DataValue`].
///
/// Type mapping:
/// - `null`                  -> `DataValue::Null`
/// - `bool`                  -> `DataValue::Bool`
/// - `number` (integer)      -> `DataValue::Int64`
/// - `number` (float)        -> `DataValue::Float64`
/// - `string`                -> `DataValue::Text` (or `DateTime` if ISO8601)
/// - `array` of numbers      -> `DataValue::Vector`
/// - `{ "lat", "lon" }`      -> `DataValue::Geo`
/// - `{ "x", "y", "z" }`     -> `DataValue::GeoEcef` (3D ECEF Cartesian, meters)
///
/// # Arguments
///
/// * `value` - A JSON value.
///
/// # Returns
///
/// The corresponding [`DataValue`].
pub fn json_to_data_value(value: &Value) -> Result<DataValue, JsValue> {
    match value {
        Value::Null => Ok(DataValue::Null),
        Value::Bool(b) => Ok(DataValue::Bool(*b)),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(DataValue::Int64(i))
            } else if let Some(f) = n.as_f64() {
                Ok(DataValue::Float64(f))
            } else {
                Err(JsValue::from_str("Invalid number value"))
            }
        }
        Value::String(s) => {
            // Try parsing as DateTime
            if let Ok(dt) = s.parse::<DateTime<chrono::Utc>>() {
                return Ok(DataValue::DateTime(dt));
            }
            Ok(DataValue::Text(s.clone()))
        }
        Value::Array(arr) => {
            // Try as vector of numbers
            let vec: Result<Vec<f32>, _> = arr
                .iter()
                .map(|v| {
                    v.as_f64().map(|f| f as f32).ok_or_else(|| {
                        JsValue::from_str("Array elements must be numbers for vector fields")
                    })
                })
                .collect();
            Ok(DataValue::Vector(vec?))
        }
        Value::Object(obj) => {
            // Check for geo { lat, lon }
            if let (Some(lat), Some(lon)) = (
                obj.get("lat").and_then(|v| v.as_f64()),
                obj.get("lon").and_then(|v| v.as_f64()),
            ) {
                let point = laurus::lexical::GeoPoint::try_new(lat, lon)
                    .map_err(|e| JsValue::from_str(&format!("invalid geo point: {e}")))?;
                return Ok(DataValue::Geo(point));
            }
            // Check for geo3d { x, y, z } (must come after the {lat, lon}
            // check so existing 2D Geo semantics are preserved).
            if let (Some(x), Some(y), Some(z)) = (
                obj.get("x").and_then(|v| v.as_f64()),
                obj.get("y").and_then(|v| v.as_f64()),
                obj.get("z").and_then(|v| v.as_f64()),
            ) {
                return Ok(DataValue::GeoEcef(laurus::GeoEcefPoint::new(x, y, z)));
            }
            Err(JsValue::from_str(
                "Cannot convert JSON object to DataValue: expected { lat, lon } or { x, y, z }",
            ))
        }
    }
}

/// Convert a [`Document`] to a `serde_json::Value`.
///
/// # Arguments
///
/// * `doc` - The document to convert.
///
/// # Returns
///
/// A JSON object with fields from the document.
pub fn document_to_json(doc: &Document) -> Value {
    let mut map = serde_json::Map::new();
    for (field, value) in &doc.fields {
        map.insert(field.clone(), data_value_to_json(value));
    }
    Value::Object(map)
}

/// Convert a [`DataValue`] to a `serde_json::Value`.
///
/// # Arguments
///
/// * `value` - The data value to convert.
///
/// # Returns
///
/// The corresponding JSON value.
pub fn data_value_to_json(value: &DataValue) -> Value {
    match value {
        DataValue::Null => Value::Null,
        DataValue::Bool(b) => Value::Bool(*b),
        DataValue::Int64(i) => serde_json::json!(*i),
        DataValue::Float64(f) => serde_json::json!(*f),
        DataValue::Text(s) => Value::String(s.clone()),
        DataValue::Bytes(b, _) => {
            Value::Array(b.iter().map(|byte| serde_json::json!(*byte)).collect())
        }
        DataValue::Vector(v) => Value::Array(v.iter().map(|f| serde_json::json!(*f)).collect()),
        DataValue::DateTime(dt) => Value::String(dt.to_rfc3339()),
        DataValue::Geo(p) => {
            serde_json::json!({ "lat": p.lat, "lon": p.lon })
        }
        DataValue::GeoEcef(p) => {
            serde_json::json!({ "x": p.x, "y": p.y, "z": p.z })
        }
        DataValue::Int64Array(arr) => {
            Value::Array(arr.iter().map(|v| serde_json::json!(*v)).collect())
        }
        DataValue::Float64Array(arr) => {
            Value::Array(arr.iter().map(|v| serde_json::json!(*v)).collect())
        }
    }
}
