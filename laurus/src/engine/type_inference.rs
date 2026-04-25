//! Type inference for the dynamic schema feature.
//!
//! When a document is ingested against a schema with
//! [`DynamicFieldPolicy::Dynamic`](super::schema::DynamicFieldPolicy::Dynamic),
//! this module infers a [`FieldOption`] for each undeclared field from the
//! value that the user provided.
//!
//! Two entry points cover the two ingestion paths:
//!
//! - [`infer_option_from_data_value`] — engine-side path; the document already
//!   carries a [`DataValue`] (gRPC `add_document`, native bindings).
//! - [`infer_from_json`] — transport-side path; used by the HTTP gateway so
//!   JSON inputs follow the same inference rules as the engine.
//!
//! Supported inferences:
//!
//! - `string` → [`FieldOption::Text`]
//! - `integer` → [`FieldOption::Integer`]
//! - `float` → [`FieldOption::Float`]
//! - `bool` → [`FieldOption::Boolean`]
//! - numeric array (all `i64`) → [`FieldOption::Integer`] with `multi_valued = true`
//! - numeric array (any non-`i64` number) → [`FieldOption::Float`] with `multi_valued = true`
//! - `object` with `lat|latitude` and `lon|lng|longitude` keys (values in
//!   range) → [`FieldOption::Geo`]
//!
//! Vector and bytes fields are never inferred; they must always be declared
//! explicitly in the schema.

use serde_json::Value as JsonValue;

use crate::data::DataValue;
use crate::error::{LaurusError, Result};
use crate::lexical::core::field::{
    BooleanOption, FloatOption, GeoOption, IntegerOption, TextOption,
};

use super::schema::FieldOption;

/// Result of attempting to infer a [`DataValue`] and [`FieldOption`] from a
/// raw JSON value.
#[derive(Debug, Clone)]
pub enum InferredValue {
    /// The value was inferred successfully.
    Inferred {
        /// The converted [`DataValue`] to store in the document.
        value: DataValue,
        /// The [`FieldOption`] to register in the schema if the field is new.
        option: FieldOption,
    },
    /// The value is known but should be silently skipped (e.g. `null`, empty
    /// array). Callers should not add the field.
    Skip,
}

/// Infer a [`FieldOption`] from an existing [`DataValue`].
///
/// Used during document ingestion when the schema's
/// [`DynamicFieldPolicy`](super::schema::DynamicFieldPolicy) is `Dynamic` and
/// the field has not been declared yet. The document already carries a
/// [`DataValue`] (the JSON → [`DataValue`] conversion happened earlier, at the
/// transport layer), so this entry point does not re-parse the value.
///
/// Supported variants:
///
/// - [`DataValue::Text`] → [`FieldOption::Text`]
/// - [`DataValue::Int64`] → [`FieldOption::Integer`]
/// - [`DataValue::Float64`] → [`FieldOption::Float`]
/// - [`DataValue::Bool`] → [`FieldOption::Boolean`]
/// - [`DataValue::DateTime`] → [`FieldOption::DateTime`]
/// - [`DataValue::Geo`] → [`FieldOption::Geo`]
/// - [`DataValue::Null`] → `Ok(None)` (caller should skip the field)
///
/// # Arguments
///
/// * `value` - The data value to inspect.
///
/// # Errors
///
/// Returns [`LaurusError::invalid_argument`] for variants that are not
/// supported by the dynamic schema:
///
/// - [`DataValue::Vector`]: vector fields must be declared explicitly.
/// - [`DataValue::Bytes`]: bytes fields must be declared explicitly.
pub fn infer_option_from_data_value(value: &DataValue) -> Result<Option<FieldOption>> {
    match value {
        DataValue::Null => Ok(None),
        DataValue::Text(_) => Ok(Some(FieldOption::Text(TextOption::default()))),
        DataValue::Int64(_) => Ok(Some(FieldOption::Integer(IntegerOption::default()))),
        DataValue::Float64(_) => Ok(Some(FieldOption::Float(FloatOption::default()))),
        DataValue::Bool(_) => Ok(Some(FieldOption::Boolean(BooleanOption::default()))),
        DataValue::DateTime(_) => Ok(Some(FieldOption::DateTime(
            crate::lexical::core::field::DateTimeOption::default(),
        ))),
        DataValue::Geo(_, _) => Ok(Some(FieldOption::Geo(GeoOption::default()))),
        DataValue::Int64Array(_) => Ok(Some(FieldOption::Integer(IntegerOption {
            multi_valued: true,
            ..Default::default()
        }))),
        DataValue::Float64Array(_) => Ok(Some(FieldOption::Float(FloatOption {
            multi_valued: true,
            ..Default::default()
        }))),
        DataValue::Vector(_) => Err(LaurusError::invalid_argument(
            "vector values require an explicit vector field declaration \
             (Hnsw, Flat, or Ivf) in the schema",
        )),
        DataValue::Bytes(_, _) => Err(LaurusError::invalid_argument(
            "bytes values require an explicit bytes field declaration in the schema",
        )),
    }
}

/// Infer a [`DataValue`] and [`FieldOption`] from a JSON value.
///
/// The mapping is:
///
/// | JSON value | DataValue | FieldOption |
/// | --- | --- | --- |
/// | `string` | [`DataValue::Text`] | [`FieldOption::Text`] |
/// | `integer` (fits in i64) | [`DataValue::Int64`] | [`FieldOption::Integer`] |
/// | `float` / large number | [`DataValue::Float64`] | [`FieldOption::Float`] |
/// | `bool` | [`DataValue::Bool`] | [`FieldOption::Boolean`] |
/// | `object` with `lat|latitude` + `lon|lng|longitude` | [`DataValue::Geo`] | [`FieldOption::Geo`] |
/// | `null` | (none) | (none) — returns [`InferredValue::Skip`] |
/// | `array` of integers | [`DataValue::Int64Array`] | [`FieldOption::Integer`] with `multi_valued = true` |
/// | `array` containing any non-i64 number | [`DataValue::Float64Array`] | [`FieldOption::Float`] with `multi_valued = true` |
/// | empty `array` | (none) | (none) — returns [`InferredValue::Skip`] |
///
/// # Arguments
///
/// * `value` - The JSON value to infer from.
///
/// # Errors
///
/// Returns [`LaurusError::invalid_argument`] if the value cannot be inferred
/// (e.g. mixed-type array, non-geo object, out-of-range geo values).
pub fn infer_from_json(value: &JsonValue) -> Result<InferredValue> {
    match value {
        JsonValue::Null => Ok(InferredValue::Skip),
        JsonValue::Bool(b) => Ok(InferredValue::Inferred {
            value: DataValue::Bool(*b),
            option: FieldOption::Boolean(BooleanOption::default()),
        }),
        JsonValue::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(InferredValue::Inferred {
                    value: DataValue::Int64(i),
                    option: FieldOption::Integer(IntegerOption::default()),
                })
            } else if let Some(f) = n.as_f64() {
                Ok(InferredValue::Inferred {
                    value: DataValue::Float64(f),
                    option: FieldOption::Float(FloatOption::default()),
                })
            } else {
                Err(LaurusError::invalid_argument(format!(
                    "number {n} cannot be represented as i64 or f64"
                )))
            }
        }
        JsonValue::String(s) => Ok(InferredValue::Inferred {
            value: DataValue::Text(s.clone()),
            option: FieldOption::Text(TextOption::default()),
        }),
        JsonValue::Array(arr) => infer_from_array(arr),
        JsonValue::Object(map) => infer_from_object(map),
    }
}

/// Infer a value from a JSON array.
///
/// Numeric-only arrays become multi-valued numeric fields. Arrays whose
/// elements all fit in `i64` map to [`DataValue::Int64Array`] backed by an
/// [`IntegerOption`] with `multi_valued = true`. Arrays containing any
/// non-`i64` number map to [`DataValue::Float64Array`] backed by a
/// [`FloatOption`] with `multi_valued = true`. Empty arrays return
/// [`InferredValue::Skip`] because their element type cannot be determined.
///
/// # Arguments
///
/// * `arr` - The JSON array to inspect.
///
/// # Errors
///
/// Returns [`LaurusError::invalid_argument`] when the array contains a
/// non-numeric or mixed-type element.
fn infer_from_array(arr: &[JsonValue]) -> Result<InferredValue> {
    if arr.is_empty() {
        return Ok(InferredValue::Skip);
    }

    let mut all_i64 = true;
    let mut all_numeric = true;
    for elem in arr {
        match elem {
            JsonValue::Number(n) => {
                if n.as_i64().is_none() {
                    all_i64 = false;
                }
            }
            _ => {
                all_numeric = false;
                break;
            }
        }
    }

    if !all_numeric {
        return Err(LaurusError::invalid_argument(
            "array fields must contain only numeric values \
             (mixed or non-numeric arrays are not supported)",
        ));
    }

    if all_i64 {
        let values: Vec<i64> = arr
            .iter()
            .map(|v| v.as_i64().expect("checked above"))
            .collect();
        Ok(InferredValue::Inferred {
            value: DataValue::Int64Array(values),
            option: FieldOption::Integer(IntegerOption {
                multi_valued: true,
                ..Default::default()
            }),
        })
    } else {
        let values: Vec<f64> = arr
            .iter()
            .map(|v| {
                v.as_f64()
                    .expect("numeric JSON values are always representable as f64")
            })
            .collect();
        Ok(InferredValue::Inferred {
            value: DataValue::Float64Array(values),
            option: FieldOption::Float(FloatOption {
                multi_valued: true,
                ..Default::default()
            }),
        })
    }
}

/// Infer a value from a JSON object.
///
/// Only objects that form a geographic point pair are accepted. A geographic
/// point must have:
///
/// - A latitude key: `lat` or `latitude`
/// - A longitude key: `lon`, `lng`, or `longitude`
/// - Both values numeric, with latitude in [-90, 90] and longitude in [-180, 180]
///
/// All other object shapes are rejected.
///
/// # Arguments
///
/// * `map` - The JSON object entries.
///
/// # Errors
///
/// Returns [`LaurusError::invalid_argument`] if the object is not a valid
/// geographic point pair.
fn infer_from_object(map: &serde_json::Map<String, JsonValue>) -> Result<InferredValue> {
    const LAT_KEYS: &[&str] = &["lat", "latitude"];
    const LON_KEYS: &[&str] = &["lon", "lng", "longitude"];

    let lat = LAT_KEYS.iter().find_map(|k| map.get(*k));
    let lon = LON_KEYS.iter().find_map(|k| map.get(*k));

    match (lat, lon) {
        (Some(lat), Some(lon)) => {
            let lat = lat
                .as_f64()
                .ok_or_else(|| LaurusError::invalid_argument("geo latitude must be a number"))?;
            let lon = lon
                .as_f64()
                .ok_or_else(|| LaurusError::invalid_argument("geo longitude must be a number"))?;
            if !(-90.0..=90.0).contains(&lat) {
                return Err(LaurusError::invalid_argument(format!(
                    "geo latitude {lat} is out of range [-90, 90]"
                )));
            }
            if !(-180.0..=180.0).contains(&lon) {
                return Err(LaurusError::invalid_argument(format!(
                    "geo longitude {lon} is out of range [-180, 180]"
                )));
            }
            Ok(InferredValue::Inferred {
                value: DataValue::Geo(lat, lon),
                option: FieldOption::Geo(GeoOption::default()),
            })
        }
        _ => Err(LaurusError::invalid_argument(
            "object values are only supported as geographic points \
             (expected keys: lat|latitude, lon|lng|longitude)",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn inferred(v: InferredValue) -> (DataValue, FieldOption) {
        match v {
            InferredValue::Inferred { value, option } => (value, option),
            InferredValue::Skip => panic!("expected Inferred, got Skip"),
        }
    }

    #[test]
    fn infer_string_to_text() {
        let (v, o) = inferred(infer_from_json(&json!("hello")).unwrap());
        assert_eq!(v, DataValue::Text("hello".into()));
        assert!(matches!(o, FieldOption::Text(_)));
    }

    #[test]
    fn infer_integer_to_integer() {
        let (v, o) = inferred(infer_from_json(&json!(42)).unwrap());
        assert_eq!(v, DataValue::Int64(42));
        assert!(matches!(o, FieldOption::Integer(_)));
    }

    #[test]
    fn infer_negative_integer() {
        let (v, o) = inferred(infer_from_json(&json!(-7)).unwrap());
        assert_eq!(v, DataValue::Int64(-7));
        assert!(matches!(o, FieldOption::Integer(_)));
    }

    #[test]
    fn infer_float_to_float() {
        let (v, o) = inferred(infer_from_json(&json!(4.5)).unwrap());
        assert_eq!(v, DataValue::Float64(4.5));
        assert!(matches!(o, FieldOption::Float(_)));
    }

    #[test]
    fn infer_bool_to_boolean() {
        let (v, o) = inferred(infer_from_json(&json!(true)).unwrap());
        assert_eq!(v, DataValue::Bool(true));
        assert!(matches!(o, FieldOption::Boolean(_)));
    }

    #[test]
    fn infer_null_skips() {
        assert!(matches!(
            infer_from_json(&JsonValue::Null).unwrap(),
            InferredValue::Skip
        ));
    }

    #[test]
    fn infer_empty_array_skips() {
        assert!(matches!(
            infer_from_json(&json!([])).unwrap(),
            InferredValue::Skip
        ));
    }

    #[test]
    fn infer_integer_array_to_int64_array() {
        let (v, o) = inferred(infer_from_json(&json!([1, 2, 3])).unwrap());
        assert_eq!(v, DataValue::Int64Array(vec![1, 2, 3]));
        match o {
            FieldOption::Integer(opt) => assert!(opt.multi_valued),
            other => panic!("expected Integer with multi_valued=true, got {other:?}"),
        }
    }

    #[test]
    fn infer_float_array_to_float64_array() {
        // Mixed integer + float (any non-i64 element flips to float array)
        let (v, o) = inferred(infer_from_json(&json!([1.0, 2.5, 3])).unwrap());
        assert_eq!(v, DataValue::Float64Array(vec![1.0, 2.5, 3.0]));
        match o {
            FieldOption::Float(opt) => assert!(opt.multi_valued),
            other => panic!("expected Float with multi_valued=true, got {other:?}"),
        }
    }

    #[test]
    fn infer_mixed_array_rejected() {
        let err = infer_from_json(&json!([1, "a"])).unwrap_err();
        assert!(err.to_string().contains("only numeric"));
    }

    #[test]
    fn infer_geo_lat_lon() {
        let (v, o) = inferred(infer_from_json(&json!({"lat": 35.1, "lon": 139.0})).unwrap());
        assert_eq!(v, DataValue::Geo(35.1, 139.0));
        assert!(matches!(o, FieldOption::Geo(_)));
    }

    #[test]
    fn infer_geo_latitude_longitude() {
        let (v, _) =
            inferred(infer_from_json(&json!({"latitude": 35.1, "longitude": 139.0})).unwrap());
        assert_eq!(v, DataValue::Geo(35.1, 139.0));
    }

    #[test]
    fn infer_geo_lng_alias() {
        let (v, _) = inferred(infer_from_json(&json!({"lat": 35.1, "lng": 139.0})).unwrap());
        assert_eq!(v, DataValue::Geo(35.1, 139.0));
    }

    #[test]
    fn infer_geo_out_of_range_lat() {
        let err = infer_from_json(&json!({"lat": 100.0, "lon": 139.0})).unwrap_err();
        assert!(err.to_string().contains("latitude"));
    }

    #[test]
    fn infer_geo_out_of_range_lon() {
        let err = infer_from_json(&json!({"lat": 35.1, "lon": 200.0})).unwrap_err();
        assert!(err.to_string().contains("longitude"));
    }

    #[test]
    fn infer_unknown_object_rejected() {
        let err = infer_from_json(&json!({"foo": 1, "bar": 2})).unwrap_err();
        assert!(err.to_string().contains("geographic"));
    }

    #[test]
    fn infer_geo_missing_lon_rejected() {
        let err = infer_from_json(&json!({"lat": 35.1})).unwrap_err();
        assert!(err.to_string().contains("geographic"));
    }
}
