//! Type coercion for existing schema fields.
//!
//! When a document is ingested, each provided value is checked against the
//! declared [`FieldOption`] of its field. If the value does not already match
//! the field's type, this module decides whether it can be coerced (converted
//! without rejecting the document) and, if so, returns the coerced value.
//!
//! The coercion rules honour the user-facing **Dynamic Schema** semantics
//! documented in [`docs/src/concepts/schema_and_fields.md`]. In particular:
//!
//! - Integer fields **truncate** incoming float values (`3.14` → `3`). This
//!   is a **silent information loss**, preferred over rejecting the document.
//! - Numeric and boolean values are parsed from their canonical string
//!   representations when they are unambiguous (`"42"`, `"true"`).
//! - Text fields accept any scalar value by stringifying it.
//! - Geographic and (future) multi-valued numeric fields have strict typing.
//!
//! Values that cannot be coerced produce an error. The caller (see
//! [`Engine`](crate::Engine)) decides what to do with the error based on the
//! [`DynamicFieldPolicy`](super::schema::DynamicFieldPolicy):
//!
//! - `Strict` → propagates the error, aborting ingestion.
//! - `Dynamic` → propagates the error. The policy's "convert when possible"
//!   rule is already embedded in this module; truly incompatible values are
//!   reported back.
//! - `Ignore` → drops the field and continues.

use crate::data::DataValue;
use crate::error::{LaurusError, Result};

use super::schema::FieldOption;

/// Attempt to coerce `value` into the type declared by `option`.
///
/// Returns the coerced (or unchanged) [`DataValue`] on success.
///
/// # Arguments
///
/// * `field_name` - The name of the field, used for error messages.
/// * `option` - The declared field option describing the target type.
/// * `value` - The incoming value from the user-supplied document.
///
/// # Errors
///
/// Returns [`LaurusError::invalid_argument`] if the value cannot be coerced
/// to the field's declared type.
pub fn coerce_value(field_name: &str, option: &FieldOption, value: DataValue) -> Result<DataValue> {
    match option {
        FieldOption::Text(_) => coerce_to_text(field_name, value),
        FieldOption::Integer(_) => coerce_to_integer(field_name, value),
        FieldOption::Float(_) => coerce_to_float(field_name, value),
        FieldOption::Boolean(_) => coerce_to_boolean(field_name, value),
        FieldOption::DateTime(_) => coerce_to_datetime(field_name, value),
        FieldOption::Geo(_) => coerce_to_geo(field_name, value),
        FieldOption::Bytes(_) => coerce_to_bytes(field_name, value),
        FieldOption::Hnsw(_) | FieldOption::Flat(_) | FieldOption::Ivf(_) => {
            coerce_to_vector(field_name, value)
        }
    }
}

fn coerce_to_text(_field_name: &str, value: DataValue) -> Result<DataValue> {
    Ok(match value {
        DataValue::Text(s) => DataValue::Text(s),
        DataValue::Int64(i) => DataValue::Text(i.to_string()),
        DataValue::Float64(f) => DataValue::Text(f.to_string()),
        DataValue::Bool(b) => DataValue::Text(b.to_string()),
        DataValue::DateTime(dt) => DataValue::Text(dt.to_rfc3339()),
        DataValue::Null => DataValue::Text(String::new()),
        // Other variants (Bytes, Vector, Geo) don't have a meaningful string
        // representation for a text field.
        other => {
            return Err(LaurusError::invalid_argument(format!(
                "cannot coerce {} to a text value",
                describe(&other)
            )));
        }
    })
}

fn coerce_to_integer(field_name: &str, value: DataValue) -> Result<DataValue> {
    match value {
        DataValue::Int64(i) => Ok(DataValue::Int64(i)),
        // Information-losing truncation: documented as intentional.
        DataValue::Float64(f) => Ok(DataValue::Int64(f as i64)),
        DataValue::Bool(b) => Ok(DataValue::Int64(if b { 1 } else { 0 })),
        DataValue::Text(s) => s.trim().parse::<i64>().map(DataValue::Int64).map_err(|_| {
            LaurusError::invalid_argument(format!(
                "field '{field_name}': cannot parse '{s}' as an integer"
            ))
        }),
        other => Err(LaurusError::invalid_argument(format!(
            "field '{field_name}': cannot coerce {} to an integer",
            describe(&other)
        ))),
    }
}

fn coerce_to_float(field_name: &str, value: DataValue) -> Result<DataValue> {
    match value {
        DataValue::Float64(f) => Ok(DataValue::Float64(f)),
        DataValue::Int64(i) => Ok(DataValue::Float64(i as f64)),
        DataValue::Bool(b) => Ok(DataValue::Float64(if b { 1.0 } else { 0.0 })),
        DataValue::Text(s) => s
            .trim()
            .parse::<f64>()
            .map(DataValue::Float64)
            .map_err(|_| {
                LaurusError::invalid_argument(format!(
                    "field '{field_name}': cannot parse '{s}' as a float"
                ))
            }),
        other => Err(LaurusError::invalid_argument(format!(
            "field '{field_name}': cannot coerce {} to a float",
            describe(&other)
        ))),
    }
}

fn coerce_to_boolean(field_name: &str, value: DataValue) -> Result<DataValue> {
    match value {
        DataValue::Bool(b) => Ok(DataValue::Bool(b)),
        DataValue::Int64(0) => Ok(DataValue::Bool(false)),
        DataValue::Int64(1) => Ok(DataValue::Bool(true)),
        DataValue::Int64(n) => Err(LaurusError::invalid_argument(format!(
            "field '{field_name}': cannot coerce integer {n} to bool (only 0 and 1 are accepted)"
        ))),
        DataValue::Text(s) => match s.trim().to_ascii_lowercase().as_str() {
            "true" => Ok(DataValue::Bool(true)),
            "false" => Ok(DataValue::Bool(false)),
            _ => Err(LaurusError::invalid_argument(format!(
                "field '{field_name}': cannot parse '{s}' as a bool (expected 'true' or 'false')"
            ))),
        },
        other => Err(LaurusError::invalid_argument(format!(
            "field '{field_name}': cannot coerce {} to a bool",
            describe(&other)
        ))),
    }
}

fn coerce_to_datetime(field_name: &str, value: DataValue) -> Result<DataValue> {
    match value {
        DataValue::DateTime(dt) => Ok(DataValue::DateTime(dt)),
        DataValue::Text(s) => {
            let parsed = chrono::DateTime::parse_from_rfc3339(s.trim()).map_err(|e| {
                LaurusError::invalid_argument(format!(
                    "field '{field_name}': cannot parse '{s}' as an RFC 3339 datetime: {e}"
                ))
            })?;
            Ok(DataValue::DateTime(parsed.with_timezone(&chrono::Utc)))
        }
        other => Err(LaurusError::invalid_argument(format!(
            "field '{field_name}': cannot coerce {} to a datetime",
            describe(&other)
        ))),
    }
}

fn coerce_to_geo(field_name: &str, value: DataValue) -> Result<DataValue> {
    match value {
        DataValue::Geo(lat, lon) => Ok(DataValue::Geo(lat, lon)),
        other => Err(LaurusError::invalid_argument(format!(
            "field '{field_name}': cannot coerce {} to a geographic point",
            describe(&other)
        ))),
    }
}

fn coerce_to_bytes(field_name: &str, value: DataValue) -> Result<DataValue> {
    match value {
        DataValue::Bytes(data, mime) => Ok(DataValue::Bytes(data, mime)),
        other => Err(LaurusError::invalid_argument(format!(
            "field '{field_name}': cannot coerce {} to bytes",
            describe(&other)
        ))),
    }
}

fn coerce_to_vector(field_name: &str, value: DataValue) -> Result<DataValue> {
    // Vector fields accept three input shapes, resolved downstream by the
    // vector store:
    //
    // - `Vector`: a pre-computed embedding, indexed verbatim.
    // - `Text` / `Bytes`: passed through unchanged so the vector store's
    //   configured embedder can turn them into vectors. The engine itself
    //   never auto-embeds, but an explicitly-configured embedder may.
    match value {
        DataValue::Vector(v) => Ok(DataValue::Vector(v)),
        DataValue::Text(s) => Ok(DataValue::Text(s)),
        DataValue::Bytes(data, mime) => Ok(DataValue::Bytes(data, mime)),
        other => Err(LaurusError::invalid_argument(format!(
            "field '{field_name}': vector fields accept Vector, Text, or Bytes \
             values; got {}",
            describe(&other)
        ))),
    }
}

fn describe(value: &DataValue) -> &'static str {
    match value {
        DataValue::Null => "null",
        DataValue::Bool(_) => "bool",
        DataValue::Int64(_) => "integer",
        DataValue::Float64(_) => "float",
        DataValue::Text(_) => "text",
        DataValue::Bytes(_, _) => "bytes",
        DataValue::Vector(_) => "vector",
        DataValue::DateTime(_) => "datetime",
        DataValue::Geo(_, _) => "geo",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexical::core::field::{
        BooleanOption, FloatOption, GeoOption, IntegerOption, TextOption,
    };

    fn integer() -> FieldOption {
        FieldOption::Integer(IntegerOption::default())
    }

    fn float() -> FieldOption {
        FieldOption::Float(FloatOption::default())
    }

    fn boolean() -> FieldOption {
        FieldOption::Boolean(BooleanOption::default())
    }

    fn text() -> FieldOption {
        FieldOption::Text(TextOption::default())
    }

    fn geo() -> FieldOption {
        FieldOption::Geo(GeoOption::default())
    }

    #[test]
    fn integer_passthrough() {
        assert_eq!(
            coerce_value("n", &integer(), DataValue::Int64(5)).unwrap(),
            DataValue::Int64(5)
        );
    }

    #[test]
    fn integer_truncates_float() {
        assert_eq!(
            coerce_value("n", &integer(), DataValue::Float64(4.7)).unwrap(),
            DataValue::Int64(4)
        );
        assert_eq!(
            coerce_value("n", &integer(), DataValue::Float64(-3.9)).unwrap(),
            DataValue::Int64(-3)
        );
    }

    #[test]
    fn integer_parses_text() {
        assert_eq!(
            coerce_value("n", &integer(), DataValue::Text("42".into())).unwrap(),
            DataValue::Int64(42)
        );
    }

    #[test]
    fn integer_rejects_unparseable_text() {
        assert!(coerce_value("n", &integer(), DataValue::Text("abc".into())).is_err());
    }

    #[test]
    fn integer_from_bool() {
        assert_eq!(
            coerce_value("n", &integer(), DataValue::Bool(true)).unwrap(),
            DataValue::Int64(1)
        );
    }

    #[test]
    fn float_from_integer() {
        assert_eq!(
            coerce_value("x", &float(), DataValue::Int64(42)).unwrap(),
            DataValue::Float64(42.0)
        );
    }

    #[test]
    fn float_parses_text() {
        assert_eq!(
            coerce_value("x", &float(), DataValue::Text("4.5".into())).unwrap(),
            DataValue::Float64(4.5)
        );
    }

    #[test]
    fn float_rejects_unparseable_text() {
        assert!(coerce_value("x", &float(), DataValue::Text("abc".into())).is_err());
    }

    #[test]
    fn boolean_from_int_zero_one() {
        assert_eq!(
            coerce_value("b", &boolean(), DataValue::Int64(0)).unwrap(),
            DataValue::Bool(false)
        );
        assert_eq!(
            coerce_value("b", &boolean(), DataValue::Int64(1)).unwrap(),
            DataValue::Bool(true)
        );
    }

    #[test]
    fn boolean_rejects_other_ints() {
        assert!(coerce_value("b", &boolean(), DataValue::Int64(2)).is_err());
    }

    #[test]
    fn boolean_from_text() {
        assert_eq!(
            coerce_value("b", &boolean(), DataValue::Text("true".into())).unwrap(),
            DataValue::Bool(true)
        );
        assert_eq!(
            coerce_value("b", &boolean(), DataValue::Text("False".into())).unwrap(),
            DataValue::Bool(false)
        );
    }

    #[test]
    fn boolean_rejects_yes() {
        assert!(coerce_value("b", &boolean(), DataValue::Text("yes".into())).is_err());
    }

    #[test]
    fn text_stringifies_any_scalar() {
        assert_eq!(
            coerce_value("t", &text(), DataValue::Int64(42)).unwrap(),
            DataValue::Text("42".into())
        );
        assert_eq!(
            coerce_value("t", &text(), DataValue::Float64(4.5)).unwrap(),
            DataValue::Text("4.5".into())
        );
        assert_eq!(
            coerce_value("t", &text(), DataValue::Bool(true)).unwrap(),
            DataValue::Text("true".into())
        );
    }

    #[test]
    fn geo_passthrough_only() {
        assert_eq!(
            coerce_value("g", &geo(), DataValue::Geo(35.1, 139.0)).unwrap(),
            DataValue::Geo(35.1, 139.0)
        );
        assert!(coerce_value("g", &geo(), DataValue::Int64(35)).is_err());
    }
}
