//! Conversion between [`laurus::Document`] and the protobuf `Document` message.
//!
//! The public [`to_proto`] and [`from_proto`] functions handle the top-level
//! document, while the private helpers convert individual [`DataValue`] fields.

use std::collections::HashMap;

use laurus::{DataValue, Document};

use crate::proto::laurus::v1;

/// Convert a laurus Document into a proto Document.
///
/// The internal `_id` system field is excluded from the output since it
/// is already available as the `id` field on the enclosing message.
pub fn to_proto(doc: &Document) -> v1::Document {
    let fields: HashMap<String, v1::Value> = doc
        .fields
        .iter()
        .filter(|(k, _)| k.as_str() != "_id")
        .map(|(k, v)| (k.clone(), data_value_to_proto(v)))
        .collect();
    v1::Document { fields }
}

/// Convert a proto Document into a laurus Document.
pub fn from_proto(proto: &v1::Document) -> Document {
    let fields: HashMap<String, DataValue> = proto
        .fields
        .iter()
        .map(|(k, v)| (k.clone(), data_value_from_proto(v)))
        .collect();
    Document { fields }
}

fn data_value_to_proto(val: &DataValue) -> v1::Value {
    use v1::value::Kind;
    let kind = match val {
        DataValue::Null => Some(Kind::NullValue(true)),
        DataValue::Bool(b) => Some(Kind::BoolValue(*b)),
        DataValue::Int64(i) => Some(Kind::Int64Value(*i)),
        DataValue::Float64(f) => Some(Kind::Float64Value(*f)),
        DataValue::Text(s) => Some(Kind::TextValue(s.clone())),
        DataValue::Bytes(b, _mime) => Some(Kind::BytesValue(b.clone())),
        DataValue::Vector(v) => Some(Kind::VectorValue(v1::VectorValue { values: v.clone() })),
        DataValue::DateTime(dt) => Some(Kind::DatetimeValue(dt.timestamp_micros())),
        DataValue::Geo(lat, lon) => Some(Kind::GeoValue(v1::GeoPoint {
            latitude: *lat,
            longitude: *lon,
        })),
        DataValue::Int64Array(arr) => Some(Kind::Int64ArrayValue(v1::Int64ArrayValue {
            values: arr.clone(),
        })),
        DataValue::Float64Array(arr) => Some(Kind::Float64ArrayValue(v1::Float64ArrayValue {
            values: arr.clone(),
        })),
    };
    v1::Value { kind }
}

fn data_value_from_proto(val: &v1::Value) -> DataValue {
    use v1::value::Kind;
    match &val.kind {
        Some(Kind::NullValue(_)) => DataValue::Null,
        Some(Kind::BoolValue(b)) => DataValue::Bool(*b),
        Some(Kind::Int64Value(i)) => DataValue::Int64(*i),
        Some(Kind::Float64Value(f)) => DataValue::Float64(*f),
        Some(Kind::TextValue(s)) => DataValue::Text(s.clone()),
        Some(Kind::BytesValue(b)) => DataValue::Bytes(b.clone(), None),
        Some(Kind::VectorValue(v)) => DataValue::Vector(v.values.clone()),
        Some(Kind::DatetimeValue(us)) => {
            let secs = us / 1_000_000;
            let nanos = ((us % 1_000_000) * 1_000) as u32;
            let dt = chrono::DateTime::from_timestamp(secs, nanos).unwrap_or_default();
            DataValue::DateTime(dt)
        }
        Some(Kind::GeoValue(g)) => DataValue::Geo(g.latitude, g.longitude),
        Some(Kind::Int64ArrayValue(arr)) => DataValue::Int64Array(arr.values.clone()),
        Some(Kind::Float64ArrayValue(arr)) => DataValue::Float64Array(arr.values.clone()),
        None => DataValue::Null,
    }
}
