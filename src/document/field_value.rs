//! Field value types for documents.

use serde::{Deserialize, Serialize};

use crate::lexical::index::inverted::query::geo::GeoPoint;

/// Numeric type for numeric range queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NumericType {
    /// Integer type (i64).
    Integer,
    /// Float type (f64).
    Float,
}

/// Represents a value for a field in a document.
///
/// Note: For bincode serialization, we wrap DateTime as timestamp internally.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FieldValue {
    /// Text value
    Text(String),
    /// Integer value
    Integer(i64),
    /// Floating point value
    Float(f64),
    /// Boolean value
    Boolean(bool),
    /// Binary data
    Binary(Vec<u8>),
    /// DateTime value
    DateTime(chrono::DateTime<chrono::Utc>),
    /// Geographic point value
    Geo(GeoPoint),
    /// Null value
    Null,
}

impl FieldValue {
    /// Convert to text if this is a text value.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            FieldValue::Text(s) => Some(s),
            _ => None,
        }
    }

    /// Convert to numeric string representation.
    pub fn as_numeric(&self) -> Option<String> {
        match self {
            FieldValue::Integer(i) => Some(i.to_string()),
            FieldValue::Float(f) => Some(f.to_string()),
            _ => None,
        }
    }

    /// Convert to datetime string representation (RFC3339).
    pub fn as_datetime(&self) -> Option<String> {
        match self {
            FieldValue::Text(s) => {
                // Try to parse as datetime and return as string if valid
                if s.parse::<chrono::DateTime<chrono::Utc>>().is_ok() {
                    Some(s.clone())
                } else {
                    None
                }
            }
            FieldValue::Integer(timestamp) => {
                // Treat as Unix timestamp
                chrono::DateTime::from_timestamp(*timestamp, 0).map(|dt| dt.to_rfc3339())
            }
            _ => None,
        }
    }

    /// Convert to boolean.
    pub fn as_boolean(&self) -> Option<bool> {
        match self {
            FieldValue::Boolean(b) => Some(*b),
            FieldValue::Text(s) => match s.to_lowercase().as_str() {
                "true" | "t" | "yes" | "y" | "1" | "on" => Some(true),
                "false" | "f" | "no" | "n" | "0" | "off" => Some(false),
                _ => None,
            },
            FieldValue::Integer(i) => Some(*i != 0),
            _ => None,
        }
    }

    /// Get the value as binary data, if possible.
    pub fn as_binary(&self) -> Option<&[u8]> {
        match self {
            FieldValue::Binary(data) => Some(data),
            _ => None,
        }
    }

    /// Convert to GeoPoint if this is a geo value.
    pub fn as_geo(&self) -> Option<&GeoPoint> {
        match self {
            FieldValue::Geo(point) => Some(point),
            _ => None,
        }
    }
}
