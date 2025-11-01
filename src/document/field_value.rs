//! Field value types for documents.
//!
//! This module defines the [`FieldValue`] enum which represents all possible
//! types of values that can be stored in document fields. It supports a variety
//! of data types for flexible schema-less indexing.
//!
//! # Supported Types
//!
//! - **Text** - String data for full-text search
//! - **Integer** - 64-bit signed integers
//! - **Float** - 64-bit floating-point numbers
//! - **Boolean** - true/false values
//! - **Binary** - Raw byte data
//! - **DateTime** - UTC timestamps with timezone
//! - **Geo** - Geographic coordinates (latitude/longitude)
//! - **Null** - Explicit null values
//!
//! # Type Conversion
//!
//! The `FieldValue` enum provides conversion methods for extracting typed values:
//!
//! ```
//! use yatagarasu::document::field_value::FieldValue;
//!
//! let text_value = FieldValue::Text("hello".to_string());
//! assert_eq!(text_value.as_text(), Some("hello"));
//!
//! let int_value = FieldValue::Integer(42);
//! assert_eq!(int_value.as_numeric(), Some("42".to_string()));
//!
//! let bool_value = FieldValue::Boolean(true);
//! assert_eq!(bool_value.as_boolean(), Some(true));
//! ```
//!
//! # Type Inference
//!
//! String values can be interpreted as different types:
//!
//! ```
//! use yatagarasu::document::field_value::FieldValue;
//!
//! // Boolean inference from text
//! let text = FieldValue::Text("true".to_string());
//! assert_eq!(text.as_boolean(), Some(true));
//!
//! let text2 = FieldValue::Text("yes".to_string());
//! assert_eq!(text2.as_boolean(), Some(true));
//! ```

use serde::{Deserialize, Serialize};

use crate::lexical::index::inverted::query::geo::GeoPoint;

/// Numeric type classification for numeric range queries.
///
/// This enum is used internally to distinguish between integer and
/// floating-point numeric types when performing range queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NumericType {
    /// Integer type (i64).
    Integer,
    /// Float type (f64).
    Float,
}

/// Represents a value for a field in a document.
///
/// This enum provides a flexible type system for document fields, supporting
/// various data types commonly used in search and indexing applications.
///
/// # Serialization
///
/// DateTime values are serialized using their UTC timestamp representation
/// for compatibility with bincode and other binary formats.
///
/// # Examples
///
/// Creating field values:
///
/// ```
/// use yatagarasu::document::field_value::FieldValue;
///
/// let text = FieldValue::Text("Rust Programming".to_string());
/// let number = FieldValue::Integer(2024);
/// let price = FieldValue::Float(39.99);
/// let active = FieldValue::Boolean(true);
/// let data = FieldValue::Binary(vec![0x00, 0x01, 0x02]);
/// ```
///
/// Extracting typed values:
///
/// ```
/// use yatagarasu::document::field_value::FieldValue;
///
/// let value = FieldValue::Integer(100);
/// assert_eq!(value.as_numeric(), Some("100".to_string()));
///
/// let text = FieldValue::Text("42".to_string());
/// assert_eq!(text.as_text(), Some("42"));
/// ```
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
