//! JSONL format document converter.
//!
//! This module provides [`JsonlDocumentConverter`] for converting JSONL (JSON Lines)
//! files into [`Document`] objects. JSONL is a line-delimited JSON format where
//! each line contains a complete, self-contained JSON object.
//!
//! # Format Requirements
//!
//! - One JSON object per line
//! - Empty lines are skipped
//! - Each object becomes a separate document
//! - Automatic type mapping from JSON to field values
//!
//! # JSONL Format Example
//!
//! ```jsonl
//! {"title": "Rust Programming", "body": "This is a tutorial", "year": 2024}
//! {"title": "Python Basics", "body": "Learn Python", "year": 2023}
//! ```
//!
//! # Type Mapping
//!
//! JSON types are automatically mapped to field values:
//! - **JSON String** → `FieldValue::Text` (with type inference for numbers/booleans)
//! - **JSON Number (integer)** → `FieldValue::Integer`
//! - **JSON Number (float)** → `FieldValue::Float`
//! - **JSON Boolean** → `FieldValue::Boolean`
//! - **JSON Object with lat/lon** → `FieldValue::Geo`
//! - **Other JSON types** → `FieldValue::Text` (stringified)
//!
//! # Geographic Coordinates
//!
//! Geographic coordinates can be specified as nested objects:
//!
//! ```jsonl
//! {"name": "Tokyo", "location": {"lat": 35.6762, "lon": 139.6503}}
//! {"name": "Paris", "location": {"lat": 48.8584, "lon": 2.2945}}
//! ```
//!
//! The converter automatically detects `{lat, lon}` objects and converts them
//! to `GeoPoint` fields.
//!
//! # Examples
//!
//! Basic JSONL conversion:
//!
//! ```no_run
//! use platypus::document::converter::DocumentConverter;
//! use platypus::document::converter::jsonl::JsonlDocumentConverter;
//!
//! let converter = JsonlDocumentConverter::new();
//!
//! for doc in converter.convert("documents.jsonl").unwrap() {
//!     let doc = doc.unwrap();
//!     println!("Document: {:?}", doc);
//! }
//! ```
//!
//! Collecting all documents:
//!
//! ```no_run
//! use platypus::document::converter::DocumentConverter;
//! use platypus::document::converter::jsonl::JsonlDocumentConverter;
//!
//! let converter = JsonlDocumentConverter::new();
//! let documents: Vec<_> = converter
//!     .convert("products.jsonl")
//!     .unwrap()
//!     .filter_map(|r| r.ok())
//!     .collect();
//!
//! println!("Loaded {} products", documents.len());
//! ```

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::Arc;

use crate::document::converter::DocumentConverter;
use crate::document::document::Document;
use crate::document::field::FieldValue;
use crate::error::{PlatypusError, Result};
use crate::lexical::index::inverted::query::geo::GeoPoint;

/// A document converter for JSONL format.
#[derive(Clone, Debug, Default)]
pub struct JsonlDocumentConverter;

impl JsonlDocumentConverter {
    /// Create a new JSONL converter.
    pub fn new() -> Self {
        JsonlDocumentConverter
    }

    /// Infer the field value type from a string.
    fn infer_field_value(&self, value: &str) -> FieldValue {
        // Try boolean
        if value.eq_ignore_ascii_case("true") {
            return FieldValue::Boolean(true);
        }
        if value.eq_ignore_ascii_case("false") {
            return FieldValue::Boolean(false);
        }

        // Try integer
        if let Ok(int_val) = value.parse::<i64>() {
            return FieldValue::Integer(int_val);
        }

        // Try float
        if let Ok(float_val) = value.parse::<f64>() {
            return FieldValue::Float(float_val);
        }

        // Default to text
        FieldValue::Text(value.to_string())
    }

    /// Parse a single JSON line into a Document.
    fn parse_json_line(&self, line: &str) -> Result<Document> {
        use serde_json::Value;

        let value: Value = serde_json::from_str(line)
            .map_err(|e| PlatypusError::parse(format!("Failed to parse JSON: {e}")))?;

        let mut doc = Document::new();

        if let Value::Object(map) = value {
            for (key, val) in map {
                let field_value = match &val {
                    Value::String(s) => self.infer_field_value(s),
                    Value::Number(n) => {
                        if let Some(i) = n.as_i64() {
                            FieldValue::Integer(i)
                        } else if let Some(f) = n.as_f64() {
                            FieldValue::Float(f)
                        } else {
                            FieldValue::Text(n.to_string())
                        }
                    }
                    Value::Bool(b) => FieldValue::Boolean(*b),
                    Value::Object(obj) => {
                        // Handle nested objects specially for geo coordinates
                        if obj.contains_key("lat") && obj.contains_key("lon") {
                            if let (Some(Value::Number(lat)), Some(Value::Number(lon))) =
                                (obj.get("lat"), obj.get("lon"))
                            {
                                if let (Some(lat_f), Some(lon_f)) = (lat.as_f64(), lon.as_f64()) {
                                    FieldValue::Geo(GeoPoint {
                                        lat: lat_f,
                                        lon: lon_f,
                                    })
                                } else {
                                    FieldValue::Text(val.to_string())
                                }
                            } else {
                                FieldValue::Text(val.to_string())
                            }
                        } else {
                            FieldValue::Text(val.to_string())
                        }
                    }
                    _ => FieldValue::Text(val.to_string()),
                };
                doc.add_field_value(key, field_value);
            }
        }

        Ok(doc)
    }
}

/// Iterator over JSONL documents.
pub struct JsonlDocumentIterator {
    reader: BufReader<File>,
    converter: Arc<JsonlDocumentConverter>,
}

impl Iterator for JsonlDocumentIterator {
    type Item = Result<Document>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut line = String::new();
        loop {
            line.clear();
            match self.reader.read_line(&mut line) {
                Ok(0) => return None, // EOF
                Ok(_) => {
                    let line = line.trim();
                    if line.is_empty() {
                        continue; // Skip empty lines
                    }
                    return Some(self.converter.parse_json_line(line));
                }
                Err(e) => {
                    return Some(Err(PlatypusError::parse(format!(
                        "Failed to read line: {}",
                        e
                    ))));
                }
            }
        }
    }
}

impl DocumentConverter for JsonlDocumentConverter {
    type Iter = JsonlDocumentIterator;

    fn convert<P: AsRef<Path>>(&self, path: P) -> Result<Self::Iter> {
        let file = File::open(path.as_ref())
            .map_err(|e| PlatypusError::parse(format!("Failed to open JSONL file: {}", e)))?;

        Ok(JsonlDocumentIterator {
            reader: BufReader::new(file),
            converter: Arc::new(self.clone()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_jsonl_parsing() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, r#"{{"title": "Test", "year": 2024}}"#).unwrap();
        file.flush().unwrap();

        let converter = JsonlDocumentConverter::new();
        let mut iter = converter.convert(file.path()).unwrap();
        let doc = iter.next().unwrap().unwrap();

        assert!(doc.has_field("title"));
        assert!(doc.has_field("year"));
    }

    #[test]
    fn test_jsonl_multiple_lines() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, r#"{{"title": "Test1", "year": 2024}}"#).unwrap();
        writeln!(file, r#"{{"title": "Test2", "year": 2023}}"#).unwrap();
        file.flush().unwrap();

        let converter = JsonlDocumentConverter::new();
        let docs: Vec<_> = converter.convert(file.path()).unwrap().collect();

        assert_eq!(docs.len(), 2);
        let doc1 = docs[0].as_ref().unwrap();
        assert_eq!(
            doc1.get_field("title").unwrap().value.as_text().unwrap(),
            "Test1"
        );

        let doc2 = docs[1].as_ref().unwrap();
        assert_eq!(
            doc2.get_field("title").unwrap().value.as_text().unwrap(),
            "Test2"
        );
    }

    #[test]
    fn test_jsonl_type_inference() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            r#"{{"title": "Test", "year": 2024, "price": 19.99, "active": true}}"#
        )
        .unwrap();
        file.flush().unwrap();

        let converter = JsonlDocumentConverter::new();
        let mut iter = converter.convert(file.path()).unwrap();
        let doc = iter.next().unwrap().unwrap();

        assert!(matches!(
            &doc.get_field("title").unwrap().value,
            FieldValue::Text(_)
        ));
        assert!(matches!(
            &doc.get_field("year").unwrap().value,
            FieldValue::Integer(2024)
        ));
        assert!(matches!(
            &doc.get_field("price").unwrap().value,
            FieldValue::Float(_)
        ));
        assert!(matches!(
            &doc.get_field("active").unwrap().value,
            FieldValue::Boolean(true)
        ));
    }

    #[test]
    fn test_jsonl_geo_point() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            r#"{{"title": "Tokyo", "location": {{"lat": 35.6762, "lon": 139.6503}}}}"#
        )
        .unwrap();
        file.flush().unwrap();

        let converter = JsonlDocumentConverter::new();
        let mut iter = converter.convert(file.path()).unwrap();
        let doc = iter.next().unwrap().unwrap();

        assert!(matches!(
            &doc.get_field("location").unwrap().value,
            FieldValue::Geo(_)
        ));
    }

    #[test]
    fn test_jsonl_empty_lines() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, r#"{{"title": "Test1"}}"#).unwrap();
        writeln!(file, "").unwrap();
        writeln!(file, r#"{{"title": "Test2"}}"#).unwrap();
        file.flush().unwrap();

        let converter = JsonlDocumentConverter::new();
        let docs: Vec<_> = converter.convert(file.path()).unwrap().collect();

        assert_eq!(docs.len(), 2);
    }
}
