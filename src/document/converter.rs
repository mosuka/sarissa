//! Document converter for creating documents from strings.
//!
//! This module provides a DocumentConverter trait and various implementations
//! that can convert different formats into Document objects.

use crate::analysis::Analyzer;
use crate::document::{Document, FieldValue};
use crate::error::{Result, SarissaError};
use std::sync::Arc;

/// A trait for converting various formats into Document objects.
///
/// This trait allows for extensible document conversion from different formats
/// like field:value, JSON, YAML, PDF, etc.
///
/// # Example
///
/// ```
/// use sarissa::document::{DocumentConverter, FieldValueDocumentConverter};
///
/// let converter = FieldValueDocumentConverter::new();
/// let doc = converter.convert("title:Rust Programming\nbody:Search engine tutorial").unwrap();
/// ```
pub trait DocumentConverter {
    /// Convert input string into a Document.
    fn convert(&self, input: &str) -> Result<Document>;
}

/// A document converter for field:value format.
///
/// Converts documents in the format:
/// ```text
/// title:Rust Programming
/// body:This is a tutorial about Rust
/// year:2024
/// ```
///
/// Supports:
/// - Text fields: `title:Rust Programming`
/// - Integer fields: `year:2024` (auto-detected)
/// - Float fields: `price:19.99` (auto-detected)
/// - Boolean fields: `active:true` or `active:false`
pub struct FieldValueDocumentConverter {
    /// Analyzer to use for text fields (optional).
    analyzer: Option<Arc<dyn Analyzer>>,
}

impl std::fmt::Debug for FieldValueDocumentConverter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FieldValueDocumentConverter")
            .field("analyzer", &self.analyzer.as_ref().map(|_| "<Analyzer>"))
            .finish()
    }
}

impl Default for FieldValueDocumentConverter {
    fn default() -> Self {
        Self::new()
    }
}

impl FieldValueDocumentConverter {
    /// Create a new field:value converter without an analyzer.
    ///
    /// Text fields will be stored as-is without tokenization.
    pub fn new() -> Self {
        FieldValueDocumentConverter { analyzer: None }
    }

    /// Create a field:value converter with a custom analyzer.
    ///
    /// The analyzer will be used to validate/process text fields.
    pub fn with_analyzer(analyzer: Arc<dyn Analyzer>) -> Self {
        FieldValueDocumentConverter {
            analyzer: Some(analyzer),
        }
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
}

impl DocumentConverter for FieldValueDocumentConverter {
    fn convert(&self, input: &str) -> Result<Document> {
        let mut doc = Document::new();

        for line in input.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue; // Skip empty lines and comments
            }

            // Parse field:value
            if let Some((field, value)) = line.split_once(':') {
                let field = field.trim();
                let value = value.trim();

                // Try to infer type
                let field_value = self.infer_field_value(value);
                doc.add_field(field, field_value);
            } else {
                return Err(SarissaError::parse(format!(
                    "Invalid line format (expected field:value): {line}"
                )));
            }
        }

        Ok(doc)
    }
}

/// A document converter for JSON format.
///
/// Converts JSON objects into Documents:
/// ```json
/// {
///   "title": "Rust Programming",
///   "body": "This is a tutorial",
///   "year": 2024
/// }
/// ```
pub struct JsonDocumentConverter {
    /// Analyzer to use for text fields (optional).
    analyzer: Option<Arc<dyn Analyzer>>,
}

impl std::fmt::Debug for JsonDocumentConverter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JsonDocumentConverter")
            .field("analyzer", &self.analyzer.as_ref().map(|_| "<Analyzer>"))
            .finish()
    }
}

impl Default for JsonDocumentConverter {
    fn default() -> Self {
        Self::new()
    }
}

impl JsonDocumentConverter {
    /// Create a new JSON converter without an analyzer.
    pub fn new() -> Self {
        JsonDocumentConverter { analyzer: None }
    }

    /// Create a JSON converter with a custom analyzer.
    pub fn with_analyzer(analyzer: Arc<dyn Analyzer>) -> Self {
        JsonDocumentConverter {
            analyzer: Some(analyzer),
        }
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
}

impl DocumentConverter for JsonDocumentConverter {
    fn convert(&self, input: &str) -> Result<Document> {
        use serde_json::Value;

        let value: Value = serde_json::from_str(input)
            .map_err(|e| SarissaError::parse(format!("Failed to parse JSON: {e}")))?;

        let mut doc = Document::new();

        if let Value::Object(map) = value {
            for (key, val) in map {
                let field_value = match val {
                    Value::String(s) => self.infer_field_value(&s),
                    Value::Number(n) => {
                        if let Some(i) = n.as_i64() {
                            FieldValue::Integer(i)
                        } else if let Some(f) = n.as_f64() {
                            FieldValue::Float(f)
                        } else {
                            FieldValue::Text(n.to_string())
                        }
                    }
                    Value::Bool(b) => FieldValue::Boolean(b),
                    _ => FieldValue::Text(val.to_string()),
                };
                doc.add_field(key, field_value);
            }
        }

        Ok(doc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_value_basic_parsing() {
        let converter = FieldValueDocumentConverter::new();
        let doc = converter
            .convert("title:Rust Programming\nbody:Search engine tutorial")
            .unwrap();

        assert_eq!(
            doc.get_field("title").unwrap().as_text().unwrap(),
            "Rust Programming"
        );
        assert_eq!(
            doc.get_field("body").unwrap().as_text().unwrap(),
            "Search engine tutorial"
        );
    }

    #[test]
    fn test_field_value_type_inference() {
        let converter = FieldValueDocumentConverter::new();
        let doc = converter
            .convert("title:Test\nyear:2024\nprice:19.99\nactive:true")
            .unwrap();

        assert!(matches!(
            doc.get_field("title").unwrap(),
            FieldValue::Text(_)
        ));
        assert!(matches!(
            doc.get_field("year").unwrap(),
            FieldValue::Integer(2024)
        ));
        assert!(matches!(
            doc.get_field("price").unwrap(),
            FieldValue::Float(_)
        ));
        assert!(matches!(
            doc.get_field("active").unwrap(),
            FieldValue::Boolean(true)
        ));
    }

    #[test]
    fn test_field_value_empty_lines_and_comments() {
        let converter = FieldValueDocumentConverter::new();
        let doc = converter
            .convert("title:Test\n\n# This is a comment\nbody:Content")
            .unwrap();

        assert_eq!(doc.len(), 2);
        assert!(doc.has_field("title"));
        assert!(doc.has_field("body"));
    }

    #[test]
    fn test_field_value_invalid_format() {
        let converter = FieldValueDocumentConverter::new();
        let result = converter.convert("invalid line without colon");
        assert!(result.is_err());
    }

    #[test]
    fn test_json_parsing() {
        let converter = JsonDocumentConverter::new();
        let json = r#"{"title": "Test", "year": 2024}"#;
        let doc = converter.convert(json).unwrap();

        assert!(doc.has_field("title"));
        assert!(doc.has_field("year"));
    }

    #[test]
    fn test_json_type_inference() {
        let converter = JsonDocumentConverter::new();
        let json = r#"{"title": "Test", "year": 2024, "price": 19.99, "active": true}"#;
        let doc = converter.convert(json).unwrap();

        assert!(matches!(
            doc.get_field("title").unwrap(),
            FieldValue::Text(_)
        ));
        assert!(matches!(
            doc.get_field("year").unwrap(),
            FieldValue::Integer(2024)
        ));
        assert!(matches!(
            doc.get_field("price").unwrap(),
            FieldValue::Float(_)
        ));
        assert!(matches!(
            doc.get_field("active").unwrap(),
            FieldValue::Boolean(true)
        ));
    }
}
