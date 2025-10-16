//! JSON format document converter.
//!
//! Converts JSON objects into Documents:
//! ```json
//! {
//!   "title": "Rust Programming",
//!   "body": "This is a tutorial",
//!   "year": 2024
//! }
//! ```

use crate::analysis::analyzer::analyzer::Analyzer;
use crate::document::converter::DocumentConverter;
use crate::document::document::Document;
use crate::document::field_value::FieldValue;
use crate::error::{Result, SageError};
use std::sync::Arc;

/// A document converter for JSON format.
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
            .map_err(|e| SageError::parse(format!("Failed to parse JSON: {e}")))?;

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
