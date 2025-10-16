//! Document structure for schema-less indexing.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::document::field_value::FieldValue;
use crate::query::geo::GeoPoint;

/// A document represents a single item to be indexed.
///
/// Documents are collections of field values in schema-less mode.
/// Fields can be added dynamically without predefined schema.
///
/// Analyzers are configured at the writer level (via `AdvancedWriterConfig`),
/// not per-document, following Lucene's design.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Document {
    /// The field values for this document
    fields: HashMap<String, FieldValue>,
}

impl Document {
    /// Create a new empty document.
    pub fn new() -> Self {
        Document {
            fields: HashMap::new(),
        }
    }

    /// Add a field value to the document.
    pub fn add_field<S: Into<String>>(&mut self, name: S, value: FieldValue) {
        self.fields.insert(name.into(), value);
    }

    /// Get a field value from the document.
    pub fn get_field(&self, name: &str) -> Option<&FieldValue> {
        self.fields.get(name)
    }

    /// Check if the document has a field.
    pub fn has_field(&self, name: &str) -> bool {
        self.fields.contains_key(name)
    }

    /// Remove a field from the document.
    pub fn remove_field(&mut self, name: &str) -> Option<FieldValue> {
        self.fields.remove(name)
    }

    /// Get all field names.
    pub fn field_names(&self) -> Vec<&str> {
        self.fields.keys().map(|s| s.as_str()).collect()
    }

    /// Get all field values.
    pub fn fields(&self) -> &HashMap<String, FieldValue> {
        &self.fields
    }

    /// Get the number of fields.
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// Check if the document is empty.
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// Create a builder for constructing documents.
    pub fn builder() -> DocumentBuilder {
        DocumentBuilder::new()
    }
}

impl Default for Document {
    fn default() -> Self {
        Self::new()
    }
}

/// A builder for constructing documents in a fluent manner.
#[derive(Debug)]
pub struct DocumentBuilder {
    document: Document,
}

impl DocumentBuilder {
    /// Create a new document builder.
    pub fn new() -> Self {
        DocumentBuilder {
            document: Document::new(),
        }
    }

    /// Add a text field to the document.
    ///
    /// When indexing, the default analyzer or field-specific analyzer configured in the writer will be used.
    pub fn add_text<S: Into<String>, T: Into<String>>(mut self, name: S, value: T) -> Self {
        self.document
            .add_field(name, FieldValue::Text(value.into()));
        self
    }

    /// Add an integer field to the document.
    pub fn add_integer<S: Into<String>>(mut self, name: S, value: i64) -> Self {
        self.document.add_field(name, FieldValue::Integer(value));
        self
    }

    /// Add a float field to the document.
    pub fn add_float<S: Into<String>>(mut self, name: S, value: f64) -> Self {
        self.document.add_field(name, FieldValue::Float(value));
        self
    }

    /// Add a boolean field to the document.
    pub fn add_boolean<S: Into<String>>(mut self, name: S, value: bool) -> Self {
        self.document.add_field(name, FieldValue::Boolean(value));
        self
    }

    /// Add a binary field to the document.
    pub fn add_binary<S: Into<String>>(mut self, name: S, value: Vec<u8>) -> Self {
        self.document.add_field(name, FieldValue::Binary(value));
        self
    }

    /// Add a numeric field to the document (convenience method for float).
    pub fn add_numeric<S: Into<String>>(mut self, name: S, value: f64) -> Self {
        self.document.add_field(name, FieldValue::Float(value));
        self
    }

    /// Add a datetime field to the document.
    pub fn add_datetime<S: Into<String>>(
        mut self,
        name: S,
        value: chrono::DateTime<chrono::Utc>,
    ) -> Self {
        self.document.add_field(name, FieldValue::DateTime(value));
        self
    }

    /// Add a geo field to the document.
    pub fn add_geo<S: Into<String>>(mut self, name: S, lat: f64, lon: f64) -> Self {
        if let Ok(point) = GeoPoint::new(lat, lon) {
            self.document.add_field(name, FieldValue::Geo(point));
        }
        self
    }

    /// Add a field with a generic value.
    ///
    /// This is a low-level method that accepts any `FieldValue` directly.
    /// For most cases, prefer using type-safe methods like `add_text`, `add_integer`, `add_float`, etc.
    ///
    /// Use this method when:
    /// - You already have a `FieldValue` instance
    /// - You need to dynamically determine the field type at runtime
    /// - You require low-level API flexibility
    pub fn add_field<S: Into<String>>(mut self, name: S, value: FieldValue) -> Self {
        self.document.add_field(name, value);
        self
    }

    /// Build the final document.
    pub fn build(self) -> Document {
        self.document
    }
}

impl Default for DocumentBuilder {
    fn default() -> Self {
        Self::new()
    }
}
