//! Document structure for schema-less indexing.

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::analysis::Analyzer;
use crate::document::FieldValue;
use crate::query::geo::GeoPoint;

/// A document represents a single item to be indexed.
///
/// Documents are collections of field values in schema-less mode.
/// Fields can be added dynamically without predefined schema.
#[derive(Clone, Serialize, Deserialize)]
pub struct Document {
    /// The field values for this document
    fields: HashMap<String, FieldValue>,
    /// Optional analyzers for specific fields
    #[serde(skip)]
    field_analyzers: HashMap<String, Arc<dyn Analyzer>>,
}

impl std::fmt::Debug for Document {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Document")
            .field("fields", &self.fields)
            .field(
                "field_analyzers",
                &format!("<{} analyzers>", self.field_analyzers.len()),
            )
            .finish()
    }
}

impl Document {
    /// Create a new empty document.
    pub fn new() -> Self {
        Document {
            fields: HashMap::new(),
            field_analyzers: HashMap::new(),
        }
    }

    /// Add a field value to the document.
    pub fn add_field<S: Into<String>>(&mut self, name: S, value: FieldValue) {
        self.fields.insert(name.into(), value);
    }

    /// Add a field value with a specific analyzer.
    pub fn add_field_with_analyzer<S: Into<String>>(
        &mut self,
        name: S,
        value: FieldValue,
        analyzer: Arc<dyn Analyzer>,
    ) {
        let field_name = name.into();
        self.fields.insert(field_name.clone(), value);
        self.field_analyzers.insert(field_name, analyzer);
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

    /// Get the analyzer for a specific field (if set).
    pub fn get_field_analyzer(&self, name: &str) -> Option<&Arc<dyn Analyzer>> {
        self.field_analyzers.get(name)
    }

    /// Get all field analyzers.
    pub fn field_analyzers(&self) -> &HashMap<String, Arc<dyn Analyzer>> {
        &self.field_analyzers
    }

    /// Set an analyzer for a specific field (internal use).
    pub(crate) fn set_field_analyzer(&mut self, field_name: String, analyzer: Arc<dyn Analyzer>) {
        self.field_analyzers.insert(field_name, analyzer);
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
    pub fn add_text<S: Into<String>, T: Into<String>>(mut self, name: S, value: T) -> Self {
        self.document
            .add_field(name, FieldValue::Text(value.into()));
        self
    }

    /// Add a text field with a specific analyzer to the document.
    pub fn add_text_with_analyzer<S: Into<String>, T: Into<String>>(
        mut self,
        name: S,
        value: T,
        analyzer: Arc<dyn Analyzer>,
    ) -> Self {
        self.document
            .add_field_with_analyzer(name, FieldValue::Text(value.into()), analyzer);
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
