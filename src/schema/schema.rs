//! Schema management for document structure definition.

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::analysis::Analyzer;
use crate::error::{Result, SarissaError};
use crate::query::geo::GeoPoint;
use crate::schema::field::{FieldDefinition, FieldType};

/// A schema defines the structure of documents in an index.
///
/// Similar to Whoosh's Schema, this defines what fields are available,
/// how they are processed, and how they are stored.
#[derive(Clone)]
pub struct Schema {
    /// Map of field names to their definitions
    fields: HashMap<String, FieldDefinition>,
    /// Ordered list of field names (for consistent ordering)
    field_names: Vec<String>,
    /// Default analyzer for text fields without explicit analyzers
    default_analyzer: Arc<dyn Analyzer>,
}

// Manual Debug implementation to handle Arc<dyn Analyzer>
impl std::fmt::Debug for Schema {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Schema")
            .field("fields", &self.fields)
            .field("field_names", &self.field_names)
            .field(
                "default_analyzer",
                &format!("<{}>", self.default_analyzer.name()),
            )
            .finish()
    }
}

impl Schema {
    /// Create a new empty schema with StandardAnalyzer as default.
    pub fn new() -> Result<Self> {
        use crate::analysis::StandardAnalyzer;
        Ok(Schema {
            fields: HashMap::new(),
            field_names: Vec::new(),
            default_analyzer: Arc::new(StandardAnalyzer::new()?),
        })
    }

    /// Create a new empty schema with a custom default analyzer.
    pub fn new_with_default_analyzer(default_analyzer: Arc<dyn Analyzer>) -> Self {
        Schema {
            fields: HashMap::new(),
            field_names: Vec::new(),
            default_analyzer,
        }
    }

    /// Add a field to the schema.
    pub fn add_field<S: Into<String>>(
        &mut self,
        name: S,
        field_type: Box<dyn FieldType>,
    ) -> Result<()> {
        let name = name.into();

        // Check if field already exists
        if self.fields.contains_key(&name) {
            return Err(SarissaError::schema(format!(
                "Field '{name}' already exists"
            )));
        }

        // Validate field name
        if name.is_empty() {
            return Err(SarissaError::schema("Field name cannot be empty"));
        }

        // Add field
        let field_def = FieldDefinition::new(name.clone(), field_type);
        self.fields.insert(name.clone(), field_def);
        self.field_names.push(name);

        Ok(())
    }

    /// Get a field definition by name.
    pub fn get_field(&self, name: &str) -> Option<&FieldDefinition> {
        self.fields.get(name)
    }

    /// Check if a field exists.
    pub fn has_field(&self, name: &str) -> bool {
        self.fields.contains_key(name)
    }

    /// Remove a field from the schema.
    pub fn remove_field(&mut self, name: &str) -> Result<()> {
        if !self.fields.contains_key(name) {
            return Err(SarissaError::schema(format!(
                "Field '{name}' does not exist"
            )));
        }

        self.fields.remove(name);
        self.field_names.retain(|n| n != name);

        Ok(())
    }

    /// Get all field names in the order they were added.
    pub fn field_names(&self) -> &[String] {
        &self.field_names
    }

    /// Get all field definitions.
    pub fn fields(&self) -> &HashMap<String, FieldDefinition> {
        &self.fields
    }

    /// Get the number of fields.
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// Check if the schema is empty.
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// Get all indexed field names.
    pub fn indexed_fields(&self) -> Vec<&str> {
        self.fields
            .iter()
            .filter(|(_, field)| field.field_type().is_indexed())
            .map(|(name, _)| name.as_str())
            .collect()
    }

    /// Get all stored field names.
    pub fn stored_fields(&self) -> Vec<&str> {
        self.fields
            .iter()
            .filter(|(_, field)| field.field_type().is_stored())
            .map(|(name, _)| name.as_str())
            .collect()
    }

    /// Get all fields that support fast access.
    pub fn fast_access_fields(&self) -> Vec<&str> {
        self.fields
            .iter()
            .filter(|(_, field)| field.field_type().supports_fast_access())
            .map(|(name, _)| name.as_str())
            .collect()
    }

    /// Validate that all required fields are present and valid.
    pub fn validate(&self) -> Result<()> {
        if self.is_empty() {
            return Err(SarissaError::schema("Schema must have at least one field"));
        }

        // Check that we have at least one indexed field
        if self.indexed_fields().is_empty() {
            return Err(SarissaError::schema(
                "Schema must have at least one indexed field",
            ));
        }

        // Validate each field
        for name in self.fields.keys() {
            if name.is_empty() {
                return Err(SarissaError::schema("Field name cannot be empty"));
            }

            // Additional field-specific validation could go here
        }

        Ok(())
    }

    /// Get the default analyzer for this schema.
    pub fn default_analyzer(&self) -> &Arc<dyn Analyzer> {
        &self.default_analyzer
    }

    /// Set the default analyzer for this schema.
    pub fn set_default_analyzer(&mut self, analyzer: Arc<dyn Analyzer>) {
        self.default_analyzer = analyzer;
    }

    /// Create a builder for constructing schemas.
    pub fn builder() -> Result<SchemaBuilder> {
        SchemaBuilder::new()
    }
}

impl Default for Schema {
    fn default() -> Self {
        // Use unwrap here as StandardAnalyzer::new() should not fail in normal circumstances
        Self::new().unwrap()
    }
}

/// A builder for constructing schemas in a fluent manner.
pub struct SchemaBuilder {
    schema: Schema,
}

// Manual Debug implementation to handle Arc<dyn Analyzer>
impl std::fmt::Debug for SchemaBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SchemaBuilder")
            .field("schema", &self.schema)
            .finish()
    }
}

impl SchemaBuilder {
    /// Create a new schema builder with StandardAnalyzer as default.
    pub fn new() -> Result<Self> {
        Ok(SchemaBuilder {
            schema: Schema::new()?,
        })
    }

    /// Create a new schema builder with a custom default analyzer.
    pub fn new_with_default_analyzer(default_analyzer: Arc<dyn Analyzer>) -> Self {
        SchemaBuilder {
            schema: Schema::new_with_default_analyzer(default_analyzer),
        }
    }

    /// Set the default analyzer for the schema being built.
    pub fn with_default_analyzer(mut self, analyzer: Arc<dyn Analyzer>) -> Self {
        self.schema.set_default_analyzer(analyzer);
        self
    }

    /// Add a field to the schema being built.
    pub fn add_field<S: Into<String>>(
        mut self,
        name: S,
        field_type: Box<dyn FieldType>,
    ) -> Result<Self> {
        self.schema.add_field(name, field_type)?;
        Ok(self)
    }

    /// Build the final schema.
    pub fn build(self) -> Result<Schema> {
        self.schema.validate()?;
        Ok(self.schema)
    }
}

impl Default for SchemaBuilder {
    fn default() -> Self {
        // Use unwrap here as StandardAnalyzer::new() should not fail in normal circumstances
        Self::new().unwrap()
    }
}

/// A document represents a single item to be indexed.
///
/// Documents are collections of field values that conform to a schema.
#[derive(Clone, Serialize, Deserialize)]
pub struct Document {
    /// The field values for this document
    fields: HashMap<String, FieldValue>,
    /// Optional analyzers for specific fields (for schema-less mode)
    #[serde(skip)]
    field_analyzers: HashMap<String, Arc<dyn Analyzer>>,
}

/// Represents a value for a field in a document.
#[derive(Debug, Clone, Serialize, Deserialize)]
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

    /// Add a field value with a specific analyzer (for schema-less mode).
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

    /// Validate the document against a schema.
    pub fn validate_against_schema(&self, schema: &Schema) -> Result<()> {
        // Check that all required fields are present
        for field_name in schema.field_names() {
            let _field_def = schema.get_field(field_name).unwrap();

            // For now, we don't have required fields, but we could add this logic
            // if self.get_field(field_name).is_none() && field_def.is_required() {
            //     return Err(SarissaError::schema(format!("Required field '{}' is missing", field_name)));
            // }
        }

        // Check that all document fields exist in the schema
        for field_name in self.field_names() {
            if !schema.has_field(field_name) {
                return Err(SarissaError::schema(format!(
                    "Field '{field_name}' is not defined in schema"
                )));
            }
        }

        Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::field::{IdField, NumericField, NumericType, TextField};

    #[test]
    fn test_schema_creation() {
        let mut schema = Schema::new().unwrap();

        assert!(schema.is_empty());
        assert_eq!(schema.len(), 0);

        schema
            .add_field("title", Box::new(TextField::new().stored(true)))
            .unwrap();
        schema.add_field("id", Box::new(IdField::new())).unwrap();

        assert!(!schema.is_empty());
        assert_eq!(schema.len(), 2);
        assert!(schema.has_field("title"));
        assert!(schema.has_field("id"));
        assert!(!schema.has_field("missing"));
    }

    #[test]
    fn test_schema_field_queries() {
        let mut schema = Schema::new().unwrap();

        schema
            .add_field(
                "title",
                Box::new(TextField::new().stored(true).indexed(true)),
            )
            .unwrap();
        schema.add_field("id", Box::new(IdField::new())).unwrap();
        schema
            .add_field(
                "count",
                Box::new(NumericField::new(NumericType::I32).fast_access(true)),
            )
            .unwrap();

        let indexed = schema.indexed_fields();
        assert_eq!(indexed.len(), 3);
        assert!(indexed.contains(&"title"));
        assert!(indexed.contains(&"id"));
        assert!(indexed.contains(&"count"));

        let stored = schema.stored_fields();
        assert_eq!(stored.len(), 2);
        assert!(stored.contains(&"title"));
        assert!(stored.contains(&"id"));

        let fast_access = schema.fast_access_fields();
        assert_eq!(fast_access.len(), 2);
        assert!(fast_access.contains(&"id"));
        assert!(fast_access.contains(&"count"));
    }

    #[test]
    fn test_schema_validation() {
        let schema = Schema::new().unwrap();
        assert!(schema.validate().is_err()); // Empty schema

        let mut schema = Schema::new().unwrap();
        schema
            .add_field(
                "stored_only",
                Box::new(TextField::new().stored(true).indexed(false)),
            )
            .unwrap();
        assert!(schema.validate().is_err()); // No indexed fields

        let mut schema = Schema::new().unwrap();
        schema
            .add_field("indexed", Box::new(TextField::new().indexed(true)))
            .unwrap();
        assert!(schema.validate().is_ok()); // Has indexed field
    }

    #[test]
    fn test_schema_builder() {
        let schema = Schema::builder()
            .unwrap()
            .add_field("title", Box::new(TextField::new().stored(true)))
            .unwrap()
            .add_field("id", Box::new(IdField::new()))
            .unwrap()
            .build()
            .unwrap();

        assert_eq!(schema.len(), 2);
        assert!(schema.has_field("title"));
        assert!(schema.has_field("id"));
    }

    #[test]
    fn test_document_creation() {
        let mut doc = Document::new();

        assert!(doc.is_empty());
        assert_eq!(doc.len(), 0);

        doc.add_field("title", FieldValue::Text("Hello World".to_string()));
        doc.add_field("id", FieldValue::Integer(123));

        assert!(!doc.is_empty());
        assert_eq!(doc.len(), 2);
        assert!(doc.has_field("title"));
        assert!(doc.has_field("id"));

        if let Some(FieldValue::Text(text)) = doc.get_field("title") {
            assert_eq!(text, "Hello World");
        } else {
            panic!("Expected text field");
        }
    }

    #[test]
    fn test_document_builder() {
        let doc = Document::builder()
            .add_text("title", "Hello World")
            .add_integer("id", 123)
            .add_float("score", 95.5)
            .add_boolean("published", true)
            .add_binary("data", vec![1, 2, 3])
            .build();

        assert_eq!(doc.len(), 5);
        assert!(doc.has_field("title"));
        assert!(doc.has_field("id"));
        assert!(doc.has_field("score"));
        assert!(doc.has_field("published"));
        assert!(doc.has_field("data"));
    }

    #[test]
    fn test_document_validation() {
        let mut schema = Schema::new().unwrap();
        schema
            .add_field("title", Box::new(TextField::new()))
            .unwrap();
        schema.add_field("id", Box::new(IdField::new())).unwrap();

        let doc = Document::builder()
            .add_text("title", "Hello")
            .add_integer("id", 123)
            .build();

        assert!(doc.validate_against_schema(&schema).is_ok());

        let doc = Document::builder()
            .add_text("title", "Hello")
            .add_text("unknown", "Should fail")
            .build();

        assert!(doc.validate_against_schema(&schema).is_err());
    }

    #[test]
    fn test_field_value_types() {
        let mut doc = Document::new();

        doc.add_field("text", FieldValue::Text("hello".to_string()));
        doc.add_field("int", FieldValue::Integer(42));
        doc.add_field("float", FieldValue::Float(std::f64::consts::PI));
        doc.add_field("bool", FieldValue::Boolean(true));
        doc.add_field("binary", FieldValue::Binary(vec![1, 2, 3]));
        doc.add_field("null", FieldValue::Null);

        assert_eq!(doc.len(), 6);

        match doc.get_field("text").unwrap() {
            FieldValue::Text(s) => assert_eq!(s, "hello"),
            _ => panic!("Expected text value"),
        }

        match doc.get_field("int").unwrap() {
            FieldValue::Integer(i) => assert_eq!(*i, 42),
            _ => panic!("Expected integer value"),
        }

        match doc.get_field("null").unwrap() {
            FieldValue::Null => {}
            _ => panic!("Expected null value"),
        }
    }
}
