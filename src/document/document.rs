//! Document structure for schema-less indexing.
//!
//! This module provides the [`Document`] structure which represents a single
//! indexable item with dynamically-typed fields. Documents follow a schema-less
//! design inspired by Apache Lucene, allowing fields to be added without
//! predefined schemas.
//!
//! # Design Philosophy
//!
//! - **Schema-less**: Fields can be added dynamically
//! - **Flexible types**: Multiple field value types (text, numbers, dates, geo, etc.)
//! - **Builder pattern**: Fluent API for document construction
//! - **Analyzer configuration**: Analyzers are configured at the writer level,
//!   not per-document (following Lucene's design)
//!
//! # Examples
//!
//! Basic document creation:
//!
//! ```
//! use yatagarasu::document::document::Document;
//! use yatagarasu::document::field_value::FieldValue;
//!
//! let mut doc = Document::new();
//! doc.add_field("title", FieldValue::Text("Rust Book".to_string()));
//! doc.add_field("year", FieldValue::Integer(2024));
//!
//! assert_eq!(doc.len(), 2);
//! assert!(doc.has_field("title"));
//! ```
//!
//! Using the builder pattern:
//!
//! ```
//! use yatagarasu::document::document::Document;
//!
//! let doc = Document::builder()
//!     .add_text("title", "Rust Programming")
//!     .add_text("author", "Jane Doe")
//!     .add_integer("year", 2024)
//!     .add_float("rating", 4.5)
//!     .add_boolean("available", true)
//!     .build();
//!
//! assert_eq!(doc.field_names().len(), 5);
//! ```
//!
//! With geographic data:
//!
//! ```
//! use yatagarasu::document::document::Document;
//!
//! let doc = Document::builder()
//!     .add_text("name", "Tokyo Tower")
//!     .add_geo("location", 35.6762, 139.6503)
//!     .build();
//!
//! assert!(doc.has_field("location"));
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::document::field_value::FieldValue;
use crate::lexical::index::inverted::query::geo::GeoPoint;

/// A document represents a single item to be indexed.
///
/// Documents are collections of field-value pairs in schema-less mode.
/// Fields can be added dynamically without a predefined schema, providing
/// flexibility similar to NoSQL document stores.
///
/// # Field Management
///
/// - Fields are stored in a `HashMap<String, FieldValue>`
/// - Field names are case-sensitive
/// - Duplicate field names overwrite previous values
/// - Fields can be added, removed, and queried at runtime
///
/// # Analysis Configuration
///
/// Analyzers are configured at the writer level (via `InvertedIndexWriterConfig`),
/// not per-document, following Lucene's design philosophy. This allows for
/// consistent analysis across all documents and per-field analyzer configuration.
///
/// # Examples
///
/// ```
/// use yatagarasu::document::document::Document;
/// use yatagarasu::document::field_value::FieldValue;
///
/// let mut doc = Document::new();
/// doc.add_field("title", FieldValue::Text("Getting Started with Rust".to_string()));
/// doc.add_field("page_count", FieldValue::Integer(250));
///
/// assert_eq!(doc.len(), 2);
/// assert_eq!(
///     doc.get_field("title").unwrap().as_text().unwrap(),
///     "Getting Started with Rust"
/// );
///
/// // Remove a field
/// doc.remove_field("page_count");
/// assert_eq!(doc.len(), 1);
/// ```
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Document {
    /// The field values for this document
    fields: HashMap<String, FieldValue>,
}

impl Document {
    /// Create a new empty document.
    ///
    /// # Examples
    ///
    /// ```
    /// use yatagarasu::document::document::Document;
    ///
    /// let doc = Document::new();
    /// assert_eq!(doc.len(), 0);
    /// assert!(doc.is_empty());
    /// ```
    pub fn new() -> Self {
        Document {
            fields: HashMap::new(),
        }
    }

    /// Add a field value to the document.
    ///
    /// If a field with the same name already exists, it will be overwritten.
    ///
    /// # Arguments
    ///
    /// * `name` - The field name (case-sensitive)
    /// * `value` - The field value
    ///
    /// # Examples
    ///
    /// ```
    /// use yatagarasu::document::document::Document;
    /// use yatagarasu::document::field_value::FieldValue;
    ///
    /// let mut doc = Document::new();
    /// doc.add_field("title", FieldValue::Text("Rust".to_string()));
    /// doc.add_field("year", FieldValue::Integer(2024));
    ///
    /// assert_eq!(doc.len(), 2);
    /// ```
    pub fn add_field<S: Into<String>>(&mut self, name: S, value: FieldValue) {
        self.fields.insert(name.into(), value);
    }

    /// Get a field value from the document.
    ///
    /// Returns `None` if the field doesn't exist.
    ///
    /// # Arguments
    ///
    /// * `name` - The field name to retrieve
    ///
    /// # Examples
    ///
    /// ```
    /// use yatagarasu::document::document::Document;
    /// use yatagarasu::document::field_value::FieldValue;
    ///
    /// let mut doc = Document::new();
    /// doc.add_field("title", FieldValue::Text("Rust".to_string()));
    ///
    /// assert_eq!(doc.get_field("title").unwrap().as_text(), Some("Rust"));
    /// assert!(doc.get_field("missing").is_none());
    /// ```
    pub fn get_field(&self, name: &str) -> Option<&FieldValue> {
        self.fields.get(name)
    }

    /// Check if the document has a field.
    ///
    /// # Arguments
    ///
    /// * `name` - The field name to check
    ///
    /// # Examples
    ///
    /// ```
    /// use yatagarasu::document::document::Document;
    /// use yatagarasu::document::field_value::FieldValue;
    ///
    /// let mut doc = Document::new();
    /// doc.add_field("title", FieldValue::Text("Rust".to_string()));
    ///
    /// assert!(doc.has_field("title"));
    /// assert!(!doc.has_field("author"));
    /// ```
    pub fn has_field(&self, name: &str) -> bool {
        self.fields.contains_key(name)
    }

    /// Remove a field from the document.
    ///
    /// Returns the removed field value, or `None` if the field didn't exist.
    ///
    /// # Arguments
    ///
    /// * `name` - The field name to remove
    ///
    /// # Examples
    ///
    /// ```
    /// use yatagarasu::document::document::Document;
    /// use yatagarasu::document::field_value::FieldValue;
    ///
    /// let mut doc = Document::new();
    /// doc.add_field("title", FieldValue::Text("Rust".to_string()));
    /// doc.add_field("year", FieldValue::Integer(2024));
    ///
    /// let removed = doc.remove_field("year");
    /// assert!(removed.is_some());
    /// assert_eq!(doc.len(), 1);
    /// ```
    pub fn remove_field(&mut self, name: &str) -> Option<FieldValue> {
        self.fields.remove(name)
    }

    /// Get all field names in the document.
    ///
    /// The order of field names is not guaranteed.
    ///
    /// # Examples
    ///
    /// ```
    /// use yatagarasu::document::document::Document;
    /// use yatagarasu::document::field_value::FieldValue;
    ///
    /// let mut doc = Document::new();
    /// doc.add_field("title", FieldValue::Text("Rust".to_string()));
    /// doc.add_field("year", FieldValue::Integer(2024));
    ///
    /// let names = doc.field_names();
    /// assert_eq!(names.len(), 2);
    /// assert!(names.contains(&"title"));
    /// assert!(names.contains(&"year"));
    /// ```
    pub fn field_names(&self) -> Vec<&str> {
        self.fields.keys().map(|s| s.as_str()).collect()
    }

    /// Get a reference to all field values.
    ///
    /// Returns the underlying HashMap containing all fields.
    ///
    /// # Examples
    ///
    /// ```
    /// use yatagarasu::document::document::Document;
    /// use yatagarasu::document::field_value::FieldValue;
    ///
    /// let mut doc = Document::new();
    /// doc.add_field("title", FieldValue::Text("Rust".to_string()));
    ///
    /// let fields = doc.fields();
    /// assert_eq!(fields.len(), 1);
    /// assert!(fields.contains_key("title"));
    /// ```
    pub fn fields(&self) -> &HashMap<String, FieldValue> {
        &self.fields
    }

    /// Get the number of fields in the document.
    ///
    /// # Examples
    ///
    /// ```
    /// use yatagarasu::document::document::Document;
    ///
    /// let doc = Document::builder()
    ///     .add_text("title", "Rust")
    ///     .add_integer("year", 2024)
    ///     .build();
    ///
    /// assert_eq!(doc.len(), 2);
    /// ```
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// Check if the document has no fields.
    ///
    /// # Examples
    ///
    /// ```
    /// use yatagarasu::document::document::Document;
    ///
    /// let doc = Document::new();
    /// assert!(doc.is_empty());
    ///
    /// let doc = Document::builder()
    ///     .add_text("title", "Rust")
    ///     .build();
    /// assert!(!doc.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// Create a builder for constructing documents.
    ///
    /// The builder provides a fluent API for adding fields to documents.
    ///
    /// # Examples
    ///
    /// ```
    /// use yatagarasu::document::document::Document;
    ///
    /// let doc = Document::builder()
    ///     .add_text("title", "Rust Programming")
    ///     .add_integer("year", 2024)
    ///     .build();
    ///
    /// assert_eq!(doc.len(), 2);
    /// ```
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
///
/// The `DocumentBuilder` provides a convenient fluent API for creating documents
/// with multiple fields. It follows the builder pattern, allowing method chaining
/// for readable document construction.
///
/// # Examples
///
/// ```
/// use yatagarasu::document::document::Document;
///
/// let doc = Document::builder()
///     .add_text("title", "Rust Programming")
///     .add_text("author", "John Doe")
///     .add_integer("year", 2024)
///     .add_float("price", 49.99)
///     .add_boolean("available", true)
///     .build();
///
/// assert_eq!(doc.len(), 5);
/// ```
///
/// With geographic coordinates:
///
/// ```
/// use yatagarasu::document::document::Document;
///
/// let doc = Document::builder()
///     .add_text("name", "Tokyo")
///     .add_geo("location", 35.6762, 139.6503)
///     .build();
///
/// assert!(doc.has_field("location"));
/// ```
#[derive(Debug)]
pub struct DocumentBuilder {
    document: Document,
}

impl DocumentBuilder {
    /// Create a new document builder.
    ///
    /// # Examples
    ///
    /// ```
    /// use yatagarasu::document::document::DocumentBuilder;
    ///
    /// let builder = DocumentBuilder::new();
    /// let doc = builder.build();
    /// assert!(doc.is_empty());
    /// ```
    pub fn new() -> Self {
        DocumentBuilder {
            document: Document::new(),
        }
    }

    /// Add a text field to the document.
    ///
    /// Text fields are analyzed during indexing using the configured analyzer.
    /// When indexing, the default analyzer or field-specific analyzer configured
    /// in the writer will be used.
    ///
    /// # Arguments
    ///
    /// * `name` - The field name
    /// * `value` - The text content
    ///
    /// # Examples
    ///
    /// ```
    /// use yatagarasu::document::document::Document;
    ///
    /// let doc = Document::builder()
    ///     .add_text("title", "Rust Programming")
    ///     .add_text("body", "Learn Rust programming language")
    ///     .build();
    ///
    /// assert!(doc.has_field("title"));
    /// assert!(doc.has_field("body"));
    /// ```
    pub fn add_text<S: Into<String>, T: Into<String>>(mut self, name: S, value: T) -> Self {
        self.document
            .add_field(name, FieldValue::Text(value.into()));
        self
    }

    /// Add an integer field to the document.
    ///
    /// Integer fields are stored as i64 values.
    ///
    /// # Arguments
    ///
    /// * `name` - The field name
    /// * `value` - The integer value
    ///
    /// # Examples
    ///
    /// ```
    /// use yatagarasu::document::document::Document;
    ///
    /// let doc = Document::builder()
    ///     .add_text("title", "Book")
    ///     .add_integer("year", 2024)
    ///     .add_integer("pages", 300)
    ///     .build();
    ///
    /// assert_eq!(doc.len(), 3);
    /// ```
    pub fn add_integer<S: Into<String>>(mut self, name: S, value: i64) -> Self {
        self.document.add_field(name, FieldValue::Integer(value));
        self
    }

    /// Add a float field to the document.
    ///
    /// Float fields are stored as f64 values.
    ///
    /// # Arguments
    ///
    /// * `name` - The field name
    /// * `value` - The floating-point value
    ///
    /// # Examples
    ///
    /// ```
    /// use yatagarasu::document::document::Document;
    ///
    /// let doc = Document::builder()
    ///     .add_text("product", "Book")
    ///     .add_float("price", 39.99)
    ///     .add_float("rating", 4.5)
    ///     .build();
    ///
    /// assert_eq!(doc.len(), 3);
    /// ```
    pub fn add_float<S: Into<String>>(mut self, name: S, value: f64) -> Self {
        self.document.add_field(name, FieldValue::Float(value));
        self
    }

    /// Add a boolean field to the document.
    ///
    /// # Arguments
    ///
    /// * `name` - The field name
    /// * `value` - The boolean value
    ///
    /// # Examples
    ///
    /// ```
    /// use yatagarasu::document::document::Document;
    ///
    /// let doc = Document::builder()
    ///     .add_text("product", "Book")
    ///     .add_boolean("in_stock", true)
    ///     .add_boolean("featured", false)
    ///     .build();
    ///
    /// assert_eq!(doc.len(), 3);
    /// ```
    pub fn add_boolean<S: Into<String>>(mut self, name: S, value: bool) -> Self {
        self.document.add_field(name, FieldValue::Boolean(value));
        self
    }

    /// Add a binary field to the document.
    ///
    /// Binary fields store raw byte data. They are stored but not indexed for search.
    ///
    /// # Arguments
    ///
    /// * `name` - The field name
    /// * `value` - The binary data as a Vec<u8>
    ///
    /// # Examples
    ///
    /// ```
    /// use yatagarasu::document::document::Document;
    ///
    /// let doc = Document::builder()
    ///     .add_text("filename", "image.png")
    ///     .add_binary("data", vec![0x89, 0x50, 0x4E, 0x47])
    ///     .build();
    ///
    /// assert_eq!(doc.len(), 2);
    /// ```
    pub fn add_binary<S: Into<String>>(mut self, name: S, value: Vec<u8>) -> Self {
        self.document.add_field(name, FieldValue::Binary(value));
        self
    }

    /// Add a numeric field to the document (convenience method for float).
    ///
    /// This is an alias for `add_float()` provided for convenience.
    ///
    /// # Arguments
    ///
    /// * `name` - The field name
    /// * `value` - The numeric value
    ///
    /// # Examples
    ///
    /// ```
    /// use yatagarasu::document::document::Document;
    ///
    /// let doc = Document::builder()
    ///     .add_numeric("temperature", 23.5)
    ///     .build();
    ///
    /// assert_eq!(doc.len(), 1);
    /// ```
    pub fn add_numeric<S: Into<String>>(mut self, name: S, value: f64) -> Self {
        self.document.add_field(name, FieldValue::Float(value));
        self
    }

    /// Add a datetime field to the document.
    ///
    /// Datetime fields store UTC timestamps.
    ///
    /// # Arguments
    ///
    /// * `name` - The field name
    /// * `value` - The datetime value (UTC)
    ///
    /// # Examples
    ///
    /// ```
    /// use yatagarasu::document::document::Document;
    /// use chrono::Utc;
    ///
    /// let doc = Document::builder()
    ///     .add_text("event", "Conference")
    ///     .add_datetime("date", Utc::now())
    ///     .build();
    ///
    /// assert_eq!(doc.len(), 2);
    /// ```
    pub fn add_datetime<S: Into<String>>(
        mut self,
        name: S,
        value: chrono::DateTime<chrono::Utc>,
    ) -> Self {
        self.document.add_field(name, FieldValue::DateTime(value));
        self
    }

    /// Add a geographic coordinate field to the document.
    ///
    /// Geographic fields store latitude/longitude coordinates as GeoPoint.
    /// If the coordinates are invalid (lat not in [-90, 90] or lon not in [-180, 180]),
    /// the field is silently skipped.
    ///
    /// # Arguments
    ///
    /// * `name` - The field name
    /// * `lat` - Latitude (-90 to 90)
    /// * `lon` - Longitude (-180 to 180)
    ///
    /// # Examples
    ///
    /// ```
    /// use yatagarasu::document::document::Document;
    ///
    /// let doc = Document::builder()
    ///     .add_text("name", "Tokyo Tower")
    ///     .add_geo("location", 35.6762, 139.6503)
    ///     .build();
    ///
    /// assert!(doc.has_field("location"));
    /// ```
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
    /// # Use this method when:
    ///
    /// - You already have a `FieldValue` instance
    /// - You need to dynamically determine the field type at runtime
    /// - You require low-level API flexibility
    ///
    /// # Arguments
    ///
    /// * `name` - The field name
    /// * `value` - The field value
    ///
    /// # Examples
    ///
    /// ```
    /// use yatagarasu::document::document::Document;
    /// use yatagarasu::document::field_value::FieldValue;
    ///
    /// let value = FieldValue::Text("Dynamic value".to_string());
    /// let doc = Document::builder()
    ///     .add_field("dynamic", value)
    ///     .build();
    ///
    /// assert_eq!(doc.len(), 1);
    /// ```
    pub fn add_field<S: Into<String>>(mut self, name: S, value: FieldValue) -> Self {
        self.document.add_field(name, value);
        self
    }

    /// Build the final document.
    ///
    /// Consumes the builder and returns the constructed document.
    ///
    /// # Examples
    ///
    /// ```
    /// use yatagarasu::document::document::Document;
    ///
    /// let doc = Document::builder()
    ///     .add_text("title", "Rust")
    ///     .add_integer("year", 2024)
    ///     .build();
    ///
    /// assert_eq!(doc.len(), 2);
    /// ```
    pub fn build(self) -> Document {
        self.document
    }
}

impl Default for DocumentBuilder {
    fn default() -> Self {
        Self::new()
    }
}
