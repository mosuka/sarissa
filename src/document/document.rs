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
//! use platypus::document::document::Document;
//! use platypus::document::field::FieldValue;
//!
//! let mut doc = Document::new();
//! doc.add_field_value("title", FieldValue::Text("Rust Book".to_string()));
//! doc.add_field_value("year", FieldValue::Integer(2024));
//!
//! assert_eq!(doc.len(), 2);
//! assert!(doc.has_field("title"));
//! ```
//!
//! Using the builder pattern:
//!
//! ```
//! use platypus::document::document::Document;
//! use platypus::document::field::{TextOption, IntegerOption, FloatOption, BooleanOption};
//!
//! let doc = Document::builder()
//!     .add_text("title", "Rust Programming", TextOption::default())
//!     .add_text("author", "Jane Doe", TextOption::default())
//!     .add_integer("year", 2024, IntegerOption::default())
//!     .add_float("rating", 4.5, FloatOption::default())
//!     .add_boolean("available", true, BooleanOption::default())
//!     .build();
//!
//! assert_eq!(doc.field_names().len(), 5);
//! ```
//!
//! With geographic data:
//!
//! ```
//! use platypus::document::document::Document;
//! use platypus::document::field::{TextOption, GeoOption};
//!
//! let doc = Document::builder()
//!     .add_text("name", "Tokyo Tower", TextOption::default())
//!     .add_geo("location", 35.6762, 139.6503, GeoOption::default())
//!     .build();
//!
//! assert!(doc.has_field("location"));
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::document::field::{
    BinaryOption, BooleanOption, DateTimeOption, Field, FieldOption, FieldValue, FloatOption,
    GeoOption, IntegerOption, TextOption, VectorOption,
};
use crate::lexical::index::inverted::query::geo::GeoPoint;

/// A document represents a single item to be indexed.
///
/// Documents are collections of fields with values and indexing options.
/// Fields can be added dynamically without a predefined schema, providing
/// flexibility similar to NoSQL document stores.
///
/// # Field Management
///
/// - Fields are stored in a `HashMap<String, Field>`
/// - Each field contains both a value (FieldValue) and indexing options (FieldOption)
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
/// ```no_run
/// use platypus::document::document::Document;
/// use platypus::document::field::{Field, FieldValue, FieldOption, TextOption};
///
/// let mut doc = Document::new();
/// doc.add_field("title", Field::new(
///     FieldValue::Text("Getting Started with Rust".to_string()),
///     FieldOption::Text(TextOption::default())
/// ));
/// doc.add_field("page_count", Field::with_default_option(
///     FieldValue::Integer(250)
/// ));
///
/// assert_eq!(doc.len(), 2);
/// assert_eq!(
///     doc.get_field("title").unwrap().value.as_text().unwrap(),
///     "Getting Started with Rust"
/// );
///
/// // Remove a field
/// doc.remove_field("page_count");
/// assert_eq!(doc.len(), 1);
/// ```
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Document {
    /// The fields in this document (each field has a value and indexing options)
    fields: HashMap<String, Field>,
}

impl Document {
    /// Create a new empty document.
    ///
    /// # Examples
    ///
    /// ```
    /// use platypus::document::document::Document;
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

    /// Add a field to the document.
    ///
    /// If a field with the same name already exists, it will be overwritten.
    ///
    /// # Arguments
    ///
    /// * `name` - The field name (case-sensitive)
    /// * `field` - The field containing value and indexing options
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use platypus::document::document::Document;
    /// use platypus::document::field::{Field, FieldValue};
    ///
    /// let mut doc = Document::new();
    /// doc.add_field("title", Field::with_default_option(FieldValue::Text("Rust".to_string())));
    /// doc.add_field("year", Field::with_default_option(FieldValue::Integer(2024)));
    ///
    /// assert_eq!(doc.len(), 2);
    /// ```
    pub fn add_field<S: Into<String>>(&mut self, name: S, field: Field) {
        self.fields.insert(name.into(), field);
    }

    /// Add a field value to the document with default indexing options.
    ///
    /// This is a convenience method that automatically infers the FieldOption
    /// from the FieldValue type using default settings.
    ///
    /// # Arguments
    ///
    /// * `name` - The field name (case-sensitive)
    /// * `value` - The field value
    ///
    /// # Examples
    ///
    /// ```
    /// use platypus::document::document::Document;
    /// use platypus::document::field::FieldValue;
    ///
    /// let mut doc = Document::new();
    /// doc.add_field_value("title", FieldValue::Text("Rust".to_string()));
    /// doc.add_field_value("year", FieldValue::Integer(2024));
    ///
    /// assert_eq!(doc.len(), 2);
    /// ```
    pub fn add_field_value<S: Into<String>>(&mut self, name: S, value: FieldValue) {
        let option = FieldOption::from_field_value(&value);
        self.fields.insert(name.into(), Field::new(value, option));
    }

    /// Get a field from the document.
    ///
    /// Returns `None` if the field doesn't exist.
    ///
    /// # Arguments
    ///
    /// * `name` - The field name to retrieve
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use platypus::document::document::Document;
    /// use platypus::document::field::{Field, FieldValue};
    ///
    /// let mut doc = Document::new();
    /// doc.add_field_value("title", FieldValue::Text("Rust".to_string()));
    ///
    /// assert_eq!(doc.get_field("title").unwrap().value.as_text(), Some("Rust"));
    /// assert!(doc.get_field("missing").is_none());
    /// ```
    pub fn get_field(&self, name: &str) -> Option<&Field> {
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
    /// use platypus::document::document::Document;
    /// use platypus::document::field::FieldValue;
    ///
    /// let mut doc = Document::new();
    /// doc.add_field_value("title", FieldValue::Text("Rust".to_string()));
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
    /// use platypus::document::document::Document;
    /// use platypus::document::field::FieldValue;
    ///
    /// let mut doc = Document::new();
    /// doc.add_field_value("title", FieldValue::Text("Rust".to_string()));
    /// doc.add_field_value("year", FieldValue::Integer(2024));
    ///
    /// let removed = doc.remove_field("year");
    /// assert!(removed.is_some());
    /// assert_eq!(doc.len(), 1);
    /// ```
    pub fn remove_field(&mut self, name: &str) -> Option<FieldValue> {
        self.fields.remove(name).map(|field| field.value)
    }

    /// Get all field names in the document.
    ///
    /// The order of field names is not guaranteed.
    ///
    /// # Examples
    ///
    /// ```
    /// use platypus::document::document::Document;
    /// use platypus::document::field::FieldValue;
    ///
    /// let mut doc = Document::new();
    /// doc.add_field_value("title", FieldValue::Text("Rust".to_string()));
    /// doc.add_field_value("year", FieldValue::Integer(2024));
    ///
    /// let names = doc.field_names();
    /// assert_eq!(names.len(), 2);
    /// assert!(names.contains(&"title"));
    /// assert!(names.contains(&"year"));
    /// ```
    pub fn field_names(&self) -> Vec<&str> {
        self.fields.keys().map(|s| s.as_str()).collect()
    }

    /// Get a reference to all fields.
    ///
    /// Returns the underlying HashMap containing all fields.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use platypus::document::document::Document;
    /// use platypus::document::field::FieldValue;
    ///
    /// let mut doc = Document::new();
    /// doc.add_field_value("title", FieldValue::Text("Rust".to_string()));
    ///
    /// let fields = doc.fields();
    /// assert_eq!(fields.len(), 1);
    /// assert!(fields.contains_key("title"));
    /// ```
    pub fn fields(&self) -> &HashMap<String, Field> {
        &self.fields
    }

    /// Get the number of fields in the document.
    ///
    /// # Examples
    ///
    /// ```
    /// use platypus::document::document::Document;
    /// use platypus::document::field::{TextOption, IntegerOption};
    ///
    /// let doc = Document::builder()
    ///     .add_text("title", "Rust", TextOption::default())
    ///     .add_integer("year", 2024, IntegerOption::default())
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
    /// use platypus::document::document::Document;
    /// use platypus::document::field::TextOption;
    ///
    /// let doc = Document::new();
    /// assert!(doc.is_empty());
    ///
    /// let doc = Document::builder()
    ///     .add_text("title", "Rust", TextOption::default())
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
    /// use platypus::document::document::Document;
    /// use platypus::document::field::{TextOption, IntegerOption};
    ///
    /// let doc = Document::builder()
    ///     .add_text("title", "Rust Programming", TextOption::default())
    ///     .add_integer("year", 2024, IntegerOption::default())
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
/// use platypus::document::document::Document;
/// use platypus::document::field::{TextOption, IntegerOption, FloatOption, BooleanOption};
///
/// let doc = Document::builder()
///     .add_text("title", "Rust Programming", TextOption::default())
///     .add_text("author", "John Doe", TextOption::default())
///     .add_integer("year", 2024, IntegerOption::default())
///     .add_float("price", 49.99, FloatOption::default())
///     .add_boolean("available", true, BooleanOption::default())
///     .build();
///
/// assert_eq!(doc.len(), 5);
/// ```
///
/// With geographic coordinates:
///
/// ```
/// use platypus::document::document::Document;
/// use platypus::document::field::{TextOption, GeoOption};
///
/// let doc = Document::builder()
///     .add_text("name", "Tokyo", TextOption::default())
///     .add_geo("location", 35.6762, 139.6503, GeoOption::default())
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
    /// use platypus::document::document::DocumentBuilder;
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

    /// Add a text field to the document with indexing options.
    ///
    /// Text fields are analyzed during indexing using the configured analyzer.
    /// The TextOption parameter controls how this field is indexed (indexed, stored, term_vectors).
    ///
    /// # Arguments
    ///
    /// * `name` - The field name
    /// * `value` - The text content
    /// * `option` - Indexing options (use `TextOption::default()` for default settings)
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use platypus::document::document::Document;
    /// use platypus::document::field::TextOption;
    ///
    /// let doc = Document::builder()
    ///     .add_text("title", "Rust Programming", TextOption::default())
    ///     .add_text("body", "Learn Rust programming language", TextOption {
    ///         indexed: true,
    ///         stored: true,
    ///         term_vectors: true,
    ///     })
    ///     .build();
    ///
    /// assert!(doc.has_field("title"));
    /// assert!(doc.has_field("body"));
    /// ```
    pub fn add_text<S: Into<String>, T: Into<String>>(
        mut self,
        name: S,
        value: T,
        option: TextOption,
    ) -> Self {
        self.document.add_field(
            name,
            Field::new(FieldValue::Text(value.into()), FieldOption::Text(option)),
        );
        self
    }

    /// Add an integer field to the document with indexing options.
    ///
    /// Integer fields are stored as i64 values.
    /// The IntegerOption parameter controls how this field is indexed (indexed, stored).
    ///
    /// # Arguments
    ///
    /// * `name` - The field name
    /// * `value` - The integer value
    /// * `option` - Indexing options (use `IntegerOption::default()` for default settings)
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use platypus::document::document::Document;
    /// use platypus::document::field::{TextOption, IntegerOption};
    ///
    /// let doc = Document::builder()
    ///     .add_text("title", "Book", TextOption::default())
    ///     .add_integer("year", 2024, IntegerOption::default())
    ///     .add_integer("pages", 300, IntegerOption::default())
    ///     .build();
    ///
    /// assert_eq!(doc.len(), 3);
    /// ```
    pub fn add_integer<S: Into<String>>(
        mut self,
        name: S,
        value: i64,
        option: IntegerOption,
    ) -> Self {
        self.document.add_field(
            name,
            Field::new(FieldValue::Integer(value), FieldOption::Integer(option)),
        );
        self
    }

    /// Add a float field to the document with indexing options.
    ///
    /// Float fields are stored as f64 values.
    /// The FloatOption parameter controls how this field is indexed (indexed, stored).
    ///
    /// # Arguments
    ///
    /// * `name` - The field name
    /// * `value` - The floating-point value
    /// * `option` - Indexing options (use `FloatOption::default()` for default settings)
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use platypus::document::document::Document;
    /// use platypus::document::field::{TextOption, FloatOption};
    ///
    /// let doc = Document::builder()
    ///     .add_text("product", "Book", TextOption::default())
    ///     .add_float("price", 39.99, FloatOption::default())
    ///     .add_float("rating", 4.5, FloatOption::default())
    ///     .build();
    ///
    /// assert_eq!(doc.len(), 3);
    /// ```
    pub fn add_float<S: Into<String>>(mut self, name: S, value: f64, option: FloatOption) -> Self {
        self.document.add_field(
            name,
            Field::new(FieldValue::Float(value), FieldOption::Float(option)),
        );
        self
    }

    /// Add a boolean field to the document with indexing options.
    ///
    /// The BooleanOption parameter controls how this field is indexed (indexed, stored).
    ///
    /// # Arguments
    ///
    /// * `name` - The field name
    /// * `value` - The boolean value
    /// * `option` - Indexing options (use `BooleanOption::default()` for default settings)
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use platypus::document::document::Document;
    /// use platypus::document::field::{TextOption, BooleanOption};
    ///
    /// let doc = Document::builder()
    ///     .add_text("product", "Book", TextOption::default())
    ///     .add_boolean("in_stock", true, BooleanOption::default())
    ///     .add_boolean("featured", false, BooleanOption::default())
    ///     .build();
    ///
    /// assert_eq!(doc.len(), 3);
    /// ```
    pub fn add_boolean<S: Into<String>>(
        mut self,
        name: S,
        value: bool,
        option: BooleanOption,
    ) -> Self {
        self.document.add_field(
            name,
            Field::new(FieldValue::Boolean(value), FieldOption::Boolean(option)),
        );
        self
    }

    /// Add a binary field to the document with indexing options.
    ///
    /// Binary fields store raw byte data. They are stored but not indexed for search.
    /// The BinaryOption parameter controls how this field is stored.
    ///
    /// # Arguments
    ///
    /// * `name` - The field name
    /// * `value` - The binary data as a Vec<u8>
    /// * `option` - Indexing options (use `BinaryOption::default()` for default settings)
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use platypus::document::document::Document;
    /// use platypus::document::field::{TextOption, BinaryOption};
    ///
    /// let doc = Document::builder()
    ///     .add_text("filename", "image.png", TextOption::default())
    ///     .add_binary("data", vec![0x89, 0x50, 0x4E, 0x47], BinaryOption::default())
    ///     .build();
    ///
    /// assert_eq!(doc.len(), 2);
    /// ```
    pub fn add_binary<S: Into<String>>(
        mut self,
        name: S,
        value: Vec<u8>,
        option: BinaryOption,
    ) -> Self {
        self.document.add_field(
            name,
            Field::new(FieldValue::Binary(value), FieldOption::Binary(option)),
        );
        self
    }

    /// Add a numeric field to the document with indexing options (convenience method for float).
    ///
    /// This is an alias for `add_float()` provided for convenience.
    /// The FloatOption parameter controls how this field is indexed (indexed, stored).
    ///
    /// # Arguments
    ///
    /// * `name` - The field name
    /// * `value` - The numeric value
    /// * `option` - Indexing options (use `FloatOption::default()` for default settings)
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use platypus::document::document::Document;
    /// use platypus::document::field::FloatOption;
    ///
    /// let doc = Document::builder()
    ///     .add_numeric("temperature", 23.5, FloatOption::default())
    ///     .build();
    ///
    /// assert_eq!(doc.len(), 1);
    /// ```
    pub fn add_numeric<S: Into<String>>(
        mut self,
        name: S,
        value: f64,
        option: FloatOption,
    ) -> Self {
        self.document.add_field(
            name,
            Field::new(FieldValue::Float(value), FieldOption::Float(option)),
        );
        self
    }

    /// Add a datetime field to the document with indexing options.
    ///
    /// Datetime fields store UTC timestamps.
    /// The DateTimeOption parameter controls how this field is indexed (indexed, stored).
    ///
    /// # Arguments
    ///
    /// * `name` - The field name
    /// * `value` - The datetime value (UTC)
    /// * `option` - Indexing options (use `DateTimeOption::default()` for default settings)
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use platypus::document::document::Document;
    /// use platypus::document::field::{TextOption, DateTimeOption};
    /// use chrono::Utc;
    ///
    /// let doc = Document::builder()
    ///     .add_text("event", "Conference", TextOption::default())
    ///     .add_datetime("date", Utc::now(), DateTimeOption::default())
    ///     .build();
    ///
    /// assert_eq!(doc.len(), 2);
    /// ```
    pub fn add_datetime<S: Into<String>>(
        mut self,
        name: S,
        value: chrono::DateTime<chrono::Utc>,
        option: DateTimeOption,
    ) -> Self {
        self.document.add_field(
            name,
            Field::new(FieldValue::DateTime(value), FieldOption::DateTime(option)),
        );
        self
    }

    /// Add a geographic coordinate field to the document with indexing options.
    ///
    /// Geographic fields store latitude/longitude coordinates as GeoPoint.
    /// If the coordinates are invalid (lat not in [-90, 90] or lon not in [-180, 180]),
    /// the field is silently skipped.
    /// The GeoOption parameter controls how this field is indexed (indexed, stored).
    ///
    /// # Arguments
    ///
    /// * `name` - The field name
    /// * `lat` - Latitude (-90 to 90)
    /// * `lon` - Longitude (-180 to 180)
    /// * `option` - Indexing options (use `GeoOption::default()` for default settings)
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use platypus::document::document::Document;
    /// use platypus::document::field::{TextOption, GeoOption};
    ///
    /// let doc = Document::builder()
    ///     .add_text("name", "Tokyo Tower", TextOption::default())
    ///     .add_geo("location", 35.6762, 139.6503, GeoOption::default())
    ///     .build();
    ///
    /// assert!(doc.has_field("location"));
    /// ```
    pub fn add_geo<S: Into<String>>(
        mut self,
        name: S,
        lat: f64,
        lon: f64,
        option: GeoOption,
    ) -> Self {
        if let Ok(point) = GeoPoint::new(lat, lon) {
            self.document.add_field(
                name,
                Field::new(FieldValue::Geo(point), FieldOption::Geo(option)),
            );
        }
        self
    }

    /// Add a vector field to the document with indexing options.
    ///
    /// Vector fields store text that will be converted to embeddings when indexed
    /// by a VectorEngine. The actual embedding conversion happens during indexing,
    /// using the embedder configured for that field.
    /// The VectorOption parameter controls how this field is indexed (indexed, stored).
    ///
    /// # Arguments
    ///
    /// * `name` - The field name
    /// * `text` - The text to be embedded
    /// * `option` - Indexing options (use `VectorOption::default()` for default settings)
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use platypus::document::document::Document;
    /// use platypus::document::field::{TextOption, VectorOption};
    ///
    /// let doc = Document::builder()
    ///     .add_text("title", "Machine Learning Basics", TextOption::default())
    ///     .add_vector("title_embedding", "Machine Learning Basics", VectorOption::default())
    ///     .build();
    ///
    /// assert!(doc.has_field("title_embedding"));
    /// ```
    pub fn add_vector<S: Into<String>, T: Into<String>>(
        mut self,
        name: S,
        text: T,
        option: VectorOption,
    ) -> Self {
        self.document.add_field(
            name,
            Field::new(FieldValue::Vector(text.into()), FieldOption::Vector(option)),
        );
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
    /// use platypus::document::document::Document;
    /// use platypus::document::field::FieldValue;
    ///
    /// let value = FieldValue::Text("Dynamic value".to_string());
    /// let doc = Document::builder()
    ///     .add_field("dynamic", value)
    ///     .build();
    ///
    /// assert_eq!(doc.len(), 1);
    /// ```
    pub fn add_field<S: Into<String>>(mut self, name: S, value: FieldValue) -> Self {
        self.document.add_field_value(name, value);
        self
    }

    /// Build the final document.
    ///
    /// Consumes the builder and returns the constructed document.
    ///
    /// # Examples
    ///
    /// ```
    /// use platypus::document::document::Document;
    /// use platypus::document::field::{TextOption, IntegerOption};
    ///
    /// let doc = Document::builder()
    ///     .add_text("title", "Rust", TextOption::default())
    ///     .add_integer("year", 2024, IntegerOption::default())
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
