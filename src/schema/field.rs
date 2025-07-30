//! Field types for schema definition.

use std::sync::Arc;

use chrono::{DateTime, NaiveDateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::analysis::analyzer::Analyzer;
use crate::query::geo::GeoPoint;

/// Trait for field types that define how fields are processed and stored.
pub trait FieldType: Send + Sync + std::fmt::Debug {
    /// Get the analyzer for this field type (if any).
    fn analyzer(&self) -> Option<&Arc<dyn Analyzer>>;

    /// Check if this field is stored (retrievable from search results).
    fn is_stored(&self) -> bool;

    /// Check if this field is indexed (searchable).
    fn is_indexed(&self) -> bool;

    /// Check if this field supports fast field access (for sorting/faceting).
    fn supports_fast_access(&self) -> bool;

    /// Get the name of this field type.
    fn type_name(&self) -> &'static str;

    /// Clone this field type.
    fn clone_box(&self) -> Box<dyn FieldType>;

    /// Check if two field types are equal.
    fn equals(&self, other: &dyn FieldType) -> bool;
}

/// A text field that supports full-text search.
#[derive(Clone)]
pub struct TextField {
    /// Whether the field value is stored
    stored: bool,
    /// Whether the field is indexed for searching
    indexed: bool,
    /// The analyzer to use for this field (we'll skip this for now due to Debug issues)
    analyzer: Option<String>, // Just store the analyzer name for now
    /// Whether phrase queries are supported
    phrase: bool,
    /// Whether to store term vectors
    term_vectors: bool,
}

impl std::fmt::Debug for TextField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TextField")
            .field("stored", &self.stored)
            .field("indexed", &self.indexed)
            .field("analyzer", &self.analyzer)
            .field("phrase", &self.phrase)
            .field("term_vectors", &self.term_vectors)
            .finish()
    }
}

impl TextField {
    /// Create a new text field with default settings.
    pub fn new() -> Self {
        TextField {
            stored: false,
            indexed: true,
            analyzer: None,
            phrase: false,
            term_vectors: false,
        }
    }

    /// Set whether this field is stored.
    pub fn stored(mut self, stored: bool) -> Self {
        self.stored = stored;
        self
    }

    /// Set whether this field is indexed.
    pub fn indexed(mut self, indexed: bool) -> Self {
        self.indexed = indexed;
        self
    }

    /// Set the analyzer for this field.
    pub fn with_analyzer<S: Into<String>>(mut self, analyzer_name: S) -> Self {
        self.analyzer = Some(analyzer_name.into());
        self
    }

    /// Set whether phrase queries are supported.
    pub fn phrase(mut self, phrase: bool) -> Self {
        self.phrase = phrase;
        self
    }

    /// Set whether to store term vectors.
    pub fn term_vectors(mut self, term_vectors: bool) -> Self {
        self.term_vectors = term_vectors;
        self
    }

    /// Check if phrase queries are supported.
    pub fn supports_phrase(&self) -> bool {
        self.phrase
    }

    /// Check if term vectors are stored.
    pub fn has_term_vectors(&self) -> bool {
        self.term_vectors
    }
}

impl Default for TextField {
    fn default() -> Self {
        Self::new()
    }
}

impl FieldType for TextField {
    fn analyzer(&self) -> Option<&Arc<dyn Analyzer>> {
        None // We'll implement this properly later
    }

    fn is_stored(&self) -> bool {
        self.stored
    }

    fn is_indexed(&self) -> bool {
        self.indexed
    }

    fn supports_fast_access(&self) -> bool {
        false // Text fields don't support fast access by default
    }

    fn type_name(&self) -> &'static str {
        "text"
    }

    fn clone_box(&self) -> Box<dyn FieldType> {
        Box::new(self.clone())
    }

    fn equals(&self, other: &dyn FieldType) -> bool {
        if other.type_name() == "text" {
            // For now, we'll just compare the type name
            true
        } else {
            false
        }
    }
}

/// An ID field for unique identifiers.
#[derive(Debug, Clone)]
pub struct IdField {
    /// Whether the field value is stored
    stored: bool,
    /// Whether the field is indexed for searching
    indexed: bool,
    /// Whether to enable unique constraint
    unique: bool,
}

impl IdField {
    /// Create a new ID field with default settings.
    pub fn new() -> Self {
        IdField {
            stored: true,
            indexed: true,
            unique: true,
        }
    }

    /// Set whether this field is stored.
    pub fn stored(mut self, stored: bool) -> Self {
        self.stored = stored;
        self
    }

    /// Set whether this field is indexed.
    pub fn indexed(mut self, indexed: bool) -> Self {
        self.indexed = indexed;
        self
    }

    /// Set whether this field has a unique constraint.
    pub fn unique(mut self, unique: bool) -> Self {
        self.unique = unique;
        self
    }

    /// Check if this field has a unique constraint.
    pub fn is_unique(&self) -> bool {
        self.unique
    }
}

impl Default for IdField {
    fn default() -> Self {
        Self::new()
    }
}

impl FieldType for IdField {
    fn analyzer(&self) -> Option<&Arc<dyn Analyzer>> {
        None // ID fields don't use analyzers
    }

    fn is_stored(&self) -> bool {
        self.stored
    }

    fn is_indexed(&self) -> bool {
        self.indexed
    }

    fn supports_fast_access(&self) -> bool {
        true // ID fields support fast access
    }

    fn type_name(&self) -> &'static str {
        "id"
    }

    fn clone_box(&self) -> Box<dyn FieldType> {
        Box::new(self.clone())
    }

    fn equals(&self, other: &dyn FieldType) -> bool {
        other.type_name() == "id"
    }
}

/// A stored field that is only stored, not indexed.
#[derive(Debug, Clone)]
pub struct StoredField {
    /// Whether compression is enabled
    compressed: bool,
}

impl StoredField {
    /// Create a new stored field with default settings.
    pub fn new() -> Self {
        StoredField { compressed: false }
    }

    /// Set whether compression is enabled.
    pub fn compressed(mut self, compressed: bool) -> Self {
        self.compressed = compressed;
        self
    }

    /// Check if compression is enabled.
    pub fn is_compressed(&self) -> bool {
        self.compressed
    }
}

impl Default for StoredField {
    fn default() -> Self {
        Self::new()
    }
}

impl FieldType for StoredField {
    fn analyzer(&self) -> Option<&Arc<dyn Analyzer>> {
        None // Stored fields don't use analyzers
    }

    fn is_stored(&self) -> bool {
        true // Always stored
    }

    fn is_indexed(&self) -> bool {
        false // Never indexed
    }

    fn supports_fast_access(&self) -> bool {
        false // Stored fields don't support fast access
    }

    fn type_name(&self) -> &'static str {
        "stored"
    }

    fn clone_box(&self) -> Box<dyn FieldType> {
        Box::new(self.clone())
    }

    fn equals(&self, other: &dyn FieldType) -> bool {
        other.type_name() == "stored"
    }
}

/// A numeric field for integers and floats.
#[derive(Debug, Clone)]
pub struct NumericField {
    /// Whether the field value is stored
    stored: bool,
    /// Whether the field is indexed for searching
    indexed: bool,
    /// Whether to enable fast field access
    fast_access: bool,
    /// The numeric type
    numeric_type: NumericType,
}

/// Supported numeric types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NumericType {
    /// 32-bit signed integer
    I32,
    /// 64-bit signed integer
    I64,
    /// 32-bit unsigned integer
    U32,
    /// 64-bit unsigned integer
    U64,
    /// 32-bit floating point
    F32,
    /// 64-bit floating point
    F64,
}

impl NumericField {
    /// Create a new numeric field with default settings.
    pub fn new(numeric_type: NumericType) -> Self {
        NumericField {
            stored: false,
            indexed: true,
            fast_access: false,
            numeric_type,
        }
    }

    /// Create a new i32 numeric field.
    pub fn i32() -> Self {
        Self::new(NumericType::I32)
    }

    /// Create a new i64 numeric field.
    pub fn i64() -> Self {
        Self::new(NumericType::I64)
    }

    /// Create a new u32 numeric field.
    pub fn u32() -> Self {
        Self::new(NumericType::U32)
    }

    /// Create a new u64 numeric field.
    pub fn u64() -> Self {
        Self::new(NumericType::U64)
    }

    /// Create a new f32 numeric field.
    pub fn f32() -> Self {
        Self::new(NumericType::F32)
    }

    /// Create a new f64 numeric field.
    pub fn f64() -> Self {
        Self::new(NumericType::F64)
    }

    /// Set whether this field is stored.
    pub fn stored(mut self, stored: bool) -> Self {
        self.stored = stored;
        self
    }

    /// Set whether this field is indexed.
    pub fn indexed(mut self, indexed: bool) -> Self {
        self.indexed = indexed;
        self
    }

    /// Set whether fast field access is enabled.
    pub fn fast_access(mut self, fast_access: bool) -> Self {
        self.fast_access = fast_access;
        self
    }

    /// Get the numeric type.
    pub fn numeric_type(&self) -> NumericType {
        self.numeric_type
    }
}

impl FieldType for NumericField {
    fn analyzer(&self) -> Option<&Arc<dyn Analyzer>> {
        None // Numeric fields don't use analyzers
    }

    fn is_stored(&self) -> bool {
        self.stored
    }

    fn is_indexed(&self) -> bool {
        self.indexed
    }

    fn supports_fast_access(&self) -> bool {
        self.fast_access
    }

    fn type_name(&self) -> &'static str {
        "numeric"
    }

    fn clone_box(&self) -> Box<dyn FieldType> {
        Box::new(self.clone())
    }

    fn equals(&self, other: &dyn FieldType) -> bool {
        other.type_name() == "numeric"
    }
}

/// A datetime field for temporal data.
#[derive(Debug, Clone)]
pub struct DateTimeField {
    /// Whether the field value is stored
    stored: bool,
    /// Whether the field is indexed for searching
    indexed: bool,
    /// Whether to enable fast field access
    fast_access: bool,
    /// Date format for parsing (default: RFC3339)
    format: Option<String>,
}

impl DateTimeField {
    /// Create a new datetime field with default settings.
    pub fn new() -> Self {
        DateTimeField {
            stored: true,
            indexed: true,
            fast_access: true,
            format: None,
        }
    }

    /// Set whether the field is stored.
    pub fn stored(mut self, stored: bool) -> Self {
        self.stored = stored;
        self
    }

    /// Set whether the field is indexed.
    pub fn indexed(mut self, indexed: bool) -> Self {
        self.indexed = indexed;
        self
    }

    /// Set whether the field supports fast access.
    pub fn fast_access(mut self, fast_access: bool) -> Self {
        self.fast_access = fast_access;
        self
    }

    /// Set the date format for parsing.
    pub fn format<S: Into<String>>(mut self, format: S) -> Self {
        self.format = Some(format.into());
        self
    }

    /// Get the date format.
    pub fn date_format(&self) -> Option<&str> {
        self.format.as_deref()
    }

    /// Parse a string into a UTC timestamp.
    pub fn parse_datetime(&self, input: &str) -> Result<DateTime<Utc>, chrono::ParseError> {
        if let Some(format) = &self.format {
            // Parse with custom format
            let naive = NaiveDateTime::parse_from_str(input, format)?;
            Ok(DateTime::from_naive_utc_and_offset(naive, Utc))
        } else {
            // Parse as RFC3339 (default)
            input.parse::<DateTime<Utc>>()
        }
    }
}

impl Default for DateTimeField {
    fn default() -> Self {
        Self::new()
    }
}

/// A keyword field for exact matching of terms (space-separated but not analyzed).
#[derive(Debug, Clone)]
pub struct KeywordField {
    /// Whether the field value is stored
    stored: bool,
    /// Whether the field is indexed for searching
    indexed: bool,
    /// Whether to enable fast field access
    fast_access: bool,
    /// Case sensitive matching
    case_sensitive: bool,
    /// Whether to store term vectors
    term_vectors: bool,
    /// Whether to tokenize on whitespace (if false, treats entire value as single term)
    tokenize: bool,
}

impl FieldType for DateTimeField {
    fn analyzer(&self) -> Option<&Arc<dyn Analyzer>> {
        None // DateTime fields don't use analyzers
    }

    fn is_stored(&self) -> bool {
        self.stored
    }

    fn is_indexed(&self) -> bool {
        self.indexed
    }

    fn supports_fast_access(&self) -> bool {
        self.fast_access
    }

    fn type_name(&self) -> &'static str {
        "datetime"
    }

    fn clone_box(&self) -> Box<dyn FieldType> {
        Box::new(self.clone())
    }

    fn equals(&self, other: &dyn FieldType) -> bool {
        other.type_name() == "datetime"
    }
}

impl KeywordField {
    /// Create a new keyword field with default settings.
    pub fn new() -> Self {
        KeywordField {
            stored: false,
            indexed: true,
            fast_access: false,
            case_sensitive: true,
            term_vectors: false,
            tokenize: true,
        }
    }

    /// Set whether the field is stored.
    pub fn stored(mut self, stored: bool) -> Self {
        self.stored = stored;
        self
    }

    /// Set whether the field is indexed.
    pub fn indexed(mut self, indexed: bool) -> Self {
        self.indexed = indexed;
        self
    }

    /// Set whether the field supports fast access.
    pub fn fast_access(mut self, fast_access: bool) -> Self {
        self.fast_access = fast_access;
        self
    }

    /// Set whether matching is case sensitive.
    pub fn case_sensitive(mut self, case_sensitive: bool) -> Self {
        self.case_sensitive = case_sensitive;
        self
    }

    /// Set whether to store term vectors.
    pub fn term_vectors(mut self, term_vectors: bool) -> Self {
        self.term_vectors = term_vectors;
        self
    }

    /// Set whether to tokenize on whitespace.
    pub fn tokenize(mut self, tokenize: bool) -> Self {
        self.tokenize = tokenize;
        self
    }

    /// Check if matching is case sensitive.
    pub fn is_case_sensitive(&self) -> bool {
        self.case_sensitive
    }

    /// Check if term vectors are stored.
    pub fn has_term_vectors(&self) -> bool {
        self.term_vectors
    }

    /// Check if tokenization is enabled.
    pub fn is_tokenized(&self) -> bool {
        self.tokenize
    }
}

impl Default for KeywordField {
    fn default() -> Self {
        Self::new()
    }
}

impl FieldType for KeywordField {
    fn analyzer(&self) -> Option<&Arc<dyn Analyzer>> {
        None // Keyword fields don't use analyzers
    }

    fn is_stored(&self) -> bool {
        self.stored
    }

    fn is_indexed(&self) -> bool {
        self.indexed
    }

    fn supports_fast_access(&self) -> bool {
        self.fast_access
    }

    fn type_name(&self) -> &'static str {
        "keyword"
    }

    fn clone_box(&self) -> Box<dyn FieldType> {
        Box::new(self.clone())
    }

    fn equals(&self, other: &dyn FieldType) -> bool {
        other.type_name() == "keyword"
    }
}

/// A boolean field for true/false values.
#[derive(Debug, Clone)]
pub struct BooleanField {
    /// Whether the field value is stored
    stored: bool,
    /// Whether the field is indexed for searching
    indexed: bool,
    /// Whether to enable fast field access
    fast_access: bool,
}

impl Default for BooleanField {
    fn default() -> Self {
        Self::new()
    }
}

impl BooleanField {
    /// Create a new boolean field with default settings.
    pub fn new() -> Self {
        BooleanField {
            stored: true,
            indexed: true,
            fast_access: true,
        }
    }

    /// Set whether the field is stored.
    pub fn stored(mut self, stored: bool) -> Self {
        self.stored = stored;
        self
    }

    /// Set whether the field is indexed.
    pub fn indexed(mut self, indexed: bool) -> Self {
        self.indexed = indexed;
        self
    }

    /// Set whether the field supports fast access.
    pub fn fast_access(mut self, fast_access: bool) -> Self {
        self.fast_access = fast_access;
        self
    }

    /// Parse a string into a boolean value.
    pub fn parse_boolean(&self, input: &str) -> Result<bool, String> {
        match input.to_lowercase().as_str() {
            "true" | "t" | "yes" | "y" | "1" | "on" => Ok(true),
            "false" | "f" | "no" | "n" | "0" | "off" => Ok(false),
            _ => Err(format!("Cannot parse '{input}' as boolean")),
        }
    }
}

impl FieldType for BooleanField {
    fn analyzer(&self) -> Option<&Arc<dyn Analyzer>> {
        None // Boolean fields don't use analyzers
    }

    fn is_stored(&self) -> bool {
        self.stored
    }

    fn is_indexed(&self) -> bool {
        self.indexed
    }

    fn supports_fast_access(&self) -> bool {
        self.fast_access
    }

    fn type_name(&self) -> &'static str {
        "boolean"
    }

    fn clone_box(&self) -> Box<dyn FieldType> {
        Box::new(self.clone())
    }

    fn equals(&self, other: &dyn FieldType) -> bool {
        other.type_name() == "boolean"
    }
}

/// A geographical field for storing location data.
#[derive(Debug, Clone)]
pub struct GeoField {
    /// Whether the field value is stored
    stored: bool,
    /// Whether the field is indexed for searching
    indexed: bool,
    /// Whether to enable fast field access
    fast_access: bool,
}

impl GeoField {
    /// Create a new geo field with default settings.
    pub fn new() -> Self {
        GeoField {
            stored: true,
            indexed: true,
            fast_access: true,
        }
    }

    /// Set whether the field is stored.
    pub fn stored(mut self, stored: bool) -> Self {
        self.stored = stored;
        self
    }

    /// Set whether the field is indexed.
    pub fn indexed(mut self, indexed: bool) -> Self {
        self.indexed = indexed;
        self
    }

    /// Set whether the field supports fast access.
    pub fn fast_access(mut self, fast_access: bool) -> Self {
        self.fast_access = fast_access;
        self
    }

    /// Parse a coordinate string into a GeoPoint.
    /// Supports formats like "lat,lon" or "lat lon"
    pub fn parse_geo_point(&self, input: &str) -> Result<GeoPoint, String> {
        let input = input.trim();

        // Try comma-separated format first
        if let Some((lat_str, lon_str)) = input.split_once(',') {
            let lat = lat_str
                .trim()
                .parse::<f64>()
                .map_err(|_| format!("Invalid latitude: '{}'", lat_str.trim()))?;
            let lon = lon_str
                .trim()
                .parse::<f64>()
                .map_err(|_| format!("Invalid longitude: '{}'", lon_str.trim()))?;

            return GeoPoint::new(lat, lon).map_err(|e| format!("Invalid coordinates: {e}"));
        }

        // Try space-separated format
        let parts: Vec<&str> = input.split_whitespace().collect();
        if parts.len() == 2 {
            let lat = parts[0]
                .parse::<f64>()
                .map_err(|_| format!("Invalid latitude: '{}'", parts[0]))?;
            let lon = parts[1]
                .parse::<f64>()
                .map_err(|_| format!("Invalid longitude: '{}'", parts[1]))?;

            return GeoPoint::new(lat, lon).map_err(|e| format!("Invalid coordinates: {e}"));
        }

        Err(format!(
            "Invalid coordinate format: '{input}'. Expected 'lat,lon' or 'lat lon'"
        ))
    }
}

impl Default for GeoField {
    fn default() -> Self {
        Self::new()
    }
}

impl FieldType for GeoField {
    fn analyzer(&self) -> Option<&Arc<dyn Analyzer>> {
        None // Geo fields don't use analyzers
    }

    fn is_stored(&self) -> bool {
        self.stored
    }

    fn is_indexed(&self) -> bool {
        self.indexed
    }

    fn supports_fast_access(&self) -> bool {
        self.fast_access
    }

    fn type_name(&self) -> &'static str {
        "geo"
    }

    fn clone_box(&self) -> Box<dyn FieldType> {
        Box::new(self.clone())
    }

    fn equals(&self, other: &dyn FieldType) -> bool {
        other.type_name() == "geo"
    }
}

/// A field definition that includes a name and type.
#[derive(Debug)]
pub struct FieldDefinition {
    /// The name of the field
    name: String,
    /// The field type
    field_type: Box<dyn FieldType>,
}

impl Clone for FieldDefinition {
    fn clone(&self) -> Self {
        FieldDefinition {
            name: self.name.clone(),
            field_type: self.field_type.clone_box(),
        }
    }
}

/// Type alias for backward compatibility.
pub type Field = FieldDefinition;

impl FieldDefinition {
    /// Create a new field definition.
    pub fn new<S: Into<String>>(name: S, field_type: Box<dyn FieldType>) -> Self {
        FieldDefinition {
            name: name.into(),
            field_type,
        }
    }

    /// Get the field name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the field type.
    pub fn field_type(&self) -> &dyn FieldType {
        self.field_type.as_ref()
    }

    /// Get the analyzer for this field type (if any).
    pub fn analyzer(&self) -> Option<&str> {
        // For now, we'll return None. In a full implementation,
        // this would return the analyzer name from the field configuration
        None
    }

    /// Check if this field is stored.
    pub fn stored(&self) -> bool {
        self.field_type.is_stored()
    }

    /// Check if this field is indexed.
    pub fn indexed(&self) -> bool {
        self.field_type.is_indexed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Datelike;
    // use crate::analysis::analyzer::StandardAnalyzer;

    #[test]
    fn test_text_field() {
        let field = TextField::new()
            .stored(true)
            .indexed(true)
            .phrase(true)
            .term_vectors(true);

        assert!(field.is_stored());
        assert!(field.is_indexed());
        assert!(field.supports_phrase());
        assert!(field.has_term_vectors());
        assert!(!field.supports_fast_access());
        assert_eq!(field.type_name(), "text");
    }

    #[test]
    fn test_text_field_with_analyzer() {
        let field = TextField::new().with_analyzer("standard");

        // For now, we just check that the analyzer name is set
        assert_eq!(field.analyzer, Some("standard".to_string()));
    }

    #[test]
    fn test_id_field() {
        let field = IdField::new().stored(true).indexed(true).unique(true);

        assert!(field.is_stored());
        assert!(field.is_indexed());
        assert!(field.is_unique());
        assert!(field.supports_fast_access());
        assert_eq!(field.type_name(), "id");
        assert!(field.analyzer().is_none());
    }

    #[test]
    fn test_stored_field() {
        let field = StoredField::new().compressed(true);

        assert!(field.is_stored());
        assert!(!field.is_indexed());
        assert!(field.is_compressed());
        assert!(!field.supports_fast_access());
        assert_eq!(field.type_name(), "stored");
        assert!(field.analyzer().is_none());
    }

    #[test]
    fn test_numeric_field() {
        let field = NumericField::new(NumericType::I64)
            .stored(true)
            .indexed(true)
            .fast_access(true);

        assert!(field.is_stored());
        assert!(field.is_indexed());
        assert!(field.supports_fast_access());
        assert_eq!(field.numeric_type(), NumericType::I64);
        assert_eq!(field.type_name(), "numeric");
        assert!(field.analyzer().is_none());
    }

    #[test]
    fn test_field_equality() {
        let field1 = TextField::new().stored(true).indexed(true);
        let field2 = TextField::new().stored(true).indexed(true);
        let field3 = TextField::new().stored(false).indexed(true);
        let id_field = IdField::new();

        // For now, all text fields are considered equal
        assert!(field1.equals(&field2));
        assert!(field1.equals(&field3)); // This passes due to our simplified comparison
        assert!(!field1.equals(&id_field));
    }

    #[test]
    fn test_field_definition() {
        let field_type = Box::new(TextField::new().stored(true));
        let field_def = FieldDefinition::new("title", field_type);

        assert_eq!(field_def.name(), "title");
        assert_eq!(field_def.field_type().type_name(), "text");
        assert!(field_def.field_type().is_stored());
    }

    #[test]
    fn test_datetime_field() {
        let field = DateTimeField::new()
            .stored(true)
            .indexed(true)
            .fast_access(true)
            .format("%Y-%m-%d %H:%M:%S");

        assert!(field.is_stored());
        assert!(field.is_indexed());
        assert!(field.supports_fast_access());
        assert_eq!(field.type_name(), "datetime");
        assert_eq!(field.date_format(), Some("%Y-%m-%d %H:%M:%S"));
        assert!(field.analyzer().is_none());
    }

    #[test]
    fn test_datetime_parsing() {
        let field = DateTimeField::new();

        // Test RFC3339 parsing (default)
        let dt = field.parse_datetime("2023-12-25T10:30:00Z").unwrap();
        assert_eq!(dt.date_naive().year(), 2023);
        assert_eq!(dt.date_naive().month(), 12);
        assert_eq!(dt.date_naive().day(), 25);

        // Test custom format parsing
        let field_custom = DateTimeField::new().format("%Y-%m-%d %H:%M:%S");
        let dt_custom = field_custom.parse_datetime("2023-12-25 10:30:00").unwrap();
        assert_eq!(dt_custom.date_naive().year(), 2023);
        assert_eq!(dt_custom.date_naive().month(), 12);
        assert_eq!(dt_custom.date_naive().day(), 25);
    }

    #[test]
    fn test_boolean_field() {
        let field = BooleanField::new()
            .stored(true)
            .indexed(true)
            .fast_access(true);

        assert!(field.is_stored());
        assert!(field.is_indexed());
        assert!(field.supports_fast_access());
        assert_eq!(field.type_name(), "boolean");
        assert!(field.analyzer().is_none());
    }

    #[test]
    fn test_boolean_parsing() {
        let field = BooleanField::new();

        // Test true values
        assert_eq!(field.parse_boolean("true").unwrap(), true);
        assert_eq!(field.parse_boolean("TRUE").unwrap(), true);
        assert_eq!(field.parse_boolean("t").unwrap(), true);
        assert_eq!(field.parse_boolean("yes").unwrap(), true);
        assert_eq!(field.parse_boolean("y").unwrap(), true);
        assert_eq!(field.parse_boolean("1").unwrap(), true);
        assert_eq!(field.parse_boolean("on").unwrap(), true);

        // Test false values
        assert_eq!(field.parse_boolean("false").unwrap(), false);
        assert_eq!(field.parse_boolean("FALSE").unwrap(), false);
        assert_eq!(field.parse_boolean("f").unwrap(), false);
        assert_eq!(field.parse_boolean("no").unwrap(), false);
        assert_eq!(field.parse_boolean("n").unwrap(), false);
        assert_eq!(field.parse_boolean("0").unwrap(), false);
        assert_eq!(field.parse_boolean("off").unwrap(), false);

        // Test invalid values
        assert!(field.parse_boolean("maybe").is_err());
        assert!(field.parse_boolean("").is_err());
    }

    #[test]
    fn test_keyword_field() {
        let field = KeywordField::new()
            .stored(true)
            .indexed(true)
            .fast_access(true)
            .case_sensitive(false)
            .term_vectors(true)
            .tokenize(true);

        assert!(field.is_stored());
        assert!(field.is_indexed());
        assert!(field.supports_fast_access());
        assert!(!field.is_case_sensitive());
        assert!(field.has_term_vectors());
        assert!(field.is_tokenized());
        assert_eq!(field.type_name(), "keyword");
        assert!(field.analyzer().is_none());
    }

    #[test]
    fn test_keyword_field_defaults() {
        let field = KeywordField::new();

        assert!(!field.is_stored());
        assert!(field.is_indexed());
        assert!(!field.supports_fast_access());
        assert!(field.is_case_sensitive());
        assert!(!field.has_term_vectors());
        assert!(field.is_tokenized());
    }

    #[test]
    fn test_numeric_field_constructors() {
        let i32_field = NumericField::i32().stored(true);
        assert_eq!(i32_field.numeric_type(), NumericType::I32);
        assert!(i32_field.is_stored());

        let i64_field = NumericField::i64().fast_access(true);
        assert_eq!(i64_field.numeric_type(), NumericType::I64);
        assert!(i64_field.supports_fast_access());

        let f32_field = NumericField::f32();
        assert_eq!(f32_field.numeric_type(), NumericType::F32);

        let f64_field = NumericField::f64();
        assert_eq!(f64_field.numeric_type(), NumericType::F64);

        let u32_field = NumericField::u32();
        assert_eq!(u32_field.numeric_type(), NumericType::U32);

        let u64_field = NumericField::u64();
        assert_eq!(u64_field.numeric_type(), NumericType::U64);
    }

    #[test]
    fn test_numeric_field_equality() {
        let field1 = NumericField::i64();
        let field2 = NumericField::f32();
        let text_field = TextField::new();

        // All numeric fields are considered equal for now
        assert!(field1.equals(&field2));
        assert!(!field1.equals(&text_field));
    }
}
