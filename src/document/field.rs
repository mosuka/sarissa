//! Field value types and field options for documents.
//!
//! This module defines:
//! - [`Field`] - A struct combining a value and its indexing options
//! - [`FieldValue`] - The value stored in a field (Text, Integer, etc.)
//! - [`FieldOption`] - Type-specific indexing options (TextOption, VectorOption, etc.)
//!
//! # Field Structure
//!
//! Each field consists of:
//! - **value**: The actual data (FieldValue)
//! - **option**: How the field should be indexed (FieldOption)
//!
//! # Supported Types
//!
//! - **Text** - String data for full-text search
//! - **Vector** - Embedding vectors for semantic search
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
//! use yatagarasu::document::field::FieldValue;
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
//! use yatagarasu::document::field::FieldValue;
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
use crate::vector::DistanceMetric;

/// A field combines a value with indexing options.
///
/// This struct represents a complete field in a document, containing both
/// the data (value) and metadata about how it should be indexed (option).
///
/// # Examples
///
/// ```
/// use yatagarasu::document::field::{Field, FieldValue, FieldOption, TextOption};
///
/// // Create a text field with custom options
/// let field = Field {
///     value: FieldValue::Text("Rust Programming".to_string()),
///     option: FieldOption::Text(TextOption {
///         indexed: true,
///         stored: true,
///         term_vectors: true,
///     }),
/// };
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Field {
    /// The field value.
    pub value: FieldValue,

    /// The field indexing options.
    pub option: FieldOption,
}

impl Field {
    /// Create a new field with a value and option.
    pub fn new(value: FieldValue, option: FieldOption) -> Self {
        Self { value, option }
    }

    /// Create a field with the option inferred from the value type.
    pub fn with_default_option(value: FieldValue) -> Self {
        let option = FieldOption::from_field_value(&value);
        Self { value, option }
    }
}

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
/// use yatagarasu::document::field::FieldValue;
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
/// use yatagarasu::document::field::FieldValue;
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
    /// Vector value (text to be embedded)
    Vector(String),
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

// ============================================================================
// Field Options - Configuration for indexing and storage
// ============================================================================

/// Options for Text fields (used by Lexical indexing).
///
/// Controls how text fields are analyzed, indexed, and stored.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextOption {
    /// Whether to index this field for search.
    #[serde(default = "default_true")]
    pub indexed: bool,

    /// Whether to store the original field value.
    #[serde(default = "default_true")]
    pub stored: bool,

    /// Whether to store term vectors (enables highlighting, more-like-this).
    #[serde(default)]
    pub term_vectors: bool,
}

impl Default for TextOption {
    fn default() -> Self {
        Self {
            indexed: true,
            stored: true,
            term_vectors: false,
        }
    }
}

/// Vector index types for semantic search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VectorIndexType {
    /// Flat index (brute-force exact search).
    /// Best for small datasets (< 100K vectors).
    Flat,

    /// HNSW index (hierarchical navigable small world graph).
    /// Best for medium to large datasets with fast approximate search.
    HNSW,

    /// IVF index (inverted file with clustering).
    /// Best for very large datasets with memory-efficient indexing.
    IVF,
}

/// Flat-specific configuration options.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct FlatOption {
    // Flat index currently has no specific options
    // This struct is here for consistency and future extensibility
}

/// HNSW-specific configuration options.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HnswOption {
    /// Number of connections per layer (M parameter).
    /// Higher values improve recall but increase memory usage.
    #[serde(default = "default_m")]
    pub m: usize,

    /// Size of the dynamic candidate list during construction.
    /// Higher values improve index quality but slow down construction.
    #[serde(default = "default_ef_construction")]
    pub ef_construction: usize,

    /// Maximum number of layers in the graph.
    #[serde(default = "default_max_layers")]
    pub max_layers: usize,
}

fn default_m() -> usize {
    16
}
fn default_ef_construction() -> usize {
    200
}
fn default_max_layers() -> usize {
    6
}

impl Default for HnswOption {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 200,
            max_layers: 6,
        }
    }
}

/// IVF-specific configuration options.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IvfOption {
    /// Number of clusters for inverted file.
    #[serde(default = "default_n_clusters")]
    pub n_clusters: usize,

    /// Number of clusters to probe during search.
    #[serde(default = "default_n_probes")]
    pub n_probes: usize,
}

fn default_n_clusters() -> usize {
    100
}
fn default_n_probes() -> usize {
    10
}

impl Default for IvfOption {
    fn default() -> Self {
        Self {
            n_clusters: 100,
            n_probes: 10,
        }
    }
}

/// Options for Vector fields (used by Vector semantic search).
///
/// Configures how text is embedded and indexed as vectors.
///
/// # Examples
///
/// ```
/// use yatagarasu::document::field::{VectorOption, VectorIndexType};
/// use yatagarasu::vector::DistanceMetric;
///
/// // Simple flat index
/// let flat = VectorOption::flat(384);
///
/// // HNSW with default settings
/// let hnsw = VectorOption::hnsw(768);
///
/// // Custom configuration
/// let custom = VectorOption {
///     index_type: VectorIndexType::HNSW,
///     dimension: 1536,
///     distance_metric: DistanceMetric::Euclidean,
///     normalize: false,
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorOption {
    /// Vector index type (Flat, HNSW, or IVF).
    pub index_type: VectorIndexType,

    /// Dimension of the embedding vectors.
    pub dimension: usize,

    /// Distance metric for similarity computation.
    pub distance_metric: DistanceMetric,

    /// Whether to normalize vectors to unit length.
    /// Recommended for cosine similarity.
    #[serde(default = "default_true")]
    pub normalize: bool,

    /// Flat-specific configuration (used when index_type = Flat).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flat: Option<FlatOption>,

    /// HNSW-specific configuration (used when index_type = HNSW).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hnsw: Option<HnswOption>,

    /// IVF-specific configuration (used when index_type = IVF).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ivf: Option<IvfOption>,
}

fn default_true() -> bool {
    true
}

impl Default for VectorOption {
    fn default() -> Self {
        Self {
            index_type: VectorIndexType::HNSW,
            dimension: 384,
            distance_metric: DistanceMetric::Cosine,
            normalize: true,
            flat: None,
            hnsw: Some(HnswOption::default()),
            ivf: None,
        }
    }
}

impl VectorOption {
    /// Create a Flat index configuration.
    ///
    /// # Arguments
    ///
    /// * `dimension` - The embedding dimension
    ///
    /// # Examples
    ///
    /// ```
    /// use yatagarasu::document::field::VectorOption;
    ///
    /// let opt = VectorOption::flat(384);
    /// ```
    pub fn flat(dimension: usize) -> Self {
        Self {
            index_type: VectorIndexType::Flat,
            dimension,
            flat: Some(FlatOption::default()),
            hnsw: None,
            ivf: None,
            ..Default::default()
        }
    }

    /// Create an HNSW index configuration with default parameters.
    ///
    /// # Arguments
    ///
    /// * `dimension` - The embedding dimension
    ///
    /// # Examples
    ///
    /// ```
    /// use yatagarasu::document::field::VectorOption;
    ///
    /// let opt = VectorOption::hnsw(768);
    /// ```
    pub fn hnsw(dimension: usize) -> Self {
        Self {
            index_type: VectorIndexType::HNSW,
            dimension,
            flat: None,
            hnsw: Some(HnswOption::default()),
            ivf: None,
            ..Default::default()
        }
    }

    /// Create an IVF index configuration with default parameters.
    ///
    /// # Arguments
    ///
    /// * `dimension` - The embedding dimension
    ///
    /// # Examples
    ///
    /// ```
    /// use yatagarasu::document::field::VectorOption;
    ///
    /// let opt = VectorOption::ivf(1536);
    /// ```
    pub fn ivf(dimension: usize) -> Self {
        Self {
            index_type: VectorIndexType::IVF,
            dimension,
            flat: None,
            hnsw: None,
            ivf: Some(IvfOption::default()),
            ..Default::default()
        }
    }
}

/// Options for Integer fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IntegerOption {
    /// Whether to index this field for range queries.
    #[serde(default = "default_true")]
    pub indexed: bool,

    /// Whether to store the original value.
    #[serde(default = "default_true")]
    pub stored: bool,
}

impl Default for IntegerOption {
    fn default() -> Self {
        Self {
            indexed: true,
            stored: true,
        }
    }
}

/// Options for Float fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FloatOption {
    /// Whether to index this field for range queries.
    #[serde(default = "default_true")]
    pub indexed: bool,

    /// Whether to store the original value.
    #[serde(default = "default_true")]
    pub stored: bool,
}

impl Default for FloatOption {
    fn default() -> Self {
        Self {
            indexed: true,
            stored: true,
        }
    }
}

/// Options for Boolean fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BooleanOption {
    /// Whether to index this field.
    #[serde(default = "default_true")]
    pub indexed: bool,

    /// Whether to store the original value.
    #[serde(default = "default_true")]
    pub stored: bool,
}

impl Default for BooleanOption {
    fn default() -> Self {
        Self {
            indexed: true,
            stored: true,
        }
    }
}

/// Options for Binary fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BinaryOption {
    /// Whether to store the binary data.
    #[serde(default = "default_true")]
    pub stored: bool,
}

impl Default for BinaryOption {
    fn default() -> Self {
        Self { stored: true }
    }
}

/// Options for DateTime fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DateTimeOption {
    /// Whether to index this field for range queries.
    #[serde(default = "default_true")]
    pub indexed: bool,

    /// Whether to store the original value.
    #[serde(default = "default_true")]
    pub stored: bool,
}

impl Default for DateTimeOption {
    fn default() -> Self {
        Self {
            indexed: true,
            stored: true,
        }
    }
}

/// Options for Geo (geographic point) fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeoOption {
    /// Whether to index this field for geo queries.
    #[serde(default = "default_true")]
    pub indexed: bool,

    /// Whether to store the original value.
    #[serde(default = "default_true")]
    pub stored: bool,
}

/// Unified field option type that wraps all field-specific options.
///
/// This enum provides a type-safe way to store configuration options
/// for different field types within a Document structure.
///
/// # Examples
///
/// ```
/// use yatagarasu::document::field::{FieldOption, TextOption, VectorOption};
///
/// // Text field with custom options
/// let text_opt = FieldOption::Text(TextOption {
///     indexed: true,
///     stored: true,
///     term_vectors: true,
/// });
///
/// // Vector field with HNSW index
/// let vector_opt = FieldOption::Vector(VectorOption::hnsw(768));
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FieldOption {
    /// Options for text fields (lexical search).
    Text(TextOption),

    /// Options for vector fields (semantic search).
    Vector(VectorOption),

    /// Options for integer fields.
    Integer(IntegerOption),

    /// Options for float fields.
    Float(FloatOption),

    /// Options for boolean fields.
    Boolean(BooleanOption),

    /// Options for binary fields.
    Binary(BinaryOption),

    /// Options for datetime fields.
    DateTime(DateTimeOption),

    /// Options for geographic point fields.
    Geo(GeoOption),
}

impl Default for FieldOption {
    fn default() -> Self {
        FieldOption::Text(TextOption::default())
    }
}

impl FieldOption {
    /// Create a default option based on the field value type.
    ///
    /// This method infers appropriate default options based on the
    /// type of field value.
    pub fn from_field_value(value: &FieldValue) -> Self {
        match value {
            FieldValue::Text(_) => FieldOption::Text(TextOption::default()),
            FieldValue::Vector(_) => FieldOption::Vector(VectorOption::default()),
            FieldValue::Integer(_) => FieldOption::Integer(IntegerOption::default()),
            FieldValue::Float(_) => FieldOption::Float(FloatOption::default()),
            FieldValue::Boolean(_) => FieldOption::Boolean(BooleanOption::default()),
            FieldValue::Binary(_) => FieldOption::Binary(BinaryOption::default()),
            FieldValue::DateTime(_) => FieldOption::DateTime(DateTimeOption::default()),
            FieldValue::Geo(_) => FieldOption::Geo(GeoOption::default()),
            FieldValue::Null => FieldOption::Text(TextOption::default()),
        }
    }
}

impl Default for GeoOption {
    fn default() -> Self {
        Self {
            indexed: true,
            stored: true,
        }
    }
}
