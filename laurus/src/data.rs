use std::collections::HashMap;

use chrono::{DateTime, Utc};
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use serde::{Deserialize, Serialize};

/// Helper for archiving DateTime as micros timestamp (i64)
pub struct MicroSeconds;

impl rkyv::with::ArchiveWith<DateTime<Utc>> for MicroSeconds {
    type Archived = rkyv::Archived<i64>;
    type Resolver = ();

    fn resolve_with(field: &DateTime<Utc>, _: (), out: rkyv::Place<Self::Archived>) {
        let ts = field.timestamp_micros();
        ts.resolve((), out);
    }
}

impl<S: rkyv::rancor::Fallible + ?Sized> rkyv::with::SerializeWith<DateTime<Utc>, S>
    for MicroSeconds
{
    fn serialize_with(
        field: &DateTime<Utc>,
        serializer: &mut S,
    ) -> Result<Self::Resolver, S::Error> {
        RkyvSerialize::serialize(&field.timestamp_micros(), serializer)
    }
}

impl<D: rkyv::rancor::Fallible + ?Sized>
    rkyv::with::DeserializeWith<rkyv::Archived<i64>, DateTime<Utc>, D> for MicroSeconds
{
    fn deserialize_with(
        archived: &rkyv::Archived<i64>,
        _deserializer: &mut D,
    ) -> Result<DateTime<Utc>, D::Error> {
        let ts: i64 = (*archived).into();
        // DateTime::from_timestamp_micros returns None only for out-of-range values.
        // Fall back to UNIX epoch for corrupted timestamps to avoid panics.
        Ok(DateTime::from_timestamp_micros(ts).unwrap_or_default())
    }
}

/// The unified value type for fields in a document.
///
/// This enum merges the concepts of `FieldValue` (from Lexical Index) and
/// `VectorValue` (from Vector Index).
#[derive(
    Debug, Clone, PartialEq, Serialize, Deserialize, Archive, RkyvSerialize, RkyvDeserialize,
)]
pub enum DataValue {
    // --- Primitive Types ---
    Null,
    Bool(bool),
    Int64(i64),
    Float64(f64),

    // --- Complex / Searchable Types ---
    /// Text content. Whether this is tokenized or treated as a keyword
    /// is determined by the schema's [`FieldOption`](crate::lexical::core::field::FieldOption)
    /// and the configured [`Analyzer`](crate::analysis::analyzer::analyzer::Analyzer).
    Text(String),

    /// Binary content (image, audio, etc.) to be embedded.
    /// Contains the raw bytes and an optional MIME type.
    Bytes(Vec<u8>, Option<String>),

    /// Pre-computed vector.
    Vector(Vec<f32>),

    /// Date and time in UTC.
    DateTime(#[rkyv(with = MicroSeconds)] chrono::DateTime<chrono::Utc>),

    /// Geographical point (latitude, longitude).
    Geo(f64, f64),

    /// Multi-valued 64-bit signed integers.
    ///
    /// Used by fields declared with
    /// [`IntegerOption::multi_valued`](crate::lexical::core::field::IntegerOption::multi_valued)
    /// set to `true`. Range queries match a document if **any** value in
    /// the array satisfies the predicate (Lucene-style "any match"
    /// semantics with constant scoring).
    Int64Array(Vec<i64>),

    /// Multi-valued 64-bit floating-point numbers.
    ///
    /// Used by fields declared with
    /// [`FloatOption::multi_valued`](crate::lexical::core::field::FloatOption::multi_valued)
    /// set to `true`. Range queries match a document if **any** value in
    /// the array satisfies the predicate (Lucene-style "any match"
    /// semantics with constant scoring).
    Float64Array(Vec<f64>),
}

impl DataValue {
    /// Returns the text value if this is a Text variant.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            DataValue::Text(s) => Some(s),
            _ => None,
        }
    }

    /// Returns the integer value if this is an Int64 variant.
    pub fn as_integer(&self) -> Option<i64> {
        match self {
            DataValue::Int64(i) => Some(*i),
            _ => None,
        }
    }

    /// Returns the float value if this is a Float64 variant.
    pub fn as_float(&self) -> Option<f64> {
        match self {
            DataValue::Float64(f) => Some(*f),
            _ => None,
        }
    }

    /// Returns the boolean value if this is a Bool variant.
    pub fn as_boolean(&self) -> Option<bool> {
        match self {
            DataValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Returns the datetime value if this is a DateTime variant.
    pub fn as_datetime(&self) -> Option<chrono::DateTime<chrono::Utc>> {
        match self {
            DataValue::DateTime(dt) => Some(*dt),
            _ => None,
        }
    }

    /// Returns the vector data if this is a Vector variant.
    pub fn as_vector(&self) -> Option<&Vec<f32>> {
        match self {
            DataValue::Vector(v) => Some(v),
            _ => None,
        }
    }

    /// Returns the bytes data if this is a Bytes variant.
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            DataValue::Bytes(b, _) => Some(b),
            _ => None,
        }
    }

    /// Returns the geographical point if this is a Geo variant.
    pub fn as_geo(&self) -> Option<(f64, f64)> {
        match self {
            DataValue::Geo(lat, lon) => Some((*lat, *lon)),
            _ => None,
        }
    }

    /// Returns the multi-valued integer slice if this is an `Int64Array` variant.
    pub fn as_int64_array(&self) -> Option<&[i64]> {
        match self {
            DataValue::Int64Array(arr) => Some(arr),
            _ => None,
        }
    }

    /// Returns the multi-valued float slice if this is a `Float64Array` variant.
    pub fn as_float64_array(&self) -> Option<&[f64]> {
        match self {
            DataValue::Float64Array(arr) => Some(arr),
            _ => None,
        }
    }
}

// --- Conversions ---

impl From<String> for DataValue {
    fn from(v: String) -> Self {
        DataValue::Text(v)
    }
}

impl From<&str> for DataValue {
    fn from(v: &str) -> Self {
        DataValue::Text(v.to_string())
    }
}

impl From<i64> for DataValue {
    fn from(v: i64) -> Self {
        DataValue::Int64(v)
    }
}

impl From<i32> for DataValue {
    fn from(v: i32) -> Self {
        DataValue::Int64(v as i64)
    }
}

impl From<f64> for DataValue {
    fn from(v: f64) -> Self {
        DataValue::Float64(v)
    }
}

impl From<f32> for DataValue {
    fn from(v: f32) -> Self {
        DataValue::Float64(v as f64)
    }
}

impl From<bool> for DataValue {
    fn from(v: bool) -> Self {
        DataValue::Bool(v)
    }
}

impl From<chrono::DateTime<chrono::Utc>> for DataValue {
    fn from(dt: chrono::DateTime<chrono::Utc>) -> Self {
        DataValue::DateTime(dt)
    }
}

impl From<Vec<f32>> for DataValue {
    fn from(v: Vec<f32>) -> Self {
        DataValue::Vector(v)
    }
}

impl From<Vec<i64>> for DataValue {
    fn from(v: Vec<i64>) -> Self {
        DataValue::Int64Array(v)
    }
}

impl From<Vec<f64>> for DataValue {
    fn from(v: Vec<f64>) -> Self {
        DataValue::Float64Array(v)
    }
}

/// Unified Document structure.
///
/// A document is a pure data container — a collection of named fields,
/// each containing a [`DataValue`]. Document identity (external ID) is
/// managed by the [`Engine`](crate::Engine), not by the document itself.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// Field data.
    pub fields: HashMap<String, DataValue>,
}

impl Document {
    /// Create a new empty document.
    pub fn new() -> Self {
        Self {
            fields: HashMap::new(),
        }
    }

    /// Get a reference to a field's value.
    pub fn get(&self, name: &str) -> Option<&DataValue> {
        self.fields.get(name)
    }

    /// Alias for [`get`](Self::get), provided for API consistency with field-based
    /// access patterns. Behaves identically to `get`.
    pub fn get_field(&self, name: &str) -> Option<&DataValue> {
        self.get(name)
    }

    /// Check if the document has a field.
    pub fn has_field(&self, name: &str) -> bool {
        self.fields.contains_key(name)
    }

    /// Get all field names.
    pub fn field_names(&self) -> Vec<&str> {
        self.fields.keys().map(|s| s.as_str()).collect()
    }

    /// Get the number of fields.
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// Check if the document is empty.
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    pub fn builder() -> DocumentBuilder {
        DocumentBuilder::default()
    }
}

impl Default for Document {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Default)]
pub struct DocumentBuilder {
    fields: HashMap<String, DataValue>,
}

impl DocumentBuilder {
    /// Add a field to the document.
    pub fn add_field(mut self, name: impl Into<String>, value: impl Into<DataValue>) -> Self {
        self.fields.insert(name.into(), value.into());
        self
    }

    /// Add a text field.
    pub fn add_text(self, name: impl Into<String>, text: impl Into<String>) -> Self {
        self.add_field(name.into(), DataValue::Text(text.into()))
    }

    /// Add an integer field.
    pub fn add_integer(self, name: impl Into<String>, value: i64) -> Self {
        self.add_field(name.into(), DataValue::Int64(value))
    }

    /// Add a float field.
    pub fn add_float(self, name: impl Into<String>, value: f64) -> Self {
        self.add_field(name.into(), DataValue::Float64(value))
    }

    /// Add a boolean field.
    pub fn add_boolean(self, name: impl Into<String>, value: bool) -> Self {
        self.add_field(name.into(), DataValue::Bool(value))
    }

    /// Add a datetime field.
    pub fn add_datetime(self, name: impl Into<String>, value: DateTime<Utc>) -> Self {
        self.add_field(name.into(), DataValue::DateTime(value))
    }

    /// Add a vector field.
    pub fn add_vector(self, name: impl Into<String>, vector: Vec<f32>) -> Self {
        self.add_field(name.into(), DataValue::Vector(vector))
    }

    /// Add a geo field (latitude, longitude).
    pub fn add_geo(self, name: impl Into<String>, lat: f64, lon: f64) -> Self {
        self.add_field(name.into(), DataValue::Geo(lat, lon))
    }

    /// Add a multi-valued integer field.
    ///
    /// The schema field must be declared with
    /// [`IntegerOption::multi_valued`](crate::lexical::core::field::IntegerOption::multi_valued)
    /// set to `true`. Range queries match if any value satisfies the
    /// predicate.
    pub fn add_int64_array(self, name: impl Into<String>, values: Vec<i64>) -> Self {
        self.add_field(name.into(), DataValue::Int64Array(values))
    }

    /// Add a multi-valued float field.
    ///
    /// The schema field must be declared with
    /// [`FloatOption::multi_valued`](crate::lexical::core::field::FloatOption::multi_valued)
    /// set to `true`. Range queries match if any value satisfies the
    /// predicate.
    pub fn add_float64_array(self, name: impl Into<String>, values: Vec<f64>) -> Self {
        self.add_field(name.into(), DataValue::Float64Array(values))
    }

    /// Add a binary data field with no MIME type.
    ///
    /// The MIME type is set to `None`. If a MIME type is needed (e.g. for
    /// multimodal embedding), use [`add_field`](Self::add_field) directly with
    /// `DataValue::Bytes(data, Some(mime))`.
    pub fn add_bytes(self, name: impl Into<String>, data: Vec<u8>) -> Self {
        self.add_field(name.into(), DataValue::Bytes(data, None))
    }

    pub fn build(self) -> Document {
        Document {
            fields: self.fields,
        }
    }
}
