//! Document-centric vector data structures.
//!
//! These types power the experimental VectorEngine feature flag. They extend
//! the existing `Vector` representation with metadata describing vector roles,
//! field-level weights, and document-wide groupings.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize, de::Deserializer, ser::Serializer};

use super::vector::Vector;

/// Metadata keys used when bridging to the legacy `Vector` representation.
pub const METADATA_EMBEDDER_ID: &str = "__platypus_vector_embedder_id";
pub const METADATA_VECTOR_TYPE: &str = "__platypus_vector_type";
pub const METADATA_WEIGHT: &str = "__platypus_vector_weight";

/// Semantic type associated with a stored vector.
///
/// This enum categorizes vectors by their source content type, enabling
/// type-specific processing and filtering during search operations.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VectorType {
    /// Vector derived from textual content (e.g., sentences, paragraphs, documents).
    /// Most common type for natural language processing and semantic search.
    Text,
    /// Vector derived from image content (e.g., photos, diagrams, screenshots).
    /// Used for visual similarity search and multimodal applications.
    Image,
    /// Vector representing user intent or query semantics.
    /// Useful for intent classification and query understanding systems.
    Intent,
    /// Vector derived from structured metadata (e.g., tags, categories, attributes).
    /// Enables semantic matching on document properties.
    Metadata,
    /// General-purpose vector without specific semantic categorization.
    /// Default type when the source content type is unknown or mixed.
    Generic,
    /// User-defined vector type with a custom label.
    /// Allows extension for domain-specific vector categories.
    Custom(String),
}

impl VectorType {
    /// Create a custom vector type label.
    pub fn custom<S: Into<String>>(label: S) -> Self {
        VectorType::Custom(label.into())
    }
}

impl fmt::Display for VectorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VectorType::Text => write!(f, "text"),
            VectorType::Image => write!(f, "image"),
            VectorType::Intent => write!(f, "intent"),
            VectorType::Metadata => write!(f, "metadata"),
            VectorType::Generic => write!(f, "generic"),
            VectorType::Custom(label) => write!(f, "{label}"),
        }
    }
}

/// Source material used to produce a vector embedding.
#[derive(Debug, Clone)]
pub enum PayloadSource {
    Text {
        value: String,
    },
    Bytes {
        bytes: Arc<[u8]>,
        mime: Option<String>,
    },
    Uri {
        uri: String,
        media_hint: Option<String>,
    },
    Vector {
        data: Arc<[f32]>,
        embedder_id: String,
    },
}

impl PayloadSource {
    pub fn text(value: impl Into<String>) -> Self {
        PayloadSource::Text {
            value: value.into(),
        }
    }
}

/// DSL unit describing a concrete segment slated for embedding.
#[derive(Debug, Clone)]
pub struct SegmentPayload {
    pub source: PayloadSource,
    pub vector_type: VectorType,
    pub weight: f32,
    pub metadata: HashMap<String, String>,
}

impl SegmentPayload {
    pub fn new(source: PayloadSource, vector_type: VectorType) -> Self {
        Self {
            source,
            vector_type,
            weight: 1.0,
            metadata: HashMap::new(),
        }
    }

    pub fn text(value: impl Into<String>) -> Self {
        Self::new(PayloadSource::text(value), VectorType::Text)
    }

    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = if weight <= 0.0 { 1.0 } else { weight };
        self
    }

    pub fn with_metadata(mut self, metadata: HashMap<String, String>) -> Self {
        self.metadata = metadata;
        self
    }
}

/// Unprocessed content destined for a vector field.
#[derive(Debug, Clone, Default)]
pub struct FieldPayload {
    pub segments: Vec<SegmentPayload>,
    pub metadata: HashMap<String, String>,
}

impl FieldPayload {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    pub fn add_segment(&mut self, segment: SegmentPayload) {
        self.segments.push(segment);
    }

    pub fn add_text_segment(&mut self, value: impl Into<String>) {
        self.segments.push(SegmentPayload::text(value));
    }

    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }
}

/// A dense vector plus metadata captured during ingestion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredVector {
    #[serde(
        serialize_with = "serialize_vector_data",
        deserialize_with = "deserialize_vector_data"
    )]
    pub data: Arc<[f32]>,
    pub embedder_id: String,
    pub vector_type: VectorType,
    pub weight: f32,
    #[serde(default)]
    pub attributes: HashMap<String, String>,
}

impl StoredVector {
    pub fn new(data: Arc<[f32]>, embedder_id: String, vector_type: VectorType) -> Self {
        Self {
            data,
            embedder_id,
            vector_type,
            weight: 1.0,
            attributes: HashMap::new(),
        }
    }

    pub fn dimension(&self) -> usize {
        self.data.len()
    }
}

impl From<Vector> for StoredVector {
    fn from(vector: Vector) -> Self {
        let data: Arc<[f32]> = vector.data.into();
        Self {
            data,
            embedder_id: String::new(),
            vector_type: VectorType::Generic,
            weight: 1.0,
            attributes: vector.metadata,
        }
    }
}

impl From<&Vector> for StoredVector {
    fn from(vector: &Vector) -> Self {
        let data: Arc<[f32]> = vector.data.clone().into();
        Self {
            data,
            embedder_id: String::new(),
            vector_type: VectorType::Generic,
            weight: 1.0,
            attributes: vector.metadata.clone(),
        }
    }
}

impl From<StoredVector> for Vector {
    fn from(stored: StoredVector) -> Self {
        stored.into_vector()
    }
}

impl From<&StoredVector> for Vector {
    fn from(stored: &StoredVector) -> Self {
        stored.to_vector()
    }
}

impl StoredVector {
    pub fn to_vector(&self) -> Vector {
        let mut metadata = self.attributes.clone();
        enrich_metadata(
            &mut metadata,
            &self.embedder_id,
            &self.vector_type,
            self.weight,
        );
        Vector {
            data: self.data.as_ref().to_vec(),
            metadata,
        }
    }

    pub fn into_vector(self) -> Vector {
        let StoredVector {
            data,
            embedder_id,
            vector_type,
            weight,
            mut attributes,
        } = self;
        enrich_metadata(&mut attributes, &embedder_id, &vector_type, weight);
        Vector {
            data: data.as_ref().to_vec(),
            metadata: attributes,
        }
    }
}

fn enrich_metadata(
    metadata: &mut HashMap<String, String>,
    embedder_id: &str,
    vector_type: &VectorType,
    weight: f32,
) {
    if !embedder_id.is_empty() {
        metadata
            .entry(METADATA_EMBEDDER_ID.to_string())
            .or_insert_with(|| embedder_id.to_string());
    }
    metadata
        .entry(METADATA_VECTOR_TYPE.to_string())
        .or_insert_with(|| vector_type.to_string());
    metadata
        .entry(METADATA_WEIGHT.to_string())
        .or_insert_with(|| weight.to_string());
}

fn serialize_vector_data<S>(data: &Arc<[f32]>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    data.as_ref().serialize(serializer)
}

fn deserialize_vector_data<'de, D>(deserializer: D) -> Result<Arc<[f32]>, D::Error>
where
    D: Deserializer<'de>,
{
    let buffer = Vec::<f32>::deserialize(deserializer)?;
    Ok(buffer.into())
}

/// All vectors associated with a single logical field.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FieldVectors {
    #[serde(default)]
    pub vectors: Vec<StoredVector>,
    #[serde(default = "FieldVectors::default_weight")]
    pub weight: f32,
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

impl FieldVectors {
    fn default_weight() -> f32 {
        1.0
    }

    pub fn vector_count(&self) -> usize {
        self.vectors.len()
    }
}

/// Document-level wrapper around field vectors and metadata (doc_id is supplied separately).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DocumentVector {
    #[serde(default)]
    pub fields: HashMap<String, FieldVectors>,
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

/// Document input model capturing raw payloads before embedding.
#[derive(Debug, Clone, Default)]
pub struct DocumentPayload {
    pub fields: HashMap<String, FieldPayload>,
    pub metadata: HashMap<String, String>,
}

impl DocumentPayload {
    pub fn new() -> Self {
        Self {
            fields: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    pub fn add_field(&mut self, field_name: impl Into<String>, payload: FieldPayload) {
        self.fields.insert(field_name.into(), payload);
    }

    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }
}

impl DocumentVector {
    pub fn new() -> Self {
        Self {
            fields: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    pub fn add_field<V: Into<String>>(&mut self, field_name: V, field: FieldVectors) {
        self.fields.insert(field_name.into(), field);
    }

    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stored_vector_conversion_enriches_metadata() {
        let stored = StoredVector {
            data: Arc::<[f32]>::from([1.0_f32, 2.0_f32]),
            embedder_id: "embedder-x".into(),
            vector_type: VectorType::Intent,
            weight: 2.5,
            attributes: HashMap::from([(String::from("chunk"), String::from("a"))]),
        };

        let vector = stored.to_vector();

        assert_eq!(vector.data, vec![1.0, 2.0]);
        assert_eq!(
            vector.metadata.get(METADATA_EMBEDDER_ID),
            Some(&"embedder-x".to_string())
        );
        assert_eq!(
            vector.metadata.get(METADATA_VECTOR_TYPE),
            Some(&"intent".to_string())
        );
        assert_eq!(
            vector.metadata.get(METADATA_WEIGHT),
            Some(&2.5_f32.to_string())
        );
        assert_eq!(vector.metadata.get("chunk"), Some(&"a".to_string()));
    }
}
