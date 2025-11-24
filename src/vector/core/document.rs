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
pub const METADATA_ROLE: &str = "__platypus_vector_role";
pub const METADATA_WEIGHT: &str = "__platypus_vector_weight";

/// Semantic role associated with a stored vector.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VectorRole {
    Text,
    Image,
    Intent,
    Metadata,
    Generic,
    Custom(String),
}

/// Unprocessed textual content destined for a vector field.
#[derive(Debug, Clone, Default)]
pub struct FieldPayload {
    /// Ordered text segments that will be embedded sequentially.
    pub text_segments: Vec<RawTextSegment>,
    /// Arbitrary metadata propagated to the resulting `FieldVectors` entry.
    pub metadata: HashMap<String, String>,
}

impl FieldPayload {
    /// Returns true when no text segments are present.
    pub fn is_empty(&self) -> bool {
        self.text_segments.is_empty()
    }

    /// Add a text segment to this payload.
    pub fn add_text_segment(&mut self, segment: RawTextSegment) {
        self.text_segments.push(segment);
    }

    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }
}

/// Raw text plus optional weighting/attributes before embedding.
#[derive(Debug, Clone)]
pub struct RawTextSegment {
    pub value: String,
    pub weight: f32,
    pub attributes: HashMap<String, String>,
}

impl RawTextSegment {
    pub fn new(value: impl Into<String>) -> Self {
        Self {
            value: value.into(),
            weight: 1.0,
            attributes: HashMap::new(),
        }
    }

    pub fn with_attributes(
        value: impl Into<String>,
        weight: f32,
        attributes: HashMap<String, String>,
    ) -> Self {
        Self {
            value: value.into(),
            weight,
            attributes,
        }
    }
}

impl VectorRole {
    /// Create a custom role label.
    pub fn custom<S: Into<String>>(label: S) -> Self {
        VectorRole::Custom(label.into())
    }
}

impl fmt::Display for VectorRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VectorRole::Text => write!(f, "text"),
            VectorRole::Image => write!(f, "image"),
            VectorRole::Intent => write!(f, "intent"),
            VectorRole::Metadata => write!(f, "metadata"),
            VectorRole::Generic => write!(f, "generic"),
            VectorRole::Custom(label) => write!(f, "{label}"),
        }
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
    pub role: VectorRole,
    pub weight: f32,
    #[serde(default)]
    pub attributes: HashMap<String, String>,
}

impl StoredVector {
    pub fn new(data: Arc<[f32]>, embedder_id: String, role: VectorRole) -> Self {
        Self {
            data,
            embedder_id,
            role,
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
            role: VectorRole::Generic,
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
            role: VectorRole::Generic,
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
        enrich_metadata(&mut metadata, &self.embedder_id, &self.role, self.weight);
        Vector {
            data: self.data.as_ref().to_vec(),
            metadata,
        }
    }

    pub fn into_vector(self) -> Vector {
        let StoredVector {
            data,
            embedder_id,
            role,
            weight,
            mut attributes,
        } = self;
        enrich_metadata(&mut attributes, &embedder_id, &role, weight);
        Vector {
            data: data.as_ref().to_vec(),
            metadata: attributes,
        }
    }
}

fn enrich_metadata(
    metadata: &mut HashMap<String, String>,
    embedder_id: &str,
    role: &VectorRole,
    weight: f32,
) {
    if !embedder_id.is_empty() {
        metadata
            .entry(METADATA_EMBEDDER_ID.to_string())
            .or_insert_with(|| embedder_id.to_string());
    }
    metadata
        .entry(METADATA_ROLE.to_string())
        .or_insert_with(|| role.to_string());
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

/// Document-level wrapper around field vectors and metadata.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DocumentVectors {
    pub doc_id: u64,
    #[serde(default)]
    pub fields: HashMap<String, FieldVectors>,
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

/// Document input model capturing raw payloads before embedding.
#[derive(Debug, Clone, Default)]
pub struct DocumentPayload {
    pub doc_id: u64,
    pub fields: HashMap<String, FieldPayload>,
    pub metadata: HashMap<String, String>,
}

impl DocumentPayload {
    pub fn new(doc_id: u64) -> Self {
        Self {
            doc_id,
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

impl DocumentVectors {
    pub fn new(doc_id: u64) -> Self {
        Self {
            doc_id,
            fields: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    pub fn add_field<V: Into<String>>(&mut self, field_name: V, field: FieldVectors) {
        self.fields.insert(field_name.into(), field);
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
            role: VectorRole::Intent,
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
            vector.metadata.get(METADATA_ROLE),
            Some(&"intent".to_string())
        );
        assert_eq!(
            vector.metadata.get(METADATA_WEIGHT),
            Some(&2.5_f32.to_string())
        );
        assert_eq!(vector.metadata.get("chunk"), Some(&"a".to_string()));
    }
}
