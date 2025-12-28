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
pub const METADATA_VECTOR_TYPE: &str = "__sarissa_vector_type";
pub const METADATA_WEIGHT: &str = "__sarissa_vector_weight";

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
    },
}

impl PayloadSource {
    /// Creates a text payload source.
    pub fn text(value: impl Into<String>) -> Self {
        PayloadSource::Text {
            value: value.into(),
        }
    }

    /// Creates a bytes payload source with optional MIME type.
    pub fn bytes(bytes: impl Into<Arc<[u8]>>, mime: Option<String>) -> Self {
        PayloadSource::Bytes {
            bytes: bytes.into(),
            mime,
        }
    }

    /// Creates a URI payload source with optional media hint.
    pub fn uri(uri: impl Into<String>, media_hint: Option<String>) -> Self {
        PayloadSource::Uri {
            uri: uri.into(),
            media_hint,
        }
    }

    /// Creates a pre-embedded vector payload source.
    pub fn vector(data: impl Into<Arc<[f32]>>) -> Self {
        PayloadSource::Vector { data: data.into() }
    }
}

/// Single payload for a field (1 field = 1 payload).
///
/// This is the simplified payload type for the flattened data model.
/// Each field in a document maps to exactly one `Payload`, which will
/// produce exactly one vector after embedding.
///
/// For long texts that need chunking, create separate documents for each chunk
/// and use metadata (e.g., `parent_doc_id`, `chunk_index`) to track relationships.
#[derive(Debug, Clone)]
pub struct Payload {
    /// The source material to be embedded.
    pub source: PayloadSource,
    /// The semantic type of the vector.
    pub vector_type: VectorType,
}

impl Payload {
    /// Creates a new payload with the given source and vector type.
    pub fn new(source: PayloadSource, vector_type: VectorType) -> Self {
        Self {
            source,
            vector_type,
        }
    }

    /// Creates a text payload with `VectorType::Text`.
    pub fn text(value: impl Into<String>) -> Self {
        Self::new(PayloadSource::text(value), VectorType::Text)
    }

    /// Creates a bytes payload with `VectorType::Image`.
    pub fn bytes(bytes: impl Into<Arc<[u8]>>, mime: Option<String>) -> Self {
        Self::new(PayloadSource::bytes(bytes, mime), VectorType::Image)
    }

    /// Creates a URI payload with `VectorType::Image`.
    pub fn uri(uri: impl Into<String>, media_hint: Option<String>) -> Self {
        Self::new(PayloadSource::uri(uri, media_hint), VectorType::Image)
    }

    /// Creates a pre-embedded vector payload with `VectorType::Generic`.
    pub fn vector(data: impl Into<Arc<[f32]>>) -> Self {
        Self::new(PayloadSource::vector(data), VectorType::Generic)
    }

    /// Sets the vector type for this payload.
    pub fn with_vector_type(mut self, vector_type: VectorType) -> Self {
        self.vector_type = vector_type;
        self
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
    pub vector_type: VectorType,
    pub weight: f32,
    #[serde(default)]
    pub attributes: HashMap<String, String>,
}

impl StoredVector {
    pub fn new(data: Arc<[f32]>, vector_type: VectorType) -> Self {
        Self {
            data,
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
        enrich_metadata(&mut metadata, &self.vector_type, self.weight);
        Vector {
            data: self.data.as_ref().to_vec(),
            metadata,
        }
    }

    pub fn into_vector(self) -> Vector {
        let StoredVector {
            data,
            vector_type,
            weight,
            mut attributes,
        } = self;
        enrich_metadata(&mut attributes, &vector_type, weight);
        Vector {
            data: data.as_ref().to_vec(),
            metadata: attributes,
        }
    }
}

fn enrich_metadata(metadata: &mut HashMap<String, String>, vector_type: &VectorType, weight: f32) {
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

/// Document with embedded vectors for each field.
///
/// Each field maps to exactly one `StoredVector`. This is the flattened model
/// where 1 field = 1 vector.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DocumentVector {
    /// Fields with their embedded vectors.
    #[serde(default)]
    pub fields: HashMap<String, StoredVector>,
    /// Document-level metadata.
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

/// Document input model capturing raw payloads before embedding.
///
/// Each field maps to exactly one `Payload`, which will produce exactly one
/// vector after embedding. For long texts that need chunking, create separate
/// documents for each chunk with metadata linking them (e.g., `parent_doc_id`).
#[derive(Debug, Clone, Default)]
pub struct DocumentPayload {
    /// Fields to embed, each containing a single payload.
    pub fields: HashMap<String, Payload>,
    /// Document-level metadata (e.g., author, source, parent_doc_id).
    pub metadata: HashMap<String, String>,
}

impl DocumentPayload {
    /// Creates a new empty document payload.
    pub fn new() -> Self {
        Self {
            fields: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    /// Sets a field with the given name and `Payload`.
    pub fn set_field(&mut self, name: impl Into<String>, payload: Payload) {
        self.fields.insert(name.into(), payload);
    }

    /// Sets a text field with `VectorType::Text`.
    ///
    /// Convenience method equivalent to `set_field(name, Payload::text(text))`.
    pub fn set_text(&mut self, name: impl Into<String>, text: impl Into<String>) {
        self.set_field(name, Payload::text(text));
    }

    /// Sets metadata for the document.
    pub fn set_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }
}

impl DocumentVector {
    /// Creates a new empty document vector.
    pub fn new() -> Self {
        Self {
            fields: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    /// Sets a field with the given name and `StoredVector`.
    pub fn set_field(&mut self, name: impl Into<String>, vector: StoredVector) {
        self.fields.insert(name.into(), vector);
    }

    /// Sets metadata for the document.
    pub fn set_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stored_vector_conversion_enriches_metadata() {
        let stored = StoredVector {
            data: Arc::<[f32]>::from([1.0_f32, 2.0_f32]),
            vector_type: VectorType::Intent,
            weight: 2.5,
            attributes: HashMap::from([(String::from("chunk"), String::from("a"))]),
        };

        let vector = stored.to_vector();

        assert_eq!(vector.data, vec![1.0, 2.0]);

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
