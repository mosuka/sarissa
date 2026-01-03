//! Document-centric vector data structures.
//!
//! These types power the experimental VectorEngine feature flag. They extend
//! the existing `Vector` representation with metadata describing vector roles,
//! field-level weights, and document-wide groupings.

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize, de::Deserializer, ser::Serializer};

use super::vector::Vector;

/// Metadata keys used when bridging to the legacy `Vector` representation.
pub const METADATA_WEIGHT: &str = "__sarissa_vector_weight";

/// Source material used to produce a vector embedding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PayloadSource {
    Text {
        value: String,
    },
    Bytes {
        #[serde(
            serialize_with = "serialize_bytes_data",
            deserialize_with = "deserialize_bytes_data"
        )]
        bytes: Arc<[u8]>,
        mime: Option<String>,
    },
    Vector {
        #[serde(
            serialize_with = "serialize_vector_data",
            deserialize_with = "deserialize_vector_data"
        )]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Payload {
    /// The source material to be embedded.
    pub source: PayloadSource,
}

impl Payload {
    /// Creates a new payload with the given source.
    pub fn new(source: PayloadSource) -> Self {
        Self { source }
    }

    /// Creates a text payload.
    pub fn text(value: impl Into<String>) -> Self {
        Self::new(PayloadSource::text(value))
    }

    /// Creates a bytes payload.
    pub fn bytes(bytes: impl Into<Arc<[u8]>>, mime: Option<String>) -> Self {
        Self::new(PayloadSource::bytes(bytes, mime))
    }

    /// Creates a pre-embedded vector payload.
    pub fn vector(data: impl Into<Arc<[f32]>>) -> Self {
        Self::new(PayloadSource::vector(data))
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
    pub weight: f32,
    #[serde(default)]
    pub attributes: HashMap<String, String>,
}

impl StoredVector {
    pub fn new(data: Arc<[f32]>) -> Self {
        Self {
            data,
            weight: 1.0,
            attributes: HashMap::new(),
        }
    }

    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
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
        enrich_metadata(&mut metadata, self.weight);
        Vector {
            data: self.data.as_ref().to_vec(),
            metadata,
        }
    }

    pub fn into_vector(self) -> Vector {
        let StoredVector {
            data,
            weight,
            mut attributes,
        } = self;
        enrich_metadata(&mut attributes, weight);
        Vector {
            data: data.as_ref().to_vec(),
            metadata: attributes,
        }
    }
}

fn enrich_metadata(metadata: &mut HashMap<String, String>, weight: f32) {
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

fn serialize_bytes_data<S>(data: &Arc<[u8]>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    data.as_ref().serialize(serializer)
}

fn deserialize_bytes_data<'de, D>(deserializer: D) -> Result<Arc<[u8]>, D::Error>
where
    D: Deserializer<'de>,
{
    let buffer = Vec::<u8>::deserialize(deserializer)?;
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
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
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

    /// Sets a text field.
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
            weight: 2.5,
            attributes: HashMap::from([(String::from("chunk"), String::from("a"))]),
        };

        let vector = stored.to_vector();

        assert_eq!(vector.data, vec![1.0, 2.0]);

        assert_eq!(
            vector.metadata.get(METADATA_WEIGHT),
            Some(&2.5_f32.to_string())
        );
        assert_eq!(vector.metadata.get("chunk"), Some(&"a".to_string()));
    }
}
