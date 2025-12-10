//! フィールドレベルのベクトル型
//!
//! このモジュールはフィールドベクトル群、フィールドペイロード、保存済みベクトルを提供する。

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize, de::Deserializer, ser::Serializer};

use crate::vector::Vector;
use crate::vector::document::payload::{SegmentPayload, VectorType, enrich_metadata};

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::document::payload::{
        METADATA_EMBEDDER_ID, METADATA_VECTOR_TYPE, METADATA_WEIGHT,
    };

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
