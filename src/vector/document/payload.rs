//! ペイロードとベクトルタイプ
//!
//! このモジュールはベクトルの種別、ペイロードソース、セグメントペイロードを提供する。

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

/// Metadata keys used when bridging to the legacy `Vector` representation.
pub const METADATA_EMBEDDER_ID: &str = "__platypus_vector_embedder_id";
pub const METADATA_VECTOR_TYPE: &str = "__platypus_vector_type";
pub const METADATA_WEIGHT: &str = "__platypus_vector_weight";

/// Semantic type associated with a stored vector.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VectorType {
    Text,
    Image,
    Intent,
    Metadata,
    Generic,
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

/// Enrich metadata with vector type and embedder information.
pub fn enrich_metadata(
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
