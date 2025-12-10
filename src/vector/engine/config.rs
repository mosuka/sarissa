//! VectorEngine 設定関連の型定義
//!
//! このモジュールはエンジン設定、フィールド設定、埋め込み設定を提供する。

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::{PlatypusError, Result};
use crate::vector::DistanceMetric;
use crate::vector::core::document::VectorType;

/// Configuration for a single vector collection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorEngineConfig {
    pub fields: HashMap<String, VectorFieldConfig>,
    #[serde(default)]
    pub embedders: HashMap<String, VectorEmbedderConfig>,
    pub default_fields: Vec<String>,
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl VectorEngineConfig {
    pub fn validate(&self) -> Result<()> {
        for field in &self.default_fields {
            if !self.fields.contains_key(field) {
                return Err(PlatypusError::invalid_config(format!(
                    "default field '{field}' is not defined"
                )));
            }
        }

        for (field_name, config) in &self.fields {
            if let Some(embedder_id) = config.embedder.as_deref()
                && !self.embedders.contains_key(embedder_id)
            {
                return Err(PlatypusError::invalid_config(format!(
                    "vector field '{field_name}' references undefined embedder '{embedder_id}'"
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorFieldConfig {
    pub dimension: usize,
    pub distance: DistanceMetric,
    pub index: VectorIndexKind,
    pub embedder_id: String,
    pub vector_type: VectorType,
    #[serde(default)]
    pub embedder: Option<String>,
    #[serde(default = "VectorFieldConfig::default_weight")]
    pub base_weight: f32,
}

impl VectorFieldConfig {
    fn default_weight() -> f32 {
        1.0
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum VectorIndexKind {
    Flat,
    Hnsw,
    Ivf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorEmbedderConfig {
    pub provider: VectorEmbedderProvider,
    pub model: String,
    #[serde(default)]
    pub options: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum VectorEmbedderProvider {
    CandleText,
    CandleMultimodal,
    OpenAiText,
    External,
}
