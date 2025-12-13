//! VectorEngine 検索リクエスト関連の型定義
//!
//! このモジュールは検索リクエスト、クエリベクトル、フィールドセレクタを提供する。

use serde::{Deserialize, Serialize};

use crate::vector::core::document::{StoredVector, VectorType};
use crate::vector::engine::filter::VectorEngineFilter;

fn default_query_limit() -> usize {
    10
}

fn default_overfetch() -> f32 {
    1.0
}

/// Request model for collection-level search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorEngineSearchRequest {
    /// Query vectors to search with.
    #[serde(default)]
    pub query_vectors: Vec<QueryVector>,
    /// Fields to search in. If None, searches all default fields.
    #[serde(default)]
    pub fields: Option<Vec<FieldSelector>>,
    /// Maximum number of results to return.
    #[serde(default = "default_query_limit")]
    pub limit: usize,
    /// How to combine scores from multiple query vectors.
    #[serde(default)]
    pub score_mode: VectorScoreMode,
    /// Overfetch factor for better result quality.
    #[serde(default = "default_overfetch")]
    pub overfetch: f32,
    /// Metadata filter to apply.
    #[serde(default)]
    pub filter: Option<VectorEngineFilter>,
    /// Minimum score threshold. Results with scores below this value are filtered out.
    /// Default is 0.0 (no filtering).
    #[serde(default)]
    pub min_score: f32,
}

impl Default for VectorEngineSearchRequest {
    fn default() -> Self {
        Self {
            query_vectors: Vec::new(),
            fields: None,
            limit: default_query_limit(),
            score_mode: VectorScoreMode::default(),
            overfetch: default_overfetch(),
            filter: None,
            min_score: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum FieldSelector {
    Exact(String),
    Prefix(String),
    VectorType(VectorType),
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum VectorScoreMode {
    #[default]
    WeightedSum,
    MaxSim,
    LateInteraction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryVector {
    pub vector: StoredVector,
    #[serde(default = "QueryVector::default_weight")]
    pub weight: f32,
}

impl QueryVector {
    fn default_weight() -> f32 {
        1.0
    }
}
