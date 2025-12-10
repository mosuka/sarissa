//! VectorEngine 検索レスポンス関連の型定義
//!
//! このモジュールは検索結果、ヒット情報、統計情報を提供する。

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::vector::field::{FieldHit, VectorFieldStats};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VectorEngineSearchResults {
    #[serde(default)]
    pub hits: Vec<VectorEngineHit>,
}

/// Aggregated statistics describing a collection and its fields.
#[derive(Debug, Clone, Default)]
pub struct VectorEngineStats {
    pub document_count: usize,
    pub fields: HashMap<String, VectorFieldStats>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorEngineHit {
    pub doc_id: u64,
    pub score: f32,
    #[serde(default)]
    pub field_hits: Vec<FieldHit>,
}
