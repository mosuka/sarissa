//! VectorEngine フィルタ関連の型定義
//!
//! このモジュールはメタデータフィルタ、エンジンフィルタ、フィルタマッチ結果を提供する。

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MetadataFilter {
    #[serde(default)]
    pub equals: HashMap<String, String>,
}

impl MetadataFilter {
    pub(crate) fn matches(&self, metadata: &HashMap<String, String>) -> bool {
        self.equals.iter().all(|(key, expected)| {
            metadata
                .get(key)
                .map(|actual| actual == expected)
                .unwrap_or(false)
        })
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.equals.is_empty()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VectorEngineFilter {
    #[serde(default)]
    pub document: MetadataFilter,
    #[serde(default)]
    pub field: MetadataFilter,
}

impl VectorEngineFilter {
    pub(crate) fn is_empty(&self) -> bool {
        self.document.is_empty() && self.field.is_empty()
    }
}

#[derive(Debug, Default)]
pub struct RegistryFilterMatches {
    pub(crate) allowed_fields: HashMap<u64, HashSet<String>>,
}

impl RegistryFilterMatches {
    pub(crate) fn is_empty(&self) -> bool {
        self.allowed_fields.is_empty()
    }

    pub(crate) fn contains_doc(&self, doc_id: u64) -> bool {
        self.allowed_fields.contains_key(&doc_id)
    }

    pub(crate) fn field_allowed(&self, doc_id: u64, field: &str) -> bool {
        self.allowed_fields
            .get(&doc_id)
            .map(|fields| fields.contains(field))
            .unwrap_or(false)
    }
}
