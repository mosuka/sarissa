//! VectorEngine スナップショット関連の型定義
//!
//! このモジュールはドキュメントスナップショット、コレクションマニフェストを提供する。

use serde::{Deserialize, Serialize};

use crate::vector::core::document::DocumentVector;
use crate::vector::engine::wal::SeqNumber;

pub const FIELD_INDEX_BASENAME: &str = "index";
pub const REGISTRY_NAMESPACE: &str = "vector_registry";
pub const REGISTRY_SNAPSHOT_FILE: &str = "registry.json";
pub const REGISTRY_WAL_FILE: &str = "wal.json";
pub const DOCUMENT_SNAPSHOT_FILE: &str = "documents.json";
pub const DOCUMENT_SNAPSHOT_TEMP_FILE: &str = "documents.tmp";
pub const COLLECTION_MANIFEST_FILE: &str = "manifest.json";
pub const COLLECTION_MANIFEST_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct DocumentSnapshot {
    #[serde(default)]
    pub(crate) last_wal_seq: SeqNumber,
    #[serde(default)]
    pub(crate) documents: Vec<SnapshotDocument>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct SnapshotDocument {
    pub(crate) doc_id: u64,
    pub(crate) document: DocumentVector,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CollectionManifest {
    pub(crate) version: u32,
    pub(crate) snapshot_wal_seq: SeqNumber,
    pub(crate) wal_last_seq: SeqNumber,
    #[serde(default)]
    pub(crate) field_configs:
        std::collections::HashMap<String, crate::vector::engine::config::VectorFieldConfig>,
}
