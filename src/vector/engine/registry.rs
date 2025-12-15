//! VectorEngine ドキュメントレジストリ関連の型定義
//!
//! このモジュールはドキュメントレジストリ、ドキュメントエントリ、フィールドエントリを提供する。

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::error::{PlatypusError, Result};
use crate::vector::core::document::DocumentVector;
use crate::vector::engine::filter::{RegistryFilterMatches, VectorFilter};

pub type RegistryVersion = u64;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldEntry {
    pub field_name: String,
    pub version: RegistryVersion,
    pub vector_count: usize,
    pub weight: f32,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentEntry {
    pub doc_id: u64,
    pub version: RegistryVersion,
    pub metadata: HashMap<String, String>,
    pub fields: HashMap<String, FieldEntry>,
}

#[derive(Debug, Default)]
pub struct DocumentVectorRegistry {
    entries: RwLock<HashMap<u64, DocumentEntry>>,
    next_version: AtomicU64,
}

impl DocumentVectorRegistry {
    pub fn upsert(
        &self,
        doc_id: u64,
        fields: &[FieldEntry],
        metadata: HashMap<String, String>,
    ) -> Result<RegistryVersion> {
        let version = self.next_version.fetch_add(1, Ordering::SeqCst) + 1;
        let mut map = HashMap::new();
        for entry in fields {
            let mut cloned = entry.clone();
            cloned.version = version;
            map.insert(cloned.field_name.clone(), cloned);
        }

        let doc_entry = DocumentEntry {
            doc_id,
            version,
            metadata,
            fields: map,
        };

        self.entries.write().insert(doc_id, doc_entry);
        Ok(version)
    }

    pub fn delete(&self, doc_id: u64) -> Result<()> {
        let mut guard = self.entries.write();
        guard
            .remove(&doc_id)
            .ok_or_else(|| PlatypusError::not_found(format!("doc_id {doc_id}")))?;
        Ok(())
    }

    pub fn get(&self, doc_id: u64) -> Option<DocumentEntry> {
        self.entries.read().get(&doc_id).cloned()
    }

    pub fn snapshot(&self) -> Result<Vec<u8>> {
        let guard = self.entries.read();
        serde_json::to_vec(&*guard).map_err(PlatypusError::from)
    }

    pub fn from_snapshot(bytes: &[u8]) -> Result<Self> {
        if bytes.is_empty() {
            return Ok(Self::default());
        }

        let entries: HashMap<u64, DocumentEntry> = serde_json::from_slice(bytes)?;
        let max_version = entries
            .values()
            .map(|entry| entry.version)
            .max()
            .unwrap_or(0);
        Ok(Self {
            entries: RwLock::new(entries),
            next_version: AtomicU64::new(max_version),
        })
    }

    pub fn document_count(&self) -> usize {
        self.entries.read().len()
    }

    pub fn filter_matches(
        &self,
        filter: &VectorFilter,
        target_fields: &[String],
    ) -> RegistryFilterMatches {
        let guard = self.entries.read();
        let mut allowed_fields: HashMap<u64, HashSet<String>> = HashMap::new();

        for entry in guard.values() {
            if !filter.document.is_empty() && !filter.document.matches(&entry.metadata) {
                continue;
            }

            let mut matched_fields: HashSet<String> = HashSet::new();
            for field_name in target_fields {
                if let Some(field_entry) = entry.fields.get(field_name)
                    && (filter.field.is_empty() || filter.field.matches(&field_entry.metadata))
                {
                    matched_fields.insert(field_name.clone());
                }
            }

            if matched_fields.is_empty() {
                continue;
            }

            allowed_fields.insert(entry.doc_id, matched_fields);
        }

        RegistryFilterMatches { allowed_fields }
    }
}

pub fn build_field_entries(document: &DocumentVector) -> Vec<FieldEntry> {
    document
        .fields
        .iter()
        .map(|(name, vector)| FieldEntry {
            field_name: name.clone(),
            version: 0,
            vector_count: 1, // Always 1 in flattened model
            weight: vector.weight,
            metadata: vector.attributes.clone(),
        })
        .collect()
}
