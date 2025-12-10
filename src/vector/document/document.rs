//! ドキュメントレベルのベクトル型
//!
//! このモジュールはドキュメントベクトルとドキュメントペイロードを提供する。

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::vector::document::field::{FieldPayload, FieldVectors};

/// Document-level wrapper around field vectors and metadata (doc_id is supplied separately).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DocumentVector {
    #[serde(default)]
    pub fields: HashMap<String, FieldVectors>,
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

impl DocumentVector {
    pub fn new() -> Self {
        Self {
            fields: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    pub fn add_field<V: Into<String>>(&mut self, field_name: V, field: FieldVectors) {
        self.fields.insert(field_name.into(), field);
    }

    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }
}

/// Document input model capturing raw payloads before embedding.
#[derive(Debug, Clone, Default)]
pub struct DocumentPayload {
    pub fields: HashMap<String, FieldPayload>,
    pub metadata: HashMap<String, String>,
}

impl DocumentPayload {
    pub fn new() -> Self {
        Self {
            fields: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    pub fn add_field(&mut self, field_name: impl Into<String>, payload: FieldPayload) {
        self.fields.insert(field_name.into(), payload);
    }

    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }
}
