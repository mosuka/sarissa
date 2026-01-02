//! VectorEngine インメモリフィールド実装
//!
//! このモジュールはインメモリでベクトルを管理するフィールド実装を提供する。

use std::cmp::Ordering as CmpOrdering;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::sync::Arc;

use parking_lot::RwLock;

use crate::error::{Result, SarissaError};
use crate::vector::Vector;
use crate::vector::core::document::StoredVector;
use crate::vector::engine::config::VectorFieldConfig;
use crate::vector::field::{
    FieldHit, FieldSearchInput, FieldSearchResults, VectorField, VectorFieldReader,
    VectorFieldStats, VectorFieldWriter,
};

#[derive(Clone, Debug)]
pub(crate) struct FieldHandle {
    pub(crate) field: Arc<dyn VectorField>,
    pub(crate) runtime: Arc<FieldRuntime>,
}

#[derive(Debug)]
pub(crate) struct FieldRuntime {
    default_reader: Arc<dyn VectorFieldReader>,
    current_reader: RwLock<Arc<dyn VectorFieldReader>>,
    writer: Arc<dyn VectorFieldWriter>,
}

impl FieldRuntime {
    pub(crate) fn new(
        reader: Arc<dyn VectorFieldReader>,
        writer: Arc<dyn VectorFieldWriter>,
    ) -> Self {
        Self {
            current_reader: RwLock::new(reader.clone()),
            default_reader: reader,
            writer,
        }
    }

    pub(crate) fn from_field(field: &Arc<dyn VectorField>) -> Arc<Self> {
        Arc::new(Self::new(field.reader_handle(), field.writer_handle()))
    }

    pub(crate) fn reader(&self) -> Arc<dyn VectorFieldReader> {
        self.current_reader.read().clone()
    }

    pub(crate) fn writer(&self) -> Arc<dyn VectorFieldWriter> {
        self.writer.clone()
    }

    pub(crate) fn replace_reader(
        &self,
        reader: Arc<dyn VectorFieldReader>,
    ) -> Arc<dyn VectorFieldReader> {
        let mut guard = self.current_reader.write();
        std::mem::replace(&mut *guard, reader)
    }

    pub(crate) fn reset_reader(&self) -> Arc<dyn VectorFieldReader> {
        self.replace_reader(self.default_reader.clone())
    }
}

#[derive(Debug)]
pub(crate) struct InMemoryVectorField {
    name: String,
    config: VectorFieldConfig,
    store: Arc<FieldStore>,
    writer: Arc<InMemoryFieldWriter>,
    reader: Arc<InMemoryFieldReader>,
}

impl InMemoryVectorField {
    pub(crate) fn new(
        name: String,
        config: VectorFieldConfig,
        delegate: Option<Arc<dyn VectorFieldWriter>>,
    ) -> Result<Self> {
        let store = Arc::new(FieldStore::default());
        let writer = Arc::new(InMemoryFieldWriter::new(
            name.clone(),
            config.clone(),
            store.clone(),
            delegate,
        ));
        let reader = Arc::new(InMemoryFieldReader::new(
            name.clone(),
            config.clone(),
            store.clone(),
        ));
        Ok(Self {
            name,
            config,
            store,
            writer,
            reader,
        })
    }

    pub(crate) fn vector_tuples(&self) -> Vec<(u64, String, Vector)> {
        self.store.vector_tuples(&self.name)
    }
}

impl VectorField for InMemoryVectorField {
    fn name(&self) -> &str {
        &self.name
    }

    fn config(&self) -> &VectorFieldConfig {
        &self.config
    }

    fn writer(&self) -> &dyn VectorFieldWriter {
        self.writer.as_ref()
    }

    fn reader(&self) -> &dyn VectorFieldReader {
        self.reader.as_ref()
    }

    fn writer_handle(&self) -> Arc<dyn VectorFieldWriter> {
        self.writer.clone()
    }

    fn reader_handle(&self) -> Arc<dyn VectorFieldReader> {
        self.reader.clone()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[derive(Debug)]
pub(crate) struct InMemoryFieldWriter {
    field_name: String,
    config: VectorFieldConfig,
    store: Arc<FieldStore>,
    delegate: Option<Arc<dyn VectorFieldWriter>>,
}

impl InMemoryFieldWriter {
    fn new(
        field_name: String,
        config: VectorFieldConfig,
        store: Arc<FieldStore>,
        delegate: Option<Arc<dyn VectorFieldWriter>>,
    ) -> Self {
        Self {
            field_name,
            config,
            store,
            delegate,
        }
    }

    fn convert_vector(&self, stored: &StoredVector) -> Result<Vector> {
        let vector = stored.to_vector();
        if vector.dimension() != self.config.dimension {
            return Err(SarissaError::invalid_argument(format!(
                "vector dimension mismatch for field '{}': expected {}, got {}",
                self.field_name,
                self.config.dimension,
                vector.dimension()
            )));
        }
        if !vector.is_valid() {
            return Err(SarissaError::invalid_argument(format!(
                "vector for field '{}' contains invalid values",
                self.field_name
            )));
        }
        Ok(vector)
    }
}

impl VectorFieldWriter for InMemoryFieldWriter {
    fn add_stored_vector(&self, doc_id: u64, vector: &StoredVector, version: u64) -> Result<()> {
        if let Some(delegate) = &self.delegate {
            delegate.add_stored_vector(doc_id, vector, version)?;
        }

        let converted = self.convert_vector(vector)?;
        self.store.replace(
            doc_id,
            FieldStoreEntry {
                vectors: vec![converted],
            },
        );
        Ok(())
    }

    fn has_storage(&self) -> bool {
        // InMemoryFieldWriter usually doesn't have disk storage in the same sense
        if let Some(delegate) = &self.delegate {
            delegate.has_storage()
        } else {
            false
        }
    }

    fn rebuild(&self, vectors: Vec<(u64, String, Vector)>) -> Result<()> {
        let vectors_clone = vectors.clone();

        if let Some(delegate) = &self.delegate {
            delegate.rebuild(vectors)?;
        }

        let mut guard = self.store.entries.write();
        guard.clear();
        for (doc_id, _, vector) in vectors_clone {
            guard.insert(
                doc_id,
                FieldStoreEntry {
                    vectors: vec![vector],
                },
            );
        }
        Ok(())
    }

    fn vectors(&self) -> Vec<(u64, String, Vector)> {
        // InMemoryFieldWriter stores vectors in self.store/field_store
        // But the trait expects a slice reference, which we can't easily return if it's in a HashMap/BTreeMap
        // For now, delegate or return empty.
        // If we want accurate Vacuum, we need access.
        // However, optimize() usually targets the persistent storage (delegate).
        if let Some(delegate) = &self.delegate {
            delegate.vectors()
        } else {
            Vec::new()
        }
    }

    fn delete_document(&self, doc_id: u64, version: u64) -> Result<()> {
        self.store.remove(doc_id);
        if let Some(delegate) = &self.delegate {
            delegate.delete_document(doc_id, version)?;
        }
        Ok(())
    }

    fn flush(&self) -> Result<()> {
        if let Some(delegate) = &self.delegate {
            delegate.flush()?;
        }
        Ok(())
    }

    fn optimize(&self) -> Result<()> {
        if let Some(delegate) = &self.delegate {
            // Rebuild delegate using vectors from RAM store
            // This ensures delegate reflects the current in-memory state
            let vectors = self.store.vector_tuples(&self.field_name);
            delegate.rebuild(vectors)?;
            delegate.flush()?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub(crate) struct InMemoryFieldReader {
    field_name: String,
    config: VectorFieldConfig,
    store: Arc<FieldStore>,
}

impl InMemoryFieldReader {
    fn new(field_name: String, config: VectorFieldConfig, store: Arc<FieldStore>) -> Self {
        Self {
            field_name,
            config,
            store,
        }
    }
}

impl VectorFieldReader for InMemoryFieldReader {
    fn search(&self, request: FieldSearchInput) -> Result<FieldSearchResults> {
        if request.field != self.field_name {
            return Err(SarissaError::invalid_argument(format!(
                "field mismatch: expected '{}', got '{}'",
                self.field_name, request.field
            )));
        }

        if request.query_vectors.is_empty() {
            return Ok(FieldSearchResults::default());
        }

        let snapshot = self.store.snapshot();
        let mut merged: HashMap<u64, FieldHit> = HashMap::new();

        for query in &request.query_vectors {
            let query_vector = query.vector.to_vector();
            if query_vector.dimension() != self.config.dimension {
                return Err(SarissaError::invalid_argument(format!(
                    "query vector dimension mismatch for field '{}': expected {}, got {}",
                    self.field_name,
                    self.config.dimension,
                    query_vector.dimension()
                )));
            }
            let effective_weight = query.weight * query.vector.weight;
            if effective_weight == 0.0 {
                continue;
            }

            for (doc_id, entry) in &snapshot {
                for vector in &entry.vectors {
                    let similarity = self
                        .config
                        .distance
                        .similarity(&query_vector.data, &vector.data)?;
                    let weighted_score = similarity * effective_weight;
                    let distance = self
                        .config
                        .distance
                        .distance(&query_vector.data, &vector.data)?;

                    match merged.entry(*doc_id) {
                        Entry::Vacant(slot) => {
                            slot.insert(FieldHit {
                                doc_id: *doc_id,
                                field: self.field_name.clone(),
                                score: weighted_score,
                                distance,
                                metadata: vector.metadata.clone(),
                            });
                        }
                        Entry::Occupied(mut slot) => {
                            let hit = slot.get_mut();
                            hit.score += weighted_score;
                            hit.distance = hit.distance.min(distance);
                            if hit.metadata.is_empty() {
                                hit.metadata = vector.metadata.clone();
                            }
                        }
                    }
                }
            }
        }

        let mut hits: Vec<FieldHit> = merged.into_values().collect();
        hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(CmpOrdering::Equal));
        if hits.len() > request.limit {
            hits.truncate(request.limit);
        }

        Ok(FieldSearchResults { hits })
    }

    fn stats(&self) -> Result<VectorFieldStats> {
        Ok(VectorFieldStats {
            vector_count: self.store.total_vectors(),
            dimension: self.config.dimension,
        })
    }
}

#[derive(Debug, Default)]
pub(crate) struct FieldStore {
    entries: RwLock<HashMap<u64, FieldStoreEntry>>,
}

impl FieldStore {
    pub(crate) fn replace(&self, doc_id: u64, entry: FieldStoreEntry) {
        self.entries.write().insert(doc_id, entry);
    }

    pub(crate) fn remove(&self, doc_id: u64) {
        self.entries.write().remove(&doc_id);
    }

    pub(crate) fn snapshot(&self) -> HashMap<u64, FieldStoreEntry> {
        self.entries.read().clone()
    }

    pub(crate) fn total_vectors(&self) -> usize {
        self.entries
            .read()
            .values()
            .map(|entry| entry.vectors.len())
            .sum()
    }

    pub(crate) fn vector_tuples(&self, field_name: &str) -> Vec<(u64, String, Vector)> {
        let guard = self.entries.read();
        let mut tuples = Vec::new();
        let name = field_name.to_string();
        for (doc_id, entry) in guard.iter() {
            for vector in &entry.vectors {
                tuples.push((*doc_id, name.clone(), vector.clone()));
            }
        }
        tuples
    }
}

#[derive(Debug, Clone)]
pub(crate) struct FieldStoreEntry {
    pub(crate) vectors: Vec<Vector>,
}
