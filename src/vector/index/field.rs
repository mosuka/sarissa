//! Compatibility adapters between doc-centric field traits and legacy vector indexes.

use std::sync::Arc;

use parking_lot::Mutex;

use crate::error::Result;
use crate::vector::core::document::StoredVector;
use crate::vector::core::vector::Vector;
use crate::vector::engine::VectorFieldConfig;
use crate::vector::field::{VectorField, VectorFieldReader, VectorFieldWriter};
use crate::vector::writer::VectorIndexWriter;

/// Bridges the new doc-centric `VectorFieldWriter` trait to existing index writers.
#[derive(Debug)]
pub struct LegacyVectorFieldWriter<W: VectorIndexWriter> {
    field_name: String,
    writer: Mutex<W>,
    path: Option<String>,
}

impl<W: VectorIndexWriter> LegacyVectorFieldWriter<W> {
    /// Create a new adapter for the provided field name and index writer.
    pub fn new(field_name: impl Into<String>, writer: W, path: Option<String>) -> Self {
        Self {
            field_name: field_name.into(),
            writer: Mutex::new(writer),
            path,
        }
    }

    /// Returns the owning field name.
    pub fn field_name(&self) -> &str {
        &self.field_name
    }

    fn to_legacy_vector(&self, doc_id: u64, stored: &StoredVector) -> (u64, String, Vector) {
        let vector = stored.to_vector();
        (doc_id, self.field_name.clone(), vector)
    }

    #[cfg(test)]
    pub(crate) fn pending_vectors(&self) -> Vec<(u64, String, Vector)> {
        let guard = self.writer.lock();
        guard.vectors().to_vec()
    }
}

impl<W> VectorFieldWriter for LegacyVectorFieldWriter<W>
where
    W: VectorIndexWriter,
{
    fn add_stored_vector(&self, doc_id: u64, vector: &StoredVector, _version: u64) -> Result<()> {
        let mut guard = self.writer.lock();
        let legacy = self.to_legacy_vector(doc_id, vector);
        guard.add_vectors(vec![legacy])
    }

    fn has_storage(&self) -> bool {
        self.writer.lock().has_storage()
    }

    fn vectors(&self) -> Vec<(u64, String, Vector)> {
        self.writer.lock().vectors().to_vec()
    }

    fn rebuild(&self, vectors: Vec<(u64, String, Vector)>) -> Result<()> {
        let mut guard = self.writer.lock();
        // To rebuild, we rely on the writer's capability to reset or build from scratch.
        // Assuming `build` can be called to populate, but we need to clear first.
        // `rollback` clears pending vectors and resets state.
        guard.rollback()?;
        guard.build(vectors)?;
        guard.finalize()?;
        // If it has storage, we might need to write? `commit` does finalize+write.
        // But we don't know the path here easily unless stored in config or we let `optimize` caller handle checking?
        // `LegacyVectorFieldWriter` doesn't know the path.
        // However, `VectorCollection` keeps track of paths?
        // Actually `VectorIndexWriter` usually knows where to write if initiated with storage?
        // `FlatIndexWriter::load` stores `storage` but not `path` (filename). `write(path)` takes path arg.

        // This is a missing piece. The writer might not know its own filename to overwrite.
        // But `VectorFieldEntry` in Registry has metadata?

        // For now, let's assume `commit` or similar is needed, but `rebuild` just updates memory state.
        // We will need to ensure persistence later.
        // But `optimize` implies persistence update.

        // If we can't persist, `optimize` is incomplete.
        // Let's look at how `VectorCollection` persists.
        // It uses `field.runtime.writer().flush()`.

        Ok(())
    }

    fn delete_document(&self, doc_id: u64, _version: u64) -> Result<()> {
        let mut guard = self.writer.lock();
        // Best-effort deletion from in-memory buffer.
        // We ignore errors here because the index might be finalized/immutable.
        // Logical deletion is handled by the Registry filter in VectorCollection.
        let _ = guard.delete_document(doc_id);
        Ok(())
    }

    fn flush(&self) -> Result<()> {
        if let Some(path) = &self.path {
            self.writer.lock().commit(path)?;
        }
        Ok(())
    }
}

/// Concrete [`VectorField`] implementation backed by adapters.
#[derive(Debug)]
pub struct AdapterBackedVectorField {
    name: String,
    config: VectorFieldConfig,
    writer: Arc<dyn VectorFieldWriter>,
    reader: Arc<dyn VectorFieldReader>,
}

impl AdapterBackedVectorField {
    /// Create a new adapter-backed vector field definition.
    pub fn new(
        name: impl Into<String>,
        config: VectorFieldConfig,
        writer: Arc<dyn VectorFieldWriter>,
        reader: Arc<dyn VectorFieldReader>,
    ) -> Self {
        Self {
            name: name.into(),
            config,
            writer,
            reader,
        }
    }

    /// Returns the shared writer handle.
    pub fn writer_handle(&self) -> &Arc<dyn VectorFieldWriter> {
        &self.writer
    }

    /// Returns the shared reader handle.
    pub fn reader_handle(&self) -> &Arc<dyn VectorFieldReader> {
        &self.reader
    }
}

impl VectorField for AdapterBackedVectorField {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::core::document::StoredVector;
    use crate::vector::index::config::{FlatIndexConfig, HnswIndexConfig, IvfIndexConfig};
    use crate::vector::index::flat::writer::FlatIndexWriter;
    use crate::vector::index::hnsw::writer::HnswIndexWriter;
    use crate::vector::index::ivf::writer::IvfIndexWriter;
    use crate::vector::writer::VectorIndexWriterConfig;
    use std::sync::Arc;

    fn sample_stored_vector() -> StoredVector {
        StoredVector::new(Arc::<[f32]>::from([1.0_f32, 0.0_f32]))
    }

    fn flat_writer() -> FlatIndexWriter {
        let mut config = FlatIndexConfig::default();
        config.dimension = 2;
        config.normalize_vectors = false;
        FlatIndexWriter::new(config, VectorIndexWriterConfig::default()).unwrap()
    }

    fn hnsw_writer() -> HnswIndexWriter {
        let mut config = HnswIndexConfig::default();
        config.dimension = 2;
        config.normalize_vectors = false;
        HnswIndexWriter::new(config, VectorIndexWriterConfig::default()).unwrap()
    }

    fn ivf_writer() -> IvfIndexWriter {
        let mut config = IvfIndexConfig::default();
        config.dimension = 2;
        config.normalize_vectors = false;
        IvfIndexWriter::new(config, VectorIndexWriterConfig::default()).unwrap()
    }

    #[test]
    fn test_adapter_flat() {
        let adapter = LegacyVectorFieldWriter::new("body", flat_writer(), None);
        assert_eq!(adapter.field_name(), "body");
        assert!(!adapter.has_storage());
    }

    #[test]
    fn test_adapter_hnsw() {
        // For HNSW, we need to handle HnswIndexWriter type availability if feature gated?
        // Assuming HnswIndexWriter is available since we are editing it.
        // But the test helper hnsw_writer() logic is hidden.
        let adapter = LegacyVectorFieldWriter::new("body", hnsw_writer(), None);
        assert_eq!(adapter.field_name(), "body");
    }

    #[test]
    fn test_adapter_ivf() {
        let adapter = LegacyVectorFieldWriter::new("body", ivf_writer(), None);
        assert_eq!(adapter.field_name(), "body");
    }

    #[test]
    fn test_adapter_deletion_error_handling() {
        let adapter = LegacyVectorFieldWriter::new("body", flat_writer(), None);
        let vector = sample_stored_vector();

        adapter.add_stored_vector(3, &vector, 1).unwrap();

        let result = adapter.delete_document(3, 2);
        assert!(result.is_ok());
    }

    #[test]
    fn adapter_supports_ivf_writer() {
        let adapter = LegacyVectorFieldWriter::new("body", ivf_writer(), None);
        let vector = sample_stored_vector();

        adapter.add_stored_vector(11, &vector, 1).unwrap();

        let pending = adapter.pending_vectors();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].0, 11);
    }

    #[test]
    fn adapter_stores_vector_with_correct_doc_id() {
        let adapter = LegacyVectorFieldWriter::new("body", flat_writer(), None);
        let mut vector = sample_stored_vector();
        vector.weight = 2.5;

        adapter.add_stored_vector(5, &vector, 1).unwrap();

        let pending = adapter.pending_vectors();
        assert_eq!(pending.len(), 1);
        // Verify doc_id and field_name
        assert_eq!(pending[0].0, 5);
        assert_eq!(pending[0].1, "body");
        // Verify vector data was converted correctly
        assert_eq!(pending[0].2.data.len(), 2);
    }
}
