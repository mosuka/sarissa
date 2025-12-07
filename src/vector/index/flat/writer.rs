//! Flat vector index builder for exact search.

use std::sync::Arc;

use rayon::prelude::*;

use crate::error::{PlatypusError, Result};
use crate::storage::Storage;
use crate::vector::core::vector::Vector;
use crate::vector::index::FlatIndexConfig;
use crate::vector::index::field::LegacyVectorFieldWriter;
use crate::vector::index::io::{read_metadata, write_metadata};
use crate::vector::writer::{VectorIndexWriter, VectorIndexWriterConfig};

/// Builder for flat vector indexes (exact search).
#[derive(Debug)]
pub struct FlatIndexWriter {
    index_config: FlatIndexConfig,
    writer_config: VectorIndexWriterConfig,
    storage: Option<Arc<dyn Storage>>,
    vectors: Vec<(u64, String, Vector)>,
    is_finalized: bool,
    total_vectors_to_add: Option<usize>,
    next_vec_id: u64,
}

impl FlatIndexWriter {
    /// Create a new flat vector index builder.
    pub fn new(
        index_config: FlatIndexConfig,
        writer_config: VectorIndexWriterConfig,
    ) -> Result<Self> {
        Ok(Self {
            index_config,
            writer_config,
            storage: None,
            vectors: Vec::new(),
            is_finalized: false,
            total_vectors_to_add: None,
            next_vec_id: 0,
        })
    }

    /// Create a new flat vector index builder with storage.
    pub fn with_storage(
        index_config: FlatIndexConfig,
        writer_config: VectorIndexWriterConfig,
        storage: Arc<dyn Storage>,
    ) -> Result<Self> {
        Ok(Self {
            index_config,
            writer_config,
            storage: Some(storage),
            vectors: Vec::new(),
            is_finalized: false,
            total_vectors_to_add: None,
            next_vec_id: 0,
        })
    }

    /// Convert this writer into a doc-centric field writer adapter.
    pub fn into_field_writer(self, field_name: impl Into<String>) -> LegacyVectorFieldWriter<Self> {
        LegacyVectorFieldWriter::new(field_name, self)
    }

    /// Load an existing flat vector index from storage.
    pub fn load(
        index_config: FlatIndexConfig,
        writer_config: VectorIndexWriterConfig,
        storage: Arc<dyn Storage>,
        path: &str,
    ) -> Result<Self> {
        use std::io::Read;

        // Open the index file
        let file_name = format!("{}.flat", path);
        let mut input = storage.open_input(&file_name)?;

        // Read metadata
        let mut num_vectors_buf = [0u8; 4];
        input.read_exact(&mut num_vectors_buf)?;
        let num_vectors = u32::from_le_bytes(num_vectors_buf) as usize;

        let mut dimension_buf = [0u8; 4];
        input.read_exact(&mut dimension_buf)?;
        let dimension = u32::from_le_bytes(dimension_buf) as usize;

        if dimension != index_config.dimension {
            return Err(PlatypusError::InvalidOperation(format!(
                "Dimension mismatch: expected {}, found {}",
                index_config.dimension, dimension
            )));
        }

        // Read vectors with field names
        let mut vectors = Vec::with_capacity(num_vectors);
        for _ in 0..num_vectors {
            let mut doc_id_buf = [0u8; 8];
            input.read_exact(&mut doc_id_buf)?;
            let doc_id = u64::from_le_bytes(doc_id_buf);

            // Read field name
            let mut field_name_len_buf = [0u8; 4];
            input.read_exact(&mut field_name_len_buf)?;
            let field_name_len = u32::from_le_bytes(field_name_len_buf) as usize;

            let mut field_name_buf = vec![0u8; field_name_len];
            input.read_exact(&mut field_name_buf)?;
            let field_name = String::from_utf8(field_name_buf).map_err(|e| {
                PlatypusError::InvalidOperation(format!("Invalid UTF-8 in field name: {}", e))
            })?;

            // Read metadata and vector data
            let metadata = read_metadata(&mut input)?;
            let mut values = vec![0.0f32; dimension];
            for value in &mut values {
                let mut value_buf = [0u8; 4];
                input.read_exact(&mut value_buf)?;
                *value = f32::from_le_bytes(value_buf);
            }

            vectors.push((doc_id, field_name, Vector::with_metadata(values, metadata)));
        }

        // Calculate next_vec_id from loaded vectors
        let max_id = vectors.iter().map(|(id, _, _)| *id).max().unwrap_or(0);
        let next_vec_id = if num_vectors > 0 { max_id + 1 } else { 0 };

        Ok(Self {
            index_config,
            writer_config,
            storage: Some(storage),
            vectors,
            is_finalized: true,
            total_vectors_to_add: Some(num_vectors),
            next_vec_id,
        })
    }

    /// Set the expected total number of vectors (for progress tracking).
    pub fn set_expected_vector_count(&mut self, count: usize) {
        self.total_vectors_to_add = Some(count);
    }

    /// Get the stored vectors (for testing/debugging).
    pub fn vectors(&self) -> &[(u64, String, Vector)] {
        &self.vectors
    }

    /// Validate vectors before adding them.
    fn validate_vectors(&self, vectors: &[(u64, String, Vector)]) -> Result<()> {
        if vectors.is_empty() {
            return Ok(());
        }

        // Check dimensions
        for (doc_id, _field_name, vector) in vectors {
            if vector.dimension() != self.index_config.dimension {
                return Err(PlatypusError::InvalidOperation(format!(
                    "Vector {} has dimension {}, expected {}",
                    doc_id,
                    vector.dimension(),
                    self.index_config.dimension
                )));
            }

            if !vector.is_valid() {
                return Err(PlatypusError::InvalidOperation(format!(
                    "Vector {doc_id} contains invalid values (NaN or infinity)"
                )));
            }
        }

        Ok(())
    }

    /// Normalize vectors if configured to do so.
    fn normalize_vectors(&self, vectors: &mut [(u64, String, Vector)]) {
        if !self.index_config.normalize_vectors {
            return;
        }

        if self.writer_config.parallel_build && vectors.len() > 100 {
            vectors.par_iter_mut().for_each(|(_, _, vector)| {
                vector.normalize();
            });
        } else {
            for (_, _, vector) in vectors {
                vector.normalize();
            }
        }
    }

    /// Check for memory limits.
    fn check_memory_limit(&self) -> Result<()> {
        if let Some(limit) = self.writer_config.memory_limit {
            let current_usage = self.estimated_memory_usage();
            if current_usage > limit {
                return Err(PlatypusError::ResourceExhausted(format!(
                    "Memory usage {current_usage} bytes exceeds limit {limit} bytes"
                )));
            }
        }
        Ok(())
    }

    /// Sort vectors by document ID and field name for better cache locality.
    fn sort_vectors(&mut self) {
        if self.writer_config.parallel_build && self.vectors.len() as u64 > 10000 {
            self.vectors
                .par_sort_by(|(doc_id_a, field_a, _), (doc_id_b, field_b, _)| {
                    doc_id_a.cmp(doc_id_b).then_with(|| field_a.cmp(field_b))
                });
        } else {
            self.vectors
                .sort_by(|(doc_id_a, field_a, _), (doc_id_b, field_b, _)| {
                    doc_id_a.cmp(doc_id_b).then_with(|| field_a.cmp(field_b))
                });
        }
    }

    /// Remove duplicate vectors (keeping the last one).
    fn deduplicate_vectors(&mut self) {
        if self.vectors.is_empty() {
            return;
        }

        // Sort first to group duplicates
        self.sort_vectors();

        // Remove duplicates, keeping the last occurrence
        let mut unique_vectors = Vec::new();
        let mut last_key: Option<(u64, String)> = None;

        for (doc_id, field_name, vector) in std::mem::take(&mut self.vectors) {
            let current_key = (doc_id, field_name.clone());
            if last_key.as_ref() != Some(&current_key) {
                unique_vectors.push((doc_id, field_name, vector));
                last_key = Some(current_key);
            } else {
                // Replace with newer vector
                if let Some((_, _, last_vector)) = unique_vectors.last_mut() {
                    *last_vector = vector;
                }
            }
        }

        self.vectors = unique_vectors;
    }
}

#[async_trait::async_trait]
impl VectorIndexWriter for FlatIndexWriter {
    fn next_vector_id(&self) -> u64 {
        self.next_vec_id
    }

    async fn add_document(
        &mut self,
        doc: crate::lexical::document::document::Document,
    ) -> Result<u64> {
        use crate::embedding::per_field::PerFieldEmbedder;
        use crate::lexical::document::field::{FieldOption, FieldValue};

        let doc_id = self.next_vec_id;
        let mut vectors = Vec::new();

        for (field_name, field) in doc.fields().iter() {
            // Check if this is a vector field and if it should be indexed
            if let FieldValue::Vector(text) = &field.value {
                let (should_index, store_value) = match &field.option {
                    FieldOption::Vector(opt) => (
                        opt.flat.is_some() || opt.hnsw.is_some() || opt.ivf.is_some(),
                        opt.stored,
                    ),
                    _ => (false, false),
                };

                if !should_index {
                    continue;
                }

                // Check if embedder is PerFieldEmbedder for field-specific embedding
                let mut vector = if let Some(per_field) = self
                    .index_config
                    .embedder
                    .as_any()
                    .downcast_ref::<PerFieldEmbedder>()
                {
                    per_field.embed_field(field_name, text.as_str()).await?
                } else {
                    self.index_config.embedder.embed(text.as_str()).await?
                };

                if store_value {
                    vector.set_original_text(text.clone());
                }

                vectors.push((doc_id, field_name.clone(), vector));
            }
        }

        if !vectors.is_empty() {
            self.add_vectors(vectors)?;
        }

        self.next_vec_id += 1;
        Ok(doc_id)
    }

    async fn add_document_with_id(
        &mut self,
        doc_id: u64,
        doc: crate::lexical::document::document::Document,
    ) -> Result<()> {
        use crate::embedding::per_field::PerFieldEmbedder;
        use crate::lexical::document::field::{FieldOption, FieldValue};

        let mut vectors = Vec::new();

        for (field_name, field) in doc.fields().iter() {
            // Check if this is a vector field and if it should be indexed
            if let FieldValue::Vector(text) = &field.value {
                let (should_index, store_value) = match &field.option {
                    FieldOption::Vector(opt) => (
                        opt.flat.is_some() || opt.hnsw.is_some() || opt.ivf.is_some(),
                        opt.stored,
                    ),
                    _ => (false, false),
                };

                if !should_index {
                    continue;
                }

                // Check if embedder is PerFieldEmbedder for field-specific embedding
                let mut vector = if let Some(per_field) = self
                    .index_config
                    .embedder
                    .as_any()
                    .downcast_ref::<PerFieldEmbedder>()
                {
                    per_field.embed_field(field_name, text.as_str()).await?
                } else {
                    self.index_config.embedder.embed(text.as_str()).await?
                };

                if store_value {
                    vector.set_original_text(text.clone());
                }

                vectors.push((doc_id, field_name.clone(), vector));
            }
        }

        if !vectors.is_empty() {
            self.add_vectors(vectors)?;
        }

        // Update next_vec_id if necessary
        if doc_id >= self.next_vec_id {
            self.next_vec_id = doc_id + 1;
        }

        Ok(())
    }

    fn build(&mut self, mut vectors: Vec<(u64, String, Vector)>) -> Result<()> {
        if self.is_finalized {
            return Err(PlatypusError::InvalidOperation(
                "Cannot build on finalized index".to_string(),
            ));
        }

        self.validate_vectors(&vectors)?;
        self.normalize_vectors(&mut vectors);

        // Update next_vec_id
        if let Some(max_id) = vectors.iter().map(|(id, _, _)| *id).max()
            && max_id >= self.next_vec_id
        {
            self.next_vec_id = max_id + 1;
        }

        self.vectors = vectors;
        self.total_vectors_to_add = Some(self.vectors.len());

        self.check_memory_limit()?;
        Ok(())
    }

    fn add_vectors(&mut self, mut vectors: Vec<(u64, String, Vector)>) -> Result<()> {
        if self.is_finalized {
            return Err(PlatypusError::InvalidOperation(
                "Cannot add vectors to finalized index".to_string(),
            ));
        }

        self.validate_vectors(&vectors)?;
        self.normalize_vectors(&mut vectors);

        // Update next_vec_id
        if let Some(max_id) = vectors.iter().map(|(id, _, _)| *id).max()
            && max_id >= self.next_vec_id
        {
            self.next_vec_id = max_id + 1;
        }

        self.vectors.extend(vectors);
        self.check_memory_limit()?;
        Ok(())
    }

    fn finalize(&mut self) -> Result<()> {
        if self.is_finalized {
            return Ok(());
        }

        // Remove duplicates and sort
        self.deduplicate_vectors();
        self.sort_vectors();

        self.is_finalized = true;
        Ok(())
    }

    fn progress(&self) -> f32 {
        if let Some(total) = self.total_vectors_to_add {
            if total == 0 {
                if self.is_finalized { 1.0 } else { 0.0 }
            } else {
                let current = self.vectors.len() as u64 as f32;
                let progress = current / total as f32;
                if self.is_finalized {
                    1.0
                } else {
                    progress.min(0.99) // Never report 100% until finalized
                }
            }
        } else if self.is_finalized {
            1.0
        } else {
            0.0
        }
    }

    fn estimated_memory_usage(&self) -> usize {
        let vector_memory = self.vectors.len()
            * (
                8 + // doc_id
            self.index_config.dimension * 4 + // f32 values
            std::mem::size_of::<Vector>()
                // Vector struct overhead
            );

        let metadata_memory = self.vectors.len() * 64; // Rough estimate for metadata

        vector_memory + metadata_memory
    }

    fn optimize(&mut self) -> Result<()> {
        if !self.is_finalized {
            return Err(PlatypusError::InvalidOperation(
                "Index must be finalized before optimization".to_string(),
            ));
        }

        // For flat index, optimization just means ensuring everything is sorted
        // and we've removed duplicates (already done in finalize)

        // Could also include memory compaction here
        self.vectors.shrink_to_fit();

        Ok(())
    }

    fn vectors(&self) -> &[(u64, String, Vector)] {
        &self.vectors
    }

    fn write(&self, path: &str) -> Result<()> {
        use std::io::Write;

        if !self.is_finalized {
            return Err(PlatypusError::InvalidOperation(
                "Index must be finalized before writing".to_string(),
            ));
        }

        let storage = self
            .storage
            .as_ref()
            .ok_or_else(|| PlatypusError::InvalidOperation("No storage configured".to_string()))?;

        // Create the index file
        let file_name = format!("{}.flat", path);
        let mut output = storage.create_output(&file_name)?;

        // Write metadata
        output.write_all(&(self.vectors.len() as u64 as u32).to_le_bytes())?;
        output.write_all(&(self.index_config.dimension as u32).to_le_bytes())?;

        // Write vectors with field names and metadata
        for (doc_id, field_name, vector) in &self.vectors {
            output.write_all(&doc_id.to_le_bytes())?;

            // Write field name length and field name
            let field_name_bytes = field_name.as_bytes();
            output.write_all(&(field_name_bytes.len() as u32).to_le_bytes())?;
            output.write_all(field_name_bytes)?;

            write_metadata(&mut output, &vector.metadata)?;

            // Write vector data
            for value in &vector.data {
                output.write_all(&value.to_le_bytes())?;
            }
        }

        output.flush()?;
        Ok(())
    }

    fn has_storage(&self) -> bool {
        self.storage.is_some()
    }

    fn delete_documents(&mut self, field: &str, value: &str) -> Result<u64> {
        // For flat index, we match based on field name and vector similarity
        // In practice, this would require metadata storage to match text values
        // For now, return 0 as this is a simplified implementation
        // TODO: Implement proper deletion with metadata storage
        let _field = field;
        let _value = value;
        Ok(0)
    }

    async fn update_document(
        &mut self,
        field: &str,
        value: &str,
        doc: crate::lexical::document::document::Document,
    ) -> Result<()> {
        // Delete existing documents
        self.delete_documents(field, value)?;

        // Add new document
        self.add_document(doc).await?;

        Ok(())
    }

    fn rollback(&mut self) -> Result<()> {
        // Clear pending vectors and reset state
        self.vectors.clear();
        self.is_finalized = false;
        self.next_vec_id = 0;
        Ok(())
    }

    fn pending_docs(&self) -> u64 {
        if self.is_finalized {
            0
        } else {
            self.vectors.len() as u64
        }
    }

    fn close(&mut self) -> Result<()> {
        // Clear all data and mark as closed
        self.vectors.clear();
        self.is_finalized = true;
        Ok(())
    }

    fn is_closed(&self) -> bool {
        // Consider closed if finalized and no pending vectors
        self.is_finalized && self.vectors.is_empty()
    }
}
