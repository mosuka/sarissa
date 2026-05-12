//! Flat vector index builder for exact search.

use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use crate::error::{LaurusError, Result};
use crate::storage::Storage;
use crate::vector::core::quantization::ScalarQuantParams;
use crate::vector::core::vector::Vector;
use crate::vector::index::FlatIndexConfig;
use crate::vector::index::field::LegacyVectorFieldWriter;
use crate::vector::index::format::{QuantHeader, VectorSegmentHeader};
use crate::vector::index::quantized_io::{
    quantize_segment, read_dequantized_vector, write_quantized_record,
};
use crate::vector::writer::{VectorIndexWriter, VectorIndexWriterConfig};

/// Builder for flat vector indexes (exact search).
#[derive(Debug)]
pub struct FlatIndexWriter {
    index_config: FlatIndexConfig,
    writer_config: VectorIndexWriterConfig,
    storage: Option<Arc<dyn Storage>>,
    path: String,
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
        path: impl Into<String>,
    ) -> Result<Self> {
        Ok(Self {
            index_config,
            writer_config,
            storage: None,
            path: path.into(),
            vectors: Vec::new(),
            is_finalized: false,
            total_vectors_to_add: None,
            next_vec_id: 0,
        })
    }

    /// Create a new flat vector index builder with storage.
    /// Create a new flat vector index builder with storage.
    ///
    /// If an existing index file is found on disk, its vectors are loaded
    /// into the writer so that the next commit preserves them. This
    /// prevents data loss across multiple commit cycles.
    pub fn with_storage(
        index_config: FlatIndexConfig,
        writer_config: VectorIndexWriterConfig,
        path: impl Into<String>,
        storage: Arc<dyn Storage>,
    ) -> Result<Self> {
        let path = path.into();
        let file_name = format!("{}.flat", path);
        if storage.file_exists(&file_name) {
            return Self::load(index_config, writer_config, storage, &path);
        }

        Ok(Self {
            index_config,
            writer_config,
            storage: Some(storage),
            path,
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
            return Err(LaurusError::InvalidOperation(format!(
                "Dimension mismatch: expected {}, found {}",
                index_config.dimension, dimension
            )));
        }

        // Read the Issue #481 Stage 1 vector segment header (LVS1).
        // Pre-Stage-1 segments are rejected with IncompatibleFormat.
        let header = VectorSegmentHeader::read_from(&mut input)?;
        let QuantHeader::Scalar8Bit(params) = header.quant;

        // Read quantized vectors, dequantizing back to f32 so the
        // in-memory writer state stays compatible with downstream
        // operations (delete_document, vectors() accessor, etc.).
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
                LaurusError::InvalidOperation(format!("Invalid UTF-8 in field name: {}", e))
            })?;

            // Read quantized payload + dequantize.
            let values = read_dequantized_vector(&mut input, dimension, &params)?;

            vectors.push((doc_id, field_name, Vector::new(values)));
        }

        // Calculate next_vec_id from loaded vectors
        let max_id = vectors.iter().map(|(id, _, _)| *id).max().unwrap_or(0);
        let next_vec_id = if num_vectors > 0 { max_id + 1 } else { 0 };

        Ok(Self {
            index_config,
            writer_config,
            storage: Some(storage),
            path: path.to_string(),
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
                return Err(LaurusError::InvalidOperation(format!(
                    "Vector {} has dimension {}, expected {}",
                    doc_id,
                    vector.dimension(),
                    self.index_config.dimension
                )));
            }

            if !vector.is_valid() {
                return Err(LaurusError::InvalidOperation(format!(
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

        #[cfg(not(target_arch = "wasm32"))]
        if self.writer_config.parallel_build && vectors.len() > 100 {
            vectors.par_iter_mut().for_each(|(_, _, vector)| {
                vector.normalize();
            });
            return;
        }

        for (_, _, vector) in vectors {
            vector.normalize();
        }
    }

    /// Check for memory limits.
    fn check_memory_limit(&self) -> Result<()> {
        if let Some(limit) = self.writer_config.memory_limit {
            let current_usage = self.estimated_memory_usage();
            if current_usage > limit {
                return Err(LaurusError::ResourceExhausted(format!(
                    "Memory usage {current_usage} bytes exceeds limit {limit} bytes"
                )));
            }
        }
        Ok(())
    }

    /// Sort vectors by document ID and field name for better cache locality.
    fn sort_vectors(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        if self.writer_config.parallel_build && self.vectors.len() as u64 > 10000 {
            self.vectors
                .par_sort_by(|(doc_id_a, field_a, _), (doc_id_b, field_b, _)| {
                    doc_id_a.cmp(doc_id_b).then_with(|| field_a.cmp(field_b))
                });
            return;
        }

        self.vectors
            .sort_by(|(doc_id_a, field_a, _), (doc_id_b, field_b, _)| {
                doc_id_a.cmp(doc_id_b).then_with(|| field_a.cmp(field_b))
            });
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

    fn build(&mut self, mut vectors: Vec<(u64, String, Vector)>) -> Result<()> {
        if self.is_finalized {
            self.is_finalized = false;
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
            self.is_finalized = false;
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

    fn vectors(&self) -> &[(u64, String, Vector)] {
        &self.vectors
    }

    fn write(&self) -> Result<()> {
        use std::io::Write;

        if !self.is_finalized {
            return Err(LaurusError::InvalidOperation(
                "Index must be finalized before writing".to_string(),
            ));
        }

        let storage = self
            .storage
            .as_ref()
            .ok_or_else(|| LaurusError::InvalidOperation("No storage configured".to_string()))?;

        // Create the index file
        let file_name = format!("{}.flat", self.path);
        let mut output = storage.create_output(&file_name)?;

        // Write metadata
        let vector_count: u32 = self.vectors.len().try_into().map_err(|_| {
            LaurusError::InvalidOperation(format!(
                "Vector count {} exceeds u32::MAX",
                self.vectors.len()
            ))
        })?;
        output.write_all(&vector_count.to_le_bytes())?;
        output.write_all(&(self.index_config.dimension as u32).to_le_bytes())?;

        // Issue #481 Stage 1, Step 7: train per-segment SQ params on
        // the f32 vectors and emit the LVS1 quantized format. Empty
        // segments fall back to neutral (0.0, 1.0) params so the
        // header is still emitted (test_put_document_* exercise this
        // path).
        let f32_vectors: Vec<Vector> = self.vectors.iter().map(|(_, _, v)| v.clone()).collect();
        let (params, records) = if f32_vectors.is_empty() {
            (
                ScalarQuantParams {
                    offset: 0.0,
                    scale: 1.0,
                },
                Vec::new(),
            )
        } else {
            quantize_segment(&f32_vectors, self.index_config.dimension)?
        };
        VectorSegmentHeader::scalar_8bit(params).write_to(&mut output)?;

        // Write vectors with field names and quantized records.
        for ((doc_id, field_name, _), (int8, meta)) in self.vectors.iter().zip(records.iter()) {
            output.write_all(&doc_id.to_le_bytes())?;

            // Write field name length and field name
            let field_name_bytes = field_name.as_bytes();
            output.write_all(&(field_name_bytes.len() as u32).to_le_bytes())?;
            output.write_all(field_name_bytes)?;

            // Write quantized payload (dim int8 + sum_q + norm_q).
            write_quantized_record(&mut output, int8, *meta)?;
        }

        output.flush()?;
        Ok(())
    }

    fn has_storage(&self) -> bool {
        self.storage.is_some()
    }

    fn delete_document(&mut self, doc_id: u64) -> Result<()> {
        if self.is_finalized {
            self.is_finalized = false;
        }

        // Logical deletion from buffer
        // Note: usage of retain might be slow for large buffers, but acceptable for this stage
        self.vectors.retain(|(id, _, _)| *id != doc_id);
        Ok(())
    }

    fn delete_documents(&mut self, _field: &str, _value: &str) -> Result<usize> {
        if self.is_finalized {
            return Err(LaurusError::InvalidOperation(
                "Cannot delete documents from finalized index".to_string(),
            ));
        }

        // Vectors no longer carry metadata; field-based deletion is not supported.
        // Use delete_document(doc_id) for document-level deletion.
        Ok(0)
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

    fn build_reader(&self) -> Result<Arc<dyn crate::vector::reader::VectorIndexReader>> {
        use crate::vector::index::flat::reader::FlatVectorIndexReader;

        let storage = self.storage.as_ref().ok_or_else(|| {
            LaurusError::InvalidOperation("Cannot build reader: storage not configured".to_string())
        })?;

        let reader = FlatVectorIndexReader::load(
            storage.clone(),
            &self.path,
            self.index_config.distance_metric,
        )?;

        Ok(Arc::new(reader))
    }
}
