//! HNSW (Hierarchical Navigable Small World) index builder for approximate search.

use std::sync::Arc;

use crate::error::{Result, YatagarasuError};
use crate::storage::Storage;
use crate::vector::core::vector::Vector;
use crate::vector::index::HnswIndexConfig;
use crate::vector::writer::{VectorIndexWriter, VectorIndexWriterConfig};

/// Builder for HNSW vector indexes (approximate search).
#[derive(Debug)]
pub struct HnswIndexWriter {
    index_config: HnswIndexConfig,
    writer_config: VectorIndexWriterConfig,
    storage: Option<Arc<dyn Storage>>,
    _ml: f64, // Level normalization factor
    vectors: Vec<(u64, String, Vector)>,
    is_finalized: bool,
    total_vectors_to_add: Option<usize>,
    next_vec_id: u64,
}

impl HnswIndexWriter {
    /// Create a new HNSW index builder.
    pub fn new(
        index_config: HnswIndexConfig,
        writer_config: VectorIndexWriterConfig,
    ) -> Result<Self> {
        Ok(Self {
            index_config,
            writer_config,
            storage: None,
            _ml: 1.0 / (2.0_f64).ln(), // 1/ln(2)
            vectors: Vec::new(),
            is_finalized: false,
            total_vectors_to_add: None,
            next_vec_id: 0,
        })
    }

    /// Create a new HNSW index builder with storage.
    pub fn with_storage(
        index_config: HnswIndexConfig,
        writer_config: VectorIndexWriterConfig,
        storage: Arc<dyn Storage>,
    ) -> Result<Self> {
        Ok(Self {
            index_config,
            writer_config,
            storage: Some(storage),
            _ml: 1.0 / (2.0_f64).ln(),
            vectors: Vec::new(),
            is_finalized: false,
            total_vectors_to_add: None,
            next_vec_id: 0,
        })
    }

    /// Load an existing HNSW index from storage.
    pub fn load(
        index_config: HnswIndexConfig,
        writer_config: VectorIndexWriterConfig,
        storage: Arc<dyn Storage>,
        path: &str,
    ) -> Result<Self> {
        use std::io::Read;

        // Open the index file
        let file_name = format!("{}.hnsw", path);
        let mut input = storage.open_input(&file_name)?;

        // Read metadata
        let mut num_vectors_buf = [0u8; 4];
        input.read_exact(&mut num_vectors_buf)?;
        let num_vectors = u32::from_le_bytes(num_vectors_buf) as usize;

        let mut dimension_buf = [0u8; 4];
        input.read_exact(&mut dimension_buf)?;
        let dimension = u32::from_le_bytes(dimension_buf) as usize;

        let mut m_buf = [0u8; 4];
        input.read_exact(&mut m_buf)?;
        let _m = u32::from_le_bytes(m_buf) as usize;

        let mut ef_construction_buf = [0u8; 4];
        input.read_exact(&mut ef_construction_buf)?;
        let _ef_construction = u32::from_le_bytes(ef_construction_buf) as usize;

        if dimension != index_config.dimension {
            return Err(YatagarasuError::InvalidOperation(format!(
                "Dimension mismatch: expected {}, found {}",
                index_config.dimension, dimension
            )));
        }

        // Read vectors
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
                YatagarasuError::InvalidOperation(format!("Invalid UTF-8 in field name: {}", e))
            })?;

            // Read vector data
            let mut values = vec![0.0f32; dimension];
            for value in &mut values {
                let mut value_buf = [0u8; 4];
                input.read_exact(&mut value_buf)?;
                *value = f32::from_le_bytes(value_buf);
            }

            vectors.push((doc_id, field_name, Vector::new(values)));
        }

        // Calculate next_vec_id from loaded vectors
        let max_id = vectors.iter().map(|(id, _, _)| *id).max().unwrap_or(0);
        let next_vec_id = if num_vectors > 0 { max_id + 1 } else { 0 };

        Ok(Self {
            index_config,
            writer_config,
            storage: Some(storage),
            _ml: 1.0 / (2.0_f64).ln(),
            vectors,
            is_finalized: true,
            total_vectors_to_add: Some(num_vectors),
            next_vec_id,
        })
    }

    /// Set HNSW-specific parameters.
    pub fn with_hnsw_params(mut self, m: usize, ef_construction: usize) -> Self {
        self.index_config.m = m;
        self.index_config.ef_construction = ef_construction;
        self
    }

    /// Set the expected total number of vectors (for progress tracking).
    pub fn set_expected_vector_count(&mut self, count: usize) {
        self.total_vectors_to_add = Some(count);
    }

    /// Calculate the layer for a new vector.
    #[allow(dead_code)]
    fn select_layer(&self) -> usize {
        let mut layer = 0;
        let mut rng = rand::rng();

        while rand::Rng::random::<f64>(&mut rng) < 0.5 && layer < 16 {
            layer += 1;
        }

        layer
    }

    /// Validate vectors before adding them.
    fn validate_vectors(&self, vectors: &[(u64, String, Vector)]) -> Result<()> {
        if vectors.is_empty() {
            return Ok(());
        }

        for (doc_id, _field_name, vector) in vectors {
            if vector.dimension() != self.index_config.dimension {
                return Err(YatagarasuError::InvalidOperation(format!(
                    "Vector {} has dimension {}, expected {}",
                    doc_id,
                    vector.dimension(),
                    self.index_config.dimension
                )));
            }

            if !vector.is_valid() {
                return Err(YatagarasuError::InvalidOperation(format!(
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

        use rayon::prelude::*;

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

    /// Build the HNSW graph structure (placeholder implementation).
    fn build_hnsw_graph(&mut self) -> Result<()> {
        // This is a placeholder for the actual HNSW graph construction
        // Real implementation would:
        // 1. Create layered graph structure
        // 2. For each vector, determine its layer
        // 3. Insert vector and create connections using greedy search
        // 4. Maintain M connections per layer with pruning

        println!(
            "Building HNSW graph with {} vectors",
            self.vectors.len() as u64
        );
        println!(
            "Parameters: M={}, efConstruction={}",
            self.index_config.m, self.index_config.ef_construction
        );

        // Placeholder: just sort vectors by ID and field for now
        self.vectors
            .sort_by_key(|(doc_id, field, _)| (*doc_id, field.clone()));

        Ok(())
    }

    /// Check for memory limits.
    fn check_memory_limit(&self) -> Result<()> {
        if let Some(limit) = self.writer_config.memory_limit {
            let current_usage = self.estimated_memory_usage();
            if current_usage > limit {
                return Err(YatagarasuError::ResourceExhausted(format!(
                    "Memory usage {current_usage} bytes exceeds limit {limit} bytes"
                )));
            }
        }
        Ok(())
    }

    /// Get the stored vectors (for testing/debugging).
    pub fn vectors(&self) -> &[(u64, String, Vector)] {
        &self.vectors
    }

    /// Get HNSW parameters.
    pub fn hnsw_params(&self) -> (usize, usize) {
        (self.index_config.m, self.index_config.ef_construction)
    }
}

#[async_trait::async_trait]
impl VectorIndexWriter for HnswIndexWriter {
    fn next_vector_id(&self) -> u64 {
        self.next_vec_id
    }

    async fn add_document(&mut self, doc: crate::document::document::Document) -> Result<u64> {
        use crate::document::field::{FieldOption, FieldValue};
        use crate::embedding::per_field::PerFieldEmbedder;

        let doc_id = self.next_vec_id;
        let mut vectors = Vec::new();

        for (field_name, field) in doc.fields().iter() {
            // Check if this is a vector field and if it should be indexed
            if let FieldValue::Vector(text) = &field.value {
                // Check FieldOption to determine if this vector should be indexed
                let should_index = match &field.option {
                    FieldOption::Vector(opt) => {
                        // Check if flat, hnsw, or ivf indexing is enabled
                        opt.flat.is_some() || opt.hnsw.is_some() || opt.ivf.is_some()
                    }
                    _ => false,
                };

                if !should_index {
                    continue;
                }

                // Check if embedder is PerFieldEmbedder for field-specific embedding
                let vector = if let Some(per_field) = self
                    .index_config
                    .embedder
                    .as_any()
                    .downcast_ref::<PerFieldEmbedder>()
                {
                    per_field.embed_field(field_name, text.as_str()).await?
                } else {
                    self.index_config.embedder.embed(text.as_str()).await?
                };

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
        doc: crate::document::document::Document,
    ) -> Result<()> {
        use crate::document::field::{FieldOption, FieldValue};
        use crate::embedding::per_field::PerFieldEmbedder;

        let mut vectors = Vec::new();

        for (field_name, field) in doc.fields().iter() {
            // Check if this is a vector field and if it should be indexed
            if let FieldValue::Vector(text) = &field.value {
                // Check FieldOption to determine if this vector should be indexed
                let should_index = match &field.option {
                    FieldOption::Vector(opt) => {
                        // Check if flat, hnsw, or ivf indexing is enabled
                        opt.flat.is_some() || opt.hnsw.is_some() || opt.ivf.is_some()
                    }
                    _ => false,
                };

                if !should_index {
                    continue;
                }

                // Check if embedder is PerFieldEmbedder for field-specific embedding
                let vector = if let Some(per_field) = self
                    .index_config
                    .embedder
                    .as_any()
                    .downcast_ref::<PerFieldEmbedder>()
                {
                    per_field.embed_field(field_name, text.as_str()).await?
                } else {
                    self.index_config.embedder.embed(text.as_str()).await?
                };

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
            return Err(YatagarasuError::InvalidOperation(
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
            return Err(YatagarasuError::InvalidOperation(
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

        // Build the actual HNSW graph structure
        self.build_hnsw_graph()?;

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

        // HNSW graph overhead (rough estimate)
        // Each vector can have up to M connections per layer
        // Average layers per vector is approximately 1/(1-p) where p=0.5
        let avg_layers = 2.0;
        let graph_memory =
            self.vectors.len() * (self.index_config.m as f32 * avg_layers * 8.0) as usize;

        let metadata_memory = self.vectors.len() * 128; // Increased for graph structure

        vector_memory + graph_memory + metadata_memory
    }

    fn optimize(&mut self) -> Result<()> {
        if !self.is_finalized {
            return Err(YatagarasuError::InvalidOperation(
                "Index must be finalized before optimization".to_string(),
            ));
        }

        // HNSW optimization could include:
        // 1. Graph pruning to remove low-quality connections
        // 2. Memory compaction
        // 3. Connection rebalancing

        println!("Optimizing HNSW index...");

        // For now, just compact memory
        self.vectors.shrink_to_fit();

        Ok(())
    }

    fn vectors(&self) -> &[(u64, String, Vector)] {
        &self.vectors
    }

    fn write(&self, path: &str) -> Result<()> {
        use std::io::Write;

        if !self.is_finalized {
            return Err(YatagarasuError::InvalidOperation(
                "Index must be finalized before writing".to_string(),
            ));
        }

        let storage = self.storage.as_ref().ok_or_else(|| {
            YatagarasuError::InvalidOperation("No storage configured".to_string())
        })?;

        // Create the index file
        let file_name = format!("{}.hnsw", path);
        let mut output = storage.create_output(&file_name)?;

        // Write metadata
        output.write_all(&(self.vectors.len() as u64 as u32).to_le_bytes())?;
        output.write_all(&(self.index_config.dimension as u32).to_le_bytes())?;
        output.write_all(&(self.index_config.m as u32).to_le_bytes())?;
        output.write_all(&(self.index_config.ef_construction as u32).to_le_bytes())?;

        // Write vectors with field names
        for (doc_id, field_name, vector) in &self.vectors {
            output.write_all(&doc_id.to_le_bytes())?;

            // Write field name length and field name
            let field_name_bytes = field_name.as_bytes();
            output.write_all(&(field_name_bytes.len() as u32).to_le_bytes())?;
            output.write_all(field_name_bytes)?;

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
        // Simplified implementation - returns 0
        // TODO: Implement proper deletion with metadata storage
        let _field = field;
        let _value = value;
        Ok(0)
    }

    async fn update_document(
        &mut self,
        field: &str,
        value: &str,
        doc: crate::document::document::Document,
    ) -> Result<()> {
        self.delete_documents(field, value)?;
        self.add_document(doc).await?;
        Ok(())
    }

    fn rollback(&mut self) -> Result<()> {
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
        self.vectors.clear();
        self.is_finalized = true;
        Ok(())
    }

    fn is_closed(&self) -> bool {
        self.is_finalized && self.vectors.is_empty()
    }
}
