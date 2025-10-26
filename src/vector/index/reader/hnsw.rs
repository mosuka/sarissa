//! HNSW vector index reader implementation.

use std::collections::HashMap;
use std::sync::Arc;

use crate::error::{Result, SageError};
use crate::storage::Storage;
use crate::vector::reader::{VectorIndexReader, VectorIterator};
use crate::vector::types::{ValidationReport, VectorIndexMetadata, VectorStats};
use crate::vector::{DistanceMetric, Vector};

/// Reader for HNSW (Hierarchical Navigable Small World) vector indexes.
pub struct HnswIndexReader {
    vectors: HashMap<u64, Vector>,
    vector_ids: Vec<u64>,
    dimension: usize,
    distance_metric: DistanceMetric,
    m: usize,
    ef_construction: usize,
}

impl HnswIndexReader {
    /// Create a reader from serialized bytes.
    pub fn from_bytes(_data: &[u8]) -> Result<Self> {
        Err(SageError::InvalidOperation(
            "from_bytes is deprecated, use load() instead".to_string(),
        ))
    }

    /// Load an HNSW vector index from storage.
    pub fn load(
        storage: Arc<dyn Storage>,
        path: &str,
        distance_metric: DistanceMetric,
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
        let m = u32::from_le_bytes(m_buf) as usize;

        let mut ef_construction_buf = [0u8; 4];
        input.read_exact(&mut ef_construction_buf)?;
        let ef_construction = u32::from_le_bytes(ef_construction_buf) as usize;

        // Read vectors
        let mut vectors = HashMap::with_capacity(num_vectors);
        let mut vector_ids = Vec::with_capacity(num_vectors);

        for _ in 0..num_vectors {
            let mut doc_id_buf = [0u8; 8];
            input.read_exact(&mut doc_id_buf)?;
            let doc_id = u64::from_le_bytes(doc_id_buf);

            let mut values = vec![0.0f32; dimension];
            for value in &mut values {
                let mut value_buf = [0u8; 4];
                input.read_exact(&mut value_buf)?;
                *value = f32::from_le_bytes(value_buf);
            }

            vector_ids.push(doc_id);
            vectors.insert(doc_id, Vector::new(values));
        }

        Ok(Self {
            vectors,
            vector_ids,
            dimension,
            distance_metric,
            m,
            ef_construction,
        })
    }

    /// Get HNSW parameters.
    pub fn hnsw_params(&self) -> (usize, usize) {
        (self.m, self.ef_construction)
    }
}

impl VectorIndexReader for HnswIndexReader {
    fn get_vector(&self, doc_id: u64) -> Result<Option<Vector>> {
        Ok(self.vectors.get(&doc_id).cloned())
    }

    fn get_vectors(&self, doc_ids: &[u64]) -> Result<Vec<Option<Vector>>> {
        Ok(doc_ids
            .iter()
            .map(|id| self.vectors.get(id).cloned())
            .collect())
    }

    fn vector_ids(&self) -> Result<Vec<u64>> {
        Ok(self.vector_ids.clone())
    }

    fn vector_count(&self) -> usize {
        self.vectors.len()
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn distance_metric(&self) -> DistanceMetric {
        self.distance_metric
    }

    fn stats(&self) -> VectorStats {
        VectorStats {
            vector_count: self.vectors.len(),
            dimension: self.dimension,
            memory_usage: self.vectors.len() * (8 + self.dimension * 4),
            build_time_ms: 0,
        }
    }

    fn contains_vector(&self, doc_id: u64) -> bool {
        self.vectors.contains_key(&doc_id)
    }

    fn get_vector_range(&self, start_doc_id: u64, end_doc_id: u64) -> Result<Vec<(u64, Vector)>> {
        Ok(self
            .vector_ids
            .iter()
            .filter(|&&id| id >= start_doc_id && id < end_doc_id)
            .filter_map(|&id| self.vectors.get(&id).map(|v| (id, v.clone())))
            .collect())
    }

    fn vector_iterator(&self) -> Result<Box<dyn VectorIterator>> {
        Ok(Box::new(HnswVectorIterator {
            vectors: self
                .vector_ids
                .iter()
                .filter_map(|&id| self.vectors.get(&id).map(|v| (id, v.clone())))
                .collect(),
            current: 0,
        }))
    }

    fn metadata(&self) -> Result<VectorIndexMetadata> {
        Ok(VectorIndexMetadata {
            index_type: "hnsw".to_string(),
            created_at: chrono::Utc::now(),
            modified_at: chrono::Utc::now(),
            version: "1".to_string(),
            build_config: serde_json::json!({}),
            custom_metadata: std::collections::HashMap::new(),
        })
    }

    fn validate(&self) -> Result<ValidationReport> {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Check for duplicate IDs
        if self.vector_ids.len() != self.vectors.len() {
            errors.push(format!(
                "Mismatch between vector_ids count ({}) and vectors count ({})",
                self.vector_ids.len(),
                self.vectors.len()
            ));
        }

        // Validate dimensions
        for (id, vector) in &self.vectors {
            if vector.dimension() != self.dimension {
                errors.push(format!(
                    "Vector {} has dimension {}, expected {}",
                    id,
                    vector.dimension(),
                    self.dimension
                ));
            }

            if !vector.is_valid() {
                errors.push(format!(
                    "Vector {} contains invalid values (NaN or infinity)",
                    id
                ));
            }
        }

        // HNSW-specific validation
        if self.m == 0 {
            warnings.push("HNSW parameter M is 0, this may indicate a corrupted index".to_string());
        }
        if self.ef_construction == 0 {
            warnings.push(
                "HNSW parameter ef_construction is 0, this may indicate a corrupted index"
                    .to_string(),
            );
        }

        Ok(ValidationReport {
            repair_suggestions: Vec::new(),
            is_valid: errors.is_empty(),
            errors,
            warnings,
        })
    }
}

/// Iterator for HNSW vector index.
struct HnswVectorIterator {
    vectors: Vec<(u64, Vector)>,
    current: usize,
}

impl VectorIterator for HnswVectorIterator {
    fn next(&mut self) -> Result<Option<(u64, Vector)>> {
        if self.current < self.vectors.len() {
            let result = self.vectors[self.current].clone();
            self.current += 1;
            Ok(Some(result))
        } else {
            Ok(None)
        }
    }

    fn skip_to(&mut self, doc_id: u64) -> Result<bool> {
        while self.current < self.vectors.len() {
            if self.vectors[self.current].0 >= doc_id {
                return Ok(true);
            }
            self.current += 1;
        }
        Ok(false)
    }

    fn position(&self) -> u64 {
        if self.current < self.vectors.len() {
            self.vectors[self.current].0
        } else {
            u64::MAX
        }
    }

    fn reset(&mut self) -> Result<()> {
        self.current = 0;
        Ok(())
    }
}
