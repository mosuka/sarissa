//! HNSW vector index reader implementation.

use std::collections::HashMap;
use std::sync::Arc;

use crate::error::{Result, YatagarasuError};
use crate::storage::Storage;
use crate::vector::index::reader::{ValidationReport, VectorIndexMetadata, VectorStats};
use crate::vector::index::reader::{VectorIndexReader, VectorIterator};
use crate::vector::{DistanceMetric, Vector};

/// Reader for HNSW (Hierarchical Navigable Small World) vector indexes.
#[derive(Debug)]
pub struct HnswIndexReader {
    vectors: HashMap<(u64, String), Vector>,
    vector_ids: Vec<(u64, String)>,
    dimension: usize,
    distance_metric: DistanceMetric,
    m: usize,
    ef_construction: usize,
}

impl HnswIndexReader {
    /// Create a reader from serialized bytes.
    pub fn from_bytes(_data: &[u8]) -> Result<Self> {
        Err(YatagarasuError::InvalidOperation(
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

        // Read vectors with field names
        let mut vectors = HashMap::with_capacity(num_vectors);
        let mut vector_ids = Vec::with_capacity(num_vectors);

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

            vector_ids.push((doc_id, field_name.clone()));
            vectors.insert((doc_id, field_name), Vector::new(values));
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
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn get_vector(&self, doc_id: u64, field_name: &str) -> Result<Option<Vector>> {
        Ok(self.vectors.get(&(doc_id, field_name.to_string())).cloned())
    }

    fn get_vectors_for_doc(&self, doc_id: u64) -> Result<Vec<(String, Vector)>> {
        Ok(self
            .vectors
            .iter()
            .filter(|((id, _), _)| *id == doc_id)
            .map(|((_, field), vec)| (field.clone(), vec.clone()))
            .collect())
    }

    fn get_vectors(&self, doc_ids: &[(u64, String)]) -> Result<Vec<Option<Vector>>> {
        Ok(doc_ids
            .iter()
            .map(|(id, field)| self.vectors.get(&(*id, field.clone())).cloned())
            .collect())
    }

    fn vector_ids(&self) -> Result<Vec<(u64, String)>> {
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

    fn contains_vector(&self, doc_id: u64, field_name: &str) -> bool {
        self.vectors.contains_key(&(doc_id, field_name.to_string()))
    }

    fn get_vector_range(
        &self,
        start_doc_id: u64,
        end_doc_id: u64,
    ) -> Result<Vec<(u64, String, Vector)>> {
        Ok(self
            .vector_ids
            .iter()
            .filter(|(id, _)| *id >= start_doc_id && *id < end_doc_id)
            .filter_map(|(id, field)| {
                self.vectors
                    .get(&(*id, field.clone()))
                    .map(|v| (*id, field.clone(), v.clone()))
            })
            .collect())
    }

    fn get_vectors_by_field(&self, field_name: &str) -> Result<Vec<(u64, Vector)>> {
        Ok(self
            .vectors
            .iter()
            .filter(|((_, field), _)| field == field_name)
            .map(|((id, _), vec)| (*id, vec.clone()))
            .collect())
    }

    fn field_names(&self) -> Result<Vec<String>> {
        use std::collections::HashSet;
        let fields: HashSet<String> = self
            .vectors
            .keys()
            .map(|(_, field)| field.clone())
            .collect();
        Ok(fields.into_iter().collect())
    }

    fn vector_iterator(&self) -> Result<Box<dyn VectorIterator>> {
        Ok(Box::new(HnswVectorIterator {
            vectors: self
                .vector_ids
                .iter()
                .filter_map(|(id, field)| {
                    self.vectors
                        .get(&(*id, field.clone()))
                        .map(|v| (*id, field.clone(), v.clone()))
                })
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
        for ((id, field), vector) in &self.vectors {
            if vector.dimension() != self.dimension {
                errors.push(format!(
                    "Vector {}:{} has dimension {}, expected {}",
                    id,
                    field,
                    vector.dimension(),
                    self.dimension
                ));
            }

            if !vector.is_valid() {
                errors.push(format!(
                    "Vector {}:{} contains invalid values (NaN or infinity)",
                    id, field
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
    vectors: Vec<(u64, String, Vector)>,
    current: usize,
}

impl VectorIterator for HnswVectorIterator {
    fn next(&mut self) -> Result<Option<(u64, String, Vector)>> {
        if self.current < self.vectors.len() {
            let result = self.vectors[self.current].clone();
            self.current += 1;
            Ok(Some(result))
        } else {
            Ok(None)
        }
    }

    fn skip_to(&mut self, doc_id: u64, field_name: &str) -> Result<bool> {
        while self.current < self.vectors.len() {
            let (id, field, _) = &self.vectors[self.current];
            if *id > doc_id || (*id == doc_id && field.as_str() >= field_name) {
                return Ok(true);
            }
            self.current += 1;
        }
        Ok(false)
    }

    fn position(&self) -> (u64, String) {
        if self.current < self.vectors.len() {
            let (id, field, _) = &self.vectors[self.current];
            (*id, field.clone())
        } else {
            (u64::MAX, String::new())
        }
    }

    fn reset(&mut self) -> Result<()> {
        self.current = 0;
        Ok(())
    }
}
