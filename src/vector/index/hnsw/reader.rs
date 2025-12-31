//! HNSW vector index reader implementation.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::error::{Result, SarissaError};
use crate::storage::{Storage, StorageInput};
use crate::vector::core::distance::DistanceMetric;
use crate::vector::core::vector::Vector;
use crate::vector::index::config::IndexLoadingMode;
use crate::vector::index::io::read_metadata;
use crate::vector::reader::{ValidationReport, VectorIndexMetadata, VectorStats};
use crate::vector::reader::{VectorIndexReader, VectorIterator};

#[derive(Debug)]
enum VectorStorage {
    Owned(HashMap<(u64, String), Vector>),
    OnDemand {
        input: Mutex<Box<dyn StorageInput>>,
        offsets: HashMap<(u64, String), u64>,
        vector_data_size: usize,
    },
}

/// Reader for HNSW (Hierarchical Navigable Small World) vector indexes.
#[derive(Debug)]
pub struct HnswIndexReader {
    storage: VectorStorage,
    vector_ids: Vec<(u64, String)>,
    dimension: usize,
    distance_metric: DistanceMetric,
    m: usize,
    ef_construction: usize,
}

impl HnswIndexReader {
    /// Create a reader from serialized bytes.
    pub fn from_bytes(_data: &[u8]) -> Result<Self> {
        Err(SarissaError::InvalidOperation(
            "from_bytes is deprecated, use load() instead".to_string(),
        ))
    }

    /// Load an HNSW vector index from storage.
    pub fn load(
        storage: Arc<dyn Storage>,
        path: &str,
        distance_metric: DistanceMetric,
        loading_mode: IndexLoadingMode,
    ) -> Result<Self> {
        use std::io::{Read, Seek};

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

        let mut vector_ids = Vec::with_capacity(num_vectors);

        let storage_impl = match loading_mode {
            IndexLoadingMode::InMemory => {
                // Read vectors with field names into memory
                let mut vectors = HashMap::with_capacity(num_vectors);

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
                        SarissaError::InvalidOperation(format!(
                            "Invalid UTF-8 in field name: {}",
                            e
                        ))
                    })?;

                    let metadata = read_metadata(&mut input)?;
                    // Read vector data
                    let mut values = vec![0.0f32; dimension];
                    for value in &mut values {
                        let mut value_buf = [0u8; 4];
                        input.read_exact(&mut value_buf)?;
                        *value = f32::from_le_bytes(value_buf);
                    }

                    vector_ids.push((doc_id, field_name.clone()));
                    vectors.insert(
                        (doc_id, field_name),
                        Vector::with_metadata(values, metadata),
                    );
                }
                VectorStorage::Owned(vectors)
            }
            IndexLoadingMode::Mmap => {
                // Scan for offsets
                let mut offsets = HashMap::with_capacity(num_vectors);
                let vector_data_size = dimension * 4;

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
                        SarissaError::InvalidOperation(format!(
                            "Invalid UTF-8 in field name: {}",
                            e
                        ))
                    })?;

                    let _metadata = read_metadata(&mut input)?;

                    // Current position is start of vector data
                    let current_pos = input.stream_position().map_err(|e| SarissaError::Io(e))?;
                    offsets.insert((doc_id, field_name.clone()), current_pos);

                    // Skip vector data (float32 * dimension)
                    input
                        .seek(std::io::SeekFrom::Current(vector_data_size as i64))
                        .map_err(|e| SarissaError::Io(e))?;

                    vector_ids.push((doc_id, field_name));
                }

                VectorStorage::OnDemand {
                    input: Mutex::new(input),
                    offsets,
                    vector_data_size,
                }
            }
        };

        Ok(Self {
            storage: storage_impl,
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

    fn retrieve_vector(&self, doc_id: u64, field_name: &str) -> Result<Option<Vector>> {
        use std::io::{Read, Seek};

        match &self.storage {
            VectorStorage::Owned(vectors) => {
                Ok(vectors.get(&(doc_id, field_name.to_string())).cloned())
            }
            VectorStorage::OnDemand {
                input,
                offsets,
                vector_data_size,
            } => {
                if let Some(&offset) = offsets.get(&(doc_id, field_name.to_string())) {
                    let mut input = input.lock().unwrap();
                    input
                        .seek(std::io::SeekFrom::Start(offset))
                        .map_err(|e| SarissaError::Io(e))?;

                    let mut values = vec![0.0f32; self.dimension];
                    // Verify size matches
                    if self.dimension * 4 != *vector_data_size {
                        return Err(SarissaError::InvalidOperation(
                            "Dimension mismatch in storage".to_string(),
                        ));
                    }

                    for value in &mut values {
                        let mut value_buf = [0u8; 4];
                        input.read_exact(&mut value_buf)?;
                        *value = f32::from_le_bytes(value_buf);
                    }

                    Ok(Some(Vector::new(values)))
                } else {
                    Ok(None)
                }
            }
        }
    }
}

impl VectorIndexReader for HnswIndexReader {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn get_vector(&self, doc_id: u64, field_name: &str) -> Result<Option<Vector>> {
        self.retrieve_vector(doc_id, field_name)
    }

    fn get_vectors_for_doc(&self, doc_id: u64) -> Result<Vec<(String, Vector)>> {
        let mut results = Vec::new();
        // Since we don't have an efficient field lookup by doc_id in OnDemand mode other than scanning ids,
        // we use vector_ids which is in memory.
        for (id, field) in &self.vector_ids {
            if *id == doc_id {
                if let Some(vec) = self.retrieve_vector(*id, field)? {
                    results.push((field.clone(), vec));
                }
            }
        }
        Ok(results)
    }

    fn get_vectors(&self, doc_ids: &[(u64, String)]) -> Result<Vec<Option<Vector>>> {
        let mut results = Vec::with_capacity(doc_ids.len());
        for (id, field) in doc_ids {
            results.push(self.retrieve_vector(*id, field)?);
        }
        Ok(results)
    }

    fn vector_ids(&self) -> Result<Vec<(u64, String)>> {
        Ok(self.vector_ids.clone())
    }

    fn vector_count(&self) -> usize {
        self.vector_ids.len()
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn distance_metric(&self) -> DistanceMetric {
        self.distance_metric
    }

    fn stats(&self) -> VectorStats {
        let memory_usage = match &self.storage {
            VectorStorage::Owned(vectors) => vectors.len() * (8 + self.dimension * 4),
            VectorStorage::OnDemand { offsets, .. } => {
                // Estimate memory for offsets map + ID list
                offsets.len() * (8 + 32 + 8) // Key + Valid + Offset roughly
            }
        };

        VectorStats {
            vector_count: self.vector_ids.len(),
            dimension: self.dimension,
            memory_usage,
            build_time_ms: 0,
        }
    }

    fn contains_vector(&self, doc_id: u64, field_name: &str) -> bool {
        match &self.storage {
            VectorStorage::Owned(vectors) => {
                vectors.contains_key(&(doc_id, field_name.to_string()))
            }
            VectorStorage::OnDemand { offsets, .. } => {
                offsets.contains_key(&(doc_id, field_name.to_string()))
            }
        }
    }

    fn get_vector_range(
        &self,
        start_doc_id: u64,
        end_doc_id: u64,
    ) -> Result<Vec<(u64, String, Vector)>> {
        let mut results = Vec::new();
        for (id, field) in &self.vector_ids {
            if *id >= start_doc_id && *id < end_doc_id {
                if let Some(vec) = self.retrieve_vector(*id, field)? {
                    results.push((*id, field.clone(), vec));
                }
            }
        }
        Ok(results)
    }

    fn get_vectors_by_field(&self, field_name: &str) -> Result<Vec<(u64, Vector)>> {
        let mut results = Vec::new();
        for (id, field) in &self.vector_ids {
            if field == field_name {
                if let Some(vec) = self.retrieve_vector(*id, field)? {
                    results.push((*id, vec));
                }
            }
        }
        Ok(results)
    }

    fn field_names(&self) -> Result<Vec<String>> {
        use std::collections::HashSet;
        let fields: HashSet<String> = self
            .vector_ids
            .iter()
            .map(|(_, field)| field.clone())
            .collect();
        Ok(fields.into_iter().collect())
    }

    fn vector_iterator(&self) -> Result<Box<dyn VectorIterator>> {
        match &self.storage {
            VectorStorage::Owned(vectors) => Ok(Box::new(HnswVectorIterator {
                vectors: self
                    .vector_ids
                    .iter()
                    .filter_map(|(id, field)| {
                        vectors
                            .get(&(*id, field.clone()))
                            .map(|v| (*id, field.clone(), v.clone()))
                    })
                    .collect(),
                current: 0,
            })),
            VectorStorage::OnDemand {
                input,
                offsets,
                vector_data_size: _,
            } => {
                // Clone the input for independent iteration
                let iter_input = input.lock().unwrap().clone_input()?;

                Ok(Box::new(OnDemandVectorIterator {
                    input: iter_input,
                    offsets: offsets.clone(),
                    vector_ids: self.vector_ids.clone(),
                    dimension: self.dimension,
                    current: 0,
                }))
            }
        }
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
        if self.vector_ids.len() != self.storage_len() {
            errors.push(format!(
                "Mismatch between vector_ids count ({}) and vectors count ({})",
                self.vector_ids.len(),
                self.storage_len()
            ));
        }

        // Validate dimensions - on demand check skipped for performance in full validation?
        // Or we should sample?
        // Implementation similar to FlatIndexReader but iterating.

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
            is_valid: errors.is_empty(),
            errors,
            warnings,
            repair_suggestions: vec![],
        })
    }
}

impl HnswIndexReader {
    fn storage_len(&self) -> usize {
        match &self.storage {
            VectorStorage::Owned(vectors) => vectors.len(),
            VectorStorage::OnDemand { offsets, .. } => offsets.len(),
        }
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

/// Iterator for on-demand vector loading.
pub struct OnDemandVectorIterator {
    input: Box<dyn StorageInput>,
    offsets: HashMap<(u64, String), u64>,
    vector_ids: Vec<(u64, String)>,
    dimension: usize,
    current: usize,
}

impl VectorIterator for OnDemandVectorIterator {
    fn next(&mut self) -> Result<Option<(u64, String, Vector)>> {
        use std::io::{Read, Seek};

        if self.current < self.vector_ids.len() {
            let (id, field) = &self.vector_ids[self.current];

            if let Some(&offset) = self.offsets.get(&(*id, field.clone())) {
                self.input
                    .seek(std::io::SeekFrom::Start(offset))
                    .map_err(|e| SarissaError::Io(e))?;

                let mut values = vec![0.0f32; self.dimension];
                for value in &mut values {
                    let mut value_buf = [0u8; 4];
                    self.input.read_exact(&mut value_buf)?;
                    *value = f32::from_le_bytes(value_buf);
                }

                let vector = Vector::new(values);
                self.current += 1;
                Ok(Some((*id, field.clone(), vector)))
            } else {
                self.current += 1;
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }

    fn skip_to(&mut self, doc_id: u64, field_name: &str) -> Result<bool> {
        while self.current < self.vector_ids.len() {
            let (id, field) = &self.vector_ids[self.current];
            if *id > doc_id || (*id == doc_id && field.as_str() >= field_name) {
                return Ok(true);
            }
            self.current += 1;
        }
        Ok(false)
    }

    fn position(&self) -> (u64, String) {
        if self.current < self.vector_ids.len() {
            let (id, field) = &self.vector_ids[self.current];
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
