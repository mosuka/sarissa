//! IVF vector index reader implementation.

use std::collections::HashMap;
use std::sync::Arc;

use crate::error::{Result, SarissaError};
use crate::storage::Storage;
use crate::vector::core::distance::DistanceMetric;
use crate::vector::core::vector::Vector;
use crate::vector::index::io::read_metadata;
use crate::vector::index::storage::VectorStorage;
use crate::vector::reader::{ValidationReport, VectorIndexMetadata, VectorStats};
use crate::vector::reader::{VectorIndexReader, VectorIterator};
use std::io::SeekFrom;
use std::sync::Mutex;

/// Reader for IVF (Inverted File) vector indexes.
#[derive(Debug)]
pub struct IvfIndexReader {
    vectors: VectorStorage,
    vector_ids: Vec<(u64, String)>,
    dimension: usize,
    distance_metric: DistanceMetric,
    n_clusters: usize,
    n_probe: usize,
    centroids: Vec<Vector>,
}

impl IvfIndexReader {
    /// Create a reader from serialized bytes.
    pub fn from_bytes(_data: &[u8]) -> Result<Self> {
        Err(SarissaError::InvalidOperation(
            "from_bytes is deprecated, use load() instead".to_string(),
        ))
    }

    /// Load an IVF vector index from storage.
    pub fn load(
        storage: &dyn Storage,
        path: &str,
        distance_metric: DistanceMetric,
    ) -> Result<Self> {
        use std::io::{Read, Seek};

        // Open the index file
        let file_name = format!("{}.ivf", path);
        let mut input = storage.open_input(&file_name)?;

        // Read metadata
        let mut num_vectors_buf = [0u8; 4];
        input.read_exact(&mut num_vectors_buf)?;
        let num_vectors = u32::from_le_bytes(num_vectors_buf) as usize;

        let mut dimension_buf = [0u8; 4];
        input.read_exact(&mut dimension_buf)?;
        let dimension = u32::from_le_bytes(dimension_buf) as usize;

        let mut n_clusters_buf = [0u8; 4];
        input.read_exact(&mut n_clusters_buf)?;
        let n_clusters = u32::from_le_bytes(n_clusters_buf) as usize;

        let mut n_probe_buf = [0u8; 4];
        input.read_exact(&mut n_probe_buf)?;
        let n_probe = u32::from_le_bytes(n_probe_buf) as usize;

        // Read centroids
        let mut centroids = Vec::with_capacity(n_clusters);
        for _ in 0..n_clusters {
            let mut values = vec![0.0f32; dimension];
            for value in &mut values {
                let mut value_buf = [0u8; 4];
                input.read_exact(&mut value_buf)?;
                *value = f32::from_le_bytes(value_buf);
            }
            centroids.push(Vector::new(values));
        }

        // Read inverted lists
        let (vectors, vector_ids) = match storage.loading_mode() {
            crate::storage::LoadingMode::Eager => {
                let mut vectors = HashMap::with_capacity(num_vectors);
                let mut vector_ids = Vec::with_capacity(num_vectors);

                for _ in 0..n_clusters {
                    let mut list_size_buf = [0u8; 4];
                    input.read_exact(&mut list_size_buf)?;
                    let list_size = u32::from_le_bytes(list_size_buf) as usize;

                    for _ in 0..list_size {
                        let mut doc_id_buf = [0u8; 8];
                        input.read_exact(&mut doc_id_buf)?;
                        let doc_id = u64::from_le_bytes(doc_id_buf);

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
                }
                (VectorStorage::Owned(Arc::new(vectors)), vector_ids)
            }
            crate::storage::LoadingMode::Lazy => {
                let mut offsets = HashMap::with_capacity(num_vectors);
                let mut vector_ids = Vec::with_capacity(num_vectors);

                for _ in 0..n_clusters {
                    let mut list_size_buf = [0u8; 4];
                    input.read_exact(&mut list_size_buf)?;
                    let list_size = u32::from_le_bytes(list_size_buf) as usize;

                    for _ in 0..list_size {
                        let start_offset = input.stream_position().map_err(SarissaError::Io)?;

                        let mut doc_id_buf = [0u8; 8];
                        input.read_exact(&mut doc_id_buf)?;
                        let doc_id = u64::from_le_bytes(doc_id_buf);

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

                        offsets.insert((doc_id, field_name.clone()), start_offset);
                        vector_ids.push((doc_id, field_name));

                        // Skip metadata
                        let _ = read_metadata(&mut input)?;

                        // Skip vector
                        input
                            .seek(SeekFrom::Current((dimension * 4) as i64))
                            .map_err(SarissaError::Io)?;
                    }
                }
                (
                    VectorStorage::OnDemand {
                        input: Arc::new(Mutex::new(input)),
                        offsets: Arc::new(offsets),
                    },
                    vector_ids,
                )
            }
        };

        Ok(Self {
            vectors,
            vector_ids,
            dimension,
            distance_metric,
            n_clusters,
            n_probe,
            centroids,
        })
    }

    /// Get IVF parameters.
    pub fn ivf_params(&self) -> (usize, usize) {
        (self.n_clusters, self.n_probe)
    }

    /// Get centroids.
    pub fn centroids(&self) -> &[Vector] {
        &self.centroids
    }
}

impl VectorIndexReader for IvfIndexReader {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn get_vector(&self, doc_id: u64, field_name: &str) -> Result<Option<Vector>> {
        self.vectors
            .get(&(doc_id, field_name.to_string()), self.dimension)
    }

    fn get_vectors_for_doc(&self, doc_id: u64) -> Result<Vec<(String, Vector)>> {
        let mut result = Vec::new();
        for (id, field) in &self.vector_ids {
            if *id == doc_id {
                if let Some(vec) = self.vectors.get(&(*id, field.clone()), self.dimension)? {
                    result.push((field.clone(), vec));
                }
            }
        }
        Ok(result)
    }

    fn get_vectors(&self, doc_ids: &[(u64, String)]) -> Result<Vec<Option<Vector>>> {
        let mut result = Vec::with_capacity(doc_ids.len());
        for (id, field) in doc_ids {
            result.push(self.vectors.get(&(*id, field.clone()), self.dimension)?);
        }
        Ok(result)
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
            memory_usage: self.vectors.len() * (8 + self.dimension * 4)
                + self.centroids.len() * self.dimension * 4,
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
        let mut result = Vec::new();
        for (id, field) in &self.vector_ids {
            if *id >= start_doc_id && *id < end_doc_id {
                if let Some(vec) = self.vectors.get(&(*id, field.clone()), self.dimension)? {
                    result.push((*id, field.clone(), vec));
                }
            }
        }
        Ok(result)
    }

    fn get_vectors_by_field(&self, field_name: &str) -> Result<Vec<(u64, Vector)>> {
        let mut result = Vec::new();
        for (id, field) in &self.vector_ids {
            if field == field_name {
                if let Some(vec) = self.vectors.get(&(*id, field.clone()), self.dimension)? {
                    result.push((*id, vec));
                }
            }
        }
        Ok(result)
    }

    fn field_names(&self) -> Result<Vec<String>> {
        use std::collections::HashSet;
        let fields: HashSet<String> = self.vector_ids.iter().map(|val| val.1.clone()).collect();
        Ok(fields.into_iter().collect())
    }

    fn vector_iterator(&self) -> Result<Box<dyn VectorIterator>> {
        Ok(Box::new(IvfVectorIterator {
            storage: self.vectors.clone(),
            keys: self.vector_ids.clone(),
            current: 0,
            dimension: self.dimension,
        }))
    }

    fn metadata(&self) -> Result<VectorIndexMetadata> {
        Ok(VectorIndexMetadata {
            index_type: "ivf".to_string(),
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

        if self.vector_ids.len() != self.vectors.len() {
            errors.push(format!(
                "Mismatch between vector_ids count ({}) and vectors count ({})",
                self.vector_ids.len(),
                self.vectors.len()
            ));
        }

        match &self.vectors {
            VectorStorage::Owned(map) => {
                for ((id, field), vector) in map.iter() {
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
            }
            VectorStorage::OnDemand { offsets, .. } => {
                for (id, field) in &self.vector_ids {
                    if !offsets.contains_key(&(*id, field.clone())) {
                        errors.push(format!(
                            "Vector {}:{} in ids but missing in storage",
                            id, field
                        ));
                    }
                }
                warnings.push("OnDemand mode: Deep vector validation skipped".to_string());
            }
        }

        for (idx, centroid) in self.centroids.iter().enumerate() {
            if centroid.dimension() != self.dimension {
                errors.push(format!(
                    "Centroid {} has dimension {}, expected {}",
                    idx,
                    centroid.dimension(),
                    self.dimension
                ));
            }

            if !centroid.is_valid() {
                errors.push(format!(
                    "Centroid {} contains invalid values (NaN or infinity)",
                    idx
                ));
            }
        }

        if self.n_clusters == 0 {
            errors.push("IVF parameter n_clusters is 0".to_string());
        }
        if self.n_probe == 0 {
            warnings.push("IVF parameter n_probe is 0".to_string());
        }
        if self.centroids.len() != self.n_clusters {
            errors.push(format!(
                "Number of centroids ({}) does not match n_clusters ({})",
                self.centroids.len(),
                self.n_clusters
            ));
        }

        Ok(ValidationReport {
            repair_suggestions: Vec::new(),
            is_valid: errors.is_empty(),
            errors,
            warnings,
        })
    }
}

/// Iterator for IVF vector index.
struct IvfVectorIterator {
    storage: VectorStorage,
    keys: Vec<(u64, String)>,
    current: usize,
    dimension: usize,
}

impl VectorIterator for IvfVectorIterator {
    fn next(&mut self) -> Result<Option<(u64, String, Vector)>> {
        if self.current < self.keys.len() {
            let (doc_id, field) = &self.keys[self.current];
            if let Some(vec) = self
                .storage
                .get(&(*doc_id, field.clone()), self.dimension)?
            {
                self.current += 1;
                Ok(Some((*doc_id, field.clone(), vec)))
            } else {
                Err(SarissaError::internal(format!(
                    "Vector {}:{} found in keys but missing in storage",
                    doc_id, field
                )))
            }
        } else {
            Ok(None)
        }
    }

    fn skip_to(&mut self, doc_id: u64, field_name: &str) -> Result<bool> {
        while self.current < self.keys.len() {
            let (id, field) = &self.keys[self.current];
            if *id > doc_id || (*id == doc_id && field.as_str() >= field_name) {
                return Ok(true);
            }
            self.current += 1;
        }
        Ok(false)
    }

    fn position(&self) -> (u64, String) {
        if self.current < self.keys.len() {
            self.keys[self.current].clone()
        } else {
            (u64::MAX, String::new())
        }
    }

    fn reset(&mut self) -> Result<()> {
        self.current = 0;
        Ok(())
    }
}
