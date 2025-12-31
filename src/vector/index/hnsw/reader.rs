//! HNSW vector index reader implementation.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::error::{Result, SarissaError};
use crate::storage::{Storage, StorageInput};
use crate::vector::core::distance::DistanceMetric;
use crate::vector::core::vector::Vector;
use crate::vector::index::hnsw::graph::HnswGraph;
use crate::vector::index::io::read_metadata;
use crate::vector::reader::{ValidationReport, VectorIndexMetadata, VectorStats};
use crate::vector::reader::{VectorIndexReader, VectorIterator};

/// Storage for vectors (in-memory or on-demand).
use crate::vector::index::storage::VectorStorage;

/// Reader for HNSW (Hierarchical Navigable Small World) vector indexes.
#[derive(Debug)]
pub struct HnswIndexReader {
    vectors: VectorStorage,
    vector_ids: Vec<(u64, String)>,
    dimension: usize,
    distance_metric: DistanceMetric,
    m: usize,
    ef_construction: usize,
    pub graph: Option<Arc<HnswGraph>>,
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
        storage: &dyn Storage,
        path: &str,
        distance_metric: DistanceMetric,
    ) -> Result<Self> {
        use std::io::{Read, Seek};

        // Open the index file
        let file_name = format!("{}.hnsw", path);
        let mut input = storage.open_input(&file_name)?;

        // Read metadata
        let mut num_vectors_buf = [0u8; 4];
        input.read_exact(&mut num_vectors_buf)?;
        let num_vectors = u32::from_le_bytes(num_vectors_buf) as usize;

        // We already have dimension from argument, but file has it too.
        // Let's read it to advance cursor, and verify?
        // Or strictly trust file? FlatIndexReader reads it.
        // Let's read it.
        let mut dimension_buf = [0u8; 4];
        input.read_exact(&mut dimension_buf)?;
        let dimension = u32::from_le_bytes(dimension_buf) as usize;

        let mut m_buf = [0u8; 4];
        input.read_exact(&mut m_buf)?;
        let m = u32::from_le_bytes(m_buf) as usize;

        let mut ef_construction_buf = [0u8; 4];
        input.read_exact(&mut ef_construction_buf)?;
        let ef_construction = u32::from_le_bytes(ef_construction_buf) as usize;

        // Helper to read graph
        let read_graph =
            |input: &mut dyn crate::storage::StorageInput| -> Result<Option<Arc<HnswGraph>>> {
                let mut has_graph_buf = [0u8; 1];
                if input.read_exact(&mut has_graph_buf).is_ok() && has_graph_buf[0] == 1 {
                    let mut entry_point_buf = [0u8; 8];
                    input.read_exact(&mut entry_point_buf)?;
                    let entry_point_raw = u64::from_le_bytes(entry_point_buf);
                    let entry_point = if entry_point_raw == u64::MAX {
                        None
                    } else {
                        Some(entry_point_raw)
                    };

                    let mut max_level_buf = [0u8; 4];
                    input.read_exact(&mut max_level_buf)?;
                    let max_level = u32::from_le_bytes(max_level_buf) as usize;

                    let mut node_count_buf = [0u8; 4];
                    input.read_exact(&mut node_count_buf)?;
                    let node_count = u32::from_le_bytes(node_count_buf) as usize;

                    let mut nodes = HashMap::with_capacity(node_count);

                    for _ in 0..node_count {
                        let mut doc_id_buf = [0u8; 8];
                        input.read_exact(&mut doc_id_buf)?;
                        let doc_id = u64::from_le_bytes(doc_id_buf);

                        let mut layer_count_buf = [0u8; 4];
                        input.read_exact(&mut layer_count_buf)?;
                        let layer_count = u32::from_le_bytes(layer_count_buf) as usize;

                        let mut layers = Vec::with_capacity(layer_count);
                        for _ in 0..layer_count {
                            let mut neighbor_count_buf = [0u8; 4];
                            input.read_exact(&mut neighbor_count_buf)?;
                            let neighbor_count = u32::from_le_bytes(neighbor_count_buf) as usize;

                            let mut neighbors = Vec::with_capacity(neighbor_count);
                            for _ in 0..neighbor_count {
                                let mut neighbor_buf = [0u8; 8];
                                input.read_exact(&mut neighbor_buf)?;
                                neighbors.push(u64::from_le_bytes(neighbor_buf));
                            }
                            layers.push(neighbors);
                        }
                        nodes.insert(doc_id, layers);
                    }

                    Ok(Some(Arc::new(HnswGraph {
                        entry_point,
                        max_level,
                        nodes,
                        m,
                        m_max: m,
                        m_max_0: m * 2,
                        ef_construction,
                        level_mult: 1.0 / (m as f64).ln(),
                    })))
                } else {
                    Ok(None)
                }
            };

        let (vectors, vector_ids, graph) = match storage.loading_mode() {
            crate::storage::LoadingMode::Eager => {
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

                let graph = read_graph(&mut input)?;
                (VectorStorage::Owned(Arc::new(vectors)), vector_ids, graph)
            }
            crate::storage::LoadingMode::Lazy => {
                let mut offsets = HashMap::with_capacity(num_vectors);
                let mut vector_ids = Vec::with_capacity(num_vectors);

                let start_pos = 4 + 4 + 4 + 4; // num_vectors, dimension, m, ef

                // Seek to start of vectors
                input
                    .seek(std::io::SeekFrom::Start(start_pos as u64))
                    .map_err(SarissaError::Io)?;

                for _ in 0..num_vectors {
                    let start_offset = input.stream_position().map_err(SarissaError::Io)?;
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

                    // Record offset for on-demand loading
                    // We record offset where the entry starts (doc_id)
                    offsets.insert((doc_id, field_name.clone()), start_offset);
                    vector_ids.push((doc_id, field_name.clone()));

                    // Skip metadata
                    let _ = read_metadata(&mut input)?;

                    // Skip vector data (dimension * 4 bytes)
                    input
                        .seek(std::io::SeekFrom::Current((dimension * 4) as i64))
                        .map_err(SarissaError::Io)?;
                }

                let graph = read_graph(&mut input)?;

                (
                    VectorStorage::OnDemand {
                        input: Arc::new(Mutex::new(input)),
                        offsets: Arc::new(offsets),
                    },
                    vector_ids,
                    graph,
                )
            }
        };

        Ok(Self {
            vectors,
            vector_ids,
            dimension,
            distance_metric,
            m,
            ef_construction,
            graph,
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
        self.vector_ids.len()
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn distance_metric(&self) -> DistanceMetric {
        self.distance_metric
    }

    fn stats(&self) -> VectorStats {
        let memory_usage = match &self.vectors {
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
        match &self.vectors {
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
        Ok(Box::new(HnswVectorIterator {
            storage: self.vectors.clone(),
            keys: self.vector_ids.clone(),
            current: 0,
            dimension: self.dimension,
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

        if self.vector_ids.len() != self.vectors.len() {
            errors.push(format!(
                "Mismatch between vector_ids count ({}) and vectors count ({})",
                self.vector_ids.len(),
                self.storage_len()
            ));
        }

        match &self.vectors {
            VectorStorage::Owned(map) => {
                for (id, field) in &self.vector_ids {
                    if let Some(vector) = map.get(&(*id, field.clone())) {
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
                    } else {
                        errors.push(format!(
                            "Vector {}:{} found in keys but missing in storage",
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
    storage: VectorStorage,
    keys: Vec<(u64, String)>,
    current: usize,
    dimension: usize,
}

impl VectorIterator for HnswVectorIterator {
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
