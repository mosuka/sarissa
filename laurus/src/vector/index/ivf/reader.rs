//! IVF vector index reader implementation.

use std::collections::HashMap;
use std::sync::Arc;

use crate::error::{LaurusError, Result};
use crate::maintenance::deletion::DeletionBitmap;
use crate::storage::Storage;
use crate::vector::core::distance::DistanceMetric;
use crate::vector::core::quantization::QuantizedVectorMeta;
use crate::vector::core::vector::Vector;
use crate::vector::index::format::{QuantHeader, VectorSegmentHeader};
use crate::vector::index::quantized_io::quantized_record_payload_size;
use crate::vector::index::quantized_storage::QuantizedVectorPool;
use crate::vector::index::storage::VectorStorage;
use crate::vector::reader::{ValidationReport, VectorIndexMetadata, VectorStats};
use crate::vector::reader::{VectorIndexReader, VectorIterator};
use std::io::SeekFrom;

/// Reader for IVF (Inverted File) vector indexes.
///
/// Maintains a per-cluster inverted list (`cluster_to_vectors`) so that the
/// [`IvfSearcher`](super::searcher::IvfSearcher) can restrict distance
/// computations to vectors belonging to the `n_probe` nearest clusters.
#[derive(Debug)]
pub struct IvfIndexReader {
    vectors: VectorStorage,
    vector_ids: Vec<(u64, String)>,
    dimension: usize,
    distance_metric: DistanceMetric,
    n_clusters: usize,
    n_probe: usize,
    centroids: Vec<Vector>,
    /// Per-cluster inverted list: `cluster_to_vectors[i]` contains the
    /// `(doc_id, field_name)` pairs assigned to cluster `i`.
    cluster_to_vectors: Vec<Vec<(u64, String)>>,
    deletion_bitmap: Option<Arc<DeletionBitmap>>,
    /// Pre-built per-field doc-id list (`field_name → Arc<[u64]>`). Built
    /// once at load so `doc_ids_for_field` returns a refcount-shared
    /// slice without re-cloning `vector_ids`. #405.
    vector_ids_by_field: HashMap<String, Arc<[u64]>>,
}

/// Group `vector_ids` by field name into refcount-shared slices.
fn build_vector_ids_by_field(vector_ids: &[(u64, String)]) -> HashMap<String, Arc<[u64]>> {
    let mut by_field: HashMap<String, Vec<u64>> = HashMap::new();
    for (doc_id, field_name) in vector_ids {
        by_field
            .entry(field_name.clone())
            .or_default()
            .push(*doc_id);
    }
    by_field
        .into_iter()
        .map(|(field, ids)| (field, Arc::<[u64]>::from(ids)))
        .collect()
}

impl IvfIndexReader {
    /// Create a reader from serialized bytes.
    pub fn from_bytes(_data: &[u8]) -> Result<Self> {
        Err(LaurusError::InvalidOperation(
            "from_bytes is deprecated, use load() instead".to_string(),
        ))
    }

    /// Load an IVF vector index from storage.
    ///
    /// # Arguments
    ///
    /// * `storage` - Shared storage backend (cloned into `OnDemand` for concurrent reads).
    /// * `path` - Base path/name for the index file (`.ivf` extension is appended).
    /// * `distance_metric` - Distance metric used for similarity computations.
    ///
    /// # Returns
    ///
    /// A new `IvfIndexReader` instance.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError`] on I/O or format errors.
    pub fn load(
        storage: Arc<dyn Storage>,
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

        // Read the Issue #481 Stage 1 vector segment header (LVS1)
        // before the inverted lists. Pre-Stage-1 segments are
        // rejected with IncompatibleFormat.
        let header = VectorSegmentHeader::read_from(&mut input)?;
        let params = match header.quant {
            QuantHeader::Scalar8Bit(p) => p,
            QuantHeader::ProductQuantization { .. } => {
                return Err(crate::error::LaurusError::NotImplemented(
                    "Product quantization (Issue #481 Stage 3) is HNSW-only; \
                     the IVF reader does not support PQ segments yet"
                        .to_string(),
                ));
            }
        };

        // Read inverted lists, preserving per-cluster grouping.
        let mut cluster_to_vectors: Vec<Vec<(u64, String)>> = Vec::with_capacity(n_clusters);

        let (vectors, vector_ids) = match storage.loading_mode() {
            crate::storage::LoadingMode::Eager => {
                let mut vector_ids = Vec::with_capacity(num_vectors);
                let mut records: Vec<(u64, String, Vec<u8>, QuantizedVectorMeta)> =
                    Vec::with_capacity(num_vectors);

                for _ in 0..n_clusters {
                    let mut list_size_buf = [0u8; 4];
                    input.read_exact(&mut list_size_buf)?;
                    let list_size = u32::from_le_bytes(list_size_buf) as usize;
                    let mut cluster_vecs = Vec::with_capacity(list_size);

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
                            LaurusError::InvalidOperation(format!(
                                "Invalid UTF-8 in field name: {}",
                                e
                            ))
                        })?;

                        // Read int8 + meta directly (no dequantize).
                        let mut int8 = vec![0u8; dimension];
                        input.read_exact(&mut int8)?;
                        let mut sum_q_buf = [0u8; 4];
                        let mut norm_q_buf = [0u8; 4];
                        input.read_exact(&mut sum_q_buf)?;
                        input.read_exact(&mut norm_q_buf)?;
                        let meta = QuantizedVectorMeta {
                            sum_q: u32::from_le_bytes(sum_q_buf),
                            norm_q: f32::from_le_bytes(norm_q_buf),
                        };

                        let key = (doc_id, field_name.clone());
                        cluster_vecs.push(key.clone());
                        vector_ids.push(key.clone());
                        records.push((doc_id, field_name, int8, meta));
                    }
                    cluster_to_vectors.push(cluster_vecs);
                }
                let pool = QuantizedVectorPool::build(params, dimension, records);
                (VectorStorage::OwnedQuantized(Arc::new(pool)), vector_ids)
            }
            crate::storage::LoadingMode::Lazy => {
                let mut offsets = HashMap::with_capacity(num_vectors);
                let mut vector_ids = Vec::with_capacity(num_vectors);
                let quant_payload_size = quantized_record_payload_size(dimension) as i64;

                for _ in 0..n_clusters {
                    let mut list_size_buf = [0u8; 4];
                    input.read_exact(&mut list_size_buf)?;
                    let list_size = u32::from_le_bytes(list_size_buf) as usize;
                    let mut cluster_vecs = Vec::with_capacity(list_size);

                    for _ in 0..list_size {
                        let start_offset = input.stream_position().map_err(LaurusError::Io)?;

                        let mut doc_id_buf = [0u8; 8];
                        input.read_exact(&mut doc_id_buf)?;
                        let doc_id = u64::from_le_bytes(doc_id_buf);

                        let mut field_name_len_buf = [0u8; 4];
                        input.read_exact(&mut field_name_len_buf)?;
                        let field_name_len = u32::from_le_bytes(field_name_len_buf) as usize;

                        let mut field_name_buf = vec![0u8; field_name_len];
                        input.read_exact(&mut field_name_buf)?;
                        let field_name = String::from_utf8(field_name_buf).map_err(|e| {
                            LaurusError::InvalidOperation(format!(
                                "Invalid UTF-8 in field name: {}",
                                e
                            ))
                        })?;

                        let key = (doc_id, field_name);
                        offsets.insert(key.clone(), start_offset);
                        cluster_vecs.push(key.clone());
                        vector_ids.push(key);

                        // Skip int8 payload + per-vector meta.
                        input
                            .seek(SeekFrom::Current(quant_payload_size))
                            .map_err(LaurusError::Io)?;
                    }
                    cluster_to_vectors.push(cluster_vecs);
                }
                (
                    VectorStorage::OnDemand {
                        storage: storage.clone(),
                        file_name: file_name.clone(),
                        offsets: Arc::new(offsets),
                        quant_params: Some(params),
                    },
                    vector_ids,
                )
            }
        };

        let vector_ids_by_field = build_vector_ids_by_field(&vector_ids);
        Ok(Self {
            vectors,
            vector_ids,
            dimension,
            distance_metric,
            n_clusters,
            n_probe,
            centroids,
            cluster_to_vectors,
            deletion_bitmap: None,
            vector_ids_by_field,
        })
    }

    pub fn set_deletion_bitmap(&mut self, bitmap: Arc<DeletionBitmap>) {
        self.deletion_bitmap = Some(bitmap);
    }

    /// Borrow the underlying [`VectorStorage`] so the IVF searcher
    /// can detect the [`VectorStorage::OwnedQuantized`] variant and
    /// switch to the int8 hot path (Issue #481 Stage 1 Step 7).
    pub fn vectors(&self) -> &VectorStorage {
        &self.vectors
    }

    fn is_deleted(&self, doc_id: u64) -> bool {
        if let Some(bitmap) = &self.deletion_bitmap {
            bitmap.is_deleted(doc_id)
        } else {
            false
        }
    }

    /// Get IVF parameters.
    pub fn ivf_params(&self) -> (usize, usize) {
        (self.n_clusters, self.n_probe)
    }

    /// Get centroids.
    pub fn centroids(&self) -> &[Vector] {
        &self.centroids
    }

    /// Returns the vector IDs assigned to the given cluster index.
    ///
    /// # Arguments
    ///
    /// * `cluster_idx` - Zero-based cluster index.
    ///
    /// # Returns
    ///
    /// A slice of `(doc_id, field_name)` pairs, or an empty slice if
    /// `cluster_idx` is out of range.
    pub fn cluster_vectors(&self, cluster_idx: usize) -> &[(u64, String)] {
        self.cluster_to_vectors
            .get(cluster_idx)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }
}

impl VectorIndexReader for IvfIndexReader {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn get_vector(&self, doc_id: u64, field_name: &str) -> Result<Option<Vector>> {
        if self.is_deleted(doc_id) {
            return Ok(None);
        }
        self.vectors
            .get(&(doc_id, field_name.to_string()), self.dimension)
    }

    fn get_vectors_for_doc(&self, doc_id: u64) -> Result<Vec<(String, Vector)>> {
        let mut result = Vec::new();
        for (id, field) in &self.vector_ids {
            if *id == doc_id
                && !self.is_deleted(*id)
                && let Some(vec) = self.vectors.get(&(*id, field.clone()), self.dimension)?
            {
                result.push((field.clone(), vec));
            }
        }
        Ok(result)
    }

    fn get_vectors(&self, doc_ids: &[(u64, String)]) -> Result<Vec<Option<Vector>>> {
        let mut result = Vec::with_capacity(doc_ids.len());
        for (id, field) in doc_ids {
            if self.is_deleted(*id) {
                result.push(None);
            } else {
                result.push(self.vectors.get(&(*id, field.clone()), self.dimension)?);
            }
        }
        Ok(result)
    }

    fn vector_ids(&self) -> Result<Vec<(u64, String)>> {
        Ok(self.vector_ids.clone())
    }

    fn doc_ids_for_field(&self, field_name: &str) -> Arc<[u64]> {
        // O(1) HashMap lookup + Arc clone (refcount bump). Default impl
        // would clone `vector_ids` (Vec<(u64, String)>) and filter
        // linearly per call. #405.
        self.vector_ids_by_field
            .get(field_name)
            .cloned()
            .unwrap_or_else(|| Vec::<u64>::new().into())
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
            if *id >= start_doc_id
                && *id < end_doc_id
                && !self.is_deleted(*id)
                && let Some(vec) = self.vectors.get(&(*id, field.clone()), self.dimension)?
            {
                result.push((*id, field.clone(), vec));
            }
        }
        Ok(result)
    }

    fn get_vectors_by_field(&self, field_name: &str) -> Result<Vec<(u64, Vector)>> {
        let mut result = Vec::new();
        for (id, field) in &self.vector_ids {
            if field == field_name
                && !self.is_deleted(*id)
                && let Some(vec) = self.vectors.get(&(*id, field.clone()), self.dimension)?
            {
                result.push((*id, vec));
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
            deletion_bitmap: self.deletion_bitmap.clone(),
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
            VectorStorage::OwnedQuantized(pool) => {
                for (id, field) in &self.vector_ids {
                    if !pool.contains(*id, field) {
                        errors.push(format!(
                            "Vector {}:{} found in keys but missing in quantized pool",
                            id, field
                        ));
                    }
                }
                warnings.push(
                    "OwnedQuantized mode: dimension / NaN checks skipped (int8 storage \
                     guarantees finite values within [offset, offset + 255*scale])"
                        .to_string(),
                );
            }
            VectorStorage::OwnedPq(pool) => {
                for (id, field) in &self.vector_ids {
                    if !pool.contains(*id, field) {
                        errors.push(format!(
                            "Vector {}:{} found in keys but missing in PQ pool",
                            id, field
                        ));
                    }
                }
                warnings.push(
                    "OwnedPq mode: dimension / NaN checks skipped (codes index into \
                     the trained codebook)"
                        .to_string(),
                );
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
    deletion_bitmap: Option<Arc<DeletionBitmap>>,
}

impl VectorIterator for IvfVectorIterator {
    fn next(&mut self) -> Result<Option<(u64, String, Vector)>> {
        // Use a loop instead of recursion to avoid stack overflow when
        // many consecutive entries are deleted.
        while self.current < self.keys.len() {
            let (doc_id, field) = &self.keys[self.current];

            // Skip deleted entries
            if let Some(bitmap) = &self.deletion_bitmap
                && bitmap.is_deleted(*doc_id)
            {
                self.current += 1;
                continue;
            }

            if let Some(vec) = self
                .storage
                .get(&(*doc_id, field.clone()), self.dimension)?
            {
                self.current += 1;
                return Ok(Some((*doc_id, field.clone(), vec)));
            } else {
                return Err(LaurusError::internal(format!(
                    "Vector {}:{} found in keys but missing in storage",
                    doc_id, field
                )));
            }
        }

        Ok(None)
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
