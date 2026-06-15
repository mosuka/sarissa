//! Flat vector index reader implementation.

use std::collections::HashMap;
use std::sync::Arc;

use crate::error::{LaurusError, Result};
use crate::storage::Storage;
use crate::vector::core::distance::DistanceMetric;
use crate::vector::core::quantization::QuantizedVectorMeta;
use crate::vector::core::vector::Vector;
use crate::vector::index::format::{QuantHeader, VectorSegmentHeader};
use crate::vector::index::quantized_io::quantized_record_payload_size;
use crate::vector::index::quantized_storage::QuantizedVectorPool;
use crate::vector::reader::{ValidationReport, VectorIndexMetadata, VectorStats};
use crate::vector::reader::{VectorIndexReader, VectorIterator};

use crate::maintenance::deletion::DeletionBitmap;
/// Storage for vectors (in-memory or on-demand).
use crate::vector::index::storage::VectorStorage;

/// Reader for flat (brute-force) vector indexes.
#[derive(Debug)]
pub struct FlatVectorIndexReader {
    vectors: VectorStorage,
    vector_ids: Vec<(u64, String)>,
    dimension: usize,
    distance_metric: DistanceMetric,
    deletion_bitmap: Option<Arc<DeletionBitmap>>,
    /// Pre-built per-field doc-id list (`field_name → Arc<[u64]>`). Built
    /// once at load so `doc_ids_for_field` returns a refcount-shared
    /// slice without re-cloning `vector_ids`. #405.
    vector_ids_by_field: std::collections::HashMap<String, Arc<[u64]>>,
}

/// Group `vector_ids` by field name into refcount-shared slices.
fn build_vector_ids_by_field(
    vector_ids: &[(u64, String)],
) -> std::collections::HashMap<String, Arc<[u64]>> {
    let mut by_field: std::collections::HashMap<String, Vec<u64>> =
        std::collections::HashMap::new();
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

impl FlatVectorIndexReader {
    /// Create a reader from serialized bytes.
    pub fn from_bytes(_data: &[u8]) -> Result<Self> {
        Err(LaurusError::InvalidOperation(
            "from_bytes is deprecated, use load() instead".to_string(),
        ))
    }

    /// Load a flat vector index from storage.
    ///
    /// # Arguments
    ///
    /// * `storage` - Shared storage backend (cloned into `OnDemand` for concurrent reads).
    /// * `path` - Base path/name for the index file (`.flat` extension is appended).
    /// * `distance_metric` - Distance metric used for similarity computations.
    ///
    /// # Returns
    ///
    /// A new `FlatIndexReader` instance.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError`] on I/O or format errors.
    pub fn load(
        storage: Arc<dyn Storage>,
        path: &str,
        distance_metric: DistanceMetric,
    ) -> Result<Self> {
        use crate::vector::index::alloc_bounds::{checked_capacity, checked_len};
        use std::io::{Read, Seek};

        // Open the index file
        let file_name = format!("{}.flat", path);
        let mut input = storage.open_input(&file_name)?;

        // Ground truth for bounding allocations sized from the unverified
        // header counts below (Issue #806). The `.flat` reader has no
        // pre-parse checksum, so every header count reaches its allocation
        // unverified.
        let file_size = input.size()?;

        // Read metadata
        let mut num_vectors_buf = [0u8; 4];
        input.read_exact(&mut num_vectors_buf)?;
        let num_vectors = u32::from_le_bytes(num_vectors_buf) as usize;

        let mut dimension_buf = [0u8; 4];
        input.read_exact(&mut dimension_buf)?;
        let dimension = u32::from_le_bytes(dimension_buf) as usize;

        // Read the Issue #481 Stage 1 vector segment header (LVS1).
        // Pre-Stage-1 segments are rejected with IncompatibleFormat.
        let header = VectorSegmentHeader::read_from(&mut input)?;
        let params = match header.quant {
            QuantHeader::Scalar8Bit(p) => p,
            QuantHeader::ProductQuantization { .. } => {
                return Err(crate::error::LaurusError::NotImplemented(
                    "Product quantization (Issue #481 Stage 3) is HNSW-only; \
                     the Flat reader does not support PQ segments yet"
                        .to_string(),
                ));
            }
            #[cfg(feature = "pq-fastscan")]
            QuantHeader::ProductQuantizationFastScan { .. } => {
                return Err(crate::error::LaurusError::NotImplemented(
                    "PQ FastScan (#695) is HNSW-only; the Flat reader does not \
                     support PQ FastScan segments"
                        .to_string(),
                ));
            }
        };

        // Bytes left for the per-vector records section, captured once at its
        // start (Issue #806). Each record is at least doc_id (8) +
        // field_name_len (4) + the fixed quantized payload (dim int8 + 8 meta),
        // so this stride also bounds the per-record `dimension`-sized int8 read.
        let records_remaining =
            file_size.saturating_sub(input.stream_position().map_err(LaurusError::Io)?);
        let record_stride = 12 + quantized_record_payload_size(dimension) as u64;

        let (vectors, vector_ids) = match storage.loading_mode() {
            crate::storage::LoadingMode::Eager => {
                // Step 7 of #481 Stage 1: load vectors as int8 + meta
                // directly into a QuantizedVectorPool.
                checked_capacity(
                    num_vectors,
                    record_stride,
                    records_remaining,
                    "flat num_vectors",
                )?;
                let mut vector_ids = Vec::with_capacity(num_vectors);
                let mut records: Vec<(u64, String, Vec<u8>, QuantizedVectorMeta)> =
                    Vec::with_capacity(num_vectors);

                for _ in 0..num_vectors {
                    let mut doc_id_buf = [0u8; 8];
                    input.read_exact(&mut doc_id_buf)?;
                    let doc_id = u64::from_le_bytes(doc_id_buf);

                    // Read field name
                    let mut field_name_len_buf = [0u8; 4];
                    input.read_exact(&mut field_name_len_buf)?;
                    let field_name_len = u32::from_le_bytes(field_name_len_buf) as usize;
                    checked_len(field_name_len, records_remaining, "flat field_name_len")?;

                    let mut field_name_buf = vec![0u8; field_name_len];
                    input.read_exact(&mut field_name_buf)?;
                    let field_name = String::from_utf8(field_name_buf).map_err(|e| {
                        LaurusError::InvalidOperation(format!("Invalid UTF-8 in field name: {}", e))
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

                    vector_ids.push((doc_id, field_name.clone()));
                    records.push((doc_id, field_name, int8, meta));
                }
                let pool = QuantizedVectorPool::build(params, dimension, records);
                (VectorStorage::OwnedQuantized(Arc::new(pool)), vector_ids)
            }
            crate::storage::LoadingMode::Lazy => {
                checked_capacity(
                    num_vectors,
                    record_stride,
                    records_remaining,
                    "flat num_vectors",
                )?;
                let mut offsets = HashMap::with_capacity(num_vectors);
                let mut vector_ids = Vec::with_capacity(num_vectors);

                // Seek to start of per-vector entries: Flat preamble
                // (count u32 + dim u32 = 8 bytes) + VectorSegmentHeader
                // (Stage-1 Scalar8Bit = 24 bytes) = 32 bytes.
                let start_pos =
                    8u64 + VectorSegmentHeader::scalar_8bit(params).serialized_size() as u64;
                input
                    .seek(std::io::SeekFrom::Start(start_pos))
                    .map_err(LaurusError::Io)?;

                let quant_payload_size = quantized_record_payload_size(dimension) as i64;

                for _ in 0..num_vectors {
                    let start_offset = input.stream_position().map_err(LaurusError::Io)?;

                    let mut doc_id_buf = [0u8; 8];
                    input.read_exact(&mut doc_id_buf)?;
                    let doc_id = u64::from_le_bytes(doc_id_buf);

                    let mut field_name_len_buf = [0u8; 4];
                    input.read_exact(&mut field_name_len_buf)?;
                    let field_name_len = u32::from_le_bytes(field_name_len_buf) as usize;
                    checked_len(field_name_len, records_remaining, "flat field_name_len")?;

                    let mut field_name_buf = vec![0u8; field_name_len];
                    input.read_exact(&mut field_name_buf)?;
                    let field_name = String::from_utf8(field_name_buf).map_err(|e| {
                        LaurusError::InvalidOperation(format!("Invalid UTF-8 in field name: {}", e))
                    })?;

                    offsets.insert((doc_id, field_name.clone()), start_offset);
                    vector_ids.push((doc_id, field_name));

                    // Skip int8 payload + per-vector meta.
                    input
                        .seek(std::io::SeekFrom::Current(quant_payload_size))
                        .map_err(LaurusError::Io)?;
                }

                (
                    VectorStorage::OnDemand {
                        storage: storage.clone(),
                        file_name: file_name.clone(),
                        offsets: Arc::new(offsets),
                        quant_params: Some(params),
                        cached_input: Arc::new(std::sync::RwLock::new(None)),
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
            deletion_bitmap: None,
            vector_ids_by_field,
        })
    }

    pub fn set_deletion_bitmap(&mut self, bitmap: Arc<DeletionBitmap>) {
        self.deletion_bitmap = Some(bitmap);
    }

    /// Borrow the underlying [`VectorStorage`] so the Flat searcher
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
}

impl VectorIndexReader for FlatVectorIndexReader {
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
        let _memory_usage = match &self.vectors {
            VectorStorage::Owned(vectors) => vectors.len() * (8 + self.dimension * 4),
            VectorStorage::OwnedQuantized(pool) => pool.data.len(),
            VectorStorage::OwnedPq(pool) => pool.data.len() + pool.codebook.len() * 4,
            #[cfg(feature = "pq-fastscan")]
            VectorStorage::OwnedPqFastScan(_) => {
                unreachable!("Flat reader rejects PQ FastScan at the segment header (HNSW-only)")
            }
            VectorStorage::OnDemand { offsets, .. } => {
                // Estimate memory for offsets map + ID list
                offsets.len() * (8 + 32 + 8) // Key + Valid + Offset roughly
            }
        };
        VectorStats {
            vector_count: self.vectors.len(),
            dimension: self.dimension,
            memory_usage: self.vectors.len() * (8 + self.dimension * 4),
            build_time_ms: 0,
        }
    }

    fn contains_vector(&self, doc_id: u64, field_name: &str) -> bool {
        match &self.vectors {
            VectorStorage::Owned(vectors) => {
                vectors.contains_key(&(doc_id, field_name.to_string()))
            }
            VectorStorage::OwnedQuantized(pool) => pool.contains(doc_id, field_name),
            VectorStorage::OwnedPq(pool) => pool.contains(doc_id, field_name),
            #[cfg(feature = "pq-fastscan")]
            VectorStorage::OwnedPqFastScan(_) => {
                unreachable!("Flat reader rejects PQ FastScan at the segment header (HNSW-only)")
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
        Ok(Box::new(FlatVectorIterator {
            storage: self.vectors.clone(),
            keys: self.vector_ids.clone(),
            current: 0,
            dimension: self.dimension,
            deletion_bitmap: self.deletion_bitmap.clone(),
        }))
    }

    fn metadata(&self) -> Result<VectorIndexMetadata> {
        Ok(VectorIndexMetadata {
            index_type: "flat".to_string(),
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
                     the trained codebook which is bounded by construction)"
                        .to_string(),
                );
            }
            #[cfg(feature = "pq-fastscan")]
            VectorStorage::OwnedPqFastScan(_) => {
                unreachable!("Flat reader rejects PQ FastScan at the segment header (HNSW-only)")
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

        Ok(ValidationReport {
            repair_suggestions: Vec::new(),
            is_valid: errors.is_empty(),
            errors,
            warnings,
        })
    }
}

/// Iterator for flat vector index.
struct FlatVectorIterator {
    storage: VectorStorage,
    keys: Vec<(u64, String)>,
    current: usize,
    dimension: usize,
    deletion_bitmap: Option<Arc<DeletionBitmap>>,
}

impl VectorIterator for FlatVectorIterator {
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

#[cfg(test)]
mod alloc_bound_tests {
    use super::*;
    use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};
    use crate::vector::core::quantization::ScalarQuantParams;
    use std::io::Write;

    /// Build an in-memory storage holding `bytes` under `name`.
    fn storage_with(name: &str, bytes: Vec<u8>) -> Arc<dyn Storage> {
        let storage = MemoryStorage::new(MemoryStorageConfig::default());
        let mut out = storage.create_output(name).unwrap();
        out.write_all(&bytes).unwrap();
        out.flush_and_sync().unwrap();
        Arc::new(storage)
    }

    /// Serialized neutral LVS1 (Scalar8Bit) header bytes.
    fn neutral_header_bytes() -> Vec<u8> {
        let mut buf = Vec::new();
        VectorSegmentHeader::scalar_8bit(ScalarQuantParams {
            offset: 0.0,
            scale: 1.0,
        })
        .write_to(&mut buf)
        .unwrap();
        buf
    }

    #[test]
    fn load_rejects_oversized_num_vectors_without_aborting() {
        // A `.flat` segment whose `num_vectors` field is corrupted to a huge
        // value while the file holds no records must be rejected cleanly,
        // never drive a multi-GiB `Vec::with_capacity` that aborts the
        // process via `handle_alloc_error` (Issue #806).
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&u32::MAX.to_le_bytes()); // num_vectors (corrupt)
        bytes.extend_from_slice(&4u32.to_le_bytes()); // dimension
        bytes.extend_from_slice(&neutral_header_bytes()); // LVS1 header, no records

        let storage = storage_with("corrupt.flat", bytes);
        let err = FlatVectorIndexReader::load(storage, "corrupt", DistanceMetric::Cosine)
            .expect_err("oversized num_vectors must be rejected as corruption");
        match err {
            LaurusError::Index(msg) => {
                assert!(msg.contains("num_vectors"), "got: {msg}");
                assert!(msg.contains("corrupted"), "got: {msg}");
            }
            other => panic!("expected Index error, got {other:?}"),
        }
    }
}
