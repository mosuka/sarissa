//! HNSW vector index reader implementation.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use crate::error::{LaurusError, Result};
use crate::storage::Storage;
use crate::vector::core::distance::DistanceMetric;
use crate::vector::core::quantization::QuantizedVectorMeta;
use crate::vector::core::vector::Vector;
use crate::vector::index::format::{QuantHeader, VectorSegmentHeader};
use crate::vector::index::hnsw::graph::HnswGraph;
use crate::vector::index::quantized_io::quantized_record_payload_size;
use crate::vector::index::quantized_storage::QuantizedVectorPool;
use crate::vector::index::rerank_sidecar::read_sidecar;
use crate::vector::index::rerank_storage::RerankStoragePool;
use crate::vector::reader::{ValidationReport, VectorIndexMetadata, VectorStats};
use crate::vector::reader::{VectorIndexReader, VectorIterator};

use crate::maintenance::deletion::DeletionBitmap;
/// Storage for vectors (in-memory or on-demand).
use crate::vector::index::storage::VectorStorage;

/// Build a `field_name → Arc<[u64]>` lookup from a flat `Vec<(u64, String)>`.
///
/// The grouping happens once at reader load so search-time
/// `doc_ids_for_field` is a single HashMap lookup + Arc clone.
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
    deletion_bitmap: Option<Arc<DeletionBitmap>>,
    /// Pre-built lookup table: `field_name → (doc_id → Vec<f32> base address as usize)`.
    ///
    /// Populated at load time for in-memory (`Owned`) storage; empty for
    /// on-demand (disk-backed) storage.  Enables zero-allocation, O(1) software
    /// prefetch hints on the HNSW search hot-path without the per-call
    /// `String` allocation that a direct `HashMap<(u64, String), _>` lookup
    /// would require.
    ///
    /// The address is stored as `usize` so that the struct remains `Send + Sync`.
    /// The pointer is valid for the lifetime of `self` because the backing
    /// `Arc<Vec<f32>>` is kept alive by `VectorStorage::Owned`.
    prefetch_index: HashMap<String, HashMap<u64, usize>>,

    /// Pre-built per-field doc-id list (`field_name → Arc<[u64]>`).
    ///
    /// Built once at load time so `doc_ids_for_field` returns a refcount-
    /// shared slice (no per-call allocation, no `Vec<(u64, String)>` clone,
    /// no linear filter). #405 (per-field `vector_ids` cache).
    vector_ids_by_field: HashMap<String, Arc<[u64]>>,

    /// Optional Stage 2 rerank storage pool (Issue #481).
    ///
    /// `Some(_)` only when (a) the LRS1 sidecar exists alongside the
    /// main `.hnsw` file and (b) the storage loading mode is Eager.
    /// Lazy mode silently skips the sidecar so the on-disk read pattern
    /// stays small; Stage 1 segments without a sidecar always yield
    /// `None`. The HNSW searcher reads this to decide whether to
    /// activate the two-stage rerank flow.
    pub rerank_storage: Option<Arc<RerankStoragePool>>,
}

impl HnswIndexReader {
    /// Create a reader from serialized bytes.
    pub fn from_bytes(_data: &[u8]) -> Result<Self> {
        Err(LaurusError::InvalidOperation(
            "from_bytes is deprecated, use load() instead".to_string(),
        ))
    }

    /// Verify the CRC-32 footer of a `.hnsw` segment if present (Issue #786).
    ///
    /// New segments end with `[magic u32][crc-32 u32]` over all preceding
    /// bytes (written via [`crate::storage::checksum::CrcWriter`]). The footer
    /// is detected by the magic at `size - 8`; legacy segments without it skip
    /// verification (back-compat). On a present footer the CRC over the
    /// content is recomputed and compared. The input is left rewound to the
    /// start so the caller's structural parse runs unchanged.
    ///
    /// # Arguments
    ///
    /// * `input` - The opened `.hnsw` segment stream.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError::index`] if a footer is present but its checksum
    /// does not match the content (corruption), or on I/O failure.
    fn verify_checksum_footer(input: &mut dyn crate::storage::StorageInput) -> Result<()> {
        use std::io::{Read, SeekFrom};

        let size = input.size()?;
        if size >= crate::vector::index::hnsw::HNSW_FOOTER_LEN {
            let content_len = size - crate::vector::index::hnsw::HNSW_FOOTER_LEN;
            input.seek(SeekFrom::Start(content_len))?;
            let mut footer = [0u8; 8];
            input.read_exact(&mut footer)?;
            let magic = u32::from_le_bytes([footer[0], footer[1], footer[2], footer[3]]);
            if magic == crate::vector::index::hnsw::HNSW_FOOTER_MAGIC {
                let stored = u32::from_le_bytes([footer[4], footer[5], footer[6], footer[7]]);
                input.seek(SeekFrom::Start(0))?;
                let mut crc_in = crate::storage::checksum::CrcReader::new(&mut *input);
                let mut remaining = content_len;
                let mut buf = [0u8; 64 * 1024];
                while remaining > 0 {
                    let want = remaining.min(buf.len() as u64) as usize;
                    crc_in.read_exact(&mut buf[..want])?;
                    remaining -= want as u64;
                }
                let computed = crc_in.checksum();
                if computed != stored {
                    return Err(LaurusError::index(
                        "HNSW segment checksum mismatch: .hnsw file is corrupted",
                    ));
                }
            }
        }
        input.seek(SeekFrom::Start(0))?;
        Ok(())
    }

    /// Load an HNSW vector index from storage.
    ///
    /// # Arguments
    ///
    /// * `storage` - Shared storage backend (cloned into `OnDemand` for concurrent reads).
    /// * `path` - Base path/name for the index file (`.hnsw` extension is appended).
    /// * `distance_metric` - Distance metric used for similarity computations.
    ///
    /// # Returns
    ///
    /// A new `HnswIndexReader` instance.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError`] on I/O or format errors.
    pub fn load(
        storage: Arc<dyn Storage>,
        path: &str,
        distance_metric: DistanceMetric,
    ) -> Result<Self> {
        use std::io::Read;

        // Open the index file
        let file_name = format!("{}.hnsw", path);
        let mut input = storage.open_input(&file_name)?;

        // Verify the CRC-32 footer if present (Issue #786), then rewind. This
        // is an independent sequential pass that does not touch the structural
        // parse below (which stops at the end of the graph and never reads the
        // trailing footer), so it works for every loading mode and leaves
        // legacy footer-less segments untouched.
        Self::verify_checksum_footer(&mut *input)?;

        // Read metadata (vector count stored as u64)
        let mut num_vectors_buf = [0u8; 8];
        input.read_exact(&mut num_vectors_buf)?;
        let num_vectors = u64::from_le_bytes(num_vectors_buf) as usize;

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

                    let mut node_count_buf = [0u8; 8];
                    input.read_exact(&mut node_count_buf)?;
                    let node_count = u64::from_le_bytes(node_count_buf) as usize;

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

                    Ok(Some(Arc::new(HnswGraph::new(
                        entry_point,
                        max_level,
                        nodes,
                        m,
                        m,
                        m * 2,
                        ef_construction,
                        1.0 / (m as f64).ln(),
                    ))))
                } else {
                    Ok(None)
                }
            };

        // Read the Issue #481 vector segment header (LVS1). Pre-Stage-1
        // segments are rejected with IncompatibleFormat. Both Scalar8Bit
        // (Stage 1) and ProductQuantization (Stage 3) variants are
        // handled here; Lazy mode silently degrades PQ to "not
        // supported" because the OnDemand path's offsets table only
        // carries Scalar8Bit params today.
        let header = VectorSegmentHeader::read_from(&mut input)?;

        let (vectors, vector_ids, graph) = match (&header.quant, storage.loading_mode()) {
            (QuantHeader::Scalar8Bit(params), crate::storage::LoadingMode::Eager) => {
                // Step 6 of #481 Stage 1: load vectors as int8 + meta
                // directly into a QuantizedVectorPool so the search
                // hot loop can use distance_quantized without per-call
                // dequantization. The legacy
                // VectorIndexReader::get_vector API still works via
                // VectorStorage::OwnedQuantized's dequantize-on-get.
                let mut vector_ids = Vec::with_capacity(num_vectors);
                let mut records: Vec<(u64, String, Vec<u8>, QuantizedVectorMeta)> =
                    Vec::with_capacity(num_vectors);

                for _ in 0..num_vectors {
                    let mut doc_id_buf = [0u8; 8];
                    input.read_exact(&mut doc_id_buf)?;
                    let doc_id = u64::from_le_bytes(doc_id_buf);

                    let mut field_name_len_buf = [0u8; 4];
                    input.read_exact(&mut field_name_len_buf)?;
                    let field_name_len = u32::from_le_bytes(field_name_len_buf) as usize;
                    let mut field_name_buf = vec![0u8; field_name_len];
                    input.read_exact(&mut field_name_buf)?;
                    let field_name = String::from_utf8(field_name_buf).map_err(|e| {
                        LaurusError::InvalidOperation(format!("Invalid UTF-8 in field name: {}", e))
                    })?;

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
                let graph = read_graph(&mut input)?;
                let pool = QuantizedVectorPool::build(*params, dimension, records);
                (
                    VectorStorage::OwnedQuantized(Arc::new(pool)),
                    vector_ids,
                    graph,
                )
            }
            (QuantHeader::Scalar8Bit(params), crate::storage::LoadingMode::Lazy) => {
                let mut offsets = HashMap::with_capacity(num_vectors);
                let mut vector_ids = Vec::with_capacity(num_vectors);

                // Seek to start of per-vector entries: HNSW preamble
                // (count u64 + dim u32 + m u32 + ef u32 = 20 bytes)
                // followed by VectorSegmentHeader (Stage-1, 24 bytes
                // for Scalar8Bit) = 44 bytes.
                let start_pos =
                    20u64 + VectorSegmentHeader::scalar_8bit(*params).serialized_size() as u64;
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

                    let mut field_name_buf = vec![0u8; field_name_len];
                    input.read_exact(&mut field_name_buf)?;
                    let field_name = String::from_utf8(field_name_buf).map_err(|e| {
                        LaurusError::InvalidOperation(format!("Invalid UTF-8 in field name: {}", e))
                    })?;

                    offsets.insert((doc_id, field_name.clone()), start_offset);
                    vector_ids.push((doc_id, field_name.clone()));

                    input
                        .seek(std::io::SeekFrom::Current(quant_payload_size))
                        .map_err(LaurusError::Io)?;
                }
                let graph = read_graph(&mut input)?;

                (
                    VectorStorage::OnDemand {
                        storage: storage.clone(),
                        file_name: file_name.clone(),
                        offsets: Arc::new(offsets),
                        quant_params: Some(*params),
                        cached_input: Arc::new(std::sync::RwLock::new(None)),
                    },
                    vector_ids,
                    graph,
                )
            }
            (
                QuantHeader::ProductQuantization {
                    params: pq_params,
                    codebook,
                },
                _loading_mode,
            ) => {
                // Stage 3 (#481): read M-byte codes per vector into a
                // PqVectorPool. Lazy mode is not yet supported for PQ
                // segments (the OnDemand path's offsets table only
                // carries Scalar8Bit params), so we eagerly load the
                // codes regardless of `loading_mode`.
                let mut vector_ids = Vec::with_capacity(num_vectors);
                let mut records: Vec<(u64, String, Vec<u8>)> = Vec::with_capacity(num_vectors);
                let codes_size = pq_params.m as usize;

                for _ in 0..num_vectors {
                    let mut doc_id_buf = [0u8; 8];
                    input.read_exact(&mut doc_id_buf)?;
                    let doc_id = u64::from_le_bytes(doc_id_buf);

                    let mut field_name_len_buf = [0u8; 4];
                    input.read_exact(&mut field_name_len_buf)?;
                    let field_name_len = u32::from_le_bytes(field_name_len_buf) as usize;
                    let mut field_name_buf = vec![0u8; field_name_len];
                    input.read_exact(&mut field_name_buf)?;
                    let field_name = String::from_utf8(field_name_buf).map_err(|e| {
                        LaurusError::InvalidOperation(format!("Invalid UTF-8 in field name: {}", e))
                    })?;

                    let mut codes = vec![0u8; codes_size];
                    input.read_exact(&mut codes)?;

                    vector_ids.push((doc_id, field_name.clone()));
                    records.push((doc_id, field_name, codes));
                }
                let graph = read_graph(&mut input)?;
                let pool = crate::vector::index::pq_storage::PqVectorPool::build(
                    *pq_params,
                    codebook.clone(),
                    records,
                );
                (VectorStorage::OwnedPq(Arc::new(pool)), vector_ids, graph)
            }
            #[cfg(feature = "pq-fastscan")]
            (
                QuantHeader::ProductQuantizationFastScan {
                    params: pq_params,
                    codebook,
                },
                _loading_mode,
            ) => {
                // Mirror the K=256 PQ load path but read 4-bit packed
                // codes via `pq_fastscan_io::read_pq_fastscan_record`
                // and build a `PqFastScanPool` for the SIMD-friendly
                // block-transposed in-memory layout.
                let mut vector_ids = Vec::with_capacity(num_vectors);
                let mut records: Vec<(u64, String, Vec<u8>)> = Vec::with_capacity(num_vectors);

                for _ in 0..num_vectors {
                    let mut doc_id_buf = [0u8; 8];
                    input.read_exact(&mut doc_id_buf)?;
                    let doc_id = u64::from_le_bytes(doc_id_buf);

                    let mut field_name_len_buf = [0u8; 4];
                    input.read_exact(&mut field_name_len_buf)?;
                    let field_name_len = u32::from_le_bytes(field_name_len_buf) as usize;
                    let mut field_name_buf = vec![0u8; field_name_len];
                    input.read_exact(&mut field_name_buf)?;
                    let field_name = String::from_utf8(field_name_buf).map_err(|e| {
                        LaurusError::InvalidOperation(format!("Invalid UTF-8 in field name: {}", e))
                    })?;

                    let codes = crate::vector::index::pq_fastscan_io::read_pq_fastscan_record(
                        &mut input, *pq_params,
                    )?;

                    vector_ids.push((doc_id, field_name.clone()));
                    records.push((doc_id, field_name, codes));
                }
                let graph = read_graph(&mut input)?;
                let pool = crate::vector::index::pq_fastscan_storage::PqFastScanPool::build(
                    *pq_params,
                    codebook.clone(),
                    records,
                )?;
                (
                    VectorStorage::OwnedPqFastScan(Arc::new(pool)),
                    vector_ids,
                    graph,
                )
            }
        };

        // Build zero-allocation prefetch lookup. For OwnedQuantized
        // (Step 6+), the address points to the int8 record start
        // inside the contiguous AoS buffer; the int8 payload + 8 bytes
        // of meta will be prefetched per neighbour. For Owned (legacy
        // f32, kept for symmetry), the address points to the Vec<f32>
        // data. Empty for OnDemand where CPU cache hints don't apply.
        let prefetch_index: HashMap<String, HashMap<u64, usize>> = match &vectors {
            VectorStorage::Owned(map) => {
                let mut idx: HashMap<String, HashMap<u64, usize>> = HashMap::new();
                for ((doc_id, field_name), vector) in map.iter() {
                    idx.entry(field_name.clone())
                        .or_default()
                        .insert(*doc_id, vector.data.as_ptr() as usize);
                }
                idx
            }
            VectorStorage::OwnedQuantized(pool) => {
                let mut idx: HashMap<String, HashMap<u64, usize>> = HashMap::new();
                let record_size = QuantizedVectorPool::record_size(pool.dim);
                let base = pool.data.as_ptr();
                for (field_name, doc_map) in pool.field_index.iter() {
                    let entry = idx.entry(field_name.clone()).or_default();
                    for (&doc_id, &pos) in doc_map.iter() {
                        // SAFETY: pool is held alive by self.vectors
                        // (Arc) for the lifetime of self; pos is in
                        // bounds because it was populated from data.len()
                        // / record_size at build time.
                        let addr = unsafe { base.add(pos as usize * record_size) } as usize;
                        entry.insert(doc_id, addr);
                    }
                }
                idx
            }
            VectorStorage::OwnedPq(_) => {
                // PQ records are M bytes each (8-32 bytes) — small
                // enough that prefetching adds no benefit; the LUT is
                // the more important cache occupant.
                HashMap::new()
            }
            #[cfg(feature = "pq-fastscan")]
            VectorStorage::OwnedPqFastScan(_) => {
                // FastScan uses 4-bit packed codes; the SIMD kernel
                // streams entire 32-vector blocks (~16M bytes/block
                // for typical M) so per-vector prefetch hints add
                // nothing on top of the natural sequential access.
                HashMap::new()
            }
            VectorStorage::OnDemand { .. } => HashMap::new(),
        };

        // Per-field doc-id lookup (#405). Built from `vector_ids` so it
        // reflects the same set the search path used to filter at every
        // call; finalised into `Arc<[u64]>` for cheap clone semantics.
        let vector_ids_by_field = build_vector_ids_by_field(&vector_ids);

        // Stage 2 rerank sidecar (Issue #481). Loaded eagerly into a
        // RerankStoragePool when (a) a `<file_name>.f32` sidecar exists
        // and (b) we are in Eager loading mode. Lazy mode skips the
        // sidecar to honor its memory-savings promise — Stage 2 segments
        // opened in Lazy mode silently degrade to Stage 1 (no rerank).
        // The pool's vector positions are paired with `vector_ids` (the
        // same order as the LVS1 segment), giving an identity mapping.
        let rerank_storage = match storage.loading_mode() {
            crate::storage::LoadingMode::Eager => {
                let sidecar_name = format!("{}.f32", file_name);
                if storage.file_exists(&sidecar_name) {
                    let mut sidecar_in = storage.open_input(&sidecar_name)?;
                    let (header, payload) = read_sidecar(&mut sidecar_in)?;
                    if header.dim as usize != dimension {
                        return Err(LaurusError::InvalidOperation(format!(
                            "rerank sidecar dim mismatch: LVS1 segment uses {dimension}, \
                             sidecar uses {}",
                            header.dim
                        )));
                    }
                    if header.vector_count as usize != vector_ids.len() {
                        return Err(LaurusError::InvalidOperation(format!(
                            "rerank sidecar vector_count mismatch: LVS1 segment has {} vectors, \
                             sidecar has {}",
                            vector_ids.len(),
                            header.vector_count
                        )));
                    }
                    let pool = RerankStoragePool::from_sidecar_payload(
                        header.storage_kind,
                        dimension,
                        header.vector_count as usize,
                        payload,
                        &vector_ids,
                    );
                    Some(Arc::new(pool))
                } else {
                    None
                }
            }
            crate::storage::LoadingMode::Lazy => None,
        };

        Ok(Self {
            vectors,
            vector_ids,
            dimension,
            distance_metric,
            m,
            ef_construction,
            graph,
            deletion_bitmap: None,
            prefetch_index,
            vector_ids_by_field,
            rerank_storage,
        })
    }

    pub fn set_deletion_bitmap(&mut self, bitmap: Arc<DeletionBitmap>) {
        self.deletion_bitmap = Some(bitmap);
    }

    /// Returns whether `doc_id` has been logically deleted.
    ///
    /// `pub(crate)` so the HNSW searcher can consult it during graph traversal
    /// (Issue #665): a deleted node must not be admitted to the result heap,
    /// otherwise it consumes an `ef_search` slot and leaks into results (the
    /// quantized distance path never calls `get_vector`, the only place
    /// deletions were previously honoured).
    ///
    /// # Arguments
    ///
    /// * `doc_id` - Internal document ID to test.
    ///
    /// # Returns
    ///
    /// `true` if a deletion bitmap is attached and marks `doc_id` deleted;
    /// `false` when no bitmap is attached.
    pub(crate) fn is_deleted(&self, doc_id: u64) -> bool {
        if let Some(bitmap) = &self.deletion_bitmap {
            bitmap.is_deleted(doc_id)
        } else {
            false
        }
    }

    /// Returns whether this reader has any logically deleted documents.
    ///
    /// Used by [`HnswSearcher`](crate::vector::index::hnsw::HnswSearcher) to
    /// decide whether graph traversal needs the per-neighbour deletion
    /// bookkeeping at all. When no bitmap is attached, or it is attached but
    /// empty (e.g. a freshly initialized segment), this returns `false` so the
    /// pristine, no-bookkeeping search path is preserved unchanged.
    ///
    /// # Returns
    ///
    /// `true` if a deletion bitmap is attached and holds at least one deleted
    /// document; `false` otherwise.
    pub(crate) fn has_deletions(&self) -> bool {
        self.deletion_bitmap
            .as_ref()
            .is_some_and(|bitmap| bitmap.deleted_count.load(Ordering::Relaxed) > 0)
    }

    /// Get HNSW parameters.
    pub fn hnsw_params(&self) -> (usize, usize) {
        (self.m, self.ef_construction)
    }

    /// Returns a reference to the per-field prefetch lookup table for `field_name`.
    ///
    /// The returned map provides O(1), zero-allocation access from `doc_id` to
    /// the base address of the corresponding `Vec<f32>` data, allowing
    /// [`HnswSearcher`] to issue software prefetch hints on the search hot-path
    /// without the `String` allocation that a direct `HashMap<(u64, String), _>`
    /// lookup would require.
    ///
    /// Returns `None` for on-demand (disk-backed) storage where prefetching is
    /// not applicable.
    ///
    /// # Arguments
    ///
    /// * `field_name` - The name of the vector field.
    pub(crate) fn field_prefetch_index(&self, field_name: &str) -> Option<&HashMap<u64, usize>> {
        self.prefetch_index.get(field_name)
    }

    /// Borrow the underlying
    /// [`crate::vector::index::storage::VectorStorage`].
    ///
    /// Intended for the HNSW searcher to detect the
    /// [`crate::vector::index::storage::VectorStorage::OwnedQuantized`]
    /// variant and switch to the int8 hot path (Issue #481 Stage 1
    /// Step 6).
    pub fn vectors(&self) -> &crate::vector::index::storage::VectorStorage {
        &self.vectors
    }

    /// Borrow the optional Stage 2 rerank storage pool.
    ///
    /// Returns `Some(_)` only when this reader was loaded against a
    /// segment whose LRS1 sidecar was present and the storage loading
    /// mode allowed eager sidecar load. The HNSW searcher consults
    /// this to decide whether the two-stage rerank flow is available.
    pub fn rerank_storage(&self) -> Option<&Arc<RerankStoragePool>> {
        self.rerank_storage.as_ref()
    }
}

impl VectorIndexReader for HnswIndexReader {
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
        // O(1) HashMap lookup + Arc clone (refcount bump). Compared to
        // the default impl this avoids the per-call `Vec<(u64, String)>`
        // clone of the full corpus and the linear filter scan. #405.
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
        let memory_usage = match &self.vectors {
            VectorStorage::Owned(vectors) => vectors.len() * (8 + self.dimension * 4),
            VectorStorage::OwnedQuantized(pool) => pool.data.len(),
            VectorStorage::OwnedPq(pool) => pool.data.len() + pool.codebook.len() * 4,
            #[cfg(feature = "pq-fastscan")]
            VectorStorage::OwnedPqFastScan(pool) => pool.packed.len() + pool.codebook.len() * 4,
            VectorStorage::OnDemand { offsets, .. } => {
                // Estimate memory for offsets map + ID list
                offsets.len() * (8 + 32 + 8) // Key + Valid + Offset roughly
            }
        };

        VectorStats {
            vector_count: self.vectors.len(),
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
            VectorStorage::OwnedQuantized(pool) => pool.contains(doc_id, field_name),
            VectorStorage::OwnedPq(pool) => pool.contains(doc_id, field_name),
            #[cfg(feature = "pq-fastscan")]
            VectorStorage::OwnedPqFastScan(pool) => pool.contains(doc_id, field_name),
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
        Ok(Box::new(HnswVectorIterator {
            storage: self.vectors.clone(),
            keys: self.vector_ids.clone(),
            current: 0,
            dimension: self.dimension,
            deletion_bitmap: self.deletion_bitmap.clone(),
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
            VectorStorage::OwnedPqFastScan(pool) => {
                for (id, field) in &self.vector_ids {
                    if !pool.contains(*id, field) {
                        errors.push(format!(
                            "Vector {}:{} found in keys but missing in PQ FastScan pool",
                            id, field
                        ));
                    }
                }
                warnings.push(
                    "OwnedPqFastScan mode: dimension / NaN checks skipped (4-bit codes \
                     index into the trained K=16 codebook which is bounded by construction)"
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
    storage: VectorStorage,
    keys: Vec<(u64, String)>,
    current: usize,
    dimension: usize,
    deletion_bitmap: Option<Arc<DeletionBitmap>>,
}

impl VectorIterator for HnswVectorIterator {
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
