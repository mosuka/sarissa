//! In-memory rerank storage pool used by the Stage 2 two-stage rerank
//! flow (Issue #481 Stage 2).
//!
//! Sibling type to [`super::quantized_storage::QuantizedVectorPool`].
//! While `QuantizedVectorPool` holds the int8 + meta payload that
//! drives the wide candidate fetch, this pool holds the matching
//! full-precision vectors that the rerank kernel rescores against.
//!
//! Built once at reader load time from the LRS1 sidecar (see
//! [`super::rerank_sidecar`]) and shared across search threads via
//! `Arc<RerankStoragePool>`. All fields are immutable after
//! construction.
//!
//! Memory footprint: `dim * bytes_per_element` per vector
//! (`bytes_per_element` = 4 for [`RerankStorageKind::F32`]) plus
//! `O(field_count + vector_count)` for the lookup tables.
//!
//! # Endianness
//!
//! The on-disk LRS1 payload is little-endian and the in-memory
//! [`Self::data`] buffer holds those bytes verbatim. The
//! [`Self::get_f32_slice`] hot-path accessor reinterprets the bytes
//! as `&[f32]` via [`std::slice::from_raw_parts`], which is sound
//! only on little-endian hosts. A `compile_error!` below catches
//! any attempt to build for a big-endian target.

#[cfg(not(target_endian = "little"))]
compile_error!(
    "RerankStoragePool requires a little-endian host because the LRS1 sidecar's f32 payload \
     is stored little-endian and reinterpreted in place"
);

use std::collections::HashMap;
use std::sync::Arc;

use crate::vector::core::rerank::RerankStorageKind;

/// In-memory rerank storage for one segment's full-precision vectors.
///
/// Index-agnostic (used by HNSW first; Flat / IVF could follow). The
/// payload layout matches the LRS1 sidecar: `vector_count * dim *
/// bytes_per_element` bytes laid out as a flat AoS buffer with each
/// vector stored back-to-back in the same order as the matching LVS1
/// segment.
#[derive(Debug)]
pub struct RerankStoragePool {
    /// On-disk encoding of each stored element.
    pub kind: RerankStorageKind,
    /// Vector dimension.
    pub dim: usize,
    /// Tightly-packed AoS payload: for vector index `i`,
    /// `data[i * record_size .. (i + 1) * record_size]` holds the
    /// per-vector bytes in the encoding selected by [`Self::kind`].
    pub data: Vec<u8>,
    /// Per-field doc_id -> vector position.
    ///
    /// Mirrors [`super::quantized_storage::QuantizedVectorPool::field_index`]
    /// so the search hot loop can reuse the same lookup pattern: hold
    /// the inner `Arc<HashMap<u64, u32>>` once per field, then do an
    /// O(1) `u64` lookup per candidate.
    pub field_index: HashMap<String, Arc<HashMap<u64, u32>>>,
    /// Total vector count (matches `data.len() / record_size`).
    pub vector_count: usize,
}

impl RerankStoragePool {
    /// Bytes occupied by one vector record under the given encoding.
    #[inline]
    pub const fn record_size(dim: usize, kind: RerankStorageKind) -> usize {
        dim * kind.bytes_per_element()
    }

    /// Build from a sequence of `(doc_id, field_name, vector)` records.
    ///
    /// The records may be in any order; the field index is built from
    /// the iteration order, and each vector's payload is written at
    /// the position equal to the iteration index. Callers that need
    /// the position to match the matching LVS1 segment must feed the
    /// records in the same order they fed the LVS1 writer.
    ///
    /// # Panics
    ///
    /// Panics if any input vector's length does not equal `dim`.
    pub fn build(
        kind: RerankStorageKind,
        dim: usize,
        records: impl IntoIterator<Item = (u64, String, Vec<f32>)>,
    ) -> Self {
        let record_size = Self::record_size(dim, kind);
        let mut data: Vec<u8> = Vec::new();
        let mut by_field: HashMap<String, HashMap<u64, u32>> = HashMap::new();

        for (doc_id, field, vector) in records {
            assert_eq!(
                vector.len(),
                dim,
                "rerank storage record dim mismatch: expected {dim}, got {}",
                vector.len()
            );
            let pos = (data.len() / record_size) as u32;
            match kind {
                RerankStorageKind::F32 => {
                    data.reserve(record_size);
                    for v in &vector {
                        data.extend_from_slice(&v.to_le_bytes());
                    }
                }
            }
            by_field.entry(field).or_default().insert(doc_id, pos);
        }

        let vector_count = data.len() / record_size;
        let field_index: HashMap<String, Arc<HashMap<u64, u32>>> = by_field
            .into_iter()
            .map(|(field, map)| (field, Arc::new(map)))
            .collect();

        Self {
            kind,
            dim,
            data,
            field_index,
            vector_count,
        }
    }

    /// Build directly from a parsed LRS1 sidecar payload.
    ///
    /// `payload` must be `vector_count * dim * bytes_per_element` long
    /// (which is what [`super::rerank_sidecar::read_sidecar`] returns).
    /// `field_assignment` provides the per-position `(doc_id,
    /// field_name)` pairing so the pool can reconstruct the
    /// `field_index` without re-encoding the bytes. The assignment
    /// must be `vector_count` long and ordered by position.
    ///
    /// # Panics
    ///
    /// Panics if `payload.len()` does not equal `vector_count * dim *
    /// bytes_per_element` or if `field_assignment.len()` does not
    /// equal `vector_count`.
    pub fn from_sidecar_payload(
        kind: RerankStorageKind,
        dim: usize,
        vector_count: usize,
        payload: Vec<u8>,
        field_assignment: &[(u64, String)],
    ) -> Self {
        let record_size = Self::record_size(dim, kind);
        assert_eq!(
            payload.len(),
            vector_count * record_size,
            "sidecar payload length mismatch"
        );
        assert_eq!(
            field_assignment.len(),
            vector_count,
            "field assignment length mismatch"
        );

        let mut by_field: HashMap<String, HashMap<u64, u32>> = HashMap::new();
        for (pos, (doc_id, field)) in field_assignment.iter().enumerate() {
            by_field
                .entry(field.clone())
                .or_default()
                .insert(*doc_id, pos as u32);
        }

        let field_index: HashMap<String, Arc<HashMap<u64, u32>>> = by_field
            .into_iter()
            .map(|(field, map)| (field, Arc::new(map)))
            .collect();

        Self {
            kind,
            dim,
            data: payload,
            field_index,
            vector_count,
        }
    }

    /// Borrow the f32 slice for `(doc_id, field)`.
    ///
    /// Returns `None` if the key is not present in this segment.
    /// The returned slice has length [`Self::dim`].
    #[inline]
    pub fn get_f32_slice(&self, doc_id: u64, field: &str) -> Option<&[f32]> {
        let pos = self.field_index.get(field)?.get(&doc_id).copied()?;
        Some(self.f32_slice_at(pos))
    }

    /// Borrow the f32 slice at vector position `pos`.
    ///
    /// Lower-level than [`Self::get_f32_slice`]; useful when the
    /// caller has already cached the per-field doc_id -> position map
    /// (the search hot loop does this once per search via
    /// [`Self::field_position_index`]).
    ///
    /// # Panics
    ///
    /// Panics if `pos >= self.vector_count`.
    #[inline]
    pub fn f32_slice_at(&self, pos: u32) -> &[f32] {
        debug_assert_eq!(self.kind, RerankStorageKind::F32);
        let record_size = Self::record_size(self.dim, self.kind);
        let start = (pos as usize) * record_size;
        let end = start + record_size;
        let bytes = &self.data[start..end];
        // SAFETY:
        // - `bytes` is a `&[u8]` slice that originated from a
        //   `Vec<u8>` whose allocator returns memory aligned to at
        //   least the alignment of `f32` (the global allocator is at
        //   least 8-byte aligned).
        // - `start` is a multiple of `record_size`, which itself is
        //   `dim * 4`, so the slice start is 4-byte aligned.
        // - The byte length equals `self.dim * 4`, exactly enough for
        //   `self.dim` `f32` values.
        // - The underlying bytes were written via `f32::to_le_bytes`
        //   (see `Self::build`) or copied from a little-endian LRS1
        //   sidecar payload. The host is little-endian (enforced by
        //   the module-level `compile_error!`), so the bit pattern
        //   matches `f32`'s native representation.
        // - The pool is shared via `Arc<Self>`; `&self` lifetime ties
        //   the returned slice to the pool's borrow, so no aliasing
        //   with mutable accessors is possible.
        unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<f32>(), self.dim) }
    }

    /// Cheap O(1) lookup of the per-field doc_id -> position map,
    /// returning the inner `Arc<HashMap>` so the search hot loop can
    /// hold it without cloning the field name on each call.
    #[inline]
    pub fn field_position_index(&self, field: &str) -> Option<Arc<HashMap<u64, u32>>> {
        self.field_index.get(field).cloned()
    }

    /// Whether the segment contains the given key.
    #[inline]
    pub fn contains(&self, doc_id: u64, field: &str) -> bool {
        self.field_index
            .get(field)
            .is_some_and(|m| m.contains_key(&doc_id))
    }

    /// Number of fields with at least one vector.
    #[inline]
    pub fn field_count(&self) -> usize {
        self.field_index.len()
    }

    /// Sorted list of field names present in this segment.
    pub fn field_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.field_index.keys().cloned().collect();
        names.sort();
        names
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_size_is_dim_times_four_for_f32() {
        assert_eq!(RerankStoragePool::record_size(0, RerankStorageKind::F32), 0);
        assert_eq!(
            RerankStoragePool::record_size(128, RerankStorageKind::F32),
            512
        );
    }

    #[test]
    fn build_packs_records_in_iteration_order() {
        let pool = RerankStoragePool::build(
            RerankStorageKind::F32,
            3,
            vec![
                (10, "embedding".to_string(), vec![1.0, 2.0, 3.0]),
                (20, "embedding".to_string(), vec![-1.0, 0.0, 1.0]),
            ],
        );
        assert_eq!(pool.vector_count, 2);
        assert_eq!(pool.dim, 3);
        assert_eq!(pool.data.len(), 2 * 3 * 4);

        let v0 = pool.get_f32_slice(10, "embedding").unwrap();
        assert_eq!(v0, &[1.0, 2.0, 3.0]);

        let v1 = pool.get_f32_slice(20, "embedding").unwrap();
        assert_eq!(v1, &[-1.0, 0.0, 1.0]);
    }

    #[test]
    fn get_f32_slice_returns_none_for_missing_keys() {
        let pool = RerankStoragePool::build(
            RerankStorageKind::F32,
            2,
            vec![(1, "f".to_string(), vec![5.0, 6.0])],
        );
        assert!(pool.get_f32_slice(2, "f").is_none(), "missing doc_id");
        assert!(pool.get_f32_slice(1, "other").is_none(), "missing field");
    }

    #[test]
    fn field_position_index_supports_hot_loop_lookup() {
        let pool = RerankStoragePool::build(
            RerankStorageKind::F32,
            2,
            vec![
                (1, "embedding".to_string(), vec![5.0, 6.0]),
                (2, "embedding".to_string(), vec![7.0, 8.0]),
                (1, "thumbnail".to_string(), vec![9.0, 10.0]),
            ],
        );
        let idx = pool.field_position_index("embedding").unwrap();
        assert_eq!(idx.len(), 2);
        let pos = *idx.get(&2).unwrap();
        assert_eq!(pool.f32_slice_at(pos), &[7.0, 8.0]);
    }

    #[test]
    fn contains_reflects_field_and_doc_id() {
        let pool = RerankStoragePool::build(
            RerankStorageKind::F32,
            1,
            vec![(1, "f".to_string(), vec![1.5])],
        );
        assert!(pool.contains(1, "f"));
        assert!(!pool.contains(2, "f"));
        assert!(!pool.contains(1, "g"));
    }

    #[test]
    fn field_names_sorted_alphabetically() {
        let pool = RerankStoragePool::build(
            RerankStorageKind::F32,
            1,
            vec![
                (1, "z".to_string(), vec![0.0]),
                (1, "a".to_string(), vec![0.0]),
                (1, "m".to_string(), vec![0.0]),
            ],
        );
        assert_eq!(pool.field_names(), vec!["a", "m", "z"]);
    }

    #[test]
    fn field_count_matches_distinct_fields() {
        let pool = RerankStoragePool::build(
            RerankStorageKind::F32,
            1,
            vec![
                (1, "a".to_string(), vec![0.0]),
                (2, "a".to_string(), vec![0.0]),
                (1, "b".to_string(), vec![0.0]),
            ],
        );
        assert_eq!(pool.field_count(), 2);
    }

    #[test]
    #[should_panic(expected = "dim mismatch")]
    fn build_panics_on_dim_mismatch() {
        RerankStoragePool::build(
            RerankStorageKind::F32,
            3,
            vec![(1, "f".to_string(), vec![1.0, 2.0])],
        );
    }

    #[test]
    fn from_sidecar_payload_round_trips_full_precision() {
        let dim = 4;
        let vectors: Vec<f32> = vec![
            1.5, 2.5, 3.5, 4.5, // pos 0
            -1.0, 0.0, 1.0, 2.0, // pos 1
        ];
        let mut payload = Vec::new();
        for v in &vectors {
            payload.extend_from_slice(&v.to_le_bytes());
        }
        let assignment = vec![(7u64, "f".to_string()), (9u64, "f".to_string())];
        let pool = RerankStoragePool::from_sidecar_payload(
            RerankStorageKind::F32,
            dim,
            2,
            payload,
            &assignment,
        );
        assert_eq!(pool.vector_count, 2);
        assert_eq!(pool.get_f32_slice(7, "f").unwrap(), &vectors[0..4]);
        assert_eq!(pool.get_f32_slice(9, "f").unwrap(), &vectors[4..8]);
    }

    #[test]
    #[should_panic(expected = "payload length mismatch")]
    fn from_sidecar_payload_panics_on_wrong_payload_size() {
        RerankStoragePool::from_sidecar_payload(
            RerankStorageKind::F32,
            4,
            2,
            vec![0u8; 10], // expects 32 bytes
            &[(1u64, "f".to_string()), (2u64, "f".to_string())],
        );
    }

    #[test]
    #[should_panic(expected = "field assignment length mismatch")]
    fn from_sidecar_payload_panics_on_wrong_assignment_size() {
        RerankStoragePool::from_sidecar_payload(
            RerankStorageKind::F32,
            4,
            2,
            vec![0u8; 32],
            &[(1u64, "f".to_string())], // wrong length
        );
    }

    #[test]
    fn f32_slice_pointer_is_aligned_to_four() {
        let pool = RerankStoragePool::build(
            RerankStorageKind::F32,
            8,
            (0..16).map(|i| (i as u64, "f".to_string(), vec![i as f32; 8])),
        );
        for pos in 0..16u32 {
            let slice = pool.f32_slice_at(pos);
            let addr = slice.as_ptr() as usize;
            assert_eq!(addr % std::mem::align_of::<f32>(), 0, "pos {pos}");
            assert_eq!(slice.len(), 8);
        }
    }
}
