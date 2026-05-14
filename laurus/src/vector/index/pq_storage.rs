//! In-memory PQ vector pool used by the Eager-loading HNSW reader
//! path (Issue #481 Stage 3, parallel to
//! [`crate::vector::index::quantized_storage::QuantizedVectorPool`]
//! for the Scalar8Bit variant).
//!
//! # Memory footprint
//!
//! - One **codebook** per segment: `m * k * sub_dim * 4` bytes (e.g.
//!   `16 * 256 * 8 * 4 = 128 KB` for SIFT-class dim=128 / M=16).
//! - **Per vector**: `m` u8 codes (8 bytes / vector at M = 8, …,
//!   32 bytes / vector at M = 32) plus the lookup tables.
//!
//! Compared to the Scalar8Bit pool which is `dim + 8` bytes / vector
//! (~136 bytes at dim = 128), PQ shrinks the per-vector footprint by
//! `dim / m` while adding a single per-segment codebook overhead.

use std::collections::HashMap;
use std::sync::Arc;

use crate::vector::core::quantization::{PqParams, pq_decode};
use crate::vector::core::vector::Vector;

/// In-memory PQ representation of one segment's vectors.
///
/// Built once at reader load time and shared across search threads via
/// `Arc<PqVectorPool>`. All fields are immutable after construction.
#[derive(Debug)]
pub struct PqVectorPool {
    /// Per-segment PQ parameters (`m`, `k`, `sub_dim`).
    pub params: PqParams,
    /// Original vector dimension (`m * sub_dim`). Kept as a cached
    /// field so the searcher does not need to multiply on every record
    /// access.
    pub dim: usize,
    /// Per-segment codebook in row-major layout. Length is
    /// `params.codebook_len()`.
    pub codebook: Vec<f32>,
    /// Tightly-packed per-vector codes: for vector index `i`,
    /// `data[i * m .. (i + 1) * m]` is the `m`-byte code.
    pub data: Vec<u8>,
    /// Per-field doc_id -> vector position. Mirrors the API of
    /// [`crate::vector::index::quantized_storage::QuantizedVectorPool::field_index`].
    pub field_index: HashMap<String, Arc<HashMap<u64, u32>>>,
    /// Total vector count (matches `data.len() / m`).
    pub vector_count: usize,
}

impl PqVectorPool {
    /// Bytes occupied by one PQ record. Equal to the number of
    /// sub-vectors (`m`).
    #[inline]
    pub const fn record_size(m: u16) -> usize {
        m as usize
    }

    /// Build from a sequence of `(doc_id, field_name, codes)` records
    /// and the per-segment PQ params and codebook.
    ///
    /// The records may be in any order; the field index is built from
    /// the iteration order, and the codes are written at the position
    /// equal to the iteration index.
    pub fn build(
        params: PqParams,
        codebook: Vec<f32>,
        records: impl IntoIterator<Item = (u64, String, Vec<u8>)>,
    ) -> Self {
        debug_assert_eq!(codebook.len(), params.codebook_len());
        let mut data = Vec::new();
        let mut by_field: HashMap<String, HashMap<u64, u32>> = HashMap::new();
        let record_size = Self::record_size(params.m);

        for (doc_id, field, codes) in records {
            debug_assert_eq!(codes.len(), record_size);
            let pos = (data.len() / record_size) as u32;
            data.extend_from_slice(&codes);
            by_field.entry(field).or_default().insert(doc_id, pos);
        }

        let vector_count = data.len().checked_div(record_size).unwrap_or(0);
        let field_index: HashMap<String, Arc<HashMap<u64, u32>>> = by_field
            .into_iter()
            .map(|(field, map)| (field, Arc::new(map)))
            .collect();

        Self {
            params,
            dim: params.original_dim(),
            codebook,
            data,
            field_index,
            vector_count,
        }
    }

    /// Borrow the `m`-byte code payload for `(doc_id, field)`.
    #[inline]
    pub fn get_codes(&self, doc_id: u64, field: &str) -> Option<&[u8]> {
        let pos = self.field_index.get(field)?.get(&doc_id).copied()?;
        Some(self.codes_at(pos))
    }

    /// Borrow the `m`-byte code payload at vector position `pos`.
    #[inline]
    pub fn codes_at(&self, pos: u32) -> &[u8] {
        let record_size = Self::record_size(self.params.m);
        let start = (pos as usize) * record_size;
        &self.data[start..start + record_size]
    }

    /// Cheap O(1) lookup of the per-field doc_id -> position map.
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

    /// Iterate over `(doc_id, field_name)` pairs in this segment.
    pub fn keys(&self) -> Vec<(u64, String)> {
        let mut keys: Vec<(u64, String)> = self
            .field_index
            .iter()
            .flat_map(|(field, map)| map.keys().map(move |id| (*id, field.clone())))
            .collect();
        keys.sort_by_key(|(id, _)| *id);
        keys
    }

    /// Reconstruct an approximate f32 vector for `(doc_id, field)`
    /// from the PQ codes + codebook. Used by the legacy
    /// [`crate::vector::reader::VectorIndexReader::get_vector`] API
    /// path; the search hot loop never calls this.
    pub fn dequantize_to_vector(&self, doc_id: u64, field: &str) -> Option<Vector> {
        let codes = self.get_codes(doc_id, field)?;
        let data = pq_decode(codes, self.params, &self.codebook);
        Some(Vector::new(data))
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

    /// Per-field doc-id list (sorted).
    pub fn doc_ids_for_field(&self, field: &str) -> Arc<[u64]> {
        let Some(map) = self.field_index.get(field) else {
            return Arc::<[u64]>::from(Vec::<u64>::new());
        };
        let mut ids: Vec<u64> = map.keys().copied().collect();
        ids.sort_unstable();
        Arc::<[u64]>::from(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::core::quantization::pq_train_codebook;

    fn small_setup() -> (PqParams, Vec<f32>) {
        let dim = 4;
        let m = 2;
        let params = PqParams::from_dim_and_m(dim, m).unwrap();
        let training = vec![
            Vector::new(vec![10.0, 10.0, 20.0, 20.0]),
            Vector::new(vec![-10.0, -10.0, -20.0, -20.0]),
        ];
        let codebook = pq_train_codebook(dim, params, &training).unwrap();
        (params, codebook)
    }

    #[test]
    fn build_packs_codes_in_iteration_order() {
        let (params, codebook) = small_setup();
        let pool = PqVectorPool::build(
            params,
            codebook,
            vec![
                (10, "embedding".to_string(), vec![0u8, 1]),
                (20, "embedding".to_string(), vec![1u8, 0]),
            ],
        );
        assert_eq!(pool.vector_count, 2);
        assert_eq!(pool.dim, 4);
        assert_eq!(pool.data.len(), 2 * PqVectorPool::record_size(params.m));

        assert_eq!(pool.get_codes(10, "embedding"), Some(&[0u8, 1][..]));
        assert_eq!(pool.get_codes(20, "embedding"), Some(&[1u8, 0][..]));
    }

    #[test]
    fn get_codes_returns_none_for_missing_keys() {
        let (params, codebook) = small_setup();
        let pool = PqVectorPool::build(params, codebook, vec![(1, "f".to_string(), vec![0u8, 0])]);
        assert!(pool.get_codes(2, "f").is_none(), "missing doc_id");
        assert!(pool.get_codes(1, "other").is_none(), "missing field");
    }

    #[test]
    fn field_position_index_supports_hot_loop_lookup() {
        let (params, codebook) = small_setup();
        let pool = PqVectorPool::build(
            params,
            codebook,
            vec![
                (1, "embedding".to_string(), vec![5u8, 6]),
                (2, "embedding".to_string(), vec![7u8, 8]),
            ],
        );
        let idx = pool.field_position_index("embedding").unwrap();
        assert_eq!(idx.len(), 2);
        let pos = *idx.get(&2).unwrap();
        assert_eq!(pool.codes_at(pos), &[7u8, 8]);
    }

    #[test]
    fn dequantize_to_vector_inverts_encode() {
        let (params, codebook) = small_setup();
        let pool = PqVectorPool::build(params, codebook, vec![(1, "f".to_string(), vec![0u8, 0])]);
        let v = pool.dequantize_to_vector(1, "f").unwrap();
        assert_eq!(v.data.len(), 4);
    }

    #[test]
    fn keys_sorted_by_doc_id() {
        let (params, codebook) = small_setup();
        let pool = PqVectorPool::build(
            params,
            codebook,
            vec![
                (5, "a".to_string(), vec![0u8, 0]),
                (1, "a".to_string(), vec![0u8, 0]),
                (3, "b".to_string(), vec![0u8, 0]),
            ],
        );
        let keys = pool.keys();
        let ids: Vec<u64> = keys.iter().map(|(id, _)| *id).collect();
        assert_eq!(ids, vec![1, 3, 5]);
    }

    #[test]
    fn record_size_matches_m() {
        assert_eq!(PqVectorPool::record_size(0), 0);
        assert_eq!(PqVectorPool::record_size(16), 16);
    }
}
