//! In-memory int8 vector pool used by the Eager-loading reader paths
//! of HNSW / Flat / IVF (Issue #481 Stage 1, Step 6 onward).
//!
//! Replaces the per-index `Arc<HashMap<(u64, String), Vector>>` that
//! held f32 vectors until Step 5. Vectors are kept as int8 in one
//! tightly-packed AoS buffer plus a per-field doc_id -> position
//! index. The search hot loop pulls `(int8 slice,
//! QuantizedVectorMeta)` directly out of this buffer and feeds it to
//! [`crate::vector::core::distance_quantized::distance_quantized`].
//!
//! Memory footprint: `dim + 8` bytes per vector (int8 payload +
//! `sum_q` u32 + `norm_q` f32) plus `O(field_count + vector_count)`
//! for the lookup tables.

use std::collections::HashMap;
use std::sync::Arc;

use crate::vector::core::quantization::{QuantizedVectorMeta, ScalarQuantParams};
use crate::vector::core::vector::Vector;

/// In-memory quantized representation of one segment's vectors.
///
/// Index-agnostic (used by HNSW Step 6 first; Flat / IVF in Step 7).
/// Built once at reader load time and shared across search threads via
/// `Arc<QuantizedVectorPool>`. All fields are immutable after
/// construction.
#[derive(Debug)]
pub struct QuantizedVectorPool {
    /// Per-segment quantization params (`offset`, `scale`).
    pub params: ScalarQuantParams,
    /// Vector dimension.
    pub dim: usize,
    /// Tightly-packed AoS payload: for vector index `i`,
    /// `data[i * record_size .. i * record_size + dim]` is the int8
    /// payload and the next 8 bytes are the meta (`sum_q` u32 LE +
    /// `norm_q` f32 LE).
    pub data: Vec<u8>,
    /// Per-field doc_id -> vector position.
    ///
    /// `field_index[field][doc_id] = i` such that the int8 payload
    /// for `(doc_id, field)` lives at `i * record_size`. The outer
    /// `String` key allocation only happens on the first lookup per
    /// search; the inner `u64` lookup is the per-candidate hot-path
    /// op.
    pub field_index: HashMap<String, Arc<HashMap<u64, u32>>>,
    /// Total vector count (matches `data.len() / record_size`).
    pub vector_count: usize,
}

impl QuantizedVectorPool {
    /// Bytes occupied by one vector record (int8 + meta).
    #[inline]
    pub const fn record_size(dim: usize) -> usize {
        dim + QuantizedVectorMeta::SERIALIZED_SIZE
    }

    /// Build from a sequence of `(doc_id, field_name, int8_data, meta)`
    /// records and the per-segment quantization params.
    ///
    /// The records may be in any order; the field index is built from
    /// the iteration order, and the int8 payload is written at the
    /// position equal to the iteration index.
    pub fn build(
        params: ScalarQuantParams,
        dim: usize,
        records: impl IntoIterator<Item = (u64, String, Vec<u8>, QuantizedVectorMeta)>,
    ) -> Self {
        let mut data = Vec::new();
        let mut by_field: HashMap<String, HashMap<u64, u32>> = HashMap::new();
        let record_size = Self::record_size(dim);

        for (doc_id, field, int8, meta) in records {
            let pos = (data.len() / record_size) as u32;
            data.extend_from_slice(&int8);
            data.extend_from_slice(&meta.sum_q.to_le_bytes());
            data.extend_from_slice(&meta.norm_q.to_le_bytes());
            by_field.entry(field).or_default().insert(doc_id, pos);
        }

        let vector_count = data.len() / record_size;
        let field_index: HashMap<String, Arc<HashMap<u64, u32>>> = by_field
            .into_iter()
            .map(|(field, map)| (field, Arc::new(map)))
            .collect();

        Self {
            params,
            dim,
            data,
            field_index,
            vector_count,
        }
    }

    /// Borrow the int8 payload + decoded meta for `(doc_id, field)`.
    ///
    /// Returns `None` if the key is not present in this segment.
    /// The int8 slice has length [`Self::dim`].
    #[inline]
    pub fn get_record(&self, doc_id: u64, field: &str) -> Option<(&[u8], QuantizedVectorMeta)> {
        let pos = self.field_index.get(field)?.get(&doc_id).copied()?;
        Some(self.record_at(pos))
    }

    /// Borrow the int8 payload + meta at vector position `pos`.
    ///
    /// Lower-level than [`Self::get_record`]; useful when the caller
    /// has already cached the per-field doc_id -> position map (the
    /// search hot loop does this once per search via
    /// [`Self::field_position_index`]).
    #[inline]
    pub fn record_at(&self, pos: u32) -> (&[u8], QuantizedVectorMeta) {
        let record_size = Self::record_size(self.dim);
        let start = (pos as usize) * record_size;
        let int8 = &self.data[start..start + self.dim];
        let meta_bytes = &self.data[start + self.dim..start + record_size];
        let sum_q = u32::from_le_bytes(meta_bytes[0..4].try_into().expect("4 bytes"));
        let norm_q = f32::from_le_bytes(meta_bytes[4..8].try_into().expect("4 bytes"));
        (int8, QuantizedVectorMeta { sum_q, norm_q })
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

    /// Dequantize the vector for `(doc_id, field)` into a fresh
    /// `Vector` (f32). Used by the legacy
    /// [`crate::vector::reader::VectorIndexReader::get_vector`] API
    /// path; the search hot loop never calls this.
    pub fn dequantize_to_vector(&self, doc_id: u64, field: &str) -> Option<Vector> {
        let (int8, _meta) = self.get_record(doc_id, field)?;
        let data: Vec<f32> = int8
            .iter()
            .map(|&b| self.params.dequantize_value(b))
            .collect();
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

    /// Per-field doc-id list (sorted). Used to back
    /// [`crate::vector::reader::VectorIndexReader::doc_ids_for_field`].
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

    fn meta(sum_q: u32, norm_q: f32) -> QuantizedVectorMeta {
        QuantizedVectorMeta { sum_q, norm_q }
    }

    fn sample_params() -> ScalarQuantParams {
        ScalarQuantParams {
            offset: -1.0,
            scale: 2.0 / 255.0,
        }
    }

    #[test]
    fn build_packs_records_in_iteration_order() {
        let q = QuantizedVectorPool::build(
            sample_params(),
            4,
            vec![
                (10, "embedding".to_string(), vec![0, 1, 2, 3], meta(6, 1.0)),
                (
                    20,
                    "embedding".to_string(),
                    vec![10, 20, 30, 40],
                    meta(100, 2.0),
                ),
            ],
        );
        assert_eq!(q.vector_count, 2);
        assert_eq!(q.dim, 4);
        assert_eq!(q.data.len(), 2 * QuantizedVectorPool::record_size(4));

        let (int8_0, meta_0) = q.get_record(10, "embedding").unwrap();
        assert_eq!(int8_0, &[0u8, 1, 2, 3]);
        assert_eq!(meta_0.sum_q, 6);
        assert_eq!(meta_0.norm_q, 1.0);

        let (int8_1, meta_1) = q.get_record(20, "embedding").unwrap();
        assert_eq!(int8_1, &[10u8, 20, 30, 40]);
        assert_eq!(meta_1.sum_q, 100);
        assert_eq!(meta_1.norm_q, 2.0);
    }

    #[test]
    fn get_record_returns_none_for_missing_keys() {
        let q = QuantizedVectorPool::build(
            sample_params(),
            2,
            vec![(1, "f".to_string(), vec![5, 6], meta(11, 1.0))],
        );
        assert!(q.get_record(2, "f").is_none(), "missing doc_id");
        assert!(q.get_record(1, "other").is_none(), "missing field");
    }

    #[test]
    fn field_position_index_supports_hot_loop_lookup() {
        let q = QuantizedVectorPool::build(
            sample_params(),
            2,
            vec![
                (1, "embedding".to_string(), vec![5, 6], meta(11, 1.0)),
                (2, "embedding".to_string(), vec![7, 8], meta(15, 1.5)),
                (1, "thumbnail".to_string(), vec![9, 10], meta(19, 0.5)),
            ],
        );
        let idx = q.field_position_index("embedding").unwrap();
        assert_eq!(idx.len(), 2);
        let pos = *idx.get(&2).unwrap();
        let (int8, meta_back) = q.record_at(pos);
        assert_eq!(int8, &[7u8, 8]);
        assert_eq!(meta_back.sum_q, 15);
    }

    #[test]
    fn dequantize_to_vector_inverts_quantize_value() {
        let q = QuantizedVectorPool::build(
            sample_params(),
            3,
            vec![(1, "f".to_string(), vec![0, 128, 255], meta(383, 1.0))],
        );
        let v = q.dequantize_to_vector(1, "f").unwrap();
        // offset = -1.0, scale = 2/255
        // u8 = 0   -> -1.0
        // u8 = 128 -> -1.0 + 128 * 2/255 = ~0.0039
        // u8 = 255 -> -1.0 + 255 * 2/255 = 1.0
        assert!((v.data[0] - (-1.0)).abs() < 1e-6);
        assert!((v.data[1] - (-1.0 + 128.0 * 2.0 / 255.0)).abs() < 1e-6);
        assert!((v.data[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn keys_sorted_by_doc_id() {
        let q = QuantizedVectorPool::build(
            sample_params(),
            1,
            vec![
                (5, "a".to_string(), vec![10], meta(10, 0.0)),
                (1, "a".to_string(), vec![20], meta(20, 0.0)),
                (3, "b".to_string(), vec![30], meta(30, 0.0)),
            ],
        );
        let keys = q.keys();
        let ids: Vec<u64> = keys.iter().map(|(id, _)| *id).collect();
        assert_eq!(ids, vec![1, 3, 5]);
    }

    #[test]
    fn doc_ids_for_field_returns_sorted_arc() {
        let q = QuantizedVectorPool::build(
            sample_params(),
            1,
            vec![
                (5, "f".to_string(), vec![10], meta(10, 0.0)),
                (1, "f".to_string(), vec![20], meta(20, 0.0)),
                (3, "f".to_string(), vec![30], meta(30, 0.0)),
            ],
        );
        let ids = q.doc_ids_for_field("f");
        assert_eq!(ids.as_ref(), &[1u64, 3, 5][..]);
        assert!(q.doc_ids_for_field("missing").is_empty());
    }

    #[test]
    fn field_names_sorted_alphabetically() {
        let q = QuantizedVectorPool::build(
            sample_params(),
            1,
            vec![
                (1, "z".to_string(), vec![0], meta(0, 0.0)),
                (1, "a".to_string(), vec![0], meta(0, 0.0)),
                (1, "m".to_string(), vec![0], meta(0, 0.0)),
            ],
        );
        assert_eq!(q.field_names(), vec!["a", "m", "z"]);
    }

    #[test]
    fn record_size_matches_dim_plus_eight() {
        assert_eq!(QuantizedVectorPool::record_size(0), 8);
        assert_eq!(QuantizedVectorPool::record_size(128), 136);
    }

    #[test]
    fn contains_reflects_field_and_doc_id() {
        let q = QuantizedVectorPool::build(
            sample_params(),
            1,
            vec![(1, "f".to_string(), vec![0], meta(0, 0.0))],
        );
        assert!(q.contains(1, "f"));
        assert!(!q.contains(2, "f"));
        assert!(!q.contains(1, "g"));
    }
}
