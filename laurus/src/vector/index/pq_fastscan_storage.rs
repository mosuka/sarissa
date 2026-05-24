//! FastScan-style PQ vector pool (Issue
//! [#651](https://github.com/mosuka/laurus/issues/651) / part A
//! [#692](https://github.com/mosuka/laurus/issues/692)).
//!
//! Parallel to
//! [`crate::vector::index::pq_storage::PqVectorPool`] (the existing
//! K=256 8-bit PQ storage); the difference is the on-disk / in-memory
//! layout of the per-vector codes:
//!
//! - `PqVectorPool` stores `M` bytes per vector in array-of-structs
//!   (AoS) order. Each byte is one K=256 code.
//! - `PqFastScanPool` stores 4-bit codes (K=16) packed two-per-byte
//!   and **transposed across 32-vector blocks** so the SIMD shuffle
//!   kernel introduced in parts B (#693, AVX2) and C (#694, NEON) can
//!   evaluate 32 candidates per iteration with a single `pshufb` /
//!   `vqtbl1q_u8`.
//!
//! Part A of #651 only builds the storage layout and a scalar
//! reference kernel; production wiring lives in part D (#695).
//!
//! # Block layout
//!
//! For a pool of `n_vectors` vectors and `M` sub-quantisers, the
//! `packed` buffer holds `ceil(n_vectors / BLOCK_SIZE)` blocks. Each
//! block occupies `M * BYTES_PER_SUB_PER_BLOCK` bytes (i.e.
//! `M * 16` bytes), and within a block the byte at offset
//! `m * BYTES_PER_SUB_PER_BLOCK + j` (0 ≤ j < 16) carries:
//!
//! - the **low nibble** of vector `block_idx * BLOCK_SIZE + j`'s code
//!   for sub-quantiser `m`;
//! - the **high nibble** of vector `block_idx * BLOCK_SIZE + j + 16`'s
//!   code for sub-quantiser `m`.
//!
//! Vectors past `n_vectors` in the trailing block are zero-padded.
//! The kernel must not read past `n_vectors`; callers enforce this via
//! [`PqFastScanPool::n_vectors`] and the field-position index.

use std::collections::HashMap;
use std::sync::Arc;

use crate::error::{LaurusError, Result};
use crate::vector::core::quantization::PqParams;

/// Number of vectors packed into one FastScan block.
///
/// Chosen to match FAISS `IndexPQFastScan`: a 128-bit SIMD register
/// holds 16 nibbles, and packing the low + high nibbles per byte
/// doubles that to 32 vectors per SIMD iteration.
pub const BLOCK_SIZE: usize = 32;

/// Bytes occupied by one sub-quantiser's slice within a single block.
/// Equal to `BLOCK_SIZE / 2` because each byte carries two nibbles
/// (i.e. two 4-bit codes for two different vectors).
pub const BYTES_PER_SUB_PER_BLOCK: usize = BLOCK_SIZE / 2;

/// In-memory FastScan PQ representation of one segment's vectors.
///
/// Built once at reader load time and shared across search threads via
/// `Arc<PqFastScanPool>`. All fields are immutable after construction.
#[derive(Debug)]
pub struct PqFastScanPool {
    /// Per-segment PQ parameters. `params.k` must be `16`; the
    /// constructor returns an error otherwise.
    pub params: PqParams,
    /// Original vector dimension (`m * sub_dim`). Cached so callers
    /// don't multiply on every access.
    pub dim: usize,
    /// Per-segment codebook in row-major layout. Length is
    /// `params.codebook_len()` (= `m * 16 * sub_dim` for K=16).
    pub codebook: Vec<f32>,
    /// 4-bit packed codes in block-transposed layout. Length is
    /// `block_count() * params.m * BYTES_PER_SUB_PER_BLOCK`.
    pub packed: Vec<u8>,
    /// Number of vectors stored in the pool. The trailing block may
    /// hold fewer than `BLOCK_SIZE` real vectors; the remainder is
    /// zero-padded.
    pub n_vectors: usize,
    /// Per-field doc_id → vector position. Position is the index into
    /// `0..n_vectors`; the kernel derives (block_idx, in_block_idx)
    /// from it.
    pub field_index: HashMap<String, Arc<HashMap<u64, u32>>>,
}

impl PqFastScanPool {
    /// Number of blocks the pool spans (ceil(n_vectors / BLOCK_SIZE)).
    #[inline]
    pub fn block_count(&self) -> usize {
        n_blocks_for(self.n_vectors)
    }

    /// Byte size of one block (`M * BYTES_PER_SUB_PER_BLOCK`).
    #[inline]
    pub fn block_stride(&self) -> usize {
        self.params.m as usize * BYTES_PER_SUB_PER_BLOCK
    }

    /// Build a [`PqFastScanPool`] from a sequence of
    /// `(doc_id, field_name, codes)` records, where each `codes` is a
    /// `Vec<u8>` of length `params.m` with every value in `[0, 15]`.
    ///
    /// The records may be in any order; the field index is built from
    /// the iteration order, and the codes are written at the position
    /// equal to the iteration index.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError::InvalidOperation`] if `params.k != 16`,
    /// if any code is out of the 4-bit range, or if any per-record
    /// `codes` length disagrees with `params.m`.
    pub fn build(
        params: PqParams,
        codebook: Vec<f32>,
        records: impl IntoIterator<Item = (u64, String, Vec<u8>)>,
    ) -> Result<Self> {
        if params.k != 16 {
            return Err(LaurusError::InvalidOperation(format!(
                "PqFastScanPool requires PqParams::k == 16 (got {})",
                params.k
            )));
        }
        if codebook.len() != params.codebook_len() {
            return Err(LaurusError::InvalidOperation(format!(
                "PqFastScanPool: codebook length {} does not match params.codebook_len() {}",
                codebook.len(),
                params.codebook_len()
            )));
        }

        let m = params.m as usize;
        let mut all_codes: Vec<Vec<u8>> = Vec::new();
        let mut by_field: HashMap<String, HashMap<u64, u32>> = HashMap::new();

        for (doc_id, field, codes) in records {
            if codes.len() != m {
                return Err(LaurusError::InvalidOperation(format!(
                    "PqFastScanPool: code length {} does not match params.m {}",
                    codes.len(),
                    m
                )));
            }
            for &c in &codes {
                if c >= 16 {
                    return Err(LaurusError::InvalidOperation(format!(
                        "PqFastScanPool: code value {c} exceeds 4-bit range [0, 15]"
                    )));
                }
            }
            let pos = all_codes.len() as u32;
            all_codes.push(codes);
            by_field.entry(field).or_default().insert(doc_id, pos);
        }

        let n_vectors = all_codes.len();
        let packed = pack_codes_into_blocks(&all_codes, m);

        let field_index: HashMap<String, Arc<HashMap<u64, u32>>> = by_field
            .into_iter()
            .map(|(field, map)| (field, Arc::new(map)))
            .collect();

        Ok(Self {
            params,
            dim: params.original_dim(),
            codebook,
            packed,
            n_vectors,
            field_index,
        })
    }

    /// Decode the M codes for `vec_idx` from the packed layout.
    ///
    /// Primarily for tests and the scalar reference kernel; the SIMD
    /// kernel (#693/#694) walks the packed buffer directly without
    /// going through this helper.
    pub fn codes_at(&self, vec_idx: usize) -> Vec<u8> {
        let m = self.params.m as usize;
        let mut out = vec![0u8; m];
        let block_idx = vec_idx / BLOCK_SIZE;
        let in_block = vec_idx % BLOCK_SIZE;
        let block_base = block_idx * self.block_stride();
        let (j, shift) = if in_block < 16 {
            (in_block, 0)
        } else {
            (in_block - 16, 4)
        };
        for (sub, slot) in out.iter_mut().enumerate().take(m) {
            let byte = self.packed[block_base + sub * BYTES_PER_SUB_PER_BLOCK + j];
            *slot = (byte >> shift) & 0x0F;
        }
        out
    }
}

/// Number of FastScan blocks needed to hold `n_vectors` vectors.
#[inline]
pub fn n_blocks_for(n_vectors: usize) -> usize {
    n_vectors.div_ceil(BLOCK_SIZE)
}

/// Pack a flat slice of per-vector codes into the FastScan block
/// layout described in the module docs.
///
/// `codes[i]` must have length `m` and each entry must be in `[0,
/// 15]`. The returned `Vec<u8>` has length `n_blocks * m * BYTES_PER_SUB_PER_BLOCK`,
/// zero-padded for vectors past `codes.len()`.
///
/// The caller is responsible for input validation (see
/// [`PqFastScanPool::build`]).
pub fn pack_codes_into_blocks(codes: &[Vec<u8>], m: usize) -> Vec<u8> {
    let n_vectors = codes.len();
    let n_blocks = n_blocks_for(n_vectors);
    let block_stride = m * BYTES_PER_SUB_PER_BLOCK;
    let mut packed = vec![0u8; n_blocks * block_stride];

    for (vec_idx, v_codes) in codes.iter().enumerate().take(n_vectors) {
        let block_idx = vec_idx / BLOCK_SIZE;
        let in_block = vec_idx % BLOCK_SIZE;
        let (j, shift) = if in_block < 16 {
            (in_block, 0)
        } else {
            (in_block - 16, 4)
        };
        let block_base = block_idx * block_stride;
        for (sub, &code) in v_codes.iter().enumerate() {
            // Caller already validated codes are 4-bit; mask just in
            // case to avoid corrupting the neighbouring nibble if a
            // bad caller slips through.
            let nibble = code & 0x0F;
            packed[block_base + sub * BYTES_PER_SUB_PER_BLOCK + j] |= nibble << shift;
        }
    }

    packed
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a tiny K=16 codebook with deterministic content.
    fn dummy_codebook(m: usize, sub_dim: usize) -> Vec<f32> {
        let k = 16usize;
        let len = m * k * sub_dim;
        (0..len).map(|i| i as f32 * 0.01).collect()
    }

    #[test]
    fn round_trip_packs_and_unpacks_codes_for_n_below_block() {
        let m = 4;
        let codes: Vec<Vec<u8>> = vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7], vec![15, 0, 8, 9]];
        let packed = pack_codes_into_blocks(&codes, m);
        let params = PqParams::new(m as u16, 16, 2).unwrap();
        let pool = PqFastScanPool::build(
            params,
            dummy_codebook(m, 2),
            codes
                .iter()
                .enumerate()
                .map(|(i, c)| (i as u64, "f".to_string(), c.clone())),
        )
        .unwrap();
        assert_eq!(pool.packed, packed);
        assert_eq!(pool.codes_at(0), vec![0, 1, 2, 3]);
        assert_eq!(pool.codes_at(1), vec![4, 5, 6, 7]);
        assert_eq!(pool.codes_at(2), vec![15, 0, 8, 9]);
    }

    #[test]
    fn round_trip_packs_and_unpacks_codes_at_block_boundaries() {
        let m = 8;
        // Build n=33 so we span block boundary (32 → 33) and exercise
        // both nibble halves plus a sparsely-populated second block.
        let n = 33;
        let codes: Vec<Vec<u8>> = (0..n)
            .map(|i| (0..m).map(|sub| ((i * 7 + sub * 3) % 16) as u8).collect())
            .collect();
        let params = PqParams::new(m as u16, 16, 2).unwrap();
        let pool = PqFastScanPool::build(
            params,
            dummy_codebook(m, 2),
            codes
                .iter()
                .enumerate()
                .map(|(i, c)| (i as u64, "f".to_string(), c.clone())),
        )
        .unwrap();
        assert_eq!(pool.block_count(), 2, "n=33 spans 2 blocks");
        for (i, expected) in codes.iter().enumerate().take(n) {
            assert_eq!(&pool.codes_at(i), expected, "vector {i} mismatch");
        }
    }

    #[test]
    fn round_trip_works_for_exactly_one_block() {
        let m = 4;
        let n = BLOCK_SIZE;
        let codes: Vec<Vec<u8>> = (0..n)
            .map(|i| (0..m).map(|sub| ((i + sub * 5) % 16) as u8).collect())
            .collect();
        let params = PqParams::new(m as u16, 16, 1).unwrap();
        let pool = PqFastScanPool::build(
            params,
            dummy_codebook(m, 1),
            codes
                .iter()
                .enumerate()
                .map(|(i, c)| (i as u64, "f".to_string(), c.clone())),
        )
        .unwrap();
        assert_eq!(pool.block_count(), 1);
        for (i, expected) in codes.iter().enumerate().take(n) {
            assert_eq!(&pool.codes_at(i), expected, "vector {i} mismatch");
        }
    }

    #[test]
    fn round_trip_works_for_two_full_blocks() {
        let m = 2;
        let n = BLOCK_SIZE * 2;
        let codes: Vec<Vec<u8>> = (0..n)
            .map(|i| vec![(i % 16) as u8, ((i / 2) % 16) as u8])
            .collect();
        let params = PqParams::new(m as u16, 16, 2).unwrap();
        let pool = PqFastScanPool::build(
            params,
            dummy_codebook(m, 2),
            codes
                .iter()
                .enumerate()
                .map(|(i, c)| (i as u64, "f".to_string(), c.clone())),
        )
        .unwrap();
        assert_eq!(pool.block_count(), 2);
        for (i, expected) in codes.iter().enumerate().take(n) {
            assert_eq!(&pool.codes_at(i), expected);
        }
    }

    #[test]
    fn build_rejects_wrong_k() {
        let params = PqParams::new(4, 256, 2).unwrap();
        let codebook = vec![0.0f32; params.codebook_len()];
        let err = PqFastScanPool::build(
            params,
            codebook,
            std::iter::empty::<(u64, String, Vec<u8>)>(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("k == 16"));
    }

    #[test]
    fn build_rejects_out_of_range_codes() {
        let m = 4;
        let params = PqParams::new(m as u16, 16, 2).unwrap();
        let codebook = dummy_codebook(m, 2);
        let err = PqFastScanPool::build(
            params,
            codebook,
            std::iter::once((0u64, "f".to_string(), vec![0u8, 1, 16, 3])),
        )
        .unwrap_err();
        assert!(err.to_string().contains("exceeds 4-bit range"));
    }

    #[test]
    fn build_rejects_code_length_mismatch() {
        let m = 4;
        let params = PqParams::new(m as u16, 16, 2).unwrap();
        let codebook = dummy_codebook(m, 2);
        let err = PqFastScanPool::build(
            params,
            codebook,
            std::iter::once((0u64, "f".to_string(), vec![0u8, 1, 2])),
        )
        .unwrap_err();
        assert!(err.to_string().contains("does not match params.m"));
    }

    #[test]
    fn build_zero_pads_trailing_partial_block() {
        let m = 2;
        let n = 5; // one partial block of 5/32 vectors
        let codes: Vec<Vec<u8>> = (0..n)
            .map(|i| vec![(i % 16) as u8, ((i + 1) % 16) as u8])
            .collect();
        let params = PqParams::new(m as u16, 16, 1).unwrap();
        let pool = PqFastScanPool::build(
            params,
            dummy_codebook(m, 1),
            codes
                .iter()
                .enumerate()
                .map(|(i, c)| (i as u64, "f".to_string(), c.clone())),
        )
        .unwrap();
        assert_eq!(pool.block_count(), 1);
        assert_eq!(
            pool.packed.len(),
            pool.block_count() * pool.block_stride(),
            "packed buffer is sized to whole blocks"
        );
        for (i, expected) in codes.iter().enumerate().take(n) {
            assert_eq!(&pool.codes_at(i), expected);
        }
        // Padding region: codes for vec_idx in [n, 32) decode as all
        // zeros (codebook entry 0 for every sub-quantiser).
        for i in n..BLOCK_SIZE {
            assert!(
                pool.codes_at(i).iter().all(|&c| c == 0),
                "vec {i} in padding should be all-zero codes"
            );
        }
    }
}
