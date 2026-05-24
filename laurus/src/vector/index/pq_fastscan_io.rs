//! Writer / reader helpers for the HNSW PQ FastScan segment payload
//! (Issue [#695](https://github.com/mosuka/laurus/issues/695) / part D
//! of [#651](https://github.com/mosuka/laurus/issues/651)).
//!
//! Parallel to [`crate::vector::index::pq_io`] (the K=256 8-bit PQ
//! variant). The on-disk layout for the FastScan region is identical
//! to the K=256 PQ region in the segment header (LVS1 with
//! `quant_kind = PRODUCT_QUANTIZATION_FASTSCAN` and the same
//! `m / k / sub_dim` + codebook payload), but each per-vector record
//! stores its `m` codes packed into `ceil(m / 2)` bytes (2 codes per
//! byte, low nibble first):
//!
//! ```text
//! [ VectorSegmentHeader  (LVS1 + PQ params + codebook, k = 16) ]
//! repeat num_vectors times:
//!   [ doc_id            u64 LE  8 bytes ]
//!   [ field_name_len    u32 LE  4 bytes ]
//!   [ field_name              field_name_len bytes (UTF-8) ]
//!   [ codes_packed             ceil(m / 2) bytes (4-bit packed) ]
//! ```
//!
//! The 4-bit packing matches the FAISS `pq4_pack_codes` convention so
//! the reader can hand the codes straight to
//! [`crate::vector::index::pq_fastscan_storage::PqFastScanPool::build`]
//! after unpacking back to `Vec<u8>` (one byte per sub-quantiser code,
//! values in `[0, 15]`).

use std::io::{Read, Write};

use crate::error::{LaurusError, Result};
use crate::vector::core::quantization::{PqParams, QuantizationMethod, VectorQuantizer, pq_decode};
use crate::vector::core::vector::Vector;

/// Bytes consumed by the per-vector codes section (everything after
/// the field_name string ends), i.e. `ceil(m / 2)`.
///
/// Currently only used by in-module round-trip tests; the writer and
/// reader inline `m.div_ceil(2)` since they already have `params.m`
/// in scope.
#[inline]
#[cfg(test)]
pub(super) const fn pq_fastscan_record_payload_size(m: u16) -> usize {
    (m as usize).div_ceil(2)
}

/// Train a K=16 PQ codebook on the given vectors and encode each one
/// into 4-bit codes.
///
/// Returns `(params, codebook, codes)` where `codes[i]` is a `Vec<u8>`
/// of length `params.m`, each entry in `[0, 15]`. The caller pairs
/// each entry back with its `(doc_id, field_name)` triple before
/// writing through [`write_pq_fastscan_record`].
///
/// # Errors
///
/// Forwards any error from [`VectorQuantizer::train`] /
/// [`VectorQuantizer::quantize`] (e.g. empty input, dimension
/// mismatch, `dim % m != 0`). Returns
/// [`LaurusError::InvalidOperation`] if a sub-quantiser code ends up
/// outside `[0, 15]` (should never happen with `K = 16` but we
/// validate to fail loudly rather than silently truncating during
/// packing).
pub(super) fn quantize_segment_pq_fastscan(
    vectors: &[Vector],
    dim: usize,
    subvector_count: usize,
) -> Result<(PqParams, Vec<f32>, Vec<Vec<u8>>)> {
    let mut quantizer = VectorQuantizer::new(
        QuantizationMethod::ProductQuantizationFastScan { subvector_count },
        dim,
    );
    quantizer.train(vectors)?;
    let (params, codebook_slice) = quantizer
        .pq_state()
        .expect("PQ FastScan quantizer trained successfully implies pq_state is set");
    let params = *params;
    if params.k != 16 {
        return Err(LaurusError::InvalidOperation(format!(
            "PQ FastScan must use k == 16, got k = {}",
            params.k
        )));
    }
    let codebook: Vec<f32> = codebook_slice.to_vec();
    let codes: Vec<Vec<u8>> = vectors
        .iter()
        .map(|v| quantizer.quantize(v).map(|(c, _)| c))
        .collect::<Result<_>>()?;
    // Validate every code is a valid 4-bit value before we let the
    // packer truncate.
    for (i, c) in codes.iter().enumerate() {
        for (j, &code) in c.iter().enumerate() {
            if code >= 16 {
                return Err(LaurusError::InvalidOperation(format!(
                    "FastScan code {code} (vector {i}, sub {j}) exceeds 4-bit range"
                )));
            }
        }
    }
    Ok((params, codebook, codes))
}

/// Pack `m` 4-bit codes into `ceil(m / 2)` bytes, low nibble first.
///
/// Mirrors the FAISS `pq4_pack_codes_1` per-vector helper. Codes must
/// already be validated to fit in 4 bits (see
/// [`quantize_segment_pq_fastscan`]).
#[inline]
fn pack_one_record(codes: &[u8]) -> Vec<u8> {
    let m = codes.len();
    let mut packed = vec![0u8; m.div_ceil(2)];
    for (i, &c) in codes.iter().enumerate() {
        let nibble = c & 0x0F;
        if i % 2 == 0 {
            packed[i / 2] |= nibble;
        } else {
            packed[i / 2] |= nibble << 4;
        }
    }
    packed
}

/// Unpack `ceil(m / 2)` bytes back into `m` 4-bit codes (`Vec<u8>` of
/// length `m`, each entry in `[0, 15]`).
#[inline]
fn unpack_one_record(packed: &[u8], m: usize) -> Vec<u8> {
    let mut codes = vec![0u8; m];
    for (i, slot) in codes.iter_mut().enumerate().take(m) {
        let byte = packed[i / 2];
        *slot = if i % 2 == 0 {
            byte & 0x0F
        } else {
            (byte >> 4) & 0x0F
        };
    }
    codes
}

/// Write the 4-bit packed codes tail of one FastScan record.
///
/// The caller is responsible for writing the preceding `doc_id` /
/// `field_name_len` / `field_name` fields and the per-segment
/// [`crate::vector::index::format::VectorSegmentHeader`].
pub(super) fn write_pq_fastscan_record<W: Write>(output: &mut W, codes: &[u8]) -> Result<()> {
    let packed = pack_one_record(codes);
    output.write_all(&packed)?;
    Ok(())
}

/// Read the 4-bit packed codes tail of one FastScan record and unpack
/// it into a `Vec<u8>` of length `m`, ready to feed
/// [`crate::vector::index::pq_fastscan_storage::PqFastScanPool::build`].
pub(super) fn read_pq_fastscan_record<R: Read>(input: &mut R, params: PqParams) -> Result<Vec<u8>> {
    let m = params.m as usize;
    let packed_len = m.div_ceil(2);
    let mut packed = vec![0u8; packed_len];
    input.read_exact(&mut packed)?;
    Ok(unpack_one_record(&packed, m))
}

/// Read the 4-bit packed codes tail of one FastScan record and
/// reconstruct it back to a `Vec<f32>` of length `params.original_dim()`
/// via the codebook. Used by HNSW writer's load path that keeps the
/// in-memory representation as f32 on rebuild — mirrors
/// [`crate::vector::index::pq_io::read_dequantized_pq_vector`] for the
/// 8-bit PQ variant.
pub(super) fn read_dequantized_pq_fastscan_vector<R: Read>(
    input: &mut R,
    params: PqParams,
    codebook: &[f32],
) -> Result<Vec<f32>> {
    let codes = read_pq_fastscan_record(input, params)?;
    Ok(pq_decode(&codes, params, codebook))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn vec_of(values: &[f32]) -> Vector {
        Vector::new(values.to_vec())
    }

    #[test]
    fn payload_size_is_ceil_m_over_two() {
        assert_eq!(pq_fastscan_record_payload_size(0), 0);
        assert_eq!(pq_fastscan_record_payload_size(1), 1);
        assert_eq!(pq_fastscan_record_payload_size(2), 1);
        assert_eq!(pq_fastscan_record_payload_size(3), 2);
        assert_eq!(pq_fastscan_record_payload_size(8), 4);
        assert_eq!(pq_fastscan_record_payload_size(16), 8);
    }

    #[test]
    fn pack_unpack_round_trip_even_m() {
        let codes = vec![0, 1, 2, 15, 8, 7, 4, 3];
        let packed = pack_one_record(&codes);
        assert_eq!(packed.len(), 4);
        let unpacked = unpack_one_record(&packed, codes.len());
        assert_eq!(unpacked, codes);
    }

    #[test]
    fn pack_unpack_round_trip_odd_m() {
        let codes = vec![5, 10, 15];
        let packed = pack_one_record(&codes);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_one_record(&packed, codes.len());
        assert_eq!(unpacked, codes);
    }

    #[test]
    fn quantize_segment_pq_fastscan_returns_k_16_codes() {
        let vectors = vec![
            vec_of(&[10.0, 10.0, 20.0, 20.0]),
            vec_of(&[-10.0, -10.0, -20.0, -20.0]),
            vec_of(&[10.1, 10.1, 20.1, 20.1]),
        ];
        let (params, codebook, codes) = quantize_segment_pq_fastscan(&vectors, 4, 2).unwrap();
        assert_eq!(params.m, 2);
        assert_eq!(params.k, 16);
        assert_eq!(params.sub_dim, 2);
        assert_eq!(codebook.len(), params.codebook_len());
        assert_eq!(codes.len(), 3);
        for c in &codes {
            assert_eq!(c.len(), 2);
            for &code in c {
                assert!(code < 16, "code {code} exceeds 4-bit range");
            }
        }
    }

    #[test]
    fn write_then_read_pq_fastscan_round_trips_codes() {
        let codes: Vec<Vec<u8>> = vec![vec![0, 1, 2, 3], vec![15, 14, 13, 12], vec![7, 8, 9, 10]];
        let mut buf = Vec::new();
        for c in &codes {
            write_pq_fastscan_record(&mut buf, c).unwrap();
        }
        // 4 codes per record → 2 bytes per record.
        assert_eq!(buf.len(), codes.len() * pq_fastscan_record_payload_size(4));

        let params = PqParams::new(4, 16, 1).unwrap();
        let mut cursor = Cursor::new(&buf);
        for original in &codes {
            let recovered = read_pq_fastscan_record(&mut cursor, params).unwrap();
            assert_eq!(&recovered, original);
        }
    }
}
