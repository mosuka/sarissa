//! Shared writer / reader helpers for the HNSW PQ segment payload
//! (Issue #481 Stage 3, parallel to
//! [`crate::vector::index::quantized_io`] for the Scalar8Bit variant).
//!
//! # On-disk layout for the PQ region
//!
//! ```text
//! [ VectorSegmentHeader        (LVS1 + PQ params + codebook) ]
//! repeat num_vectors times:
//!   [ doc_id            u64 LE  8 bytes ]
//!   [ field ref: v3+ = field_id u16 (per-segment dictionary,     ]
//!   [   Issue #633); v1/v2 = name_len u32 + UTF-8 name           ]
//!   [ codes                   m bytes (per sub-vector centroid index) ]
//! ```
//!
//! Unlike the Stage 1 layout, PQ records have no per-vector metadata
//! block — the codebook covers all the affine corrections SQ needs to
//! reconstruct distances.

use std::io::{Read, Write};

use crate::error::Result;
use crate::vector::core::quantization::{PqParams, QuantizationMethod, VectorQuantizer, pq_decode};
use crate::vector::core::vector::Vector;
use crate::vector::index::pq_codebook::SharedPqCodebook;

/// Bytes consumed by the per-vector codes (everything after the
/// field_name string ends). Equal to `m`.
///
/// Currently only used by the in-module roundtrip test; the writer
/// (Phase 3b) inlines `codes.len()` and the reader inlines
/// `params.m as usize` since both already carry that information.
#[inline]
#[cfg(test)]
pub(super) const fn pq_record_payload_size(m: u16) -> usize {
    m as usize
}

/// Encode `vectors` against a PQ codebook, either training a fresh one
/// or reusing a pre-trained [`SharedPqCodebook`] (Issue #631).
///
/// Returns `(params, codebook, codes)` where `codes` is one
/// `Vec<u8>` of length `params.m` per input vector, in the same order
/// as the input slice. The caller pairs each entry back with its
/// `(doc_id, field_name)` triple before writing.
///
/// # Arguments
///
/// * `vectors` - The vectors to encode.
/// * `dim` - Original vector dimension.
/// * `subvector_count` - PQ `m` (must divide `dim`).
/// * `shared` - When `Some`, skip training entirely and encode against
///   this pre-trained codebook via [`VectorQuantizer::from_pq_codebook`]
///   (Issue #631 — training a fresh codebook on every segment/merge
///   costs multiple seconds and recurs at every tier of the
///   segment-per-commit merge hierarchy). When `None`, train a fresh
///   codebook on `vectors` as before.
///
/// # Errors
///
/// * Forwards [`SharedPqCodebook::validate_for`]'s error if `shared`'s
///   geometry does not match `dim`/`subvector_count`.
/// * Otherwise forwards any error from [`VectorQuantizer::train`] /
///   [`VectorQuantizer::quantize`] (e.g. empty input, dimension
///   mismatch, non-finite values, `dim % m != 0`).
pub(super) fn quantize_segment_pq(
    vectors: &[Vector],
    dim: usize,
    subvector_count: usize,
    shared: Option<&SharedPqCodebook>,
) -> Result<(PqParams, Vec<f32>, Vec<Vec<u8>>)> {
    let quantizer = match shared {
        Some(cb) => {
            cb.validate_for(dim, subvector_count)?;
            VectorQuantizer::from_pq_codebook(dim, cb.params, cb.codebook.clone())?
        }
        None => {
            let mut quantizer = VectorQuantizer::new(
                QuantizationMethod::ProductQuantization { subvector_count },
                dim,
            );
            quantizer.train(vectors)?;
            quantizer
        }
    };
    let (params, codebook_slice) = quantizer
        .pq_state()
        .expect("PQ quantizer is trained or built from an existing codebook, so pq_state is set");
    let params = *params;
    let codebook: Vec<f32> = codebook_slice.to_vec();
    let codes: Vec<Vec<u8>> = vectors
        .iter()
        .map(|v| quantizer.quantize(v).map(|(c, _)| c))
        .collect::<Result<_>>()?;
    Ok((params, codebook, codes))
}

/// Write the PQ codes tail of one vector record.
///
/// The caller is responsible for writing the preceding `doc_id` /
/// `field_name_len` / `field_name` fields and the per-segment
/// [`crate::vector::index::format::VectorSegmentHeader`].
pub(super) fn write_pq_record<W: Write>(output: &mut W, codes: &[u8]) -> Result<()> {
    output.write_all(codes)?;
    Ok(())
}

/// Read the PQ codes tail of one vector record and reconstruct it
/// back to a `Vec<f32>` of length `params.original_dim()` via the
/// codebook. Used by load paths that keep the in-memory representation
/// as f32 (the HNSW writer's `load` path on rebuild).
pub(super) fn read_dequantized_pq_vector<R: Read>(
    input: &mut R,
    params: PqParams,
    codebook: &[f32],
) -> Result<Vec<f32>> {
    let mut codes = vec![0u8; params.m as usize];
    input.read_exact(&mut codes)?;
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
    fn quantize_segment_pq_returns_params_codebook_codes() {
        let vectors = vec![
            vec_of(&[10.0, 10.0, 20.0, 20.0]),
            vec_of(&[-10.0, -10.0, -20.0, -20.0]),
            vec_of(&[10.1, 10.1, 20.1, 20.1]),
        ];
        let (params, codebook, codes) = quantize_segment_pq(&vectors, 4, 2, None).unwrap();
        assert_eq!(params.m, 2);
        assert_eq!(params.k, 256);
        assert_eq!(params.sub_dim, 2);
        assert_eq!(codebook.len(), params.codebook_len());
        assert_eq!(codes.len(), 3);
        for c in &codes {
            assert_eq!(c.len(), 2);
        }
    }

    #[test]
    fn write_then_read_pq_roundtrips_to_codebook() {
        let vectors = vec![
            vec_of(&[10.0, 10.0, 20.0, 20.0]),
            vec_of(&[-10.0, -10.0, -20.0, -20.0]),
        ];
        let (params, codebook, codes) = quantize_segment_pq(&vectors, 4, 2, None).unwrap();

        let mut buf = Vec::new();
        for c in &codes {
            write_pq_record(&mut buf, c).unwrap();
        }
        assert_eq!(buf.len(), codes.len() * pq_record_payload_size(params.m));

        let mut cursor = Cursor::new(&buf);
        for (i, original) in vectors.iter().enumerate() {
            let recovered = read_dequantized_pq_vector(&mut cursor, params, &codebook).unwrap();
            assert_eq!(recovered.len(), 4);
            // PQ is lossy; just require that the reconstruction is on
            // the correct side of zero for this well-separated fixture.
            assert_eq!(
                recovered[0].signum(),
                original.data[0].signum(),
                "vector {i} sign flip on first coord"
            );
        }
    }

    #[test]
    fn payload_size_equals_m() {
        assert_eq!(pq_record_payload_size(0), 0);
        assert_eq!(pq_record_payload_size(16), 16);
    }

    /// Issue #631: encoding against a pre-trained [`SharedPqCodebook`]
    /// must produce byte-identical output to training a fresh codebook
    /// on the exact same corpus -- the strongest cheap correctness
    /// check available, since `pq_train_codebook` is deterministic
    /// (seeded k-means).
    #[test]
    fn shared_codebook_produces_byte_identical_codes_to_fresh_training() {
        let mut state: u64 = 0x1234_5678_9ABC_DEF0;
        let vectors: Vec<Vector> = (0..300)
            .map(|_| {
                let data: Vec<f32> = (0..32)
                    .map(|_| {
                        state = state
                            .wrapping_mul(6_364_136_223_846_793_005)
                            .wrapping_add(1_442_695_040_888_963_407);
                        ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
                    })
                    .collect();
                Vector::new(data)
            })
            .collect();

        let (trained_params, trained_codebook, trained_codes) =
            quantize_segment_pq(&vectors, 32, 4, None).unwrap();

        let shared = crate::vector::index::pq_codebook::SharedPqCodebook {
            params: trained_params,
            codebook: trained_codebook.clone(),
        };
        let (shared_params, shared_codebook, shared_codes) =
            quantize_segment_pq(&vectors, 32, 4, Some(&shared)).unwrap();

        assert_eq!(shared_params, trained_params);
        assert_eq!(shared_codebook, trained_codebook);
        assert_eq!(shared_codes, trained_codes);
    }
}
