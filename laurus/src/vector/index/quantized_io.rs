//! Shared writer / reader helpers for the HNSW quantized vector
//! payload (Issue #481 Stage 1, Step 5).
//!
//! The HNSW segment layout puts `(doc_id, field_name)` and the
//! quantized vector record together for each entry, which prevents
//! reusing [`crate::vector::index::quantized_segment::QuantizedSegmentVectors`]
//! directly (that type assumes a homogeneous AoS payload).
//!
//! This module centralises the per-vector record encoding so
//! `HnswIndexWriter::write` (writer) and `HnswIndexWriter::load` /
//! `HnswIndexReader::load` (readers) cannot drift apart.
//!
//! # On-disk layout for the quantized region
//!
//! ```text
//! [ VectorSegmentHeader  24 bytes ]   <- written / read separately
//! repeat num_vectors times:
//!   [ doc_id           u64  LE   8 bytes ]
//!   [ field ref: v3+ = field_id u16 (per-segment dictionary,     ]
//!   [   Issue #633); v1/v2 = name_len u32 + UTF-8 name           ]
//!   [ int8 data            dim bytes ]
//!   [ sum_q             u32  LE   4 bytes ]
//!   [ norm_q            f32  LE   4 bytes ]
//! ```

use std::io::{Read, Write};

use crate::error::Result;
use crate::vector::core::quantization::{
    QuantizationMethod, QuantizedVectorMeta, ScalarQuantParams, VectorQuantizer,
};
use crate::vector::core::vector::Vector;

/// One quantized vector record returned by [`quantize_segment`]:
/// the int8 payload bytes paired with their per-vector meta.
pub(super) type QuantizedRecord = (Vec<u8>, QuantizedVectorMeta);

/// Bytes consumed by the int8 + meta portion of one vector record
/// (everything after the field_name string ends).
#[inline]
pub(super) const fn quantized_record_payload_size(dim: usize) -> usize {
    dim + QuantizedVectorMeta::SERIALIZED_SIZE
}

/// Train segment-level [`ScalarQuantParams`] on the given vectors and
/// quantize each one.
///
/// The returned records are in the same order as the input, so the
/// caller can pair them back with `(doc_id, field_name)` triples.
///
/// # Errors
///
/// Forwards any error from [`VectorQuantizer::train`] /
/// [`VectorQuantizer::quantize`] (e.g. empty input, dimension mismatch,
/// non-finite values).
pub(super) fn quantize_segment(
    vectors: &[Vector],
    dim: usize,
) -> Result<(ScalarQuantParams, Vec<QuantizedRecord>)> {
    let mut quantizer = VectorQuantizer::new(QuantizationMethod::Scalar8Bit, dim);
    quantizer.train(vectors)?;
    let params = *quantizer
        .params()
        .expect("quantizer trained successfully implies params are set");
    let records: Vec<QuantizedRecord> = vectors
        .iter()
        .map(|v| quantizer.quantize(v))
        .collect::<Result<_>>()?;
    Ok((params, records))
}

/// Write the int8 + meta tail of one vector record.
///
/// The caller is responsible for writing the preceding `doc_id` /
/// `field_name_len` / `field_name` fields.
pub(super) fn write_quantized_record<W: Write>(
    output: &mut W,
    int8_data: &[u8],
    meta: QuantizedVectorMeta,
) -> Result<()> {
    output.write_all(int8_data)?;
    output.write_all(&meta.sum_q.to_le_bytes())?;
    output.write_all(&meta.norm_q.to_le_bytes())?;
    Ok(())
}

/// Read the int8 + meta tail of one vector record and dequantize it
/// back to a `Vec<f32>` of length `dim`.
///
/// Used by load paths that keep the in-memory representation as f32
/// (Step 5 of #481 Stage 1; Step 6 will switch the in-memory form to
/// int8 directly).
pub(super) fn read_dequantized_vector<R: Read>(
    input: &mut R,
    dim: usize,
    params: &ScalarQuantParams,
) -> Result<Vec<f32>> {
    let mut int8_buf = vec![0u8; dim];
    input.read_exact(&mut int8_buf)?;
    // Skip the 8-byte meta (sum_q + norm_q): the in-memory form is
    // f32 in this step, so we don't need them here. They will become
    // load-time state in Step 6.
    let mut meta_buf = [0u8; QuantizedVectorMeta::SERIALIZED_SIZE];
    input.read_exact(&mut meta_buf)?;
    Ok(int8_buf
        .iter()
        .map(|&b| params.dequantize_value(b))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn vec_of(values: &[f32]) -> Vector {
        Vector::new(values.to_vec())
    }

    #[test]
    fn quantize_segment_returns_params_and_records() {
        let vectors = vec![
            vec_of(&[-1.0, 0.0, 1.0]),
            vec_of(&[-0.5, 0.5, 0.25]),
            vec_of(&[0.1, -0.4, 0.9]),
        ];
        let (params, records) = quantize_segment(&vectors, 3).unwrap();
        assert!(params.scale > 0.0);
        assert_eq!(records.len(), 3);
        for (i, (q, meta)) in records.iter().enumerate() {
            assert_eq!(q.len(), 3, "vector {i}");
            let expected = QuantizedVectorMeta::from_quantized(q, &params);
            assert_eq!(meta.sum_q, expected.sum_q);
            assert!((meta.norm_q - expected.norm_q).abs() < 1e-5);
        }
    }

    #[test]
    fn write_then_read_dequantized_roundtrips_within_scale() {
        let dim = 8;
        let vectors = vec![
            vec_of(&[-1.0, -0.7, -0.3, 0.0, 0.2, 0.5, 0.8, 1.0]),
            vec_of(&[0.1, 0.2, 0.3, 0.4, -0.4, -0.3, -0.2, -0.1]),
        ];
        let (params, records) = quantize_segment(&vectors, dim).unwrap();
        let mut buf = Vec::new();
        for (q, meta) in &records {
            write_quantized_record(&mut buf, q, *meta).unwrap();
        }
        assert_eq!(
            buf.len(),
            records.len() * quantized_record_payload_size(dim)
        );

        let mut cursor = Cursor::new(&buf);
        for (i, original) in vectors.iter().enumerate() {
            let recovered = read_dequantized_vector(&mut cursor, dim, &params).unwrap();
            for (j, (orig, rec)) in original.data.iter().zip(recovered.iter()).enumerate() {
                assert!(
                    (orig - rec).abs() <= params.scale + 1e-6,
                    "vector {i} dim {j}: orig = {orig}, rec = {rec}, scale = {}",
                    params.scale
                );
            }
        }
    }

    #[test]
    fn payload_size_is_dim_plus_eight() {
        assert_eq!(quantized_record_payload_size(0), 8);
        assert_eq!(quantized_record_payload_size(128), 136);
    }
}
