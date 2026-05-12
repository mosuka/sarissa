//! Vector quantization for memory-efficient storage and fast distance.
//!
//! This module implements **per-segment global affine** scalar quantization
//! for dense vectors:
//!
//! - One `(offset, scale)` pair per segment (f32, segment-global, **not**
//!   per-dimension and **not** per-vector).
//! - Each f32 element is encoded as a `u8`:
//!   `q = clamp(round((v - offset) / scale), 0, 255)`.
//! - Each vector is stored on disk as `dim` bytes plus a small
//!   [`QuantizedVectorMeta`] (`sum_q: u32` + `norm_q: f32` = 8 bytes) used
//!   by the int8 distance kernel to recover Cosine / DotProduct values
//!   without per-element dequantization.
//!
//! The trade-off is intentional: per-segment global affine is less
//! accurate than per-dimension affine, but allows the distance hot loop
//! to collapse into one int8 SIMD multiply-accumulate plus three scalar
//! corrections — the speed win that makes the 2× HNSW search target of
//! Issue #481 Stage 1 reachable.
//!
//! # Variants
//!
//! - [`QuantizationMethod::Scalar8Bit`]: implemented here. Default.
//! - [`QuantizationMethod::ProductQuantization`]: reserved for Stage 3 of
//!   Issue #481. Constructing a [`VectorQuantizer`] with this method
//!   succeeds, but [`VectorQuantizer::train`] / [`VectorQuantizer::quantize`]
//!   return [`LaurusError::NotImplemented`].

use serde::{Deserialize, Serialize};

use crate::error::{LaurusError, Result};
use crate::vector::core::vector::Vector;

/// Quantization methods for compressing dense vectors.
///
/// Used in [`crate::vector::core::field`] options to declare the
/// per-field quantization choice. The default is [`Scalar8Bit`](Self::Scalar8Bit).
///
/// `None` (no-quantization) is intentionally not represented — Issue
/// #481 Stage 1 mandates int8 quantization for all vector fields. The
/// on-disk format reserves `quant_kind = 0` for a future re-introduction
/// of an unquantized variant if ever needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum QuantizationMethod {
    /// Scalar quantization to 8-bit integers using per-segment global
    /// affine `(offset, scale)`. 4× memory reduction vs unquantized
    /// f32; ≈ 2× search latency improvement at recall ≥ 0.95.
    #[default]
    Scalar8Bit,
    /// Product quantization. **Stage 3 of Issue #481 — currently
    /// returns [`LaurusError::NotImplemented`].** The variant is kept
    /// in the enum so callers (CLI / proto / bindings) can pre-select
    /// it and surface a clear error until the implementation lands.
    ProductQuantization {
        /// Number of sub-vectors. Stage-3 implementation will use this
        /// to derive the codebook layout.
        subvector_count: usize,
    },
}

/// Segment-level scalar quantization parameters.
///
/// All vectors in one segment share a single `(offset, scale)` pair.
/// Computed at flush time from the vectors being persisted, then
/// written to the segment header so readers can reconstruct
/// approximate f32 values via [`Self::dequantize_value`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ScalarQuantParams {
    /// Affine offset shared by all vectors in this segment.
    pub offset: f32,
    /// Affine scale shared by all vectors in this segment. Strictly
    /// positive after a successful [`Self::train`] (constant input
    /// data falls back to `scale = 1.0`).
    pub scale: f32,
}

impl ScalarQuantParams {
    /// Train per-segment global affine parameters from a set of vectors.
    ///
    /// Scans every element of every input vector to find the global
    /// `min` and `max`, then derives `offset = min` and
    /// `scale = (max - min) / 255`.
    ///
    /// # Arguments
    ///
    /// * `vectors` - Non-empty slice of training vectors. Empty slices
    ///   yield an error since min / max are undefined.
    ///
    /// # Returns
    ///
    /// `(offset, scale)` such that `quantize_value` is well-defined for
    /// the entire input range. If the input is constant (all elements
    /// equal), returns `(min, 1.0)` so quantization still produces a
    /// well-defined `0` for every element.
    ///
    /// # Errors
    ///
    /// * [`LaurusError::InvalidOperation`] if `vectors` is empty.
    /// * [`LaurusError::InvalidOperation`] if any element is non-finite
    ///   (`NaN` or `±inf`), since quantization parameters would be
    ///   meaningless.
    pub fn train(vectors: &[Vector]) -> Result<Self> {
        if vectors.is_empty() {
            return Err(LaurusError::InvalidOperation(
                "Cannot train scalar quantization on an empty vector set".to_string(),
            ));
        }
        let mut min_v = f32::INFINITY;
        let mut max_v = f32::NEG_INFINITY;
        let mut total_count: usize = 0;
        for v in vectors {
            for &x in v.data.iter() {
                if !x.is_finite() {
                    return Err(LaurusError::InvalidOperation(
                        "Training vectors contain NaN or infinite values".to_string(),
                    ));
                }
                if x < min_v {
                    min_v = x;
                }
                if x > max_v {
                    max_v = x;
                }
                total_count += 1;
            }
        }
        if total_count == 0 {
            return Err(LaurusError::InvalidOperation(
                "Training vectors are all empty (zero dimensions)".to_string(),
            ));
        }
        let range = max_v - min_v;
        if range <= 0.0 {
            // Constant data: pick a neutral scale so quantize_value
            // yields a well-defined u8 (always 0) for every element.
            return Ok(Self {
                offset: min_v,
                scale: 1.0,
            });
        }
        Ok(Self {
            offset: min_v,
            scale: range / 255.0,
        })
    }

    /// Quantize one f32 element to a `u8`.
    ///
    /// Saturates outside `[offset, offset + 255 * scale]` rather than
    /// wrapping. This is the right behavior for query-side quantization
    /// where a query may have values just outside the training range.
    #[inline]
    pub fn quantize_value(&self, v: f32) -> u8 {
        let normalized = (v - self.offset) / self.scale;
        normalized.round().clamp(0.0, 255.0) as u8
    }

    /// Dequantize one `u8` back to f32.
    #[inline]
    pub fn dequantize_value(&self, q: u8) -> f32 {
        self.offset + self.scale * (q as f32)
    }

    /// Quantize a whole [`Vector`] into a `Vec<u8>` of the same length.
    pub fn quantize(&self, vector: &Vector) -> Vec<u8> {
        vector
            .data
            .iter()
            .map(|&v| self.quantize_value(v))
            .collect()
    }

    /// Quantize a slice of f32 values.
    pub fn quantize_slice(&self, data: &[f32]) -> Vec<u8> {
        data.iter().map(|&v| self.quantize_value(v)).collect()
    }

    /// Dequantize a slice of `u8` back into a `Vec<f32>`.
    pub fn dequantize(&self, q: &[u8]) -> Vec<f32> {
        q.iter().map(|&qi| self.dequantize_value(qi)).collect()
    }
}

/// Per-vector metadata persisted alongside the int8 vector data.
///
/// Lives next to each quantized vector on disk (8 bytes / vector) and
/// is read into memory together with the int8 payload. The fields are
/// computed at quantization time so the search hot loop can avoid
/// per-element dequantization for Cosine and Dot-product distances.
///
/// Layout (when persisted):
/// ```text
/// [ int8 data (dim bytes) | sum_q (4 bytes, u32 LE) | norm_q (4 bytes, f32 LE) ]
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[repr(C)]
pub struct QuantizedVectorMeta {
    /// `Σ q[i]` over all elements of the quantized vector. Used by the
    /// distance kernel to compute the cross-term correction
    /// `scale * offset * sum_q` without a hot-loop pass over `q`.
    pub sum_q: u32,
    /// L2 norm of the dequantized representation,
    /// `sqrt(Σ dequantize(q[i])^2)`. Used as the `||a||` denominator in
    /// Cosine distance so cosine never falls back to per-element
    /// dequantize during search.
    pub norm_q: f32,
}

impl QuantizedVectorMeta {
    /// Compute metadata from an already-quantized vector.
    ///
    /// # Arguments
    ///
    /// * `q` - The quantized representation (u8 slice).
    /// * `params` - The same segment-level params used to produce `q`.
    ///   Required to recover the f32 values for `norm_q`.
    pub fn from_quantized(q: &[u8], params: &ScalarQuantParams) -> Self {
        let mut sum_q: u32 = 0;
        let mut norm_sq: f32 = 0.0;
        for &qi in q {
            sum_q += qi as u32;
            let dq = params.dequantize_value(qi);
            norm_sq += dq * dq;
        }
        Self {
            sum_q,
            norm_q: norm_sq.sqrt(),
        }
    }

    /// Serialized size in bytes (8 = 4 for sum_q + 4 for norm_q).
    pub const SERIALIZED_SIZE: usize = 8;
}

/// High-level quantizer that pairs a [`QuantizationMethod`] with
/// segment-level state ([`ScalarQuantParams`]).
///
/// Typical use:
/// 1. `let mut q = VectorQuantizer::new(method, dim);`
/// 2. `q.train(&segment_vectors)?;`
/// 3. for each vector to persist: `let (bytes, meta) = q.quantize(&v)?;`
/// 4. write `bytes` and `meta` to the segment, plus `q.params()` to the
///    segment header.
///
/// On read-back, construct via [`Self::from_params`] using the params
/// recovered from the header.
#[derive(Debug, Clone)]
pub struct VectorQuantizer {
    method: QuantizationMethod,
    dimension: usize,
    params: Option<ScalarQuantParams>,
}

impl VectorQuantizer {
    /// Create an untrained quantizer for `dimension`-d vectors.
    ///
    /// Calling [`quantize`](Self::quantize) before [`train`](Self::train)
    /// returns [`LaurusError::InvalidOperation`].
    pub fn new(method: QuantizationMethod, dimension: usize) -> Self {
        Self {
            method,
            dimension,
            params: None,
        }
    }

    /// Construct a quantizer from already-known params (e.g. loaded
    /// from a segment header at read time).
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError::NotImplemented`] for
    /// [`QuantizationMethod::ProductQuantization`].
    pub fn from_params(
        method: QuantizationMethod,
        dimension: usize,
        params: ScalarQuantParams,
    ) -> Result<Self> {
        match method {
            QuantizationMethod::Scalar8Bit => Ok(Self {
                method,
                dimension,
                params: Some(params),
            }),
            QuantizationMethod::ProductQuantization { .. } => Err(LaurusError::NotImplemented(
                "Product quantization (Issue #481 Stage 3) is not yet implemented".to_string(),
            )),
        }
    }

    /// Train segment-level quantization params on a representative set
    /// of vectors.
    ///
    /// For [`QuantizationMethod::Scalar8Bit`] this delegates to
    /// [`ScalarQuantParams::train`]. For
    /// [`QuantizationMethod::ProductQuantization`] this returns
    /// [`LaurusError::NotImplemented`].
    pub fn train(&mut self, vectors: &[Vector]) -> Result<()> {
        match self.method {
            QuantizationMethod::Scalar8Bit => {
                let params = ScalarQuantParams::train(vectors)?;
                self.params = Some(params);
                Ok(())
            }
            QuantizationMethod::ProductQuantization { .. } => Err(LaurusError::NotImplemented(
                "Product quantization (Issue #481 Stage 3) is not yet implemented".to_string(),
            )),
        }
    }

    /// Quantize a single vector and produce the per-vector metadata.
    ///
    /// # Errors
    ///
    /// * [`LaurusError::InvalidOperation`] if the quantizer has not
    ///   been trained yet.
    /// * [`LaurusError::InvalidOperation`] if `vector.dimension()`
    ///   differs from the dimension passed at construction time.
    /// * [`LaurusError::NotImplemented`] for
    ///   [`QuantizationMethod::ProductQuantization`].
    pub fn quantize(&self, vector: &Vector) -> Result<(Vec<u8>, QuantizedVectorMeta)> {
        match self.method {
            QuantizationMethod::Scalar8Bit => {
                let params = self.params.as_ref().ok_or_else(|| {
                    LaurusError::InvalidOperation(
                        "Quantizer must be trained before quantizing vectors".to_string(),
                    )
                })?;
                if vector.dimension() != self.dimension {
                    return Err(LaurusError::InvalidOperation(format!(
                        "Vector dimension mismatch: expected {}, got {}",
                        self.dimension,
                        vector.dimension()
                    )));
                }
                let q = params.quantize(vector);
                let meta = QuantizedVectorMeta::from_quantized(&q, params);
                Ok((q, meta))
            }
            QuantizationMethod::ProductQuantization { .. } => Err(LaurusError::NotImplemented(
                "Product quantization (Issue #481 Stage 3) is not yet implemented".to_string(),
            )),
        }
    }

    /// Get the segment-level quantization params, if trained.
    pub fn params(&self) -> Option<&ScalarQuantParams> {
        self.params.as_ref()
    }

    /// Get the configured quantization method.
    pub fn method(&self) -> QuantizationMethod {
        self.method
    }

    /// Get the configured vector dimension.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Whether the quantizer has been trained (or constructed from
    /// known params).
    pub fn is_trained(&self) -> bool {
        self.params.is_some()
    }

    /// Compression ratio vs unquantized f32. `4.0` for
    /// [`QuantizationMethod::Scalar8Bit`].
    ///
    /// Note: the ratio counts only the int8 data path. The 8-byte
    /// per-vector metadata adds `8 / (dim * 4)` overhead, e.g. ~6 % at
    /// `dim = 32`, ~1 % at `dim = 192`, ~0.5 % at `dim = 384`.
    pub fn compression_ratio(&self) -> f32 {
        match self.method {
            QuantizationMethod::Scalar8Bit => 4.0,
            QuantizationMethod::ProductQuantization { subvector_count } => {
                if subvector_count == 0 {
                    1.0
                } else {
                    (self.dimension * 4) as f32 / subvector_count as f32
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vec_of(values: &[f32]) -> Vector {
        Vector::new(values.to_vec())
    }

    #[test]
    fn train_picks_min_and_max_across_all_vectors() {
        let vectors = vec![
            vec_of(&[0.0, 1.0, 2.0]),
            vec_of(&[-1.5, 3.0, 0.5]),
            vec_of(&[5.0, -2.0, 1.0]),
        ];
        let params = ScalarQuantParams::train(&vectors).unwrap();
        assert_eq!(params.offset, -2.0); // global min
        assert!((params.scale - (5.0 - -2.0) / 255.0).abs() < 1e-7);
    }

    #[test]
    fn train_on_empty_set_is_invalid_operation() {
        let err = ScalarQuantParams::train(&[]).unwrap_err();
        assert!(matches!(err, LaurusError::InvalidOperation(_)));
    }

    #[test]
    fn train_on_constant_data_uses_unit_scale() {
        let vectors = vec![vec_of(&[3.5, 3.5, 3.5]); 4];
        let params = ScalarQuantParams::train(&vectors).unwrap();
        assert_eq!(params.offset, 3.5);
        assert_eq!(params.scale, 1.0);
        // Every element quantizes to 0 (since (3.5 - 3.5) / 1.0 = 0).
        for &v in &[3.5, 3.5, 3.5] {
            assert_eq!(params.quantize_value(v), 0);
        }
    }

    #[test]
    fn train_rejects_non_finite_values() {
        let vectors = vec![vec_of(&[1.0, f32::NAN, 2.0])];
        let err = ScalarQuantParams::train(&vectors).unwrap_err();
        assert!(matches!(err, LaurusError::InvalidOperation(_)));

        let vectors = vec![vec_of(&[1.0, f32::INFINITY, 2.0])];
        let err = ScalarQuantParams::train(&vectors).unwrap_err();
        assert!(matches!(err, LaurusError::InvalidOperation(_)));
    }

    #[test]
    fn quantize_value_roundtrips_within_scale() {
        let params = ScalarQuantParams {
            offset: -1.0,
            scale: 2.0 / 255.0, // covers [-1.0, 1.0]
        };
        for v in [-1.0_f32, -0.5, 0.0, 0.25, 0.99] {
            let q = params.quantize_value(v);
            let dq = params.dequantize_value(q);
            assert!(
                (v - dq).abs() <= params.scale,
                "v = {v}, dq = {dq}, scale = {}",
                params.scale
            );
        }
    }

    #[test]
    fn quantize_value_saturates_outside_training_range() {
        let params = ScalarQuantParams {
            offset: 0.0,
            scale: 1.0 / 255.0, // covers [0.0, 1.0]
        };
        // Below range saturates to 0.
        assert_eq!(params.quantize_value(-5.0), 0);
        // Above range saturates to 255.
        assert_eq!(params.quantize_value(5.0), 255);
    }

    #[test]
    fn quantize_vector_roundtrip_within_scale() {
        let vectors = vec![
            vec_of(&[-1.0, -0.5, 0.0, 0.5, 1.0]),
            vec_of(&[0.1, -0.2, 0.3, -0.4, 0.5]),
        ];
        let params = ScalarQuantParams::train(&vectors).unwrap();
        for v in &vectors {
            let q = params.quantize(v);
            let dq = params.dequantize(&q);
            for (orig, recovered) in v.data.iter().zip(dq.iter()) {
                assert!(
                    (orig - recovered).abs() <= params.scale,
                    "orig = {orig}, recovered = {recovered}, scale = {}",
                    params.scale
                );
            }
        }
    }

    #[test]
    fn meta_sum_q_matches_summed_quantized_bytes() {
        let params = ScalarQuantParams {
            offset: 0.0,
            scale: 1.0 / 255.0,
        };
        let q: Vec<u8> = vec![0, 64, 128, 200, 255];
        let meta = QuantizedVectorMeta::from_quantized(&q, &params);
        let expected_sum: u32 = q.iter().map(|&x| x as u32).sum();
        assert_eq!(meta.sum_q, expected_sum);
    }

    #[test]
    fn meta_norm_q_matches_dequantized_norm() {
        let params = ScalarQuantParams {
            offset: -1.0,
            scale: 2.0 / 255.0,
        };
        let q: Vec<u8> = vec![0, 64, 128, 192, 255];
        let meta = QuantizedVectorMeta::from_quantized(&q, &params);
        let dq = params.dequantize(&q);
        let expected: f32 = dq.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((meta.norm_q - expected).abs() < 1e-5);
    }

    #[test]
    fn quantizer_requires_training_before_quantize() {
        let q = VectorQuantizer::new(QuantizationMethod::Scalar8Bit, 3);
        let err = q.quantize(&vec_of(&[1.0, 2.0, 3.0])).unwrap_err();
        assert!(matches!(err, LaurusError::InvalidOperation(_)));
    }

    #[test]
    fn quantizer_train_then_quantize_returns_data_and_meta() {
        let mut q = VectorQuantizer::new(QuantizationMethod::Scalar8Bit, 3);
        let training = vec![vec_of(&[-1.0, 0.0, 1.0]), vec_of(&[-0.5, 0.5, 0.25])];
        q.train(&training).unwrap();
        assert!(q.is_trained());
        let (bytes, meta) = q.quantize(&vec_of(&[0.0, 0.5, -0.25])).unwrap();
        assert_eq!(bytes.len(), 3);
        let expected_sum: u32 = bytes.iter().map(|&x| x as u32).sum();
        assert_eq!(meta.sum_q, expected_sum);
        assert!(meta.norm_q.is_finite() && meta.norm_q >= 0.0);
    }

    #[test]
    fn quantizer_rejects_dimension_mismatch() {
        let mut q = VectorQuantizer::new(QuantizationMethod::Scalar8Bit, 3);
        q.train(&[vec_of(&[1.0, 2.0, 3.0])]).unwrap();
        let err = q.quantize(&vec_of(&[1.0, 2.0])).unwrap_err();
        assert!(matches!(err, LaurusError::InvalidOperation(_)));
    }

    #[test]
    fn product_quantization_train_returns_not_implemented() {
        let mut q = VectorQuantizer::new(
            QuantizationMethod::ProductQuantization { subvector_count: 8 },
            128,
        );
        let err = q.train(&[vec_of(&[1.0; 128])]).unwrap_err();
        assert!(matches!(err, LaurusError::NotImplemented(_)));
    }

    #[test]
    fn product_quantization_from_params_returns_not_implemented() {
        let err = VectorQuantizer::from_params(
            QuantizationMethod::ProductQuantization { subvector_count: 8 },
            128,
            ScalarQuantParams {
                offset: 0.0,
                scale: 1.0,
            },
        )
        .unwrap_err();
        assert!(matches!(err, LaurusError::NotImplemented(_)));
    }

    #[test]
    fn from_params_roundtrips_for_scalar8bit() {
        let params = ScalarQuantParams {
            offset: -2.0,
            scale: 4.0 / 255.0,
        };
        let q = VectorQuantizer::from_params(QuantizationMethod::Scalar8Bit, 5, params).unwrap();
        assert_eq!(q.method(), QuantizationMethod::Scalar8Bit);
        assert_eq!(q.dimension(), 5);
        assert_eq!(q.params(), Some(&params));
    }

    #[test]
    fn default_method_is_scalar_8bit() {
        let m: QuantizationMethod = Default::default();
        assert_eq!(m, QuantizationMethod::Scalar8Bit);
    }

    #[test]
    fn compression_ratio_scalar_is_4x() {
        let q = VectorQuantizer::new(QuantizationMethod::Scalar8Bit, 128);
        assert_eq!(q.compression_ratio(), 4.0);
    }

    #[test]
    fn meta_serialized_size_is_eight() {
        assert_eq!(QuantizedVectorMeta::SERIALIZED_SIZE, 8);
    }
}
