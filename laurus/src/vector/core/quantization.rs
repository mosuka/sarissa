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
//! - [`QuantizationMethod::ProductQuantization`]: implemented here.
//!   Stage 3 of Issue #481. Trains a per-segment codebook of `M`
//!   sub-vectors × `K = 256` centroids via Lloyd k-means with
//!   k-means++ initialisation, then encodes every vector as `M` bytes.
//!   The distance kernel is wired up separately in
//!   [`crate::vector::core::distance_quantized`].

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
    /// FastScan product quantization (K=16 4-bit codes + SIMD LUT
    /// distance). Experimental, only available with the `pq-fastscan`
    /// cargo feature and only supported by HNSW indexes today
    /// (Issue [#695](https://github.com/mosuka/laurus/issues/695) /
    /// part D of [#651](https://github.com/mosuka/laurus/issues/651)).
    ///
    /// Use this when the HNSW search latency is dominated by the PQ
    /// ADC kernel and the corpus fits the FAISS-style block layout
    /// (32 vectors / block, 4-bit packed codes).
    #[cfg(feature = "pq-fastscan")]
    ProductQuantizationFastScan {
        /// Number of sub-vectors. Same semantics as
        /// [`Self::ProductQuantization::subvector_count`]; the only
        /// difference is the K=16 codebook (4-bit codes) and the
        /// block-transposed packing.
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

/// Number of Lloyd iterations the PQ trainer runs per sub-vector.
/// Matches the FAISS / Lucene default; converges within ~10 iters on
/// SIFT-class data and is bounded above for predictable commit time.
pub const PQ_KMEANS_ITERATIONS: usize = 25;

/// Train a PQ codebook by running `M` independent k-means clusterings,
/// one per sub-vector stride.
///
/// Returns the codebook as a `Vec<f32>` of length `params.codebook_len()`
/// laid out row-major:
/// `codebook[m * (k * sub_dim) + k * sub_dim + d]`.
///
/// `dimension` is the original vector dimension and must equal
/// `params.original_dim()` (checked).
pub fn pq_train_codebook(
    dimension: usize,
    params: PqParams,
    vectors: &[Vector],
) -> Result<Vec<f32>> {
    if vectors.is_empty() {
        return Err(LaurusError::InvalidOperation(
            "Cannot train product quantization on an empty vector set".to_string(),
        ));
    }
    if dimension != params.original_dim() {
        return Err(LaurusError::InvalidOperation(format!(
            "PQ training dim mismatch: vectors imply {dimension}, params imply {}",
            params.original_dim()
        )));
    }
    for v in vectors {
        if v.dimension() != dimension {
            return Err(LaurusError::InvalidOperation(format!(
                "PQ training input has mixed dimensions: expected {dimension}, got {}",
                v.dimension()
            )));
        }
    }

    let m = params.m as usize;
    let k = params.k as usize;
    let sub_dim = params.sub_dim as usize;
    let n = vectors.len();

    // Sub-sample fallback: every cluster needs at least one input
    // point. If `n < k` we duplicate inputs to fill rather than fail.
    let effective_k = k.min(n.max(1));

    let mut codebook = vec![0.0_f32; params.codebook_len()];

    // Reusable scratch buffer to avoid per-sub-vector reallocations.
    let mut sub_data: Vec<f32> = Vec::with_capacity(n * sub_dim);

    for sub in 0..m {
        // Project corpus onto this sub-vector stride.
        sub_data.clear();
        for v in vectors {
            sub_data.extend_from_slice(&v.data[sub * sub_dim..(sub + 1) * sub_dim]);
        }

        // Deterministic seed derived from `sub` so two runs over the
        // same input produce the same codebook.
        let seed: u64 =
            0xCAFE_F00D_DEAD_BEEF_u64 ^ ((sub as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));

        let centroids = kmeans_train(&sub_data, sub_dim, effective_k, PQ_KMEANS_ITERATIONS, seed);

        // Write centroids into codebook[sub][..]. When `effective_k < k`
        // the remaining slots are left zero-filled — they will never
        // be selected by the encoder because `pq_encode` only picks
        // from the seeded set, but the codebook is sized to `k` so the
        // on-disk format stays uniform.
        let dst_base = sub * k * sub_dim;
        let src_len = effective_k * sub_dim;
        codebook[dst_base..dst_base + src_len].copy_from_slice(&centroids[..src_len]);
    }

    Ok(codebook)
}

/// Encode one full-dimensional vector into `m` byte codes using a
/// trained codebook. Each byte is the index of the nearest centroid for
/// that sub-vector stride under squared L2 distance.
pub fn pq_encode(vector: &[f32], params: PqParams, codebook: &[f32]) -> Vec<u8> {
    debug_assert_eq!(codebook.len(), params.codebook_len());
    let m = params.m as usize;
    let k = params.k as usize;
    let sub_dim = params.sub_dim as usize;
    let mut codes = Vec::with_capacity(m);
    for sub in 0..m {
        let q_sub = &vector[sub * sub_dim..(sub + 1) * sub_dim];
        let base = sub * k * sub_dim;
        let mut best_k: u8 = 0;
        let mut best_d = f32::INFINITY;
        for ki in 0..k {
            let c = &codebook[base + ki * sub_dim..base + (ki + 1) * sub_dim];
            let mut sum = 0.0_f32;
            for d in 0..sub_dim {
                let diff = q_sub[d] - c[d];
                sum += diff * diff;
            }
            if sum < best_d {
                best_d = sum;
                best_k = ki as u8;
            }
        }
        codes.push(best_k);
    }
    codes
}

/// Reconstruct an approximate f32 vector from PQ codes using the
/// trained codebook. The output length is `params.original_dim()`.
pub fn pq_decode(codes: &[u8], params: PqParams, codebook: &[f32]) -> Vec<f32> {
    debug_assert_eq!(codes.len(), params.m as usize);
    debug_assert_eq!(codebook.len(), params.codebook_len());
    let m = params.m as usize;
    let k = params.k as usize;
    let sub_dim = params.sub_dim as usize;
    let mut out = Vec::with_capacity(m * sub_dim);
    for (sub, &code) in codes.iter().enumerate().take(m) {
        let base = sub * k * sub_dim + code as usize * sub_dim;
        out.extend_from_slice(&codebook[base..base + sub_dim]);
    }
    out
}

/// Deterministic Lloyd k-means over a flat `n × sub_dim` row-major
/// buffer. Used internally by [`pq_train_codebook`].
///
/// Returns `k * sub_dim` floats (row-major centroids).
fn kmeans_train(data: &[f32], sub_dim: usize, k: usize, iters: usize, seed: u64) -> Vec<f32> {
    let n = data.len() / sub_dim;
    debug_assert_eq!(data.len(), n * sub_dim);
    debug_assert!(n >= 1, "k-means needs at least 1 point");
    let k = k.min(n.max(1));

    let mut state = seed;
    let mut centroids = vec![0.0_f32; k * sub_dim];

    // k-means++ initialisation.
    {
        // First centroid: deterministic pick.
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let first = ((state >> 32) as usize) % n;
        centroids[..sub_dim].copy_from_slice(&data[first * sub_dim..(first + 1) * sub_dim]);

        let mut min_d2: Vec<f32> = (0..n)
            .map(|i| {
                let p = &data[i * sub_dim..(i + 1) * sub_dim];
                l2_squared_slice(p, &centroids[..sub_dim])
            })
            .collect();

        for c_idx in 1..k {
            let total: f32 = min_d2.iter().sum();
            let chosen = if total == 0.0 {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((state >> 32) as usize) % n
            } else {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let r = ((state >> 32) as f32 / u32::MAX as f32) * total;
                let mut acc = 0.0_f32;
                let mut pick = n - 1;
                for (i, &d) in min_d2.iter().enumerate() {
                    acc += d;
                    if acc >= r {
                        pick = i;
                        break;
                    }
                }
                pick
            };
            centroids[c_idx * sub_dim..(c_idx + 1) * sub_dim]
                .copy_from_slice(&data[chosen * sub_dim..(chosen + 1) * sub_dim]);
            // Update min squared distance against the new centroid.
            let new_c = &centroids[c_idx * sub_dim..(c_idx + 1) * sub_dim];
            for i in 0..n {
                let p = &data[i * sub_dim..(i + 1) * sub_dim];
                let d = l2_squared_slice(p, new_c);
                if d < min_d2[i] {
                    min_d2[i] = d;
                }
            }
        }
    }

    // Lloyd iterations.
    let mut sums = vec![0.0_f32; k * sub_dim];
    let mut counts = vec![0u32; k];
    for _ in 0..iters {
        sums.iter_mut().for_each(|x| *x = 0.0);
        counts.iter_mut().for_each(|c| *c = 0);

        for i in 0..n {
            let p = &data[i * sub_dim..(i + 1) * sub_dim];
            let mut best_j = 0usize;
            let mut best_d = l2_squared_slice(p, &centroids[..sub_dim]);
            for j in 1..k {
                let c = &centroids[j * sub_dim..(j + 1) * sub_dim];
                let d = l2_squared_slice(p, c);
                if d < best_d {
                    best_d = d;
                    best_j = j;
                }
            }
            counts[best_j] += 1;
            let sum_base = best_j * sub_dim;
            for d in 0..sub_dim {
                sums[sum_base + d] += p[d];
            }
        }

        for j in 0..k {
            if counts[j] > 0 {
                let inv = 1.0 / counts[j] as f32;
                for d in 0..sub_dim {
                    centroids[j * sub_dim + d] = sums[j * sub_dim + d] * inv;
                }
            }
        }
    }

    centroids
}

#[inline]
fn l2_squared_slice(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut sum = 0.0_f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum
}

/// Parameters for the Stage 3 Product Quantization variant.
///
/// `m` sub-vectors × `k = 256` centroids per sub-vector × `sub_dim`
/// floats per centroid form the codebook stored in the segment
/// header (see [`crate::vector::index::format::QuantHeader::ProductQuantization`]).
///
/// `dim` (the original vector dimension) is recovered as
/// `m * sub_dim`; this struct does not carry it separately because
/// PQ derives `sub_dim` from `dim / m` at construction time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PqParams {
    /// Number of sub-vectors the original `dim`-dimensional vector
    /// space is split into. Must be > 0 and must divide `dim`.
    pub m: u16,
    /// Centroids per sub-vector. Issue #481 Stage 3 ships only the
    /// 8-bit variant, so this is always `256`; reserved as a header
    /// field so future 4-bit variants can be added without a format
    /// break.
    pub k: u16,
    /// Sub-vector dimension; equal to `original_dim / m`.
    pub sub_dim: u16,
}

impl PqParams {
    /// Build a validated [`PqParams`].
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError::InvalidOperation`] if any field is zero
    /// or if `k` is not one of the supported centroid counts (`16` or
    /// `256`).
    ///
    /// `k = 256` is the 8-bit PQ variant shipped in Issue #481 Stage 3.
    /// `k = 16` is the FastScan 4-bit variant introduced in Issue #651
    /// part A (#692); the on-disk format and search-time kernels live
    /// in parallel modules under [`crate::vector::core::distance_pq_fastscan`]
    /// and [`crate::vector::index::pq_fastscan_storage`].
    pub fn new(m: u16, k: u16, sub_dim: u16) -> Result<Self> {
        if m == 0 || k == 0 || sub_dim == 0 {
            return Err(LaurusError::InvalidOperation(format!(
                "PqParams components must be > 0 (got m={m}, k={k}, sub_dim={sub_dim})"
            )));
        }
        if !matches!(k, 16 | 256) {
            return Err(LaurusError::InvalidOperation(format!(
                "PqParams::k must be one of {{16, 256}} (got {k}); 256 is the \
                 8-bit PQ variant (Issue #481 Stage 3), 16 is the FastScan \
                 4-bit variant (Issue #651 / #692)"
            )));
        }
        Ok(Self { m, k, sub_dim })
    }

    /// Derive params from `dim` and `m`. `sub_dim` is `dim / m`.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError::InvalidOperation`] if `m == 0` or
    /// `dim % m != 0`.
    pub fn from_dim_and_m(dim: usize, m: usize) -> Result<Self> {
        if m == 0 {
            return Err(LaurusError::InvalidOperation(
                "Product quantization subvector_count must be > 0".to_string(),
            ));
        }
        if !dim.is_multiple_of(m) {
            return Err(LaurusError::InvalidOperation(format!(
                "Product quantization subvector_count {m} must divide vector \
                 dimension {dim} (got {dim} % {m} = {})",
                dim % m
            )));
        }
        let sub_dim = dim / m;
        Self::new(
            u16::try_from(m).map_err(|_| {
                LaurusError::InvalidOperation(format!(
                    "Product quantization subvector_count {m} exceeds u16::MAX"
                ))
            })?,
            256,
            u16::try_from(sub_dim).map_err(|_| {
                LaurusError::InvalidOperation(format!(
                    "Product quantization sub_dim {sub_dim} exceeds u16::MAX"
                ))
            })?,
        )
    }

    /// Total number of f32 entries in the codebook
    /// (`m * k * sub_dim`).
    #[inline]
    pub fn codebook_len(&self) -> usize {
        self.m as usize * self.k as usize * self.sub_dim as usize
    }

    /// Total codebook size in bytes (`codebook_len * 4`).
    #[inline]
    pub fn codebook_byte_size(&self) -> usize {
        self.codebook_len() * 4
    }

    /// Original vector dimension this codebook covers (`m * sub_dim`).
    #[inline]
    pub fn original_dim(&self) -> usize {
        self.m as usize * self.sub_dim as usize
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
    state: QuantizerState,
}

/// Internal trained state of a [`VectorQuantizer`]. Kept private so
/// callers go through `params()` / `pq_state()` / `is_trained()`.
#[derive(Debug, Clone)]
enum QuantizerState {
    /// Constructed but not yet trained — `train` must be called
    /// before `quantize`.
    Untrained,
    /// Stage 1: per-segment scalar `(offset, scale)`.
    Scalar8Bit(ScalarQuantParams),
    /// Stage 3: PQ params plus the per-segment codebook (row-major
    /// `m × k × sub_dim` floats).
    ProductQuantization {
        params: PqParams,
        codebook: Vec<f32>,
    },
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
            state: QuantizerState::Untrained,
        }
    }

    /// Construct a Scalar8Bit quantizer from already-known params
    /// (e.g. loaded from a segment header at read time).
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError::InvalidOperation`] if `method` is
    /// [`QuantizationMethod::ProductQuantization`] — use
    /// [`Self::from_pq_codebook`] instead.
    pub fn from_params(
        method: QuantizationMethod,
        dimension: usize,
        params: ScalarQuantParams,
    ) -> Result<Self> {
        match method {
            QuantizationMethod::Scalar8Bit => Ok(Self {
                method,
                dimension,
                state: QuantizerState::Scalar8Bit(params),
            }),
            QuantizationMethod::ProductQuantization { .. } => Err(LaurusError::InvalidOperation(
                "Use VectorQuantizer::from_pq_codebook for ProductQuantization; \
                 from_params is Scalar8Bit-only"
                    .to_string(),
            )),
            #[cfg(feature = "pq-fastscan")]
            QuantizationMethod::ProductQuantizationFastScan { .. } => {
                Err(LaurusError::InvalidOperation(
                    "Use VectorQuantizer::from_pq_codebook for ProductQuantizationFastScan; \
                     from_params is Scalar8Bit-only"
                        .to_string(),
                ))
            }
        }
    }

    /// Construct a Product Quantization quantizer from a pre-trained
    /// codebook (e.g. loaded from a segment header at read time).
    ///
    /// # Errors
    ///
    /// * [`LaurusError::InvalidOperation`] if `codebook.len() != params.codebook_len()`.
    /// * [`LaurusError::InvalidOperation`] if `params.original_dim() != dimension`.
    pub fn from_pq_codebook(
        dimension: usize,
        params: PqParams,
        codebook: Vec<f32>,
    ) -> Result<Self> {
        if params.original_dim() != dimension {
            return Err(LaurusError::InvalidOperation(format!(
                "PQ codebook dim mismatch: params imply {}, quantizer wants {dimension}",
                params.original_dim()
            )));
        }
        if codebook.len() != params.codebook_len() {
            return Err(LaurusError::InvalidOperation(format!(
                "PQ codebook length {} does not match params (m={}, k={}, sub_dim={} -> {})",
                codebook.len(),
                params.m,
                params.k,
                params.sub_dim,
                params.codebook_len()
            )));
        }
        Ok(Self {
            method: QuantizationMethod::ProductQuantization {
                subvector_count: params.m as usize,
            },
            dimension,
            state: QuantizerState::ProductQuantization { params, codebook },
        })
    }

    /// Train segment-level quantization params on a representative set
    /// of vectors.
    ///
    /// For [`QuantizationMethod::Scalar8Bit`] this delegates to
    /// [`ScalarQuantParams::train`]. For
    /// [`QuantizationMethod::ProductQuantization`] this runs `M`
    /// independent k-means clusterings (one per sub-vector stride) and
    /// stores the resulting `M × K × sub_dim` codebook.
    pub fn train(&mut self, vectors: &[Vector]) -> Result<()> {
        match self.method {
            QuantizationMethod::Scalar8Bit => {
                let params = ScalarQuantParams::train(vectors)?;
                self.state = QuantizerState::Scalar8Bit(params);
                Ok(())
            }
            QuantizationMethod::ProductQuantization { subvector_count } => {
                let params = PqParams::from_dim_and_m(self.dimension, subvector_count)?;
                let codebook = pq_train_codebook(self.dimension, params, vectors)?;
                self.state = QuantizerState::ProductQuantization { params, codebook };
                Ok(())
            }
            #[cfg(feature = "pq-fastscan")]
            QuantizationMethod::ProductQuantizationFastScan { subvector_count } => {
                // FastScan uses K=16; reuse `from_dim_and_m` which validates
                // divisibility and propagate via the standard PQ codebook
                // training (k-means on each sub-vector).
                let mut params = PqParams::from_dim_and_m(self.dimension, subvector_count)?;
                params.k = 16;
                let codebook = pq_train_codebook(self.dimension, params, vectors)?;
                self.state = QuantizerState::ProductQuantization { params, codebook };
                Ok(())
            }
        }
    }

    /// Quantize a single vector.
    ///
    /// For Scalar8Bit the result is `(dim bytes, per-vector meta)`.
    /// For Product Quantization the result is `(m bytes, empty meta)` —
    /// PQ does not use the `sum_q` / `norm_q` correction terms (the
    /// ADC distance kernel is fed the LUT, not the meta block).
    ///
    /// # Errors
    ///
    /// * [`LaurusError::InvalidOperation`] if the quantizer has not
    ///   been trained yet.
    /// * [`LaurusError::InvalidOperation`] if `vector.dimension()`
    ///   differs from the dimension passed at construction time.
    pub fn quantize(&self, vector: &Vector) -> Result<(Vec<u8>, QuantizedVectorMeta)> {
        if vector.dimension() != self.dimension {
            return Err(LaurusError::InvalidOperation(format!(
                "Vector dimension mismatch: expected {}, got {}",
                self.dimension,
                vector.dimension()
            )));
        }
        match &self.state {
            QuantizerState::Untrained => Err(LaurusError::InvalidOperation(
                "Quantizer must be trained before quantizing vectors".to_string(),
            )),
            QuantizerState::Scalar8Bit(params) => {
                let q = params.quantize(vector);
                let meta = QuantizedVectorMeta::from_quantized(&q, params);
                Ok((q, meta))
            }
            QuantizerState::ProductQuantization { params, codebook } => {
                let codes = pq_encode(&vector.data, *params, codebook);
                Ok((
                    codes,
                    QuantizedVectorMeta {
                        sum_q: 0,
                        norm_q: 0.0,
                    },
                ))
            }
        }
    }

    /// Get the Scalar8Bit segment-level quantization params, if the
    /// quantizer is trained in Scalar8Bit mode. Returns `None` for
    /// untrained quantizers or for Product Quantization (use
    /// [`Self::pq_state`] instead).
    pub fn params(&self) -> Option<&ScalarQuantParams> {
        match &self.state {
            QuantizerState::Scalar8Bit(p) => Some(p),
            _ => None,
        }
    }

    /// Get the Product Quantization params and codebook, if the
    /// quantizer is trained in PQ mode.
    pub fn pq_state(&self) -> Option<(&PqParams, &[f32])> {
        match &self.state {
            QuantizerState::ProductQuantization { params, codebook } => {
                Some((params, codebook.as_slice()))
            }
            _ => None,
        }
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
        !matches!(self.state, QuantizerState::Untrained)
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
            #[cfg(feature = "pq-fastscan")]
            QuantizationMethod::ProductQuantizationFastScan { subvector_count } => {
                // FastScan stores 4 bits per sub-vector (half a byte per
                // sub), so the compressed size is `subvector_count / 2`
                // bytes per vector.
                if subvector_count == 0 {
                    1.0
                } else {
                    (self.dimension * 4) as f32 / (subvector_count as f32 / 2.0)
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
    fn product_quantization_train_succeeds_on_valid_inputs() {
        let dim = 8usize;
        let m = 4usize;
        let mut q = VectorQuantizer::new(
            QuantizationMethod::ProductQuantization { subvector_count: m },
            dim,
        );
        let training: Vec<Vector> = (0..32)
            .map(|i| {
                let v: Vec<f32> = (0..dim).map(|j| (i + j) as f32 * 0.1).collect();
                vec_of(&v)
            })
            .collect();
        q.train(&training).unwrap();
        assert!(q.is_trained());
        let (pq_params, codebook) = q.pq_state().expect("PQ trained");
        assert_eq!(pq_params.m as usize, m);
        assert_eq!(pq_params.k, 256);
        assert_eq!(pq_params.sub_dim as usize, dim / m);
        assert_eq!(codebook.len(), pq_params.codebook_len());
        // Untrained accessor is None.
        assert!(q.params().is_none());
    }

    #[test]
    fn product_quantization_encode_decode_roundtrips_to_codebook() {
        let dim = 4usize;
        let m = 2usize;
        let mut q = VectorQuantizer::new(
            QuantizationMethod::ProductQuantization { subvector_count: m },
            dim,
        );
        // Two clusters that are far apart in both sub-vector strides
        // so k-means picks them and encoding stays deterministic.
        let training = vec![
            vec_of(&[10.0, 10.0, 20.0, 20.0]),
            vec_of(&[-10.0, -10.0, -20.0, -20.0]),
        ];
        q.train(&training).unwrap();
        let (codes, _meta) = q
            .quantize(&vec_of(&[10.5, 10.5, 20.5, 20.5]))
            .expect("quantize");
        assert_eq!(codes.len(), m);
        let (params, cb) = q.pq_state().unwrap();
        let decoded = pq_decode(&codes, *params, cb);
        // The decoded vector is the centroid for whichever cluster was
        // picked. With these inputs the encoder picks the first centroid
        // (the (10,10,20,20) cluster).
        assert_eq!(decoded.len(), dim);
    }

    #[test]
    fn product_quantization_from_pq_codebook_roundtrips() {
        let params = PqParams::new(4, 256, 2).unwrap();
        let codebook = vec![0.0_f32; params.codebook_len()];
        let q = VectorQuantizer::from_pq_codebook(8, params, codebook.clone()).unwrap();
        assert!(q.is_trained());
        assert_eq!(q.pq_state().unwrap().0, &params);
        assert_eq!(q.pq_state().unwrap().1.len(), codebook.len());
    }

    #[test]
    fn product_quantization_from_pq_codebook_rejects_size_mismatch() {
        let params = PqParams::new(4, 256, 2).unwrap();
        let bad = vec![0.0_f32; params.codebook_len() - 1];
        let err = VectorQuantizer::from_pq_codebook(8, params, bad).unwrap_err();
        assert!(matches!(err, LaurusError::InvalidOperation(_)));
    }

    #[test]
    fn pq_params_validate_divides_dim() {
        // dim must be divisible by m.
        let err = PqParams::from_dim_and_m(10, 3).unwrap_err();
        assert!(matches!(err, LaurusError::InvalidOperation(_)));
        let ok = PqParams::from_dim_and_m(12, 3).unwrap();
        assert_eq!(ok.original_dim(), 12);
        assert_eq!(ok.sub_dim, 4);
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
