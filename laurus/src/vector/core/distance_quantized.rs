//! int8 SIMD distance kernel for per-segment global affine SQ.
//!
//! Companion to [`crate::vector::core::distance`] (the f32 path) and
//! [`crate::vector::core::quantization`] (the SQ encoder). After
//! Issue #481 Stage 1 the HNSW / Flat / IVF search hot path uses
//! the functions in this module instead of the f32 kernels.
//!
//! # Algorithm
//!
//! With per-segment global affine SQ, every f32 element is encoded
//! as `q = clamp(round((v - offset) / scale), 0, 255)`, sharing one
//! `(offset, scale)` pair across the whole segment. The dot product
//! reconstructs from the quantized form via the affine identity:
//!
//! ```text
//! a · b ≈ N·offset² + scale·offset·(Σaq + Σbq) + scale²·Σ(aq · bq)
//! ```
//!
//! Of those four terms only `Σ(aq · bq)` is per-pair: it is computed
//! in i32 SIMD using [`dot_u8_to_i32`]. The other three collapse into
//! constants once the candidate's `sum_q` is loaded — that constant
//! is what makes the int8 kernel fast vs the f32 kernel.
//!
//! For Euclidean and Manhattan, the `offset` cancels in `(a - b)`, so
//! distance computation reduces to a pure int8 difference accumulator
//! plus a single `scale²` (Euclidean) or `scale` (Manhattan) post-multiply.
//!
//! # SIMD strategy
//!
//! All accumulators use [`wide::i32x8`] (8-wide i32 lanes, portable
//! across x86_64 / aarch64 / wasm32 via the `wide` crate). Per-element
//! `u8 → i32` zero-extension happens at load time. `i32` is wide
//! enough to hold any per-pair product (`u8 * u8 ≤ 65025`) and the
//! per-chunk accumulation across realistic dimensions
//! (`dim ≤ 4096` keeps the running sum well below `i32::MAX`).

use wide::i32x8;

use crate::vector::core::distance::DistanceMetric;
use crate::vector::core::quantization::{QuantizedVectorMeta, ScalarQuantParams};

/// Per-query state cached once per HNSW search invocation.
///
/// Built by [`Self::prepare`] from the f32 query and the segment-level
/// quantization params. Subsequent [`distance_quantized`] calls read
/// only from this struct (no further f32 access required for
/// Euclidean / Manhattan / DotProduct; Cosine / Angular still need
/// `norm_query` precomputed here).
#[derive(Debug, Clone)]
pub struct QuantizedQuery {
    /// Query elements quantized with the segment's `(offset, scale)`.
    /// Length equals the vector dimension.
    pub q_data: Vec<u8>,
    /// `Σ q_data[i]`, used in the affine dot-product reconstruction.
    pub sum_q: u32,
    /// `||query||_f32 = sqrt(Σ query[i]²)` computed from the original
    /// f32 query before quantization. Used as the cosine denominator
    /// (paired with `cand_meta.norm_q`, which is the f32 norm of the
    /// dequantized candidate).
    pub norm_query: f32,
    /// Cached `params.scale` so the distance kernel does not need to
    /// dereference [`ScalarQuantParams`] for each candidate.
    pub scale: f32,
    /// Cached `params.offset`.
    pub offset: f32,
    /// Vector dimension. Equal to `q_data.len()`.
    pub dim: usize,
}

impl QuantizedQuery {
    /// Quantize the f32 query under the segment's params and precompute
    /// the constants used by [`distance_quantized`].
    ///
    /// # Arguments
    ///
    /// * `query` - f32 query vector. Saturated outside the segment's
    ///   trained range (see [`ScalarQuantParams::quantize_value`]).
    /// * `params` - Segment-level quantization params recovered from
    ///   the segment header at search time.
    pub fn prepare(query: &[f32], params: &ScalarQuantParams) -> Self {
        let q_data = params.quantize_slice(query);
        let sum_q: u32 = q_data.iter().map(|&x| x as u32).sum();
        let norm_query = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        Self {
            q_data,
            sum_q,
            norm_query,
            scale: params.scale,
            offset: params.offset,
            dim: query.len(),
        }
    }
}

/// Compute distance from a [`QuantizedQuery`] to one quantized candidate.
///
/// `cand` and `cand_meta` together describe one candidate vector as
/// stored in the segment: `cand` is the int8 payload, `cand_meta`
/// carries the per-vector `sum_q` (cosine cross-term) and `norm_q`
/// (cosine denominator).
///
/// # Returns
///
/// Distance in the metric's natural scale, matching the convention of
/// [`DistanceMetric::distance`] (smaller = more similar; DotProduct
/// returns the negated dot product so higher similarity → smaller
/// distance).
///
/// # Panics
///
/// In debug mode, panics if `cand.len() != query.dim`. The caller (HNSW
/// searcher) guarantees this invariant in release mode.
pub fn distance_quantized(
    metric: DistanceMetric,
    query: &QuantizedQuery,
    cand: &[u8],
    cand_meta: QuantizedVectorMeta,
) -> f32 {
    debug_assert_eq!(cand.len(), query.dim);
    match metric {
        DistanceMetric::Cosine => {
            let approx_dot = approx_dot_product(query, cand, cand_meta.sum_q);
            let denom = cand_meta.norm_q * query.norm_query;
            if denom == 0.0 {
                1.0
            } else {
                let cosine = (approx_dot / denom).clamp(-1.0, 1.0);
                1.0 - cosine
            }
        }
        DistanceMetric::Angular => {
            let approx_dot = approx_dot_product(query, cand, cand_meta.sum_q);
            let denom = cand_meta.norm_q * query.norm_query;
            if denom == 0.0 {
                std::f32::consts::PI
            } else {
                let cosine = (approx_dot / denom).clamp(-1.0, 1.0);
                cosine.acos()
            }
        }
        DistanceMetric::Euclidean => {
            // (a - b)² with affine offset cancellation: the offset term
            // disappears, leaving scale² as the per-pair multiplier.
            let sq_diff = sq_diff_u8_to_i32(&query.q_data, cand);
            (query.scale * query.scale * sq_diff as f32).sqrt()
        }
        DistanceMetric::Manhattan => {
            // |a - b| with the same offset cancellation; the per-pair
            // multiplier is just scale (no square).
            let abs_diff = abs_diff_u8_to_i32(&query.q_data, cand);
            query.scale * abs_diff as f32
        }
        DistanceMetric::DotProduct => -approx_dot_product(query, cand, cand_meta.sum_q),
    }
}

/// Reconstruct an approximate f32 dot product from quantized data.
///
/// Implements `a·b ≈ N·offset² + scale·offset·(sum_q_a + sum_q_b)
/// + scale²·Σ(aq · bq)`. Only the last term is per-pair; the other
/// three collapse into a constant once `cand_sum_q` is loaded.
#[inline]
fn approx_dot_product(query: &QuantizedQuery, cand: &[u8], cand_sum_q: u32) -> f32 {
    let dot_q = dot_u8_to_i32(&query.q_data, cand);
    let n = query.dim as f32;
    let off = query.offset;
    let scale = query.scale;
    n * off * off
        + scale * off * (query.sum_q as f32 + cand_sum_q as f32)
        + scale * scale * dot_q as f32
}

/// SIMD `Σ a[i] * b[i]` over u8 inputs, accumulating in i32.
///
/// Uses [`wide::i32x8`] for an 8-wide accumulator. Per-element
/// `u8 → i32` zero-extension is folded into the SIMD load. The tail
/// (≤ 7 elements) falls back to scalar.
///
/// # Overflow
///
/// `u8 * u8 ≤ 65025` fits in i32. Accumulating up to `i32::MAX / 65025
/// ≈ 33,000` per-pair products fits without overflow, comfortably
/// covering all realistic vector dimensions.
#[inline]
pub fn dot_u8_to_i32(a: &[u8], b: &[u8]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = i32x8::ZERO;
    let chunks_a = a.chunks_exact(8);
    let chunks_b = b.chunks_exact(8);
    let rem_a = chunks_a.remainder();
    let rem_b = chunks_b.remainder();
    for (ca, cb) in chunks_a.zip(chunks_b) {
        let va = i32x8::from([
            ca[0] as i32,
            ca[1] as i32,
            ca[2] as i32,
            ca[3] as i32,
            ca[4] as i32,
            ca[5] as i32,
            ca[6] as i32,
            ca[7] as i32,
        ]);
        let vb = i32x8::from([
            cb[0] as i32,
            cb[1] as i32,
            cb[2] as i32,
            cb[3] as i32,
            cb[4] as i32,
            cb[5] as i32,
            cb[6] as i32,
            cb[7] as i32,
        ]);
        acc += va * vb;
    }
    let mut total: i32 = acc.reduce_add();
    for (x, y) in rem_a.iter().zip(rem_b.iter()) {
        total += (*x as i32) * (*y as i32);
    }
    total
}

/// SIMD `Σ (a[i] - b[i])²` over u8 inputs, accumulating in i32.
///
/// The subtraction widens to i32 first so the result remains correct
/// even when `a[i] < b[i]` (would underflow in u8). Per-pair
/// squared diff is at most `255² = 65025`, identical to the dot-product
/// kernel's per-pair bound, so the same overflow analysis applies.
#[inline]
pub fn sq_diff_u8_to_i32(a: &[u8], b: &[u8]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = i32x8::ZERO;
    let chunks_a = a.chunks_exact(8);
    let chunks_b = b.chunks_exact(8);
    let rem_a = chunks_a.remainder();
    let rem_b = chunks_b.remainder();
    for (ca, cb) in chunks_a.zip(chunks_b) {
        let va = i32x8::from([
            ca[0] as i32,
            ca[1] as i32,
            ca[2] as i32,
            ca[3] as i32,
            ca[4] as i32,
            ca[5] as i32,
            ca[6] as i32,
            ca[7] as i32,
        ]);
        let vb = i32x8::from([
            cb[0] as i32,
            cb[1] as i32,
            cb[2] as i32,
            cb[3] as i32,
            cb[4] as i32,
            cb[5] as i32,
            cb[6] as i32,
            cb[7] as i32,
        ]);
        let diff = va - vb;
        acc += diff * diff;
    }
    let mut total: i32 = acc.reduce_add();
    for (x, y) in rem_a.iter().zip(rem_b.iter()) {
        let d = (*x as i32) - (*y as i32);
        total += d * d;
    }
    total
}

/// SIMD `Σ |a[i] - b[i]|` over u8 inputs, accumulating in i32.
///
/// Uses widened `i32` subtraction + [`i32x8::abs`] for the absolute
/// value. Per-pair magnitude is at most `255`, so accumulation
/// safety is the same as the other kernels.
#[inline]
pub fn abs_diff_u8_to_i32(a: &[u8], b: &[u8]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = i32x8::ZERO;
    let chunks_a = a.chunks_exact(8);
    let chunks_b = b.chunks_exact(8);
    let rem_a = chunks_a.remainder();
    let rem_b = chunks_b.remainder();
    for (ca, cb) in chunks_a.zip(chunks_b) {
        let va = i32x8::from([
            ca[0] as i32,
            ca[1] as i32,
            ca[2] as i32,
            ca[3] as i32,
            ca[4] as i32,
            ca[5] as i32,
            ca[6] as i32,
            ca[7] as i32,
        ]);
        let vb = i32x8::from([
            cb[0] as i32,
            cb[1] as i32,
            cb[2] as i32,
            cb[3] as i32,
            cb[4] as i32,
            cb[5] as i32,
            cb[6] as i32,
            cb[7] as i32,
        ]);
        let diff = va - vb;
        acc += diff.abs();
    }
    let mut total: i32 = acc.reduce_add();
    for (x, y) in rem_a.iter().zip(rem_b.iter()) {
        total += ((*x as i32) - (*y as i32)).abs();
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::core::vector::Vector;

    /// Reference scalar implementations used to validate the SIMD
    /// kernels regardless of the underlying instruction set.
    fn scalar_dot(a: &[u8], b: &[u8]) -> i32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x as i32) * (y as i32))
            .sum()
    }
    fn scalar_sq_diff(a: &[u8], b: &[u8]) -> i32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let d = (x as i32) - (y as i32);
                d * d
            })
            .sum()
    }
    fn scalar_abs_diff(a: &[u8], b: &[u8]) -> i32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| ((x as i32) - (y as i32)).abs())
            .sum()
    }

    fn pseudo_random_u8(seed: u32, len: usize) -> Vec<u8> {
        // Tiny LCG, deterministic across runs. Good enough for the
        // SIMD vs scalar consistency check.
        let mut state = seed.wrapping_mul(0x9E37_79B9).wrapping_add(0xDEAD_BEEF);
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1103515245).wrapping_add(12345);
                (state >> 16) as u8
            })
            .collect()
    }

    fn pseudo_random_f32(seed: u32, len: usize, lo: f32, hi: f32) -> Vec<f32> {
        let bytes = pseudo_random_u8(seed, len);
        let range = hi - lo;
        bytes
            .into_iter()
            .map(|b| lo + (b as f32 / 255.0) * range)
            .collect()
    }

    #[test]
    fn dot_simd_matches_scalar_for_various_lengths() {
        for &dim in &[1, 7, 8, 9, 16, 17, 64, 65, 128, 384, 768] {
            let a = pseudo_random_u8(1, dim);
            let b = pseudo_random_u8(2, dim);
            assert_eq!(
                dot_u8_to_i32(&a, &b),
                scalar_dot(&a, &b),
                "dim = {dim}: SIMD dot disagrees with scalar"
            );
        }
    }

    #[test]
    fn sq_diff_simd_matches_scalar_for_various_lengths() {
        for &dim in &[1, 7, 8, 9, 16, 17, 64, 65, 128, 384, 768] {
            let a = pseudo_random_u8(3, dim);
            let b = pseudo_random_u8(4, dim);
            assert_eq!(
                sq_diff_u8_to_i32(&a, &b),
                scalar_sq_diff(&a, &b),
                "dim = {dim}: SIMD sq_diff disagrees with scalar"
            );
        }
    }

    #[test]
    fn abs_diff_simd_matches_scalar_for_various_lengths() {
        for &dim in &[1, 7, 8, 9, 16, 17, 64, 65, 128, 384, 768] {
            let a = pseudo_random_u8(5, dim);
            let b = pseudo_random_u8(6, dim);
            assert_eq!(
                abs_diff_u8_to_i32(&a, &b),
                scalar_abs_diff(&a, &b),
                "dim = {dim}: SIMD abs_diff disagrees with scalar"
            );
        }
    }

    #[test]
    fn dot_extremes_do_not_overflow_at_realistic_dim() {
        // Worst case: every element 255. dim = 4096 keeps Σ 255*255
        // ≈ 266M, well below i32::MAX (~2.1B).
        let dim = 4096;
        let a = vec![255u8; dim];
        let b = vec![255u8; dim];
        let expected = 255i32 * 255 * dim as i32;
        assert_eq!(dot_u8_to_i32(&a, &b), expected);
    }

    /// End-to-end correctness: quantized cosine distance must agree
    /// with the f32 cosine distance to within a small per-pair error
    /// bound. The bound scales with `params.scale`, so we use vectors
    /// in a fixed `[-1, 1]` range and assert `|delta| < 0.02` on the
    /// returned 1 - cos(θ) scalar.
    #[test]
    fn cosine_quantized_matches_f32_within_tolerance() {
        let dim = 128;
        let a_f32 = pseudo_random_f32(11, dim, -1.0, 1.0);
        let b_f32 = pseudo_random_f32(12, dim, -1.0, 1.0);
        let training = vec![Vector::new(a_f32.clone()), Vector::new(b_f32.clone())];
        let params = ScalarQuantParams::train(&training).unwrap();

        let q_a = params.quantize_slice(&a_f32);
        let meta_a = QuantizedVectorMeta::from_quantized(&q_a, &params);

        let prepared = QuantizedQuery::prepare(&b_f32, &params);
        let approx_cosine_dist =
            distance_quantized(DistanceMetric::Cosine, &prepared, &q_a, meta_a);

        let exact_cosine_dist = DistanceMetric::Cosine.distance(&a_f32, &b_f32).unwrap();
        let delta = (approx_cosine_dist - exact_cosine_dist).abs();
        assert!(
            delta < 0.02,
            "cosine: quantized = {approx_cosine_dist}, f32 = {exact_cosine_dist}, |Δ| = {delta}"
        );
    }

    #[test]
    fn euclidean_quantized_matches_f32_within_tolerance() {
        let dim = 128;
        let a_f32 = pseudo_random_f32(21, dim, -1.0, 1.0);
        let b_f32 = pseudo_random_f32(22, dim, -1.0, 1.0);
        let params =
            ScalarQuantParams::train(&[Vector::new(a_f32.clone()), Vector::new(b_f32.clone())])
                .unwrap();

        let q_a = params.quantize_slice(&a_f32);
        let meta_a = QuantizedVectorMeta::from_quantized(&q_a, &params);
        let prepared = QuantizedQuery::prepare(&b_f32, &params);

        let approx_dist = distance_quantized(DistanceMetric::Euclidean, &prepared, &q_a, meta_a);
        let exact_dist = DistanceMetric::Euclidean.distance(&a_f32, &b_f32).unwrap();
        let rel_err = (approx_dist - exact_dist).abs() / exact_dist.max(1e-6);
        assert!(
            rel_err < 0.05,
            "euclidean: quantized = {approx_dist}, f32 = {exact_dist}, rel_err = {rel_err}"
        );
    }

    #[test]
    fn manhattan_quantized_matches_f32_within_tolerance() {
        let dim = 128;
        let a_f32 = pseudo_random_f32(31, dim, -1.0, 1.0);
        let b_f32 = pseudo_random_f32(32, dim, -1.0, 1.0);
        let params =
            ScalarQuantParams::train(&[Vector::new(a_f32.clone()), Vector::new(b_f32.clone())])
                .unwrap();

        let q_a = params.quantize_slice(&a_f32);
        let meta_a = QuantizedVectorMeta::from_quantized(&q_a, &params);
        let prepared = QuantizedQuery::prepare(&b_f32, &params);

        let approx_dist = distance_quantized(DistanceMetric::Manhattan, &prepared, &q_a, meta_a);
        let exact_dist = DistanceMetric::Manhattan.distance(&a_f32, &b_f32).unwrap();
        let rel_err = (approx_dist - exact_dist).abs() / exact_dist.max(1e-6);
        assert!(
            rel_err < 0.02,
            "manhattan: quantized = {approx_dist}, f32 = {exact_dist}, rel_err = {rel_err}"
        );
    }

    #[test]
    fn dot_product_quantized_matches_f32_within_tolerance() {
        let dim = 128;
        let a_f32 = pseudo_random_f32(41, dim, -1.0, 1.0);
        let b_f32 = pseudo_random_f32(42, dim, -1.0, 1.0);
        let params =
            ScalarQuantParams::train(&[Vector::new(a_f32.clone()), Vector::new(b_f32.clone())])
                .unwrap();

        let q_a = params.quantize_slice(&a_f32);
        let meta_a = QuantizedVectorMeta::from_quantized(&q_a, &params);
        let prepared = QuantizedQuery::prepare(&b_f32, &params);

        let approx_dist = distance_quantized(DistanceMetric::DotProduct, &prepared, &q_a, meta_a);
        let exact_dist = DistanceMetric::DotProduct.distance(&a_f32, &b_f32).unwrap();
        let abs_err = (approx_dist - exact_dist).abs();
        assert!(
            abs_err < 0.5,
            "dot_product: quantized = {approx_dist}, f32 = {exact_dist}, |Δ| = {abs_err}"
        );
    }

    #[test]
    fn angular_quantized_matches_f32_within_tolerance() {
        let dim = 128;
        let a_f32 = pseudo_random_f32(51, dim, -1.0, 1.0);
        let b_f32 = pseudo_random_f32(52, dim, -1.0, 1.0);
        let params =
            ScalarQuantParams::train(&[Vector::new(a_f32.clone()), Vector::new(b_f32.clone())])
                .unwrap();

        let q_a = params.quantize_slice(&a_f32);
        let meta_a = QuantizedVectorMeta::from_quantized(&q_a, &params);
        let prepared = QuantizedQuery::prepare(&b_f32, &params);

        let approx_dist = distance_quantized(DistanceMetric::Angular, &prepared, &q_a, meta_a);
        let exact_dist = DistanceMetric::Angular.distance(&a_f32, &b_f32).unwrap();
        let abs_err = (approx_dist - exact_dist).abs();
        assert!(
            abs_err < 0.05,
            "angular: quantized = {approx_dist}, f32 = {exact_dist}, |Δ| = {abs_err}"
        );
    }

    #[test]
    fn cosine_handles_zero_norm_query() {
        let dim = 16;
        let a_f32 = pseudo_random_f32(61, dim, -1.0, 1.0);
        let zero_query: Vec<f32> = vec![0.0; dim];
        let params = ScalarQuantParams::train(&[Vector::new(a_f32.clone())]).unwrap();
        let q_a = params.quantize_slice(&a_f32);
        let meta_a = QuantizedVectorMeta::from_quantized(&q_a, &params);
        let prepared = QuantizedQuery::prepare(&zero_query, &params);

        let dist = distance_quantized(DistanceMetric::Cosine, &prepared, &q_a, meta_a);
        assert_eq!(dist, 1.0, "zero query should yield max cosine distance");
    }

    #[test]
    fn prepared_query_caches_segment_params() {
        let dim = 8;
        let query = vec![0.5_f32; dim];
        let params = ScalarQuantParams {
            offset: -1.0,
            scale: 2.0 / 255.0,
        };
        let prepared = QuantizedQuery::prepare(&query, &params);
        assert_eq!(prepared.dim, dim);
        assert_eq!(prepared.offset, params.offset);
        assert_eq!(prepared.scale, params.scale);
        assert_eq!(prepared.q_data.len(), dim);
        let expected_sum: u32 = prepared.q_data.iter().map(|&x| x as u32).sum();
        assert_eq!(prepared.sum_q, expected_sum);
    }
}
