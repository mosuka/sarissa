//! SIMD-friendly helpers that power hot paths inside Laurus.

/// SIMD-accelerated ASCII operations.
pub mod ascii {

    /// Convert ASCII characters to lowercase using optimized byte operations.
    ///
    /// This function processes bytes in chunks for better performance
    /// while maintaining the benefits of SIMD-style thinking.
    pub fn to_lowercase_optimized(input: &str) -> String {
        let bytes = input.as_bytes();
        let mut result = Vec::with_capacity(bytes.len());

        // Process 8 bytes at a time for better cache efficiency
        let (chunks, remainder) = bytes.as_chunks::<8>();

        for chunk in chunks {
            let mut processed = [0u8; 8];
            for (i, &byte) in chunk.iter().enumerate() {
                // ASCII uppercase A-Z to lowercase conversion
                if byte.is_ascii_uppercase() {
                    processed[i] = byte + 32; // Convert to lowercase
                } else {
                    processed[i] = byte;
                }
            }
            result.extend_from_slice(&processed);
        }

        // Handle remaining bytes
        for &byte in remainder {
            if byte.is_ascii_uppercase() {
                result.push(byte + 32);
            } else {
                result.push(byte);
            }
        }

        // Convert back to string (we know it's valid UTF-8 since input was ASCII)
        unsafe { String::from_utf8_unchecked(result) }
    }

    /// Fallback implementation for non-ASCII or when optimization is not beneficial.
    pub fn to_lowercase_fallback(input: &str) -> String {
        input.to_lowercase()
    }

    /// Main entry point for optimized lowercase conversion.
    ///
    /// This function automatically chooses between optimized and fallback
    /// implementations based on the input characteristics.
    pub fn to_lowercase(input: &str) -> String {
        // Check if input is ASCII for optimization
        if input.is_ascii() && input.len() >= 16 {
            to_lowercase_optimized(input)
        } else {
            to_lowercase_fallback(input)
        }
    }

    /// Find the first whitespace character position using optimized search.
    pub fn find_whitespace_optimized(input: &[u8]) -> Option<usize> {
        if input.len() < 8 {
            return input.iter().position(|&b| b.is_ascii_whitespace());
        }

        let (chunks, remainder) = input.as_chunks::<8>();
        let mut chunk_idx = 0;

        for chunk in chunks {
            for (byte_idx, &byte) in chunk.iter().enumerate() {
                if byte == b' ' || byte == b'\t' || byte == b'\n' || byte == b'\r' {
                    return Some(chunk_idx * 8 + byte_idx);
                }
            }
            chunk_idx += 1;
        }

        // Check remainder
        let base_offset = chunk_idx * 8;
        remainder
            .iter()
            .position(|&b| b.is_ascii_whitespace())
            .map(|pos| base_offset + pos)
    }

    /// Optimized whitespace finding (public API).
    pub fn find_whitespace_simd(input: &[u8]) -> Option<usize> {
        find_whitespace_optimized(input)
    }
}

/// SIMD-accelerated numerical operations for scoring.
pub mod numeric {
    use wide::f32x8;

    /// Batch BM25 TF calculation using SIMD for improved throughput.
    ///
    /// Computes TF = tf * (k1 + 1) / (tf + k1 * norm_factor) for each element.
    ///
    /// # Arguments
    /// * `term_freqs` - Slice of term frequencies
    /// * `k1` - BM25 k1 parameter
    /// * `norm_factors` - Slice of normalization factors (same length as term_freqs)
    ///
    /// # Returns
    /// Vector of BM25 TF scores.
    pub fn batch_bm25_tf(term_freqs: &[f32], k1: f32, norm_factors: &[f32]) -> Vec<f32> {
        assert_eq!(term_freqs.len(), norm_factors.len());
        let len = term_freqs.len();
        let mut results = Vec::with_capacity(len);

        let k1_plus_1 = f32x8::splat(k1 + 1.0);
        let k1_vec = f32x8::splat(k1);

        let (tf_chunks, tf_remainder) = term_freqs.as_chunks::<8>();
        let (norm_chunks, _) = norm_factors.as_chunks::<8>();

        for (tf_chunk, norm_chunk) in tf_chunks.iter().zip(norm_chunks) {
            let tf = f32x8::from(*tf_chunk);
            let norm = f32x8::from(*norm_chunk);
            // BM25 TF: tf * (k1 + 1) / (tf + k1 * norm)
            let numerator = tf * k1_plus_1;
            let denominator = tf + k1_vec * norm;
            let result = numerator / denominator;
            results.extend_from_slice(&result.to_array());
        }

        // Handle remainder with scalar fallback.
        let norm_remainder_start = len - tf_remainder.len();
        let norm_remainder = &norm_factors[norm_remainder_start..];
        for (tf, norm) in tf_remainder.iter().zip(norm_remainder.iter()) {
            results.push((tf * (k1 + 1.0)) / (tf + k1 * norm));
        }

        results
    }

    /// Batch BM25 final score calculation using SIMD.
    ///
    /// Computes final_score = idf * tf * boost for each element.
    ///
    /// # Arguments
    /// * `tf_scores` - Slice of TF scores
    /// * `idf_scores` - Slice of IDF scores
    /// * `boosts` - Slice of boost factors
    ///
    /// # Returns
    /// Vector of final BM25 scores.
    pub fn batch_bm25_final_score(
        tf_scores: &[f32],
        idf_scores: &[f32],
        boosts: &[f32],
    ) -> Vec<f32> {
        assert_eq!(tf_scores.len(), idf_scores.len());
        assert_eq!(tf_scores.len(), boosts.len());
        let len = tf_scores.len();
        let mut results = Vec::with_capacity(len);

        let (tf_chunks, tf_remainder) = tf_scores.as_chunks::<8>();
        let (idf_chunks, _) = idf_scores.as_chunks::<8>();
        let (boost_chunks, _) = boosts.as_chunks::<8>();

        for ((tf_chunk, idf_chunk), boost_chunk) in
            tf_chunks.iter().zip(idf_chunks).zip(boost_chunks)
        {
            let tf = f32x8::from(*tf_chunk);
            let idf = f32x8::from(*idf_chunk);
            let boost = f32x8::from(*boost_chunk);
            let result = idf * tf * boost;
            results.extend_from_slice(&result.to_array());
        }

        // Handle remainder with scalar fallback.
        let remainder_start = len - tf_remainder.len();
        let idf_remainder = &idf_scores[remainder_start..];
        let boost_remainder = &boosts[remainder_start..];
        for ((tf, idf), boost) in tf_remainder
            .iter()
            .zip(idf_remainder.iter())
            .zip(boost_remainder.iter())
        {
            results.push(idf * tf * boost);
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimized_lowercase_ascii() {
        let input = "HELLO WORLD THIS IS A TEST STRING FOR OPTIMIZATION";
        let expected = "hello world this is a test string for optimization";
        let result = ascii::to_lowercase(input);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_optimized_lowercase_mixed() {
        let input = "Hello World 123 ABC def";
        let expected = "hello world 123 abc def";
        let result = ascii::to_lowercase(input);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_optimized_lowercase_empty() {
        let result = ascii::to_lowercase("");
        assert_eq!(result, "");
    }

    #[test]
    fn test_optimized_lowercase_short() {
        let input = "ABC";
        let expected = "abc";
        let result = ascii::to_lowercase(input);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_fallback_for_unicode() {
        let input = "Héllo Wörld"; // Non-ASCII characters
        let result = ascii::to_lowercase(input);
        let expected = input.to_lowercase();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_simd_batch_bm25_tf() {
        // 10 elements: 8 handled by SIMD + 2 remainder
        let tfs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let norms = vec![1.0; 10];
        let k1 = 1.2;
        let result = numeric::batch_bm25_tf(&tfs, k1, &norms);
        assert_eq!(result.len(), 10);
        for (i, &r) in result.iter().enumerate() {
            let tf = tfs[i];
            let expected = (tf * (k1 + 1.0)) / (tf + k1 * 1.0);
            assert!((r - expected).abs() < 1e-5, "mismatch at index {i}");
        }
    }

    #[test]
    fn test_simd_batch_bm25_tf_exact_multiple() {
        // Exactly 8 elements (no remainder)
        let tfs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let norms = vec![0.5; 8];
        let k1 = 1.5;
        let result = numeric::batch_bm25_tf(&tfs, k1, &norms);
        assert_eq!(result.len(), 8);
        for (i, &r) in result.iter().enumerate() {
            let tf = tfs[i];
            let expected = (tf * (k1 + 1.0)) / (tf + k1 * 0.5);
            assert!((r - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn test_simd_batch_bm25_final_score() {
        let tfs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let idfs = vec![0.5, 1.0, 1.5, 2.0, 0.5, 1.0, 1.5, 2.0, 0.5];
        let boosts = vec![1.0; 9];
        let result = numeric::batch_bm25_final_score(&tfs, &idfs, &boosts);
        assert_eq!(result.len(), 9);
        for (i, &r) in result.iter().enumerate() {
            let expected = idfs[i] * tfs[i] * boosts[i];
            assert!((r - expected).abs() < 1e-5, "mismatch at index {i}");
        }
    }

    #[test]
    fn test_simd_batch_bm25_tf_empty() {
        let result = numeric::batch_bm25_tf(&[], 1.2, &[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_simd_batch_bm25_final_score_empty() {
        let result = numeric::batch_bm25_final_score(&[], &[], &[]);
        assert!(result.is_empty());
    }
}
