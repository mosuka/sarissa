//! SIMD optimization utilities for Sage.

pub mod simd_wide;

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
        let chunks = bytes.chunks_exact(8);
        let remainder = chunks.remainder();

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

    /// Optimized whitespace detection for ASCII text.
    ///
    /// Returns a boolean indicating whether any whitespace was found.
    pub fn contains_whitespace_optimized(input: &[u8]) -> bool {
        if input.len() < 8 {
            return input.iter().any(|&b| b.is_ascii_whitespace());
        }

        // Process 8 bytes at a time
        let chunks = input.chunks_exact(8);
        let remainder = chunks.remainder();

        for chunk in chunks {
            for &byte in chunk {
                if byte == b' ' || byte == b'\t' || byte == b'\n' || byte == b'\r' {
                    return true;
                }
            }
        }

        // Check remainder
        remainder.iter().any(|&b| b.is_ascii_whitespace())
    }

    /// Find the first whitespace character position using optimized search.
    pub fn find_whitespace_optimized(input: &[u8]) -> Option<usize> {
        if input.len() < 8 {
            return input.iter().position(|&b| b.is_ascii_whitespace());
        }

        let mut chunks = input.chunks_exact(8);
        let remainder = chunks.remainder();
        let mut chunk_idx = 0;

        for chunk in &mut chunks {
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

    /// Optimized whitespace detection (public API).
    pub fn contains_whitespace_simd(input: &[u8]) -> bool {
        contains_whitespace_optimized(input)
    }

    /// Optimized whitespace finding (public API).
    pub fn find_whitespace_simd(input: &[u8]) -> Option<usize> {
        find_whitespace_optimized(input)
    }
}

/// SIMD-accelerated numerical operations for scoring.
pub mod numeric {

    /// Batch BM25 score calculation for multiple documents.
    ///
    /// This function processes multiple TF values simultaneously
    /// for better performance in scoring operations.
    pub fn batch_bm25_tf(term_freqs: &[f32], k1: f32, norm_factors: &[f32]) -> Vec<f32> {
        assert_eq!(term_freqs.len(), norm_factors.len());

        let mut results = Vec::with_capacity(term_freqs.len());

        // Process 4 values at a time for better performance
        let chunks = term_freqs.chunks_exact(4);
        let remainder = chunks.remainder();
        let norm_chunks = norm_factors.chunks_exact(4);

        for (tf_chunk, norm_chunk) in chunks.zip(norm_chunks) {
            let mut batch_results = [0.0f32; 4];

            for i in 0..4 {
                let tf = tf_chunk[i];
                let norm = norm_chunk[i];

                // BM25 TF calculation: tf * (k1 + 1) / (tf + k1 * norm)
                batch_results[i] = (tf * (k1 + 1.0)) / (tf + k1 * norm);
            }

            results.extend_from_slice(&batch_results);
        }

        // Handle remaining values
        let norm_remainder = &norm_factors[norm_factors.len() - remainder.len()..];
        for (tf, norm) in remainder.iter().zip(norm_remainder.iter()) {
            let tf_score = (tf * (k1 + 1.0)) / (tf + k1 * norm);
            results.push(tf_score);
        }

        results
    }

    /// Batch IDF calculation for multiple terms.
    pub fn batch_idf(doc_freqs: &[u64], total_docs: u64) -> Vec<f32> {
        let mut results = Vec::with_capacity(doc_freqs.len());

        // Process 4 values at a time
        let chunks = doc_freqs.chunks_exact(4);
        let remainder = chunks.remainder();

        for chunk in chunks {
            let mut batch_results = [0.0f32; 4];

            for i in 0..4 {
                let df = chunk[i] as f32;
                let n = total_docs as f32;

                // IDF calculation: ln((n - df + 0.5) / (df + 0.5))
                batch_results[i] = ((n - df + 0.5) / (df + 0.5)).ln();
            }

            results.extend_from_slice(&batch_results);
        }

        // Handle remaining values
        for &df in remainder {
            let df_f = df as f32;
            let n_f = total_docs as f32;
            let idf = ((n_f - df_f + 0.5) / (df_f + 0.5)).ln();
            results.push(idf);
        }

        results
    }

    /// Batch final BM25 score calculation.
    pub fn batch_bm25_final_score(
        tf_scores: &[f32],
        idf_scores: &[f32],
        boosts: &[f32],
    ) -> Vec<f32> {
        assert_eq!(tf_scores.len(), idf_scores.len());
        assert_eq!(tf_scores.len(), boosts.len());

        let mut results = Vec::with_capacity(tf_scores.len());

        // Process 4 values at a time
        let chunks = tf_scores.chunks_exact(4);
        let remainder = chunks.remainder();
        let idf_chunks = idf_scores.chunks_exact(4);
        let boost_chunks = boosts.chunks_exact(4);

        for ((tf_chunk, idf_chunk), boost_chunk) in chunks.zip(idf_chunks).zip(boost_chunks) {
            let mut batch_results = [0.0f32; 4];

            for i in 0..4 {
                // Final BM25: IDF * TF * boost
                batch_results[i] = idf_chunk[i] * tf_chunk[i] * boost_chunk[i];
            }

            results.extend_from_slice(&batch_results);
        }

        // Handle remaining values
        let idf_remainder = &idf_scores[idf_scores.len() - remainder.len()..];
        let boost_remainder = &boosts[boosts.len() - remainder.len()..];

        for ((tf, idf), boost) in remainder
            .iter()
            .zip(idf_remainder.iter())
            .zip(boost_remainder.iter())
        {
            results.push(idf * tf * boost);
        }

        results
    }

    /// Optimized sum calculation for f32 slices.
    pub fn fast_sum(values: &[f32]) -> f32 {
        let mut sum = 0.0f32;

        // Process 8 values at a time for better performance
        let chunks = values.chunks_exact(8);
        let remainder = chunks.remainder();

        for chunk in chunks {
            // Manual unrolling for better performance
            sum += chunk[0]
                + chunk[1]
                + chunk[2]
                + chunk[3]
                + chunk[4]
                + chunk[5]
                + chunk[6]
                + chunk[7];
        }

        // Handle remainder
        for &value in remainder {
            sum += value;
        }

        sum
    }

    /// Find maximum value and its index in parallel.
    pub fn find_max_with_index(values: &[f32]) -> Option<(usize, f32)> {
        if values.is_empty() {
            return None;
        }

        let mut max_val = values[0];
        let mut max_idx = 0;

        // Process in chunks for better cache performance
        let mut chunks = values.chunks_exact(4);
        let remainder = chunks.remainder();
        let mut chunk_idx = 0;

        for chunk in &mut chunks {
            for (i, &val) in chunk.iter().enumerate() {
                if val > max_val {
                    max_val = val;
                    max_idx = chunk_idx * 4 + i;
                }
            }
            chunk_idx += 1;
        }

        // Handle remainder
        let base_idx = chunk_idx * 4;
        for (i, &val) in remainder.iter().enumerate() {
            if val > max_val {
                max_val = val;
                max_idx = base_idx + i;
            }
        }

        Some((max_idx, max_val))
    }
}

/// SIMD-accelerated variable integer encoding/decoding operations.
pub mod varint {

    /// Batch encode multiple u32 values using optimized varint encoding.
    pub fn batch_encode_u32(values: &[u32]) -> Vec<u8> {
        let mut result = Vec::new();

        // Process 4 values at a time for better performance
        let chunks = values.chunks_exact(4);
        let remainder = chunks.remainder();

        for chunk in chunks {
            // Encode each value in the chunk
            for &value in chunk {
                encode_u32_to_vec(value, &mut result);
            }
        }

        // Handle remaining values
        for &value in remainder {
            encode_u32_to_vec(value, &mut result);
        }

        result
    }

    /// Batch encode multiple u64 values using optimized varint encoding.
    pub fn batch_encode_u64(values: &[u64]) -> Vec<u8> {
        let mut result = Vec::new();

        // Process 4 values at a time for better performance
        let chunks = values.chunks_exact(4);
        let remainder = chunks.remainder();

        for chunk in chunks {
            // Encode each value in the chunk
            for &value in chunk {
                encode_u64_to_vec(value, &mut result);
            }
        }

        // Handle remaining values
        for &value in remainder {
            encode_u64_to_vec(value, &mut result);
        }

        result
    }

    /// Helper function to encode a u32 value directly to a vector.
    fn encode_u32_to_vec(mut value: u32, output: &mut Vec<u8>) {
        while value >= 0x80 {
            output.push((value & 0x7F) as u8 | 0x80);
            value >>= 7;
        }
        output.push(value as u8);
    }

    /// Helper function to encode a u64 value directly to a vector.
    fn encode_u64_to_vec(mut value: u64, output: &mut Vec<u8>) {
        while value >= 0x80 {
            output.push((value & 0x7F) as u8 | 0x80);
            value >>= 7;
        }
        output.push(value as u8);
    }

    /// Batch decode multiple varint values from a byte slice.
    /// Returns (decoded_values, bytes_consumed).
    pub fn batch_decode_u32(data: &[u8], count: usize) -> (Vec<u32>, usize) {
        let mut values = Vec::with_capacity(count);
        let mut offset = 0;

        for _ in 0..count {
            if offset >= data.len() {
                break;
            }

            let (value, bytes_read) = decode_u32_from_slice(&data[offset..]);
            if bytes_read == 0 {
                break;
            }

            values.push(value);
            offset += bytes_read;
        }

        (values, offset)
    }

    /// Batch decode multiple u64 varint values from a byte slice.
    /// Returns (decoded_values, bytes_consumed).
    pub fn batch_decode_u64(data: &[u8], count: usize) -> (Vec<u64>, usize) {
        let mut values = Vec::with_capacity(count);
        let mut offset = 0;

        for _ in 0..count {
            if offset >= data.len() {
                break;
            }

            let (value, bytes_read) = decode_u64_from_slice(&data[offset..]);
            if bytes_read == 0 {
                break;
            }

            values.push(value);
            offset += bytes_read;
        }

        (values, offset)
    }

    /// Helper function to decode a u32 varint from a byte slice.
    /// Returns (value, bytes_consumed).
    fn decode_u32_from_slice(data: &[u8]) -> (u32, usize) {
        let mut value = 0u32;
        let mut shift = 0;
        let mut bytes_read = 0;

        for &byte in data.iter().take(5) {
            // u32 varint is at most 5 bytes
            bytes_read += 1;
            value |= ((byte & 0x7F) as u32) << shift;

            if byte & 0x80 == 0 {
                return (value, bytes_read);
            }

            shift += 7;
            if shift >= 32 {
                break; // Prevent overflow
            }
        }

        (value, bytes_read)
    }

    /// Helper function to decode a u64 varint from a byte slice.
    /// Returns (value, bytes_consumed).
    fn decode_u64_from_slice(data: &[u8]) -> (u64, usize) {
        let mut value = 0u64;
        let mut shift = 0;
        let mut bytes_read = 0;

        for &byte in data.iter().take(10) {
            // u64 varint is at most 10 bytes
            bytes_read += 1;
            value |= ((byte & 0x7F) as u64) << shift;

            if byte & 0x80 == 0 {
                return (value, bytes_read);
            }

            shift += 7;
            if shift >= 64 {
                break; // Prevent overflow
            }
        }

        (value, bytes_read)
    }

    /// Optimized delta encoding for sorted sequences.
    /// Encodes the differences between consecutive values.
    pub fn delta_encode_u32(values: &[u32]) -> Vec<u8> {
        if values.is_empty() {
            return Vec::new();
        }

        let mut result = Vec::new();
        let mut prev = 0u32;

        for &value in values {
            let delta = value.wrapping_sub(prev);
            encode_u32_to_vec(delta, &mut result);
            prev = value;
        }

        result
    }

    /// Decode delta-encoded values back to original sequence.
    pub fn delta_decode_u32(data: &[u8], count: usize) -> Vec<u32> {
        let (deltas, _) = batch_decode_u32(data, count);
        let mut result = Vec::with_capacity(deltas.len());
        let mut current = 0u32;

        for delta in deltas {
            current = current.wrapping_add(delta);
            result.push(current);
        }

        result
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
    fn test_contains_whitespace() {
        assert!(ascii::contains_whitespace_simd(b"hello world"));
        assert!(ascii::contains_whitespace_simd(b"hello\tworld"));
        assert!(ascii::contains_whitespace_simd(b"hello\nworld"));
        assert!(!ascii::contains_whitespace_simd(b"helloworld"));
    }

    #[test]
    fn test_find_whitespace() {
        assert_eq!(ascii::find_whitespace_simd(b"hello world"), Some(5));
        assert_eq!(ascii::find_whitespace_simd(b"hello\tworld"), Some(5));
        assert_eq!(ascii::find_whitespace_simd(b"helloworld"), None);
        assert_eq!(ascii::find_whitespace_simd(b""), None);
    }

    #[test]
    fn test_optimized_with_long_strings() {
        // Test with strings longer than 16 characters to trigger optimized path
        let long_input = "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG AND CONTINUES RUNNING";
        let expected = "the quick brown fox jumps over the lazy dog and continues running";
        let result = ascii::to_lowercase(long_input);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_whitespace_in_long_strings() {
        let long_input = b"abcdefghijklmnopqrstuvwxyz hello world";
        assert!(ascii::contains_whitespace_simd(long_input));
        assert_eq!(ascii::find_whitespace_simd(long_input), Some(26));
    }

    #[test]
    fn test_fallback_for_unicode() {
        let input = "Héllo Wörld"; // Non-ASCII characters
        let result = ascii::to_lowercase(input);
        let expected = input.to_lowercase();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_batch_bm25_tf() {
        let term_freqs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let norm_factors = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let k1 = 1.2;

        let results = numeric::batch_bm25_tf(&term_freqs, k1, &norm_factors);

        // Verify results match individual calculations
        for (i, &tf) in term_freqs.iter().enumerate() {
            let expected = (tf * (k1 + 1.0)) / (tf + k1 * norm_factors[i]);
            assert!((results[i] - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_batch_idf() {
        let doc_freqs = vec![1, 5, 10, 50, 100];
        let total_docs = 1000;

        let results = numeric::batch_idf(&doc_freqs, total_docs);

        // Verify results match individual calculations
        for (i, &df) in doc_freqs.iter().enumerate() {
            let n = total_docs as f32;
            let df_f = df as f32;
            let expected = ((n - df_f + 0.5) / (df_f + 0.5)).ln();
            assert!((results[i] - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_batch_bm25_final_score() {
        let tf_scores = vec![0.5, 0.8, 1.2, 0.3];
        let idf_scores = vec![2.0, 1.5, 1.0, 3.0];
        let boosts = vec![1.0, 1.5, 1.0, 2.0];

        let results = numeric::batch_bm25_final_score(&tf_scores, &idf_scores, &boosts);

        // Verify results match individual calculations
        for i in 0..tf_scores.len() {
            let expected = idf_scores[i] * tf_scores[i] * boosts[i];
            assert!((results[i] - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_fast_sum() {
        let values = vec![1.0, 2.5, 3.2, 4.8, 5.1, 6.3, 7.7, 8.9, 9.1];
        let result = numeric::fast_sum(&values);
        let expected: f32 = values.iter().sum();
        assert!((result - expected).abs() < 1e-6);
    }

    #[test]
    fn test_find_max_with_index() {
        let values = vec![1.0, 5.2, 3.1, 8.7, 2.4, 6.8];
        let result = numeric::find_max_with_index(&values);
        assert_eq!(result, Some((3, 8.7)));
    }

    #[test]
    fn test_find_max_empty() {
        let values: Vec<f32> = vec![];
        let result = numeric::find_max_with_index(&values);
        assert_eq!(result, None);
    }

    #[test]
    fn test_numeric_edge_cases() {
        // Test with single element
        let single = vec![42.0];
        assert_eq!(numeric::fast_sum(&single), 42.0);
        assert_eq!(numeric::find_max_with_index(&single), Some((0, 42.0)));

        // Test with two elements
        let pair = vec![10.0, 20.0];
        assert_eq!(numeric::fast_sum(&pair), 30.0);
        assert_eq!(numeric::find_max_with_index(&pair), Some((1, 20.0)));
    }

    #[test]
    fn test_batch_varint_u32() {
        let values = vec![1, 127, 128, 16383, 16384, 2097151];
        let encoded = varint::batch_encode_u32(&values);

        let (decoded, bytes_consumed) = varint::batch_decode_u32(&encoded, values.len());

        assert_eq!(decoded, values);
        assert_eq!(bytes_consumed, encoded.len());
    }

    #[test]
    fn test_batch_varint_u64() {
        let values = vec![1u64, 127, 128, 16383, 16384, 2097151, u64::MAX / 2];
        let encoded = varint::batch_encode_u64(&values);

        let (decoded, bytes_consumed) = varint::batch_decode_u64(&encoded, values.len());

        assert_eq!(decoded, values);
        assert_eq!(bytes_consumed, encoded.len());
    }

    #[test]
    fn test_delta_encoding() {
        let values = vec![10, 15, 20, 25, 30, 35, 40];
        let encoded = varint::delta_encode_u32(&values);
        let decoded = varint::delta_decode_u32(&encoded, values.len());

        assert_eq!(decoded, values);
    }

    #[test]
    fn test_delta_encoding_unsorted() {
        let values = vec![100, 50, 200, 75];
        let encoded = varint::delta_encode_u32(&values);
        let decoded = varint::delta_decode_u32(&encoded, values.len());

        assert_eq!(decoded, values);
    }

    #[test]
    fn test_varint_empty() {
        let values: Vec<u32> = vec![];
        let encoded = varint::batch_encode_u32(&values);
        assert!(encoded.is_empty());

        let (decoded, bytes_consumed) = varint::batch_decode_u32(&encoded, 0);
        assert!(decoded.is_empty());
        assert_eq!(bytes_consumed, 0);
    }

    #[test]
    fn test_varint_large_batch() {
        let values: Vec<u32> = (0..1000).collect();
        let encoded = varint::batch_encode_u32(&values);
        let (decoded, _) = varint::batch_decode_u32(&encoded, values.len());

        assert_eq!(decoded, values);
    }
}
