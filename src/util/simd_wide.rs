//! Advanced SIMD optimizations using the `wide` crate for true vectorization.

use wide::{f32x4, f32x8, i32x4, i32x8, u32x4, u32x8};

/// True SIMD ASCII operations using wide vectors.
pub mod ascii_wide {
    use super::*;
    
    /// SIMD-accelerated lowercase conversion for ASCII text.
    pub fn to_lowercase_simd(input: &str) -> String {
        let bytes = input.as_bytes();
        
        // Only use SIMD for sufficiently large ASCII strings
        if !input.is_ascii() || bytes.len() < 32 {
            return input.to_lowercase();
        }
        
        let mut result = Vec::with_capacity(bytes.len());
        
        // Process 32 bytes at a time using u32x8 (4 bytes per lane, 8 lanes)
        let chunks = bytes.chunks_exact(32);
        let remainder = chunks.remainder();
        
        for chunk in chunks {
            // Load 32 bytes as 8 u32 values (little-endian)
            let data1 = u32x4::new([
                u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
                u32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]),
                u32::from_le_bytes([chunk[8], chunk[9], chunk[10], chunk[11]]),
                u32::from_le_bytes([chunk[12], chunk[13], chunk[14], chunk[15]]),
            ]);
            
            let data2 = u32x4::new([
                u32::from_le_bytes([chunk[16], chunk[17], chunk[18], chunk[19]]),
                u32::from_le_bytes([chunk[20], chunk[21], chunk[22], chunk[23]]),
                u32::from_le_bytes([chunk[24], chunk[25], chunk[26], chunk[27]]),
                u32::from_le_bytes([chunk[28], chunk[29], chunk[30], chunk[31]]),
            ]);
            
            // Process each u32x4
            let result1 = simd_lowercase_u32x4(data1);
            let result2 = simd_lowercase_u32x4(data2);
            
            // Convert back to bytes
            for value in result1.to_array() {
                result.extend_from_slice(&value.to_le_bytes());
            }
            for value in result2.to_array() {
                result.extend_from_slice(&value.to_le_bytes());
            }
        }
        
        // Process remainder using scalar approach
        for &byte in remainder {
            if byte >= b'A' && byte <= b'Z' {
                result.push(byte + 32);
            } else {
                result.push(byte);
            }
        }
        
        // Safety: We know the input was valid ASCII and we only changed case
        unsafe { String::from_utf8_unchecked(result) }
    }
    
    /// SIMD lowercase conversion for a u32x4 containing 4 ASCII bytes each.
    fn simd_lowercase_u32x4(data: u32x4) -> u32x4 {
        // Create masks for detecting uppercase letters (A-Z)
        // A = 0x41, Z = 0x5A, so we check if byte >= 0x41 && byte <= 0x5A
        
        // Mask for bytes >= 'A' (0x41)
        let mask_a = data.cmp_ge(u32x4::splat(0x41414141));
        
        // Mask for bytes <= 'Z' (0x5A)  
        let mask_z = data.cmp_le(u32x4::splat(0x5A5A5A5A));
        
        // Combined mask for uppercase letters
        let uppercase_mask = mask_a & mask_z;
        
        // Add 32 (0x20) to convert to lowercase
        let lowercase_offset = u32x4::splat(0x20202020);
        let converted = data + (uppercase_mask & lowercase_offset);
        
        converted
    }
    
    /// SIMD whitespace detection for large byte arrays.
    pub fn contains_whitespace_simd(input: &[u8]) -> bool {
        if input.len() < 32 {
            return input.iter().any(|&b| b.is_ascii_whitespace());
        }
        
        // Define whitespace characters as SIMD vectors
        let space = u32x8::splat(0x20202020);  // space
        let tab = u32x8::splat(0x09090909);    // tab
        let newline = u32x8::splat(0x0A0A0A0A); // newline
        let carriage = u32x8::splat(0x0D0D0D0D); // carriage return
        
        // Process 32 bytes at a time
        let chunks = input.chunks_exact(32);
        let remainder = chunks.remainder();
        
        for chunk in chunks {
            // Load chunk as u32x8
            let data = u32x8::new([
                u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
                u32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]),
                u32::from_le_bytes([chunk[8], chunk[9], chunk[10], chunk[11]]),
                u32::from_le_bytes([chunk[12], chunk[13], chunk[14], chunk[15]]),
                u32::from_le_bytes([chunk[16], chunk[17], chunk[18], chunk[19]]),
                u32::from_le_bytes([chunk[20], chunk[21], chunk[22], chunk[23]]),
                u32::from_le_bytes([chunk[24], chunk[25], chunk[26], chunk[27]]),
                u32::from_le_bytes([chunk[28], chunk[29], chunk[30], chunk[31]]),
            ]);
            
            // Check for whitespace characters
            let is_space = data.cmp_eq(space);
            let is_tab = data.cmp_eq(tab);
            let is_newline = data.cmp_eq(newline);
            let is_carriage = data.cmp_eq(carriage);
            
            let whitespace_mask = is_space | is_tab | is_newline | is_carriage;
            
            // If any lane has a match, we found whitespace
            if whitespace_mask.any() {
                return true;
            }
        }
        
        // Check remainder
        remainder.iter().any(|&b| b.is_ascii_whitespace())
    }
}

/// Advanced numerical SIMD operations using wide vectors.
pub mod numeric_wide {
    use super::*;
    
    /// SIMD-accelerated BM25 TF calculation using f32x8.
    pub fn batch_bm25_tf_simd(
        term_freqs: &[f32],
        k1: f32,
        norm_factors: &[f32],
    ) -> Vec<f32> {
        assert_eq!(term_freqs.len(), norm_factors.len());
        
        if term_freqs.len() < 8 {
            // Fall back to scalar for small inputs
            return crate::util::simd::numeric::batch_bm25_tf(term_freqs, k1, norm_factors);
        }
        
        let mut result = Vec::with_capacity(term_freqs.len());
        let k1_vec = f32x8::splat(k1);
        let one_vec = f32x8::splat(1.0);
        let k1_plus_one = k1_vec + one_vec;
        
        // Process 8 values at a time
        let chunks = term_freqs.chunks_exact(8);
        let remainder = chunks.remainder();
        let norm_chunks = norm_factors.chunks_exact(8);
        
        for (tf_chunk, norm_chunk) in chunks.zip(norm_chunks) {
            let tf_vec = f32x8::new(*tf_chunk.try_into().unwrap());
            let norm_vec = f32x8::new(*norm_chunk.try_into().unwrap());
            
            // BM25 TF: tf * (k1 + 1) / (tf + k1 * norm)
            let numerator = tf_vec * k1_plus_one;
            let denominator = tf_vec + (k1_vec * norm_vec);
            let tf_scores = numerator / denominator;
            
            result.extend_from_slice(&tf_scores.to_array());
        }
        
        // Handle remainder with scalar operations
        let norm_remainder = &norm_factors[norm_factors.len() - remainder.len()..];
        for (&tf, &norm) in remainder.iter().zip(norm_remainder.iter()) {
            let tf_score = (tf * (k1 + 1.0)) / (tf + k1 * norm);
            result.push(tf_score);
        }
        
        result
    }
    
    /// SIMD-accelerated IDF calculation using f32x8.
    pub fn batch_idf_simd(doc_freqs: &[u64], total_docs: u64) -> Vec<f32> {
        if doc_freqs.len() < 8 {
            return crate::util::simd::numeric::batch_idf(doc_freqs, total_docs);
        }
        
        let mut result = Vec::with_capacity(doc_freqs.len());
        let n_vec = f32x8::splat(total_docs as f32);
        let half_vec = f32x8::splat(0.5);
        
        // Process 8 values at a time
        let chunks = doc_freqs.chunks_exact(8);
        let remainder = chunks.remainder();
        
        for chunk in chunks {
            // Convert u64 to f32
            let df_array: [f32; 8] = [
                chunk[0] as f32, chunk[1] as f32, chunk[2] as f32, chunk[3] as f32,
                chunk[4] as f32, chunk[5] as f32, chunk[6] as f32, chunk[7] as f32,
            ];
            let df_vec = f32x8::new(df_array);
            
            // IDF: ln((n - df + 0.5) / (df + 0.5))
            let numerator = n_vec - df_vec + half_vec;
            let denominator = df_vec + half_vec;
            let ratio = numerator / denominator;
            let idf_scores = ratio.ln();
            
            result.extend_from_slice(&idf_scores.to_array());
        }
        
        // Handle remainder
        for &df in remainder {
            let df_f = df as f32;
            let n_f = total_docs as f32;
            let idf = ((n_f - df_f + 0.5) / (df_f + 0.5)).ln();
            result.push(idf);
        }
        
        result
    }
    
    /// SIMD-accelerated final BM25 score calculation using f32x8.
    pub fn batch_bm25_final_score_simd(
        tf_scores: &[f32],
        idf_scores: &[f32], 
        boosts: &[f32],
    ) -> Vec<f32> {
        assert_eq!(tf_scores.len(), idf_scores.len());
        assert_eq!(tf_scores.len(), boosts.len());
        
        if tf_scores.len() < 8 {
            return crate::util::simd::numeric::batch_bm25_final_score(tf_scores, idf_scores, boosts);
        }
        
        let mut result = Vec::with_capacity(tf_scores.len());
        
        // Process 8 values at a time
        let chunks = tf_scores.chunks_exact(8);
        let remainder = chunks.remainder();
        let idf_chunks = idf_scores.chunks_exact(8);
        let boost_chunks = boosts.chunks_exact(8);
        
        for ((tf_chunk, idf_chunk), boost_chunk) in chunks.zip(idf_chunks).zip(boost_chunks) {
            let tf_vec = f32x8::new(*tf_chunk.try_into().unwrap());
            let idf_vec = f32x8::new(*idf_chunk.try_into().unwrap());
            let boost_vec = f32x8::new(*boost_chunk.try_into().unwrap());
            
            // Final BM25: IDF * TF * boost
            let final_scores = idf_vec * tf_vec * boost_vec;
            
            result.extend_from_slice(&final_scores.to_array());
        }
        
        // Handle remainder
        let idf_remainder = &idf_scores[idf_scores.len() - remainder.len()..];
        let boost_remainder = &boosts[boosts.len() - remainder.len()..];
        
        for ((&tf, &idf), &boost) in remainder.iter().zip(idf_remainder.iter()).zip(boost_remainder.iter()) {
            result.push(idf * tf * boost);
        }
        
        result
    }
    
    /// SIMD-accelerated sum using f32x8.
    pub fn fast_sum_simd(values: &[f32]) -> f32 {
        if values.len() < 16 {
            return crate::util::simd::numeric::fast_sum(values);
        }
        
        let mut sum_vec = f32x8::splat(0.0);
        
        // Process 8 values at a time
        let chunks = values.chunks_exact(8);
        let remainder = chunks.remainder();
        
        for chunk in chunks {
            let data_vec = f32x8::new(*chunk.try_into().unwrap());
            sum_vec = sum_vec + data_vec;
        }
        
        // Sum all lanes in the vector
        let sum_array = sum_vec.to_array();
        let mut total = sum_array.iter().sum::<f32>();
        
        // Add remainder
        total += remainder.iter().sum::<f32>();
        
        total
    }
    
    /// SIMD-accelerated dot product calculation.
    pub fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        
        if a.len() < 8 {
            return a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        }
        
        let mut dot_vec = f32x8::splat(0.0);
        
        // Process 8 values at a time
        let chunks_a = a.chunks_exact(8);
        let chunks_b = b.chunks_exact(8);
        let remainder_a = chunks_a.remainder();
        let remainder_b = chunks_b.remainder();
        
        for (chunk_a, chunk_b) in chunks_a.zip(chunks_b) {
            let vec_a = f32x8::new(*chunk_a.try_into().unwrap());
            let vec_b = f32x8::new(*chunk_b.try_into().unwrap());
            let product = vec_a * vec_b;
            dot_vec = dot_vec + product;
        }
        
        // Sum all lanes
        let dot_array = dot_vec.to_array();
        let mut total = dot_array.iter().sum::<f32>();
        
        // Add remainder
        total += remainder_a.iter().zip(remainder_b.iter()).map(|(x, y)| x * y).sum::<f32>();
        
        total
    }
    
    /// SIMD-accelerated cosine similarity calculation.
    pub fn cosine_similarity_simd(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        
        if a.is_empty() {
            return 0.0;
        }
        
        let dot_product = dot_product_simd(a, b);
        let norm_a = fast_sum_simd(&a.iter().map(|x| x * x).collect::<Vec<_>>()).sqrt();
        let norm_b = fast_sum_simd(&b.iter().map(|x| x * x).collect::<Vec<_>>()).sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }
    
    /// SIMD-accelerated vector addition.
    pub fn vector_add_simd(a: &[f32], b: &[f32]) -> Vec<f32> {
        assert_eq!(a.len(), b.len());
        
        if a.len() < 8 {
            return a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
        }
        
        let mut result = Vec::with_capacity(a.len());
        
        // Process 8 values at a time
        let chunks_a = a.chunks_exact(8);
        let chunks_b = b.chunks_exact(8);
        let remainder_a = chunks_a.remainder();
        let remainder_b = chunks_b.remainder();
        
        for (chunk_a, chunk_b) in chunks_a.zip(chunks_b) {
            let vec_a = f32x8::new(*chunk_a.try_into().unwrap());
            let vec_b = f32x8::new(*chunk_b.try_into().unwrap());
            let sum = vec_a + vec_b;
            result.extend_from_slice(&sum.to_array());
        }
        
        // Handle remainder
        for (&x, &y) in remainder_a.iter().zip(remainder_b.iter()) {
            result.push(x + y);
        }
        
        result
    }
    
    /// SIMD-accelerated vector scaling.
    pub fn vector_scale_simd(vector: &[f32], scale: f32) -> Vec<f32> {
        if vector.len() < 8 {
            return vector.iter().map(|x| x * scale).collect();
        }
        
        let mut result = Vec::with_capacity(vector.len());
        let scale_vec = f32x8::splat(scale);
        
        // Process 8 values at a time
        let chunks = vector.chunks_exact(8);
        let remainder = chunks.remainder();
        
        for chunk in chunks {
            let data_vec = f32x8::new(*chunk.try_into().unwrap());
            let scaled = data_vec * scale_vec;
            result.extend_from_slice(&scaled.to_array());
        }
        
        // Handle remainder
        for &x in remainder {
            result.push(x * scale);
        }
        
        result
    }
}

/// SIMD-accelerated integer operations.
pub mod integer_wide {
    use super::*;
    
    /// SIMD-accelerated delta encoding for document IDs.
    pub fn delta_encode_simd(values: &[u32]) -> Vec<u32> {
        if values.is_empty() || values.len() < 8 {
            return delta_encode_scalar(values);
        }
        
        let mut result = Vec::with_capacity(values.len());
        
        // First value is encoded as-is
        result.push(values[0]);
        
        if values.len() == 1 {
            return result;
        }
        
        // Process remaining values in chunks of 8
        let mut prev = values[0];
        let remaining = &values[1..];
        let chunks = remaining.chunks_exact(8);
        let remainder = chunks.remainder();
        
        for chunk in chunks {
            let current_vec = u32x8::new(*chunk.try_into().unwrap());
            let prev_vec = u32x8::new([prev, chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6]]);
            
            // Calculate deltas
            let deltas = current_vec - prev_vec;
            let delta_array = deltas.to_array();
            
            result.extend_from_slice(&delta_array);
            prev = chunk[7];
        }
        
        // Handle remainder
        for &value in remainder {
            result.push(value - prev);
            prev = value;
        }
        
        result
    }
    
    /// Scalar implementation for delta encoding (fallback).
    fn delta_encode_scalar(values: &[u32]) -> Vec<u32> {
        if values.is_empty() {
            return Vec::new();
        }
        
        let mut result = Vec::with_capacity(values.len());
        result.push(values[0]);
        
        for i in 1..values.len() {
            result.push(values[i] - values[i - 1]);
        }
        
        result
    }
    
    /// SIMD-accelerated delta decoding.
    pub fn delta_decode_simd(deltas: &[u32]) -> Vec<u32> {
        if deltas.is_empty() || deltas.len() < 8 {
            return delta_decode_scalar(deltas);
        }
        
        let mut result = Vec::with_capacity(deltas.len());
        
        // First value is the delta itself
        result.push(deltas[0]);
        
        if deltas.len() == 1 {
            return result;
        }
        
        let mut current = deltas[0];
        let remaining = &deltas[1..];
        let chunks = remaining.chunks_exact(8);
        let remainder = chunks.remainder();
        
        for chunk in chunks {
            let delta_vec = u32x8::new(*chunk.try_into().unwrap());
            let mut reconstructed = [0u32; 8];
            
            // Reconstruct values by accumulating deltas
            reconstructed[0] = current + delta_vec.as_array_ref()[0];
            for i in 1..8 {
                reconstructed[i] = reconstructed[i - 1] + delta_vec.as_array_ref()[i];
            }
            
            result.extend_from_slice(&reconstructed);
            current = reconstructed[7];
        }
        
        // Handle remainder
        for &delta in remainder {
            current += delta;
            result.push(current);
        }
        
        result
    }
    
    /// Scalar implementation for delta decoding (fallback).
    fn delta_decode_scalar(deltas: &[u32]) -> Vec<u32> {
        if deltas.is_empty() {
            return Vec::new();
        }
        
        let mut result = Vec::with_capacity(deltas.len());
        let mut current = deltas[0];
        result.push(current);
        
        for &delta in &deltas[1..] {
            current += delta;
            result.push(current);
        }
        
        result
    }
    
    /// SIMD-accelerated bitwise intersection of sorted arrays.
    pub fn intersect_sorted_simd(a: &[u32], b: &[u32]) -> Vec<u32> {
        if a.is_empty() || b.is_empty() {
            return Vec::new();
        }
        
        // For small arrays, use scalar algorithm
        if a.len() < 16 || b.len() < 16 {
            return intersect_sorted_scalar(a, b);
        }
        
        let mut result = Vec::new();
        let mut i = 0;
        let mut j = 0;
        
        // Use SIMD for the main processing
        while i + 8 <= a.len() && j + 8 <= b.len() {
            let chunk_a = u32x8::new([
                a[i], a[i+1], a[i+2], a[i+3], a[i+4], a[i+5], a[i+6], a[i+7]
            ]);
            let chunk_b = u32x8::new([
                b[j], b[j+1], b[j+2], b[j+3], b[j+4], b[j+5], b[j+6], b[j+7]
            ]);
            
            // Find intersections
            for (idx_a, val_a) in chunk_a.to_array().iter().enumerate() {
                for (idx_b, val_b) in chunk_b.to_array().iter().enumerate() {
                    if val_a == val_b {
                        result.push(*val_a);
                    }
                }
            }
            
            // Advance the pointer for the smaller maximum value
            if chunk_a.as_array_ref()[7] <= chunk_b.as_array_ref()[7] {
                i += 8;
            } else {
                j += 8;
            }
        }
        
        // Handle remaining elements with scalar approach
        while i < a.len() && j < b.len() {
            if a[i] == b[j] {
                result.push(a[i]);
                i += 1;
                j += 1;
            } else if a[i] < b[j] {
                i += 1;
            } else {
                j += 1;
            }
        }
        
        result
    }
    
    /// Scalar implementation for sorted array intersection.
    fn intersect_sorted_scalar(a: &[u32], b: &[u32]) -> Vec<u32> {
        let mut result = Vec::new();
        let mut i = 0;
        let mut j = 0;
        
        while i < a.len() && j < b.len() {
            if a[i] == b[j] {
                result.push(a[i]);
                i += 1;
                j += 1;
            } else if a[i] < b[j] {
                i += 1;
            } else {
                j += 1;
            }
        }
        
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simd_lowercase() {
        let input = "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG AND RUNS FAST THROUGH THE FOREST";
        let expected = input.to_lowercase();
        let result = ascii_wide::to_lowercase_simd(input);
        assert_eq!(result, expected);
    }
    
    #[test]
    fn test_simd_whitespace_detection() {
        let input = b"abcdefghijklmnopqrstuvwxyz hello world test string";
        assert!(ascii_wide::contains_whitespace_simd(input));
        
        let no_space = b"abcdefghijklmnopqrstuvwxyz";
        assert!(!ascii_wide::contains_whitespace_simd(no_space));
    }
    
    #[test]
    fn test_simd_bm25_tf() {
        let term_freqs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let norm_factors = vec![1.0; 9];
        let k1 = 1.2;
        
        let result = numeric_wide::batch_bm25_tf_simd(&term_freqs, k1, &norm_factors);
        
        // Verify against scalar implementation
        let expected = crate::util::simd::numeric::batch_bm25_tf(&term_freqs, k1, &norm_factors);
        
        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-6);
        }
    }
    
    #[test]
    fn test_simd_idf() {
        let doc_freqs = vec![1, 5, 10, 20, 50, 100, 200, 500, 1000];
        let total_docs = 10000;
        
        let result = numeric_wide::batch_idf_simd(&doc_freqs, total_docs);
        let expected = crate::util::simd::numeric::batch_idf(&doc_freqs, total_docs);
        
        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-6);
        }
    }
    
    #[test]
    fn test_simd_sum() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
        let result = numeric_wide::fast_sum_simd(&values);
        let expected: f32 = values.iter().sum();
        assert!((result - expected).abs() < 1e-6);
    }
    
    #[test]
    fn test_simd_dot_product() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        
        let result = numeric_wide::dot_product_simd(&a, &b);
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        
        assert!((result - expected).abs() < 1e-6);
    }
    
    #[test]
    fn test_simd_vector_operations() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        
        let sum_result = numeric_wide::vector_add_simd(&a, &b);
        let expected_sum: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
        
        for (r, e) in sum_result.iter().zip(expected_sum.iter()) {
            assert!((r - e).abs() < 1e-6);
        }
        
        let scale_result = numeric_wide::vector_scale_simd(&a, 2.0);
        let expected_scale: Vec<f32> = a.iter().map(|x| x * 2.0).collect();
        
        for (r, e) in scale_result.iter().zip(expected_scale.iter()) {
            assert!((r - e).abs() < 1e-6);
        }
    }
    
    #[test]
    fn test_simd_delta_encoding() {
        let values = vec![10, 15, 20, 25, 30, 35, 40, 45, 50, 55];
        let encoded = integer_wide::delta_encode_simd(&values);
        let decoded = integer_wide::delta_decode_simd(&encoded);
        
        assert_eq!(decoded, values);
    }
    
    #[test]
    fn test_simd_intersection() {
        let a = vec![1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31];
        let b = vec![2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31];
        
        let result = integer_wide::intersect_sorted_simd(&a, &b);
        let expected = vec![3, 7, 11, 15, 19, 23, 27, 31];
        
        assert_eq!(result, expected);
    }
}