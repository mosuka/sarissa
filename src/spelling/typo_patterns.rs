//! Keyboard-aware typo patterns for spelling correction.
//!
//! This module provides domain-specific functionality for handling common typing errors
//! based on keyboard layout proximity. This is particularly useful for improving
//! spell correction accuracy by weighting errors based on physical key distances.

/// Common typo patterns for keyboard-based errors.
pub struct TypoPatterns;

impl TypoPatterns {
    /// Get nearby keys on a QWERTY keyboard for a given character.
    pub fn nearby_keys(ch: char) -> Vec<char> {
        match ch.to_ascii_lowercase() {
            'q' => vec!['w', 'a', 's'],
            'w' => vec!['q', 'e', 'a', 's', 'd'],
            'e' => vec!['w', 'r', 's', 'd', 'f'],
            'r' => vec!['e', 't', 'd', 'f', 'g'],
            't' => vec!['r', 'y', 'f', 'g', 'h'],
            'y' => vec!['t', 'u', 'g', 'h', 'j'],
            'u' => vec!['y', 'i', 'h', 'j', 'k'],
            'i' => vec!['u', 'o', 'j', 'k', 'l'],
            'o' => vec!['i', 'p', 'k', 'l'],
            'p' => vec!['o', 'l'],
            'a' => vec!['q', 'w', 's', 'z'],
            's' => vec!['a', 'd', 'w', 'e', 'z', 'x'],
            'd' => vec!['s', 'f', 'e', 'r', 'x', 'c'],
            'f' => vec!['d', 'g', 'r', 't', 'c', 'v'],
            'g' => vec!['f', 'h', 't', 'y', 'v', 'b'],
            'h' => vec!['g', 'j', 'y', 'u', 'b', 'n'],
            'j' => vec!['h', 'k', 'u', 'i', 'n', 'm'],
            'k' => vec!['j', 'l', 'i', 'o', 'm'],
            'l' => vec!['k', 'o', 'p', 'm'],
            'z' => vec!['a', 's', 'x'],
            'x' => vec!['z', 'c', 's', 'd'],
            'c' => vec!['x', 'v', 'd', 'f'],
            'v' => vec!['c', 'b', 'f', 'g'],
            'b' => vec!['v', 'n', 'g', 'h'],
            'n' => vec!['b', 'm', 'h', 'j'],
            'm' => vec!['n', 'j', 'k', 'l'],
            _ => vec![],
        }
    }

    /// Calculate keyboard distance-weighted edit distance.
    /// Substitutions between nearby keys have lower cost.
    #[allow(clippy::needless_range_loop)]
    pub fn keyboard_distance(s1: &str, s2: &str) -> f64 {
        let len1 = s1.chars().count();
        let len2 = s2.chars().count();

        if len1 == 0 {
            return len2 as f64;
        }
        if len2 == 0 {
            return len1 as f64;
        }

        let s1_chars: Vec<char> = s1.chars().collect();
        let s2_chars: Vec<char> = s2.chars().collect();

        let mut matrix = vec![vec![0.0; len2 + 1]; len1 + 1];

        // Initialize first row and column
        for i in 0..=len1 {
            matrix[i][0] = i as f64;
        }
        for j in 0..=len2 {
            matrix[0][j] = j as f64;
        }

        // Fill the matrix with weighted costs
        for i in 1..=len1 {
            for j in 1..=len2 {
                let ch1 = s1_chars[i - 1];
                let ch2 = s2_chars[j - 1];

                let substitution_cost = if ch1 == ch2 {
                    0.0
                } else {
                    // Lower cost for nearby keys
                    let nearby = Self::nearby_keys(ch1);
                    if nearby.contains(&ch2) {
                        0.5 // Nearby key substitution
                    } else {
                        1.0 // Regular substitution
                    }
                };

                matrix[i][j] = (matrix[i - 1][j] + 1.0) // deletion
                    .min(matrix[i][j - 1] + 1.0) // insertion
                    .min(matrix[i - 1][j - 1] + substitution_cost); // substitution
            }
        }

        matrix[len1][len2]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_typo_patterns_nearby_keys() {
        let nearby_q = TypoPatterns::nearby_keys('q');
        assert!(nearby_q.contains(&'w'));
        assert!(nearby_q.contains(&'a'));
        assert!(!nearby_q.contains(&'z'));

        let nearby_m = TypoPatterns::nearby_keys('m');
        assert!(nearby_m.contains(&'n'));
        assert!(nearby_m.contains(&'j'));
    }

    #[test]
    fn test_keyboard_distance() {
        // Exact match
        assert!((TypoPatterns::keyboard_distance("search", "search") - 0.0).abs() < 1e-6);

        // Nearby key substitution should be cheaper than regular substitution
        let nearby_dist = TypoPatterns::keyboard_distance("search", "searcg"); // h->g nearby
        let regular_dist = TypoPatterns::keyboard_distance("search", "searcp"); // h->p not nearby
        assert!(nearby_dist < regular_dist);
    }
}
