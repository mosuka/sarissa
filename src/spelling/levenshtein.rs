//! Levenshtein distance calculation for spelling correction.

use std::cmp::min;

/// Calculate the Levenshtein distance between two strings.
/// This is the minimum number of single-character edits (insertions, deletions, or substitutions)
/// required to change one word into another.
#[allow(clippy::needless_range_loop)]
pub fn levenshtein_distance(s1: &str, s2: &str) -> usize {
    let len1 = s1.chars().count();
    let len2 = s2.chars().count();

    if len1 == 0 {
        return len2;
    }
    if len2 == 0 {
        return len1;
    }

    let s1_chars: Vec<char> = s1.chars().collect();
    let s2_chars: Vec<char> = s2.chars().collect();

    // Create a matrix to store distances
    let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];

    // Initialize first row and column
    for i in 0..=len1 {
        matrix[i][0] = i;
    }
    for j in 0..=len2 {
        matrix[0][j] = j;
    }

    // Fill the matrix
    for i in 1..=len1 {
        for j in 1..=len2 {
            let cost = if s1_chars[i - 1] == s2_chars[j - 1] {
                0
            } else {
                1
            };

            matrix[i][j] = min(
                min(
                    matrix[i - 1][j] + 1, // deletion
                    matrix[i][j - 1] + 1, // insertion
                ),
                matrix[i - 1][j - 1] + cost, // substitution
            );
        }
    }

    matrix[len1][len2]
}

/// Calculate Levenshtein distance with a maximum threshold for early termination.
/// Returns None if the distance exceeds the threshold, which can be more efficient
/// for filtering candidates.
#[allow(clippy::needless_range_loop)]
pub fn levenshtein_distance_threshold(s1: &str, s2: &str, threshold: usize) -> Option<usize> {
    let len1 = s1.chars().count();
    let len2 = s2.chars().count();

    // Early termination if length difference exceeds threshold
    if len1.abs_diff(len2) > threshold {
        return None;
    }

    if len1 == 0 {
        return if len2 <= threshold { Some(len2) } else { None };
    }
    if len2 == 0 {
        return if len1 <= threshold { Some(len1) } else { None };
    }

    let s1_chars: Vec<char> = s1.chars().collect();
    let s2_chars: Vec<char> = s2.chars().collect();

    // Use only two rows for space optimization
    let mut prev_row = vec![0; len2 + 1];
    let mut curr_row = vec![0; len2 + 1];

    // Initialize first row
    for j in 0..=len2 {
        prev_row[j] = j;
    }

    for i in 1..=len1 {
        curr_row[0] = i;
        let mut min_in_row = i;

        for j in 1..=len2 {
            let cost = if s1_chars[i - 1] == s2_chars[j - 1] {
                0
            } else {
                1
            };

            curr_row[j] = min(
                min(
                    prev_row[j] + 1,     // deletion
                    curr_row[j - 1] + 1, // insertion
                ),
                prev_row[j - 1] + cost, // substitution
            );

            min_in_row = min(min_in_row, curr_row[j]);
        }

        // Early termination if minimum in row exceeds threshold
        if min_in_row > threshold {
            return None;
        }

        std::mem::swap(&mut prev_row, &mut curr_row);
    }

    let distance = prev_row[len2];
    if distance <= threshold {
        Some(distance)
    } else {
        None
    }
}

/// Calculate Damerau-Levenshtein distance, which also considers transpositions.
/// This is more accurate for real-world typos where adjacent characters are swapped.
#[allow(clippy::needless_range_loop)]
pub fn damerau_levenshtein_distance(s1: &str, s2: &str) -> usize {
    let len1 = s1.chars().count();
    let len2 = s2.chars().count();

    if len1 == 0 {
        return len2;
    }
    if len2 == 0 {
        return len1;
    }

    let s1_chars: Vec<char> = s1.chars().collect();
    let s2_chars: Vec<char> = s2.chars().collect();

    let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];

    // Initialize first row and column
    for i in 0..=len1 {
        matrix[i][0] = i;
    }
    for j in 0..=len2 {
        matrix[0][j] = j;
    }

    // Fill the matrix
    for i in 1..=len1 {
        for j in 1..=len2 {
            let cost = if s1_chars[i - 1] == s2_chars[j - 1] {
                0
            } else {
                1
            };

            matrix[i][j] = min(
                min(
                    matrix[i - 1][j] + 1, // deletion
                    matrix[i][j - 1] + 1, // insertion
                ),
                matrix[i - 1][j - 1] + cost, // substitution
            );

            // Check for transposition
            if i > 1
                && j > 1
                && s1_chars[i - 1] == s2_chars[j - 2]
                && s1_chars[i - 2] == s2_chars[j - 1]
            {
                matrix[i][j] = min(
                    matrix[i][j],
                    matrix[i - 2][j - 2] + cost, // transposition
                );
            }
        }
    }

    matrix[len1][len2]
}

/// Calculate normalized Levenshtein distance as a ratio between 0.0 and 1.0.
/// 0.0 means identical strings, 1.0 means completely different.
pub fn levenshtein_ratio(s1: &str, s2: &str) -> f64 {
    let len1 = s1.chars().count();
    let len2 = s2.chars().count();
    let max_len = len1.max(len2);

    if max_len == 0 {
        return 0.0;
    }

    let distance = levenshtein_distance(s1, s2);
    1.0 - (distance as f64 / max_len as f64)
}

/// Optimized version for calculating distance between a query and multiple candidates.
/// This reuses computation where possible for better performance.
pub struct LevenshteinMatcher {
    query: String,
    #[allow(dead_code)]
    query_chars: Vec<char>,
    #[allow(dead_code)]
    query_len: usize,
}

impl LevenshteinMatcher {
    /// Create a new matcher for the given query string.
    pub fn new(query: String) -> Self {
        let query_chars: Vec<char> = query.chars().collect();
        let query_len = query_chars.len();

        LevenshteinMatcher {
            query,
            query_chars,
            query_len,
        }
    }

    /// Get the original query string.
    pub fn query(&self) -> &str {
        &self.query
    }

    /// Calculate distance to a candidate string.
    pub fn distance(&self, candidate: &str) -> usize {
        levenshtein_distance(&self.query, candidate)
    }

    /// Calculate distance with threshold for early termination.
    pub fn distance_threshold(&self, candidate: &str, threshold: usize) -> Option<usize> {
        levenshtein_distance_threshold(&self.query, candidate, threshold)
    }

    /// Calculate similarity ratio (0.0 to 1.0, higher is more similar).
    pub fn similarity(&self, candidate: &str) -> f64 {
        levenshtein_ratio(&self.query, candidate)
    }

    /// Check if a candidate is within the given edit distance threshold.
    pub fn is_match(&self, candidate: &str, max_distance: usize) -> bool {
        self.distance_threshold(candidate, max_distance).is_some()
    }
}

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
    fn test_levenshtein_distance() {
        assert_eq!(levenshtein_distance("", ""), 0);
        assert_eq!(levenshtein_distance("", "a"), 1);
        assert_eq!(levenshtein_distance("a", ""), 1);
        assert_eq!(levenshtein_distance("a", "a"), 0);
        assert_eq!(levenshtein_distance("ab", "ac"), 1);
        assert_eq!(levenshtein_distance("abc", "def"), 3);
        assert_eq!(levenshtein_distance("kitten", "sitting"), 3);
        assert_eq!(levenshtein_distance("search", "serach"), 2); // transposition
    }

    #[test]
    fn test_levenshtein_distance_threshold() {
        assert_eq!(
            levenshtein_distance_threshold("kitten", "sitting", 3),
            Some(3)
        );
        assert_eq!(levenshtein_distance_threshold("kitten", "sitting", 2), None);
        assert_eq!(
            levenshtein_distance_threshold("search", "search", 0),
            Some(0)
        );
        assert_eq!(levenshtein_distance_threshold("a", "abc", 1), None);
        assert_eq!(levenshtein_distance_threshold("a", "ab", 1), Some(1));
    }

    #[test]
    fn test_damerau_levenshtein_distance() {
        assert_eq!(damerau_levenshtein_distance("", ""), 0);
        assert_eq!(damerau_levenshtein_distance("ab", "ba"), 1); // transposition
        assert_eq!(damerau_levenshtein_distance("search", "serach"), 1); // transposition
        assert_eq!(damerau_levenshtein_distance("kitten", "sitting"), 3);
    }

    #[test]
    fn test_levenshtein_ratio() {
        assert!((levenshtein_ratio("", "") - 0.0).abs() < 1e-6);
        assert!((levenshtein_ratio("abc", "abc") - 1.0).abs() < 1e-6);
        assert!((levenshtein_ratio("abc", "def") - 0.0).abs() < 1e-6);

        let ratio = levenshtein_ratio("search", "serach");
        assert!(ratio > 0.5 && ratio < 1.0);
    }

    #[test]
    fn test_levenshtein_matcher() {
        let matcher = LevenshteinMatcher::new("search".to_string());

        assert_eq!(matcher.query(), "search");
        assert_eq!(matcher.distance("search"), 0);
        assert_eq!(matcher.distance("serach"), 2);
        assert!(matcher.similarity("search") > matcher.similarity("serach"));
        assert!(matcher.is_match("serach", 2));
        assert!(!matcher.is_match("completely_different", 2));
    }

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

    #[test]
    fn test_common_typos() {
        let common_typos = vec![
            ("the", "teh"),       // transposition
            ("search", "serach"), // transposition
            ("hello", "helo"),    // deletion
            ("world", "wrold"),   // transposition
            ("quick", "quikc"),   // transposition
        ];

        for (correct, typo) in common_typos {
            let distance = levenshtein_distance(correct, typo);
            assert!(
                distance <= 2,
                "Distance too high for {} -> {}: {}",
                correct,
                typo,
                distance
            );

            let damerau_distance = damerau_levenshtein_distance(correct, typo);
            assert!(
                damerau_distance <= distance,
                "Damerau distance should be <= Levenshtein"
            );
        }
    }
}
