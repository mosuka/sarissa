//! Generic Levenshtein distance algorithms.
//!
//! This module provides efficient implementations of string edit distance algorithms
//! that can be used for various purposes including fuzzy matching, spell correction,
//! similarity scoring, and more.

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

/// Calculate Damerau-Levenshtein distance with a maximum threshold for early
/// termination. Mirrors [`levenshtein_distance_threshold`] but additionally
/// handles transpositions (adjacent character swaps count as a single edit).
///
/// Returns `None` if the distance exceeds the threshold.
///
/// Uses a 3-row sliding buffer (instead of the full N×M matrix used by
/// [`damerau_levenshtein_distance`]) because the transposition rule needs to
/// reference `matrix[i-2][j-2]`. This drops working set to O(min(N, M)) and
/// enables early termination when the per-row minimum exceeds the threshold.
#[allow(clippy::needless_range_loop)]
pub fn damerau_levenshtein_distance_threshold(
    s1: &str,
    s2: &str,
    threshold: usize,
) -> Option<usize> {
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

    // Use three rows: prev2 (i-2), prev (i-1), curr (i).
    let mut prev2_row = vec![0usize; len2 + 1];
    let mut prev_row = vec![0usize; len2 + 1];
    let mut curr_row = vec![0usize; len2 + 1];

    // Initialize first row (i = 0)
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

            // Check for transposition (needs i >= 2 and j >= 2)
            if i > 1
                && j > 1
                && s1_chars[i - 1] == s2_chars[j - 2]
                && s1_chars[i - 2] == s2_chars[j - 1]
            {
                curr_row[j] = min(curr_row[j], prev2_row[j - 2] + cost);
            }

            min_in_row = min(min_in_row, curr_row[j]);
        }

        // Early termination if minimum in row exceeds threshold
        if min_in_row > threshold {
            return None;
        }

        // Rotate rows: prev2 <- prev <- curr; reuse the old prev2 buffer as
        // the next curr to avoid reallocating.
        std::mem::swap(&mut prev2_row, &mut prev_row);
        std::mem::swap(&mut prev_row, &mut curr_row);
    }

    // After the final swap, `prev_row` holds the last computed row.
    let distance = prev_row[len2];
    if distance <= threshold {
        Some(distance)
    } else {
        None
    }
}

/// A matcher that pre-stores a query string for calculating Levenshtein distance
/// against multiple candidates.
///
/// The `distance_threshold` method delegates to [`levenshtein_distance_threshold`],
/// so it does not cache intermediate computation across calls.
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

    /// Calculate the Levenshtein distance between the stored query and
    /// `candidate`, returning `None` when the distance exceeds `threshold`
    /// (early termination).
    ///
    /// This method delegates directly to [`levenshtein_distance_threshold`],
    /// passing the pre-stored query string so callers do not need to supply
    /// it on every invocation. No intermediate computation is cached or
    /// reused across calls.
    ///
    /// # Arguments
    ///
    /// * `candidate` - The string to compare against the query.
    /// * `threshold` - Maximum acceptable distance. If the actual distance
    ///   exceeds this value, `None` is returned for early termination.
    ///
    /// # Returns
    ///
    /// `Some(distance)` if the distance is at most `threshold`, or `None`
    /// otherwise.
    pub fn distance_threshold(&self, candidate: &str, threshold: usize) -> Option<usize> {
        levenshtein_distance_threshold(&self.query, candidate, threshold)
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
    fn test_damerau_levenshtein_distance_threshold() {
        // Within threshold
        assert_eq!(
            damerau_levenshtein_distance_threshold("ab", "ba", 1),
            Some(1)
        );
        assert_eq!(
            damerau_levenshtein_distance_threshold("search", "serach", 1),
            Some(1)
        );
        assert_eq!(
            damerau_levenshtein_distance_threshold("kitten", "sitting", 3),
            Some(3)
        );

        // Exceeds threshold
        assert_eq!(
            damerau_levenshtein_distance_threshold("kitten", "sitting", 2),
            None
        );
        assert_eq!(
            damerau_levenshtein_distance_threshold("abc", "def", 2),
            None
        );

        // Edge cases
        assert_eq!(damerau_levenshtein_distance_threshold("", "", 0), Some(0));
        assert_eq!(damerau_levenshtein_distance_threshold("", "a", 1), Some(1));
        assert_eq!(damerau_levenshtein_distance_threshold("a", "", 1), Some(1));
        assert_eq!(damerau_levenshtein_distance_threshold("a", "", 0), None);

        // Length-difference early termination
        assert_eq!(damerau_levenshtein_distance_threshold("a", "abcd", 2), None);

        // Equivalence with the non-threshold variant when within threshold
        for (a, b) in &[
            ("ab", "ba"),
            ("search", "serach"),
            ("kitten", "sitting"),
            ("hello", "hallo"),
            ("programming", "programing"),
        ] {
            let full = damerau_levenshtein_distance(a, b);
            assert_eq!(
                damerau_levenshtein_distance_threshold(a, b, full),
                Some(full),
                "mismatch for ({a:?}, {b:?})"
            );
        }
    }

    #[test]
    fn test_levenshtein_matcher() {
        let matcher = LevenshteinMatcher::new("search".to_string());
        assert_eq!(matcher.distance_threshold("search", 2), Some(0));
        assert_eq!(matcher.distance_threshold("serach", 2), Some(2));
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
                "Distance too high for {correct} -> {typo}: {distance}"
            );

            let damerau_distance = damerau_levenshtein_distance(correct, typo);
            assert!(
                damerau_distance <= distance,
                "Damerau distance should be <= Levenshtein"
            );
        }
    }
}
