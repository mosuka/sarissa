//! Finite automaton support for query matching.
//!
//! This module provides Levenshtein automaton for efficient fuzzy matching.

use crate::error::Result;
use crate::lexical::core::terms::{TermStats, TermsEnum};
use crate::spelling::levenshtein::{damerau_levenshtein_distance, levenshtein_distance};

/// A simple Levenshtein automaton for fuzzy string matching.
///
/// This is a simplified implementation that checks if a candidate string
/// is within the specified edit distance from the pattern.
///
/// TODO: Implement a proper finite automaton using the Levenshtein automaton algorithm.
/// For now, this uses direct distance calculation, which is less efficient but simpler.
///
/// See Lucene's LevenshteinAutomata.java for the full implementation.
#[derive(Debug, Clone)]
pub struct LevenshteinAutomaton {
    /// The pattern string to match against
    pattern: String,
    /// Maximum edit distance
    max_edits: u32,
    /// Minimum prefix length that must match exactly
    prefix_length: usize,
    /// Whether to use Damerau-Levenshtein (transpositions count as 1 edit)
    transpositions: bool,
}

impl LevenshteinAutomaton {
    /// Create a new Levenshtein automaton.
    ///
    /// # Arguments
    ///
    /// * `pattern` - The string to match against
    /// * `max_edits` - Maximum edit distance (0, 1, or 2)
    /// * `prefix_length` - Required exact prefix length
    /// * `transpositions` - Whether transpositions count as single edits
    pub fn new(
        pattern: impl Into<String>,
        max_edits: u32,
        prefix_length: usize,
        transpositions: bool,
    ) -> Self {
        LevenshteinAutomaton {
            pattern: pattern.into(),
            max_edits,
            prefix_length,
            transpositions,
        }
    }

    /// Check if a candidate string matches the automaton.
    ///
    /// Returns true if the candidate is within max_edits edit distance from the pattern
    /// and satisfies the prefix requirement.
    pub fn matches(&self, candidate: &str) -> bool {
        // Check prefix requirement
        if self.prefix_length > 0 {
            let pattern_prefix = self
                .pattern
                .chars()
                .take(self.prefix_length)
                .collect::<String>();
            let candidate_prefix = candidate
                .chars()
                .take(self.prefix_length)
                .collect::<String>();

            if pattern_prefix != candidate_prefix {
                return false;
            }
        }

        // Calculate edit distance
        let distance = if self.transpositions {
            damerau_levenshtein_distance(&self.pattern, candidate) as u32
        } else {
            levenshtein_distance(&self.pattern, candidate) as u32
        };

        distance <= self.max_edits
    }

    /// Get the pattern string.
    pub fn pattern(&self) -> &str {
        &self.pattern
    }

    /// Get the maximum edit distance.
    pub fn max_edits(&self) -> u32 {
        self.max_edits
    }

    /// Get the prefix length.
    pub fn prefix_length(&self) -> usize {
        self.prefix_length
    }

    /// Check if transpositions are enabled.
    pub fn uses_transpositions(&self) -> bool {
        self.transpositions
    }

    /// Get the initial seek term for this automaton.
    ///
    /// This is useful for seeking to the starting position in a term dictionary.
    /// For a fuzzy query, we can start from the prefix (if any).
    pub fn initial_seek_term(&self) -> Option<String> {
        if self.prefix_length > 0 {
            Some(self.pattern.chars().take(self.prefix_length).collect())
        } else {
            None
        }
    }
}

/// A terms enum that filters terms using an automaton.
///
/// This wraps another TermsEnum and only yields terms that match the automaton.
pub struct AutomatonTermsEnum<T: TermsEnum> {
    /// The underlying terms enum
    inner: T,
    /// The automaton to filter with
    automaton: LevenshteinAutomaton,
    /// Maximum number of matching terms to return
    max_matches: Option<usize>,
    /// Number of matches found so far
    matches_found: usize,
}

impl<T: TermsEnum> AutomatonTermsEnum<T> {
    /// Create a new automaton terms enum.
    pub fn new(inner: T, automaton: LevenshteinAutomaton) -> Self {
        AutomatonTermsEnum {
            inner,
            automaton,
            max_matches: None,
            matches_found: 0,
        }
    }

    /// Set the maximum number of matching terms to return.
    pub fn with_max_matches(mut self, max_matches: usize) -> Self {
        self.max_matches = Some(max_matches);
        self
    }

    /// Get the automaton.
    pub fn automaton(&self) -> &LevenshteinAutomaton {
        &self.automaton
    }
}

impl<T: TermsEnum> TermsEnum for AutomatonTermsEnum<T> {
    fn next(&mut self) -> Result<Option<TermStats>> {
        // Check if we've reached the max matches limit
        if let Some(max) = self.max_matches
            && self.matches_found >= max
        {
            return Ok(None);
        }

        // Seek to the initial position if we haven't started yet
        if self.matches_found == 0
            && let Some(seek_term) = self.automaton.initial_seek_term()
        {
            self.inner.seek(&seek_term)?;
        }

        // Find the next matching term
        while let Some(term_stats) = self.inner.next()? {
            if self.automaton.matches(&term_stats.term) {
                self.matches_found += 1;
                return Ok(Some(term_stats));
            }

            // Optimization: if we've moved past possible matches, stop early
            // This works if the term dictionary is sorted
            if let Some(prefix) = self.automaton.initial_seek_term()
                && !term_stats.term.starts_with(&prefix)
            {
                // We've moved past all terms with the required prefix
                return Ok(None);
            }
        }

        Ok(None)
    }

    fn seek(&mut self, target: &str) -> Result<bool> {
        self.inner.seek(target)
    }

    fn seek_exact(&mut self, term: &str) -> Result<bool> {
        self.inner.seek_exact(term)
    }

    fn current(&self) -> Option<&TermStats> {
        self.inner.current()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_levenshtein_automaton() {
        let automaton = LevenshteinAutomaton::new("hello", 1, 0, true);

        assert!(automaton.matches("hello")); // exact match
        assert!(automaton.matches("helo")); // 1 deletion
        assert!(automaton.matches("hallo")); // 1 substitution
        assert!(automaton.matches("helllo")); // 1 insertion
        assert!(automaton.matches("ehllo")); // 1 transposition

        assert!(!automaton.matches("world")); // too different
        assert!(!automaton.matches("hi")); // too different (2 edits)
    }

    #[test]
    fn test_prefix_constraint() {
        let automaton = LevenshteinAutomaton::new("hello", 2, 2, true);

        assert!(automaton.matches("hello")); // exact match
        assert!(automaton.matches("heLLo")); // 2 edits, prefix "he" matches

        assert!(!automaton.matches("xello")); // prefix doesn't match
        assert!(!automaton.matches("world")); // prefix doesn't match
    }
}
