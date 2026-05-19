//! Finite automaton support for query matching.
//!
//! This module provides Levenshtein automaton for efficient fuzzy matching.

use crate::error::Result;
use crate::lexical::index::inverted::core::terms::{TermStats, TermsEnum};
use crate::util::levenshtein::{
    damerau_levenshtein_distance_threshold, levenshtein_distance_threshold,
};

/// A simple Levenshtein automaton for fuzzy string matching.
///
/// This is a simplified implementation that checks if a candidate string
/// is within the specified edit distance from the pattern.
///
use regex::Regex;

/// A trait for finite automata used in term filtering.
pub trait Automaton: Send + Sync {
    /// Check if a candidate string matches the automaton.
    fn matches(&self, candidate: &str) -> bool;

    /// Get the initial seek term for this automaton.
    ///
    /// This is useful for seeking to the starting position in a term dictionary.
    fn initial_seek_term(&self) -> Option<String>;
}

/// A simple Levenshtein automaton for fuzzy string matching.
///
/// This is a simplified implementation that checks if a candidate string
/// is within the specified edit distance from the pattern.
#[derive(Debug, Clone)]
pub struct LevenshteinAutomaton {
    /// The pattern string to match against
    pattern: String,
    /// Precomputed prefix of `pattern` of length `prefix_length`, if any. Stored
    /// to avoid recomputing `pattern.chars().take(prefix_length).collect()` on
    /// every `matches` call — that allocation showed up under inlined
    /// `AutomatonTermsEnum::next` as a per-term hot spot (see #530).
    pattern_prefix: Option<String>,
    /// Maximum edit distance
    max_edits: u32,
    /// Minimum prefix length that must match exactly
    prefix_length: usize,
    /// Whether to use Damerau-Levenshtein (transpositions count as 1 edit)
    transpositions: bool,
}

impl LevenshteinAutomaton {
    /// Create a new Levenshtein automaton.
    pub fn new(
        pattern: impl Into<String>,
        max_edits: u32,
        prefix_length: usize,
        transpositions: bool,
    ) -> Self {
        let pattern: String = pattern.into();
        let pattern_prefix = if prefix_length > 0 {
            Some(pattern.chars().take(prefix_length).collect())
        } else {
            None
        };
        LevenshteinAutomaton {
            pattern,
            pattern_prefix,
            max_edits,
            prefix_length,
            transpositions,
        }
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
}

impl Automaton for LevenshteinAutomaton {
    fn matches(&self, candidate: &str) -> bool {
        // Check prefix requirement against the precomputed prefix.
        if let Some(prefix) = &self.pattern_prefix
            && !candidate.starts_with(prefix.as_str())
        {
            return false;
        }

        // Calculate edit distance using the threshold-aware variants. They use
        // a 2-row (or 3-row for Damerau) sliding buffer instead of the full
        // N × M matrix, terminate early when the per-row minimum exceeds the
        // threshold, and avoid heap allocations for the matrix on the
        // non-matching majority of candidates.
        let threshold = self.max_edits as usize;
        let distance = if self.transpositions {
            damerau_levenshtein_distance_threshold(&self.pattern, candidate, threshold)
        } else {
            levenshtein_distance_threshold(&self.pattern, candidate, threshold)
        };
        distance.is_some()
    }

    fn initial_seek_term(&self) -> Option<String> {
        self.pattern_prefix.clone()
    }
}

/// An automaton backed by a regular expression.
#[derive(Debug, Clone)]
pub struct RegexAutomaton {
    regex: Regex,
    #[allow(dead_code)]
    pattern: String,
    initial_seek_term: Option<String>,
}

impl RegexAutomaton {
    /// Create a new regex automaton.
    pub fn new(pattern: &str) -> Result<Self> {
        let regex = Regex::new(pattern).map_err(|e| {
            crate::error::LaurusError::analysis(format!("Invalid regexp pattern: {e}"))
        })?;

        // Extract prefix optimization
        let initial_seek_term = Self::extract_prefix(pattern);

        Ok(RegexAutomaton {
            regex,
            pattern: pattern.to_string(),
            initial_seek_term,
        })
    }

    /// Create from existing compiled regex.
    pub fn from_regex(regex: Regex, pattern: String) -> Self {
        let initial_seek_term = Self::extract_prefix(&pattern);
        RegexAutomaton {
            regex,
            pattern,
            initial_seek_term,
        }
    }

    fn extract_prefix(pattern: &str) -> Option<String> {
        let mut chars = pattern.chars();
        if chars.next() != Some('^') {
            return None;
        }

        let mut prefix = String::new();
        let mut escaped = false;

        for c in chars {
            if escaped {
                prefix.push(c);
                escaped = false;
            } else {
                match c {
                    '\\' => escaped = true,
                    '.' | '+' | '*' | '?' | '(' | ')' | '|' | '[' | ']' | '{' | '}' | '^' | '$' => {
                        break;
                    }
                    _ => prefix.push(c),
                }
            }
        }

        if prefix.is_empty() {
            None
        } else {
            Some(prefix)
        }
    }
}

impl Automaton for RegexAutomaton {
    fn matches(&self, candidate: &str) -> bool {
        self.regex.is_match(candidate)
    }

    fn initial_seek_term(&self) -> Option<String> {
        self.initial_seek_term.clone()
    }
}

/// A terms enum that filters terms using an automaton.
///
/// This wraps another TermsEnum and only yields terms that match the automaton.
pub struct AutomatonTermsEnum<T: TermsEnum, A: Automaton> {
    /// The underlying terms enum
    inner: T,
    /// The automaton to filter with
    automaton: A,
    /// Maximum number of matching terms to return
    max_matches: Option<usize>,
    /// Number of matches found so far
    matches_found: usize,
}

impl<T: TermsEnum, A: Automaton> AutomatonTermsEnum<T, A> {
    /// Create a new automaton terms enum.
    pub fn new(inner: T, automaton: A) -> Self {
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
    pub fn automaton(&self) -> &A {
        &self.automaton
    }
}

impl<T: TermsEnum, A: Automaton> TermsEnum for AutomatonTermsEnum<T, A> {
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
            let _ = self.inner.seek(&seek_term)?;
            // Note: We ignore the bool result of seek because we need to check matches anyway.
            // If seek returns false (exact match not found), it still positions at the insertion point,
            // which is correct for iterating forward.
        }

        // Find the next matching term
        while let Some(term_stats) = self.inner.next()? {
            if self.automaton.matches(&term_stats.term) {
                self.matches_found += 1;
                return Ok(Some(term_stats));
            }

            // Optimization: if we've moved past possible matches, stop early
            // This works if the term dictionary is sorted and we have a prefix constraint
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
