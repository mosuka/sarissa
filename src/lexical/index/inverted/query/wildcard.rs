//! Wildcard query implementation for pattern matching.

use std::fmt::Debug;
use std::sync::Arc;

use regex::Regex;

use crate::error::Result;
use crate::lexical::index::inverted::query::Query;
use crate::lexical::index::inverted::query::matcher::{EmptyMatcher, Matcher};
use crate::lexical::index::inverted::query::scorer::Scorer;
use crate::lexical::reader::LexicalIndexReader;

/// A query that matches documents containing terms that match a wildcard pattern.
///
/// Supports the following wildcards:
/// - `*` matches zero or more characters
/// - `?` matches exactly one character
/// - `\*` and `\?` match literal `*` and `?` characters
#[derive(Debug, Clone)]
pub struct WildcardQuery {
    /// The field to search in.
    field: String,
    /// The wildcard pattern.
    pattern: String,
    /// The compiled regex for matching.
    regex: Arc<Regex>,
    /// The boost factor for this query.
    boost: f32,
}

impl WildcardQuery {
    /// Create a new wildcard query.
    pub fn new<S: Into<String>>(field: S, pattern: S) -> Result<Self> {
        let field = field.into();
        let pattern = pattern.into();
        let regex = Self::compile_pattern(&pattern)?;

        Ok(WildcardQuery {
            field,
            pattern,
            regex: Arc::new(regex),
            boost: 1.0,
        })
    }

    /// Set the boost factor for this query.
    pub fn with_boost(mut self, boost: f32) -> Self {
        self.boost = boost;
        self
    }

    /// Get the field name.
    pub fn field(&self) -> &str {
        &self.field
    }

    /// Get the wildcard pattern.
    pub fn pattern(&self) -> &str {
        &self.pattern
    }

    /// Compile a wildcard pattern into a regex.
    fn compile_pattern(pattern: &str) -> Result<Regex> {
        let mut regex_pattern = String::new();
        regex_pattern.push('^'); // Match from the beginning

        let chars: Vec<char> = pattern.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            match chars[i] {
                '\\' => {
                    // Handle escape sequences
                    if i + 1 < chars.len() {
                        match chars[i + 1] {
                            '*' => {
                                regex_pattern.push_str("\\*");
                                i += 1; // Skip the escaped character
                            }
                            '?' => {
                                regex_pattern.push_str("\\?");
                                i += 1; // Skip the escaped character
                            }
                            c => {
                                // Other escaped characters
                                regex_pattern.push('\\');
                                regex_pattern.push(c);
                                i += 1; // Skip the escaped character
                            }
                        }
                    } else {
                        regex_pattern.push('\\');
                    }
                }
                '*' => {
                    regex_pattern.push_str(".*");
                }
                '?' => {
                    regex_pattern.push('.');
                }
                // Regex special characters that need escaping
                '^' | '$' | '.' | '+' | '(' | ')' | '[' | ']' | '{' | '}' | '|' => {
                    regex_pattern.push('\\');
                    regex_pattern.push(chars[i]);
                }
                c => {
                    regex_pattern.push(c);
                }
            }
            i += 1;
        }

        regex_pattern.push('$'); // Match to the end

        Regex::new(&regex_pattern).map_err(|e| {
            crate::error::YatagarasuError::analysis(format!("Invalid wildcard pattern: {e}"))
        })
    }

    /// Check if a term matches the wildcard pattern.
    pub fn matches(&self, term: &str) -> bool {
        self.regex.is_match(term)
    }

    /// Calculate pattern complexity for scoring.
    fn calculate_pattern_complexity(&self) -> f32 {
        let mut complexity = 1.0;
        let pattern_chars: Vec<char> = self.pattern.chars().collect();

        // Base complexity factors
        let asterisk_count = pattern_chars.iter().filter(|&&c| c == '*').count();
        let question_count = pattern_chars.iter().filter(|&&c| c == '?').count();
        let literal_count = pattern_chars.len() - asterisk_count - question_count;

        // Leading wildcard increases complexity (slower)
        if pattern_chars.first() == Some(&'*') {
            complexity *= 0.7; // Leading wildcards are less precise
        }

        // Trailing wildcard is more efficient
        if pattern_chars.last() == Some(&'*') && pattern_chars.first() != Some(&'*') {
            complexity *= 1.2; // Prefix matching is efficient
        }

        // More literals make patterns more selective
        complexity *= 1.0 + (literal_count as f32 * 0.1);

        // Multiple wildcards increase complexity
        complexity *= 1.0 - ((asterisk_count + question_count) as f32 * 0.05);

        // Exact patterns (no wildcards) are most precise
        if asterisk_count == 0 && question_count == 0 {
            complexity *= 1.5;
        }

        complexity.clamp(0.1, 2.0)
    }
}

impl Query for WildcardQuery {
    fn matcher(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Matcher>> {
        // Schema-less: no field validation needed
        // All fields are treated as text fields for wildcard matching
        let mut matching_doc_ids = Vec::new();

        // Iterate through all documents and find matches
        for doc_id in 0..reader.doc_count() {
            if let Ok(Some(doc)) = reader.document(doc_id)
                && let Some(field_value) = doc.get_field(&self.field)
                && let Some(text) = field_value.value.as_text()
            {
                // Schema-less: token-based matching for all text fields
                let tokens: Vec<&str> = text.split_whitespace().collect();
                let matches = tokens.iter().any(|token| self.regex.is_match(token));

                if matches {
                    matching_doc_ids.push(doc_id);
                }
            }
        }

        if matching_doc_ids.is_empty() {
            Ok(Box::new(EmptyMatcher::new()))
        } else {
            Ok(Box::new(
                crate::lexical::index::inverted::query::matcher::PreComputedMatcher::new(
                    matching_doc_ids,
                ),
            ))
        }
    }

    fn scorer(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Scorer>> {
        // Calculate actual matching statistics for better scoring
        let total_docs = reader.doc_count();
        let mut actual_doc_freq = 0u64;
        let mut total_term_freq = 0u64;
        let mut field_lengths = Vec::new();

        // Schema-less: treat all fields as text fields for wildcard matching
        // Count actual matches and collect field statistics
        for doc_id in 0..total_docs {
            if let Ok(Some(doc)) = reader.document(doc_id)
                && let Some(field_value) = doc.get_field(&self.field)
                && let Some(text) = field_value.value.as_text()
            {
                // Token-based matching with count
                let tokens: Vec<&str> = text.split_whitespace().collect();
                let match_count = tokens
                    .iter()
                    .filter(|token| self.regex.is_match(token))
                    .count();
                let matches = match_count > 0;
                let field_len = tokens.len();

                if matches {
                    actual_doc_freq += 1;
                    total_term_freq += match_count as u64;
                }
                field_lengths.push(field_len);
            }
        }

        // Calculate average field length
        let avg_field_length = if field_lengths.is_empty() {
            10.0
        } else {
            field_lengths.iter().sum::<usize>() as f64 / field_lengths.len() as f64
        };

        // Create dynamic scorer based on pattern characteristics
        let pattern_complexity = self.calculate_pattern_complexity();
        let selectivity_boost = if actual_doc_freq > 0 {
            // More selective patterns get higher base scores
            (total_docs as f32 / actual_doc_freq as f32).ln().max(0.1)
        } else {
            0.1
        };

        Ok(Box::new(WildcardScorer::new(
            actual_doc_freq.max(1),
            total_term_freq.max(1),
            total_docs,
            avg_field_length,
            pattern_complexity,
            selectivity_boost,
            self.boost,
        )))
    }

    fn boost(&self) -> f32 {
        self.boost
    }

    fn set_boost(&mut self, boost: f32) {
        self.boost = boost;
    }

    fn description(&self) -> String {
        if self.boost == 1.0 {
            format!("{}:{}", self.field, self.pattern)
        } else {
            format!("{}:{}^{}", self.field, self.pattern, self.boost)
        }
    }

    fn clone_box(&self) -> Box<dyn Query> {
        Box::new(self.clone())
    }

    fn is_empty(&self, _reader: &dyn LexicalIndexReader) -> Result<bool> {
        Ok(self.pattern.is_empty())
    }

    fn cost(&self, _reader: &dyn LexicalIndexReader) -> Result<u64> {
        Ok(1000) // Wildcards are expensive
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Specialized scorer for wildcard queries.
#[derive(Debug, Clone)]
pub struct WildcardScorer {
    /// Document frequency of matches.
    doc_freq: u64,
    /// Total term frequency across all documents.
    total_term_freq: u64,
    /// Total number of documents in the index.
    total_docs: u64,
    /// Average field length.
    avg_field_length: f64,
    /// Pattern complexity factor.
    complexity: f32,
    /// Pattern selectivity boost.
    selectivity: f32,
    /// Boost factor.
    boost: f32,
}

impl WildcardScorer {
    /// Create a new wildcard scorer.
    pub fn new(
        doc_freq: u64,
        total_term_freq: u64,
        total_docs: u64,
        avg_field_length: f64,
        complexity: f32,
        selectivity: f32,
        boost: f32,
    ) -> Self {
        WildcardScorer {
            doc_freq,
            total_term_freq,
            total_docs,
            avg_field_length,
            complexity,
            selectivity,
            boost,
        }
    }

    /// Calculate IDF component with wildcard adjustments.
    fn calculate_idf(&self) -> f32 {
        if self.doc_freq == 0 || self.total_docs == 0 {
            return 0.1;
        }

        let n = self.total_docs as f32;
        let df = self.doc_freq as f32;

        // Standard IDF with wildcard selectivity adjustment
        let base_idf = ((n - df + 0.5) / (df + 0.5)).ln().max(0.1);

        // Apply selectivity boost for more discriminating patterns
        base_idf * self.selectivity
    }

    /// Calculate TF component with pattern complexity.
    fn calculate_tf(&self, term_freq: f32) -> f32 {
        if term_freq == 0.0 {
            return 0.0;
        }

        // BM25 TF calculation with complexity adjustment
        let k1 = 1.2;
        let b = 0.75;
        let field_length = self.avg_field_length as f32;
        let norm_factor = 1.0 - b + b * (field_length / self.avg_field_length as f32);

        let tf_component = (term_freq * (k1 + 1.0)) / (term_freq + k1 * norm_factor);

        // Apply complexity factor - more complex patterns get slight boost
        tf_component * self.complexity
    }
}

impl crate::lexical::index::inverted::query::scorer::Scorer for WildcardScorer {
    fn score(&self, _doc_id: u64, term_freq: f32, _field_length: Option<f32>) -> f32 {
        if self.doc_freq == 0 || self.total_docs == 0 {
            return 0.0;
        }

        let idf = self.calculate_idf();
        let tf = self.calculate_tf(term_freq);

        // Final score with document frequency variation
        let doc_factor = 1.0 + (self.total_term_freq as f32 / self.doc_freq.max(1) as f32) * 0.1;

        self.boost * idf * tf * doc_factor
    }

    fn boost(&self) -> f32 {
        self.boost
    }

    fn set_boost(&mut self, boost: f32) {
        self.boost = boost;
    }

    fn max_score(&self) -> f32 {
        let max_idf = self.calculate_idf();
        let max_tf = self.calculate_tf(10.0); // Assume max term frequency
        let max_doc_factor =
            1.0 + (self.total_term_freq as f32 / self.doc_freq.max(1) as f32) * 0.1;

        self.boost * max_idf * max_tf * max_doc_factor
    }

    fn name(&self) -> &'static str {
        "WildcardScorer"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wildcard_query_creation() {
        let query = WildcardQuery::new("content", "hello*").unwrap();

        assert_eq!(query.field(), "content");
        assert_eq!(query.pattern(), "hello*");
        assert_eq!(query.boost(), 1.0);
    }

    #[test]
    fn test_wildcard_query_with_boost() {
        let query = WildcardQuery::new("content", "test?")
            .unwrap()
            .with_boost(2.5);

        assert_eq!(query.boost(), 2.5);
    }

    #[test]
    fn test_wildcard_pattern_compilation() {
        // Test simple wildcard
        let query = WildcardQuery::new("field", "hello*").unwrap();
        assert!(query.matches("hello"));
        assert!(query.matches("helloworld"));
        assert!(!query.matches("hell"));

        // Test question mark
        let query = WildcardQuery::new("field", "h?llo").unwrap();
        assert!(query.matches("hello"));
        assert!(query.matches("hallo"));
        assert!(query.matches("hxllo"));
        assert!(!query.matches("heello"));

        // Test combination
        let query = WildcardQuery::new("field", "h*l?o").unwrap();
        assert!(query.matches("hello"));
        assert!(query.matches("hallo"));
        assert!(query.matches("heeello"));
        assert!(query.matches("hllo")); // Actually matches because ? can be 'l'
    }

    #[test]
    fn test_escaped_wildcards() {
        let query = WildcardQuery::new("field", "hello\\*world").unwrap();
        assert!(query.matches("hello*world"));
        assert!(!query.matches("helloworld"));
        assert!(!query.matches("hello123world"));

        let query = WildcardQuery::new("field", "hello\\?world").unwrap();
        assert!(query.matches("hello?world"));
        assert!(!query.matches("helloxworld"));
    }

    #[test]
    fn test_special_regex_characters() {
        let query = WildcardQuery::new("field", "hello.world").unwrap();
        assert!(query.matches("hello.world"));
        assert!(!query.matches("helloxworld"));

        let query = WildcardQuery::new("field", "hello+world").unwrap();
        assert!(query.matches("hello+world"));
        assert!(!query.matches("helloworld"));
    }
}
