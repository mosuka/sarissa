//! Wildcard query implementation for pattern matching.

use crate::error::Result;
use crate::index::reader::IndexReader;
use crate::query::matcher::{EmptyMatcher, Matcher};
use crate::query::scorer::{BM25Scorer, Scorer};
use crate::query::Query;
use regex::Regex;
use std::fmt::Debug;
use std::sync::Arc;

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
            crate::error::SarissaError::analysis(format!("Invalid wildcard pattern: {e}"))
        })
    }

    /// Check if a term matches the wildcard pattern.
    pub fn matches(&self, term: &str) -> bool {
        self.regex.is_match(term)
    }
}

impl Query for WildcardQuery {
    fn matcher(&self, _reader: &dyn IndexReader) -> Result<Box<dyn Matcher>> {
        // For now, return an empty matcher - full implementation would
        // require term enumeration from the index
        Ok(Box::new(EmptyMatcher::new()))
    }

    fn scorer(&self, _reader: &dyn IndexReader) -> Result<Box<dyn Scorer>> {
        Ok(Box::new(BM25Scorer::new(1, 1, 1, 1.0, 1, self.boost)))
    }

    fn boost(&self) -> f32 {
        self.boost
    }

    fn set_boost(&mut self, boost: f32) {
        self.boost = boost;
    }

    fn description(&self) -> String {
        format!(
            "WildcardQuery(field:{}, pattern:{})",
            self.field, self.pattern
        )
    }

    fn clone_box(&self) -> Box<dyn Query> {
        Box::new(self.clone())
    }

    fn is_empty(&self, _reader: &dyn IndexReader) -> Result<bool> {
        Ok(self.pattern.is_empty())
    }

    fn cost(&self, _reader: &dyn IndexReader) -> Result<u64> {
        Ok(1000) // Wildcards are expensive
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
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
