//! Phrase query implementation for exact phrase matching.

use crate::error::Result;
use crate::index::reader::IndexReader;
use crate::query::matcher::{EmptyMatcher, Matcher};
use crate::query::scorer::{BM25Scorer, Scorer};
use crate::query::Query;
use std::fmt::Debug;

/// A query that matches documents containing an exact phrase.
///
/// A phrase query finds documents where the specified terms appear
/// in the exact order with no other terms between them.
#[derive(Debug, Clone)]
pub struct PhraseQuery {
    /// The field to search in.
    field: String,
    /// The terms that make up the phrase, in order.
    terms: Vec<String>,
    /// The boost factor for this query.
    boost: f32,
    /// Optional slop - maximum allowed distance between terms (0 = exact phrase).
    slop: u32,
}

impl PhraseQuery {
    /// Create a new phrase query.
    pub fn new<S: Into<String>>(field: S, terms: Vec<String>) -> Self {
        PhraseQuery {
            field: field.into(),
            terms,
            boost: 1.0,
            slop: 0,
        }
    }

    /// Create a phrase query from a phrase string.
    pub fn from_phrase<S: Into<String>>(field: S, phrase: &str) -> Self {
        let terms: Vec<String> = phrase.split_whitespace().map(|s| s.to_string()).collect();
        Self::new(field, terms)
    }

    /// Set the boost factor for this query.
    pub fn with_boost(mut self, boost: f32) -> Self {
        self.boost = boost;
        self
    }

    /// Set the slop (maximum distance between terms).
    ///
    /// A slop of 0 means exact phrase match.
    /// A slop of 1 allows one word between phrase terms.
    pub fn with_slop(mut self, slop: u32) -> Self {
        self.slop = slop;
        self
    }

    /// Get the field name.
    pub fn field(&self) -> &str {
        &self.field
    }

    /// Get the phrase terms.
    pub fn terms(&self) -> &[String] {
        &self.terms
    }

    /// Get the slop value.
    pub fn slop(&self) -> u32 {
        self.slop
    }
}

impl Query for PhraseQuery {
    fn matcher(&self, reader: &dyn IndexReader) -> Result<Box<dyn Matcher>> {
        if self.terms.is_empty() {
            return Ok(Box::new(EmptyMatcher::new()));
        }

        // Simplified implementation: check if all terms exist in the field
        // This is not a true phrase query but at least finds documents with all terms
        use crate::query::boolean::BooleanQuery;
        use crate::query::term::TermQuery;
        
        let mut bool_query = BooleanQuery::new();
        for term in &self.terms {
            bool_query.add_must(Box::new(TermQuery::new(&self.field, term)));
        }
        
        bool_query.matcher(reader)
    }

    fn scorer(&self, _reader: &dyn IndexReader) -> Result<Box<dyn Scorer>> {
        // Create a BM25 scorer for phrase queries
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
            "PhraseQuery(field:{}, terms:{:?}, slop:{})",
            self.field, self.terms, self.slop
        )
    }

    fn clone_box(&self) -> Box<dyn Query> {
        Box::new(self.clone())
    }

    fn is_empty(&self, _reader: &dyn IndexReader) -> Result<bool> {
        Ok(self.terms.is_empty())
    }

    fn cost(&self, _reader: &dyn IndexReader) -> Result<u64> {
        Ok(self.terms.len() as u64 * 100) // Rough estimate
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phrase_query_creation() {
        let query = PhraseQuery::new("content", vec!["hello".to_string(), "world".to_string()]);

        assert_eq!(query.field(), "content");
        assert_eq!(query.terms(), &["hello", "world"]);
        assert_eq!(query.slop(), 0);
        assert_eq!(query.boost(), 1.0);
    }

    #[test]
    fn test_phrase_query_from_phrase() {
        let query = PhraseQuery::from_phrase("content", "hello world test");

        assert_eq!(query.field(), "content");
        assert_eq!(query.terms(), &["hello", "world", "test"]);
    }

    #[test]
    fn test_phrase_query_with_boost() {
        let query = PhraseQuery::new("content", vec!["hello".to_string()]).with_boost(2.5);

        assert_eq!(query.boost(), 2.5);
    }

    #[test]
    fn test_phrase_query_with_slop() {
        let query = PhraseQuery::new("content", vec!["hello".to_string(), "world".to_string()])
            .with_slop(2);

        assert_eq!(query.slop(), 2);
    }
}
