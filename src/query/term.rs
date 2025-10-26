//! Term query implementation for exact term matching.

use crate::error::Result;
use crate::lexical::reader::IndexReader;
use crate::query::matcher::{EmptyMatcher, Matcher, PostingMatcher};
use crate::query::query::Query;
use crate::query::scorer::{BM25Scorer, Scorer};

/// A query that matches documents containing a specific term.
#[derive(Debug, Clone)]
pub struct TermQuery {
    /// The field to search in.
    field: String,
    /// The term to search for.
    term: String,
    /// The boost factor for this query.
    boost: f32,
}

impl TermQuery {
    /// Create a new term query.
    ///
    /// Like Lucene, TermQuery performs exact matching and does NOT analyze the term.
    /// The term should already be in the normalized form (e.g., lowercased).
    /// Use a query parser or analyzer to normalize query strings before creating TermQuery.
    pub fn new<F, T>(field: F, term: T) -> Self
    where
        F: Into<String>,
        T: Into<String>,
    {
        TermQuery {
            field: field.into(),
            term: term.into(),
            boost: 1.0,
        }
    }

    /// Get the field name.
    pub fn field(&self) -> &str {
        &self.field
    }

    /// Get the term.
    pub fn term(&self) -> &str {
        &self.term
    }

    /// Set the boost factor.
    pub fn with_boost(mut self, boost: f32) -> Self {
        self.boost = boost;
        self
    }
}

impl Query for TermQuery {
    fn matcher(&self, reader: &dyn IndexReader) -> Result<Box<dyn Matcher>> {
        // Schema-less: no field validation needed
        // Try to get posting list for this term
        match reader.postings(&self.field, &self.term)? {
            Some(posting_iter) => {
                // Create a matcher from the posting iterator
                Ok(Box::new(PostingMatcher::new(posting_iter)))
            }
            None => {
                // Term not found in index
                Ok(Box::new(EmptyMatcher::new()))
            }
        }
    }

    fn scorer(&self, reader: &dyn IndexReader) -> Result<Box<dyn Scorer>> {
        // Get term information for BM25 scoring
        let term_info = reader.term_info(&self.field, &self.term)?;
        let field_stats = reader.field_stats(&self.field)?;

        match (term_info, field_stats) {
            (Some(term_info), Some(field_stats)) => {
                let scorer = BM25Scorer::new(
                    term_info.doc_freq,
                    term_info.total_freq,
                    field_stats.doc_count,
                    field_stats.avg_length,
                    reader.doc_count(),
                    self.boost,
                );
                Ok(Box::new(scorer))
            }
            _ => {
                // Term or field not found, return a scorer with zero scores
                let scorer = BM25Scorer::new(0, 0, 0, 0.0, 0, self.boost);
                Ok(Box::new(scorer))
            }
        }
    }

    fn boost(&self) -> f32 {
        self.boost
    }

    fn set_boost(&mut self, boost: f32) {
        self.boost = boost;
    }

    fn description(&self) -> String {
        if self.boost == 1.0 {
            format!("{}:{}", self.field, self.term)
        } else {
            format!("{}:{}^{}", self.field, self.term, self.boost)
        }
    }

    fn clone_box(&self) -> Box<dyn Query> {
        Box::new(self.clone())
    }

    fn is_empty(&self, reader: &dyn IndexReader) -> Result<bool> {
        // Schema-less: no field validation needed
        match reader.term_info(&self.field, &self.term)? {
            Some(term_info) => Ok(term_info.doc_freq == 0),
            None => Ok(true),
        }
    }

    fn cost(&self, reader: &dyn IndexReader) -> Result<u64> {
        match reader.term_info(&self.field, &self.term)? {
            Some(term_info) => Ok(term_info.doc_freq),
            None => Ok(0),
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn field(&self) -> Option<&str> {
        Some(&self.field)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexical::index::reader::inverted::{InvertedIndexReader, InvertedIndexReaderConfig};
    use crate::storage::memory::MemoryStorage;
    use crate::storage::{FileStorageConfig, MemoryStorageConfig};
    use std::sync::Arc;

    #[allow(dead_code)]
    #[test]
    fn test_term_query_creation() {
        let query = TermQuery::new("title", "hello");

        assert_eq!(query.field(), "title");
        assert_eq!(query.term(), "hello");
        assert_eq!(query.boost(), 1.0);
        assert_eq!(query.description(), "title:hello");
    }

    #[test]
    fn test_term_query_with_boost() {
        let query = TermQuery::new("title", "hello").with_boost(2.0);

        assert_eq!(query.boost(), 2.0);
        assert_eq!(query.description(), "title:hello^2");
    }

    #[test]
    fn test_term_query_matcher() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let reader =
            InvertedIndexReader::new(vec![], storage, InvertedIndexReaderConfig::default())
                .unwrap();

        let query = TermQuery::new("title", "hello");

        // Should create a matcher even for non-existent terms
        let matcher = query.matcher(&reader).unwrap();
        assert!(matcher.is_exhausted() || matcher.doc_id() != u64::MAX);
    }

    #[test]
    fn test_term_query_scorer() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let reader =
            InvertedIndexReader::new(vec![], storage, InvertedIndexReaderConfig::default())
                .unwrap();

        let query = TermQuery::new("title", "hello");

        // Should create a scorer even for non-existent terms
        let scorer = query.scorer(&reader).unwrap();
        // The scorer should handle missing terms gracefully
        assert!(scorer.score(0, 1.0, None) >= 0.0);
    }

    #[test]
    fn test_term_query_is_empty() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let reader =
            InvertedIndexReader::new(vec![], storage, InvertedIndexReaderConfig::default())
                .unwrap();

        let query = TermQuery::new("title", "hello");

        // Should be empty for non-existent terms
        assert!(query.is_empty(&reader).unwrap());

        // Should be empty for non-existent fields
        let query = TermQuery::new("nonexistent", "hello");
        assert!(query.is_empty(&reader).unwrap());
    }

    #[test]
    fn test_term_query_cost() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let reader =
            InvertedIndexReader::new(vec![], storage, InvertedIndexReaderConfig::default())
                .unwrap();

        let query = TermQuery::new("title", "hello");

        // Should return 0 cost for non-existent terms
        assert_eq!(query.cost(&reader).unwrap(), 0);
    }

    #[test]
    fn test_term_query_clone() {
        let query = TermQuery::new("title", "hello").with_boost(2.0);
        let cloned = query.clone_box();

        assert_eq!(cloned.description(), "title:hello^2");
        assert_eq!(cloned.boost(), 2.0);
    }
}
