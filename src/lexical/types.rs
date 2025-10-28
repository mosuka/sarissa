//! Type definitions for lexical search operations.
//!
//! This module contains common type definitions used across the lexical search module,
//! mirroring the structure of vector::types for consistency.

use crate::query::query::Query;

/// Sort order for search results.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortOrder {
    /// Ascending order (lowest to highest).
    Asc,
    /// Descending order (highest to lowest).
    Desc,
}

/// Field to sort search results by.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum SortField {
    /// Sort by relevance score (default).
    #[default]
    Score,
    /// Sort by a document field value.
    Field {
        /// Field name to sort by.
        name: String,
        /// Sort order.
        order: SortOrder,
    },
}

/// Configuration for search operations.
#[derive(Debug, Clone)]
pub struct LexicalSearchParams {
    /// Maximum number of documents to return.
    pub max_docs: usize,
    /// Minimum score threshold.
    pub min_score: f32,
    /// Whether to load document content.
    pub load_documents: bool,
    /// Timeout for search operations in milliseconds.
    pub timeout_ms: Option<u64>,
    /// Enable parallel search for better performance on multi-core systems.
    pub parallel: bool,
    /// Sort results by field or score.
    pub sort_by: SortField,
}

impl Default for LexicalSearchParams {
    fn default() -> Self {
        LexicalSearchParams {
            max_docs: 10,
            min_score: 0.0,
            load_documents: true,
            timeout_ms: None,
            parallel: false,
            sort_by: SortField::default(),
        }
    }
}

/// Search request containing query and configuration.
#[derive(Debug)]
pub struct LexicalSearchRequest {
    /// The query to execute.
    pub query: Box<dyn Query>,
    /// Search configuration.
    pub params: LexicalSearchParams,
}

impl Clone for LexicalSearchRequest {
    fn clone(&self) -> Self {
        LexicalSearchRequest {
            query: self.query.clone_box(),
            params: self.params.clone(),
        }
    }
}

impl LexicalSearchRequest {
    /// Create a new search request.
    pub fn new(query: Box<dyn Query>) -> Self {
        LexicalSearchRequest {
            query,
            params: LexicalSearchParams::default(),
        }
    }

    /// Set the maximum number of documents to return.
    pub fn max_docs(mut self, max_docs: usize) -> Self {
        self.params.max_docs = max_docs;
        self
    }

    /// Set the minimum score threshold.
    pub fn min_score(mut self, min_score: f32) -> Self {
        self.params.min_score = min_score;
        self
    }

    /// Set whether to load document content.
    pub fn load_documents(mut self, load_documents: bool) -> Self {
        self.params.load_documents = load_documents;
        self
    }

    /// Set the search timeout.
    pub fn timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.params.timeout_ms = Some(timeout_ms);
        self
    }

    /// Enable parallel search.
    pub fn parallel(mut self, parallel: bool) -> Self {
        self.params.parallel = parallel;
        self
    }

    /// Sort results by a field in ascending order.
    pub fn sort_by_field_asc(mut self, field: &str) -> Self {
        self.params.sort_by = SortField::Field {
            name: field.to_string(),
            order: SortOrder::Asc,
        };
        self
    }

    /// Sort results by a field in descending order.
    pub fn sort_by_field_desc(mut self, field: &str) -> Self {
        self.params.sort_by = SortField::Field {
            name: field.to_string(),
            order: SortOrder::Desc,
        };
        self
    }

    /// Sort results by relevance score (default).
    pub fn sort_by_score(mut self) -> Self {
        self.params.sort_by = SortField::Score;
        self
    }
}

/// Information about a term in the index.
#[derive(Debug, Clone)]
pub struct ReaderTermInfo {
    /// The field name.
    pub field: String,

    /// The term text.
    pub term: String,

    /// Number of documents containing this term.
    pub doc_freq: u64,

    /// Total number of occurrences of this term.
    pub total_freq: u64,

    /// Position of the term in the posting list.
    pub posting_offset: u64,

    /// Size of the posting list in bytes.
    pub posting_size: u64,
}

/// Statistics about a field in the index.
#[derive(Debug, Clone)]
pub struct FieldStats {
    /// The field name.
    pub field: String,

    /// Number of unique terms in this field.
    pub unique_terms: u64,

    /// Total number of term occurrences.
    pub total_terms: u64,

    /// Number of documents with this field.
    pub doc_count: u64,

    /// Average field length.
    pub avg_length: f64,

    /// Minimum field length.
    pub min_length: u64,

    /// Maximum field length.
    pub max_length: u64,
}

/// Simplified field statistics for query scoring.
#[derive(Debug, Clone)]
pub struct FieldStatistics {
    /// Average field length.
    pub avg_field_length: f64,

    /// Number of documents with this field.
    pub doc_count: u64,

    /// Total number of terms.
    pub total_terms: u64,
}
