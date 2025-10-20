//! Full-text search execution and result processing.
//!
//! This module handles all read operations and search execution:
//! - Query execution
//! - Scoring and ranking
//! - Result collection and processing
//! - Faceting and aggregation
//! - Highlighting and similarity

pub mod advanced_reader;
pub mod engine;
pub mod facet;
pub mod highlight;
pub mod result_processor;
pub mod scoring;
pub mod searcher;
pub mod similarity;
pub mod spell_corrected;

use crate::error::Result;
use crate::query::SearchResults;
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
pub struct SearchConfig {
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

impl Default for SearchConfig {
    fn default() -> Self {
        SearchConfig {
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
pub struct SearchRequest {
    /// The query to execute.
    pub query: Box<dyn Query>,
    /// Search configuration.
    pub config: SearchConfig,
}

impl Clone for SearchRequest {
    fn clone(&self) -> Self {
        SearchRequest {
            query: self.query.clone_box(),
            config: self.config.clone(),
        }
    }
}

impl SearchRequest {
    /// Create a new search request.
    pub fn new(query: Box<dyn Query>) -> Self {
        SearchRequest {
            query,
            config: SearchConfig::default(),
        }
    }

    /// Set the maximum number of documents to return.
    pub fn max_docs(mut self, max_docs: usize) -> Self {
        self.config.max_docs = max_docs;
        self
    }

    /// Set the minimum score threshold.
    pub fn min_score(mut self, min_score: f32) -> Self {
        self.config.min_score = min_score;
        self
    }

    /// Set whether to load document content.
    pub fn load_documents(mut self, load_documents: bool) -> Self {
        self.config.load_documents = load_documents;
        self
    }

    /// Set the search timeout.
    pub fn timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.config.timeout_ms = Some(timeout_ms);
        self
    }

    /// Enable parallel search.
    pub fn parallel(mut self, parallel: bool) -> Self {
        self.config.parallel = parallel;
        self
    }

    /// Sort results by a field in ascending order.
    pub fn sort_by_field_asc(mut self, field: &str) -> Self {
        self.config.sort_by = SortField::Field {
            name: field.to_string(),
            order: SortOrder::Asc,
        };
        self
    }

    /// Sort results by a field in descending order.
    pub fn sort_by_field_desc(mut self, field: &str) -> Self {
        self.config.sort_by = SortField::Field {
            name: field.to_string(),
            order: SortOrder::Desc,
        };
        self
    }

    /// Sort results by relevance score (default).
    pub fn sort_by_score(mut self) -> Self {
        self.config.sort_by = SortField::Score;
        self
    }

    /// Execute the search request.
    pub fn search(self, engine: &mut engine::SearchEngine) -> Result<SearchResults> {
        engine.search(self)
    }
}
