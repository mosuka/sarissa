//! Search engine for executing queries and collecting results.

pub mod engine;
pub mod facet;
pub mod highlight;
pub mod hybrid;
pub mod result_processor;
pub mod scoring;
pub mod searcher;
pub mod similarity;
pub mod spell_corrected;

pub use self::engine::SearchEngine;
pub use self::facet::*;
pub use self::highlight::*;
pub use self::hybrid::*;
pub use self::result_processor::*;
pub use self::scoring::*;
pub use self::searcher::Searcher;
pub use self::similarity::*;
pub use self::spell_corrected::*;

// Re-export advanced search functionality
pub use crate::query::{FuzzyConfig, FuzzyMatch, FuzzyQuery};
pub use crate::query::{GeoBoundingBox, GeoBoundingBoxQuery, GeoDistanceQuery, GeoMatch, GeoPoint};
pub use crate::query::{
    MoreLikeThisQuery, SimilarityAlgorithm, SimilarityConfig, SimilarityResult,
};

use crate::error::Result;
use crate::query::{Query, SearchResults};

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
}

impl Default for SearchConfig {
    fn default() -> Self {
        SearchConfig {
            max_docs: 10,
            min_score: 0.0,
            load_documents: true,
            timeout_ms: None,
            parallel: true, // Enable parallel search by default
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

    /// Set the search configuration.
    pub fn config(mut self, config: SearchConfig) -> Self {
        self.config = config;
        self
    }
}

/// Trait for search execution.
pub trait Search {
    /// Execute a search request.
    fn search(&self, request: SearchRequest) -> Result<SearchResults>;

    /// Execute a query with default configuration.
    fn search_query(&self, query: Box<dyn Query>) -> Result<SearchResults> {
        self.search(SearchRequest::new(query))
    }

    /// Count the number of documents matching a query.
    fn count(&self, query: Box<dyn Query>) -> Result<u64>;
}
