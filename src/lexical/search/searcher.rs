//! Searcher trait for lexical search execution.

use std::sync::Arc;

use crate::error::Result;
use crate::lexical::index::inverted::query::{Query, SearchResults};

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

/// Query representation that can be either a DSL string or a Query object.
#[derive(Debug)]
pub enum LexicalSearchQuery {
    /// Query specified as a DSL string (will be parsed at search time).
    Dsl(String),
    /// Query specified as a Query object.
    Obj(Box<dyn Query>),
}

/// Search request containing query and configuration.
#[derive(Debug)]
pub struct LexicalSearchRequest {
    /// The query to execute.
    pub query: LexicalSearchQuery,
    /// Search configuration.
    pub params: LexicalSearchParams,
}

impl Clone for LexicalSearchQuery {
    fn clone(&self) -> Self {
        match self {
            LexicalSearchQuery::Dsl(s) => LexicalSearchQuery::Dsl(s.clone()),
            LexicalSearchQuery::Obj(q) => LexicalSearchQuery::Obj(q.clone_box()),
        }
    }
}

impl Clone for LexicalSearchRequest {
    fn clone(&self) -> Self {
        LexicalSearchRequest {
            query: self.query.clone(),
            params: self.params.clone(),
        }
    }
}

impl From<String> for LexicalSearchQuery {
    fn from(s: String) -> Self {
        LexicalSearchQuery::Dsl(s)
    }
}

impl From<&str> for LexicalSearchQuery {
    fn from(s: &str) -> Self {
        LexicalSearchQuery::Dsl(s.to_string())
    }
}

impl From<Box<dyn Query>> for LexicalSearchQuery {
    fn from(q: Box<dyn Query>) -> Self {
        LexicalSearchQuery::Obj(q)
    }
}

impl LexicalSearchQuery {
    /// Parse DSL string into Query object using the given analyzer.
    pub fn into_query(
        self,
        analyzer: &Arc<dyn crate::analysis::analyzer::analyzer::Analyzer>,
    ) -> crate::error::Result<Box<dyn Query>> {
        match self {
            LexicalSearchQuery::Dsl(dsl_string) => {
                let parser = crate::lexical::index::inverted::query::parser::QueryParser::new(
                    analyzer.clone(),
                );
                parser.parse(&dsl_string)
            }
            LexicalSearchQuery::Obj(query) => Ok(query),
        }
    }

    /// Extract the Query object, panics if this is a DSL string.
    pub fn unwrap_query(self) -> Box<dyn Query> {
        match self {
            LexicalSearchQuery::Obj(query) => query,
            LexicalSearchQuery::Dsl(_) => panic!("Expected Query object, found DSL string"),
        }
    }
}

impl LexicalSearchRequest {
    /// Create a new search request from any query type.
    ///
    /// Accepts:
    /// - `&str`: DSL query string
    /// - `String`: DSL query string
    /// - `Box<dyn Query>`: Query object
    pub fn new(query: impl Into<LexicalSearchQuery>) -> Self {
        LexicalSearchRequest {
            query: query.into(),
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

/// Trait for lexical search implementations.
///
/// This trait defines the interface for executing searches against lexical indexes.
pub trait LexicalSearcher: Send + Sync + std::fmt::Debug {
    /// Execute a search with the given request.
    fn search(&self, request: LexicalSearchRequest) -> Result<SearchResults>;

    /// Count the number of matching documents for a query.
    fn count(&self, query: LexicalSearchQuery) -> Result<u64>;
}
