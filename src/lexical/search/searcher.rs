//! Searcher trait for lexical search execution.

use crate::error::Result;
use crate::lexical::index::inverted::query::SearchResults;
use crate::lexical::types::{LexicalSearchQuery, LexicalSearchRequest};

/// Trait for lexical search implementations.
///
/// This trait defines the interface for executing searches against lexical indexes.
pub trait LexicalSearcher: Send + Sync + std::fmt::Debug {
    /// Execute a search with the given request.
    fn search(&self, request: LexicalSearchRequest) -> Result<SearchResults>;

    /// Count the number of matching documents for a query.
    fn count(&self, query: LexicalSearchQuery) -> Result<u64>;
}
