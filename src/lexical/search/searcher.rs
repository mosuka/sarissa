//! Search execution implementations.
//!
//! This module provides various searcher implementations for executing
//! queries and collecting results from inverted indexes.

use crate::error::Result;
use crate::lexical::types::SearchRequest;
use crate::query::SearchResults;
use crate::query::query::Query;

pub mod inverted_index;

/// Trait for lexical searchers.
///
/// This trait defines the interface for all lexical search implementations,
/// similar to the VectorSearcher trait for vector search.
pub trait LexicalSearcher: Send + Sync {
    /// Execute a search with the given request.
    ///
    /// This method handles search operations including query execution,
    /// scoring, and result collection based on the search request configuration.
    fn search(&self, request: SearchRequest) -> Result<SearchResults>;

    /// Count documents matching the given query.
    ///
    /// This method returns the total number of documents that match the query
    /// without actually retrieving the documents.
    fn count(&self, query: Box<dyn Query>) -> Result<u64>;

    /// Warm up the searcher (pre-load data, caches, etc.).
    ///
    /// Default implementation does nothing.
    fn warmup(&mut self) -> Result<()> {
        Ok(())
    }
}
