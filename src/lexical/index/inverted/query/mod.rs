//! Query system for searching documents in inverted indexes.

pub mod advanced_query;
pub mod boolean;
pub mod collector;
pub mod fuzzy;
pub mod geo;
pub mod matcher;
pub mod multi_term;
pub mod parser;
pub mod phrase;
pub mod range;
pub mod scorer;
pub mod span;
pub mod term;
pub mod wildcard;

use std::any::Any;
use std::collections::HashMap;
use std::fmt::Debug;

use serde::{Deserialize, Serialize};

#[allow(unused_imports)]
use crate::document::document::Document;
use crate::error::Result;
use crate::lexical::reader::IndexReader;

use self::matcher::Matcher;
use self::scorer::Scorer;

/// A search hit containing a document and its score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hit {
    /// The document ID.
    pub doc_id: u32,
    /// The relevance score.
    pub score: f32,
    /// The document fields (if retrieved).
    pub fields: HashMap<String, String>,
}

/// A search hit containing a document and its score (legacy).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    /// The document ID.
    pub doc_id: u64,
    /// The relevance score.
    pub score: f32,
    /// The document (if retrieved).
    pub document: Option<Document>,
}

/// Search results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResults {
    /// The search hits.
    pub hits: Vec<SearchHit>,
    /// Total number of matching documents.
    pub total_hits: u64,
    /// Maximum score in the results.
    pub max_score: f32,
}

/// Query result wrapper for different result types.
#[derive(Debug, Clone)]
pub struct QueryResult {
    /// Document ID.
    pub doc_id: u32,
    /// Score.
    pub score: f32,
}

/// Trait for search queries.
pub trait Query: Send + Sync + Debug {
    /// Create a matcher for this query.
    fn matcher(&self, reader: &dyn IndexReader) -> Result<Box<dyn Matcher>>;

    /// Create a scorer for this query.
    fn scorer(&self, reader: &dyn IndexReader) -> Result<Box<dyn Scorer>>;

    /// Get the boost factor for this query.
    fn boost(&self) -> f32;

    /// Set the boost factor for this query.
    fn set_boost(&mut self, boost: f32);

    /// Get a human-readable description of this query.
    fn description(&self) -> String;

    /// Clone this query.
    fn clone_box(&self) -> Box<dyn Query>;

    /// Check if this query matches any documents.
    fn is_empty(&self, reader: &dyn IndexReader) -> Result<bool>;

    /// Get the estimated cost of executing this query.
    fn cost(&self, reader: &dyn IndexReader) -> Result<u64>;

    /// Get this query as Any for downcasting.
    fn as_any(&self) -> &dyn Any;

    /// Get the field name this query searches in, if applicable.
    /// Returns None for queries that don't target a specific field (e.g., BooleanQuery).
    fn field(&self) -> Option<&str> {
        None
    }
}
