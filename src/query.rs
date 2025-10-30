//! Query system for searching documents.

pub mod advanced_query;
pub mod boolean;
pub mod collector;
pub mod fuzzy;
pub mod geo;
pub mod matcher;
pub mod multi_term;
pub mod parser;
pub mod phrase;
#[allow(clippy::module_inception)]
pub mod query;
pub mod range;
pub mod scorer;
pub mod span;
pub mod term;
pub mod wildcard;

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[allow(unused_imports)]
use crate::document::document::Document;

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
