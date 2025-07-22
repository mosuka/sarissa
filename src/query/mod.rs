//! Query system for searching documents.

pub mod advanced_query;
pub mod boolean;
pub mod collector;
pub mod fuzzy;
pub mod geo;
pub mod matcher;
pub mod parser;
pub mod phrase;
#[allow(clippy::module_inception)]
pub mod query;
pub mod range;
pub mod scorer;
pub mod span;
pub mod term;
pub mod wildcard;

pub use self::advanced_query::{
    AdvancedQuery, AdvancedQueryConfig, BooleanQueryBuilder as AdvancedBooleanQueryBuilder,
    MultiFieldQuery, MultiFieldQueryType,
};
pub use self::boolean::{BooleanClause, BooleanQuery, BooleanQueryBuilder, Occur};
pub use self::collector::{Collector, CountCollector, TopDocsCollector};
pub use self::fuzzy::{FuzzyConfig, FuzzyMatch, FuzzyQuery};
pub use self::geo::{GeoBoundingBox, GeoBoundingBoxQuery, GeoDistanceQuery, GeoMatch, GeoPoint};
pub use self::matcher::Matcher;
pub use self::parser::{QueryParser, QueryParserBuilder};
pub use self::phrase::PhraseQuery;
pub use self::query::Query;
pub use self::range::{Bound, DateTimeRangeQuery, NumericRangeQuery, RangeQuery};
pub use self::scorer::{BM25Scorer, Scorer};
pub use self::span::{
    Span, SpanContainingQuery, SpanNearQuery, SpanQuery, SpanQueryBuilder, SpanQueryWrapper,
    SpanTermQuery, SpanWithinQuery,
};
pub use self::term::TermQuery;
pub use self::wildcard::WildcardQuery;

// Re-export similarity types for convenience
pub use crate::search::similarity::{
    MoreLikeThisQuery, SimilarityAlgorithm, SimilarityConfig, SimilarityResult,
};

#[allow(unused_imports)]
use crate::schema::Document;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
