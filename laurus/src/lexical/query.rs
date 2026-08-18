//! Query system for searching documents in inverted indexes.

pub mod advanced_query;
pub mod boolean;
pub mod collector;
pub mod fuzzy;
pub mod geo;
pub mod geo3d;
pub mod matcher;
pub mod multi_term;
pub mod parser;
pub mod phrase;
pub mod prefix;
pub mod range;
pub mod regexp;
pub mod scorer;
pub mod span;
pub mod term;
pub mod wildcard;

// Re-exports for cleaner API
pub use advanced_query::AdvancedQuery;
pub use boolean::{BooleanQuery, BooleanQueryBuilder};
pub use fuzzy::FuzzyQuery;
pub use geo::{GeoBoundingBox, GeoBoundingBoxQuery, GeoDistanceQuery, GeoPoint};
pub use geo3d::{
    Geo3dBoundingBoxQuery, Geo3dDistanceQuery, Geo3dMatch, Geo3dMatcher, Geo3dNearestQuery,
    Geo3dScorer,
};
pub use multi_term::MultiTermQuery;
pub use parser::LexicalQueryParser;
pub use phrase::PhraseQuery;
pub use prefix::PrefixQuery;
pub use range::NumericRangeQuery;
pub use regexp::RegexpQuery;
pub use span::{SpanNearQuery, SpanQuery, SpanTermQuery};
pub use term::TermQuery;
pub use wildcard::WildcardQuery;

use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;

use serde::{Deserialize, Serialize};

use crate::error::Result;
#[allow(unused_imports)]
use crate::lexical::core::document::Document;
use crate::lexical::reader::LexicalIndexReader;

use self::matcher::Matcher;
use self::scorer::Scorer;

/// A search hit containing a document and its score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hit {
    /// The document ID.
    pub doc_id: u64,
    /// The relevance score.
    pub score: f32,
    /// The document fields (if retrieved).
    pub fields: HashMap<String, String>,
}

/// A single search hit containing a matched document and its relevance score.
///
/// Returned as part of [`LexicalSearchResults`] to represent each document
/// that matched the search query. The `score` reflects the relevance ranking
/// computed by the scorer (e.g., BM25), and the `document` field optionally
/// holds the stored fields if they were requested.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    /// The internal document ID.
    pub doc_id: u64,
    /// The relevance score.
    pub score: f32,
    /// The document (if retrieved).
    pub document: Option<Document>,
}

/// Aggregated results from a lexical search query.
///
/// Contains the ranked list of matching documents along with summary statistics.
///
/// # Fields
///
/// - `hits` - Ranked list of [`SearchHit`] entries, ordered by descending score.
/// - `total_hits` - Total number of documents that matched the query (may exceed `hits.len()`
///   when a limit is applied).
/// - `max_score` - The highest relevance score among all results, useful for normalization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LexicalSearchResults {
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
    pub doc_id: u64,
    /// Score.
    pub score: f32,
}

/// Trait for search queries.
pub trait Query: Send + Sync + Debug {
    /// Create a matcher for this query.
    fn matcher(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Matcher>>;

    /// Create a scorer for this query.
    fn scorer(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Scorer>>;

    /// Create the matcher and scorer for this query in one pass.
    ///
    /// The default implementation builds them independently via
    /// [`matcher`](Query::matcher) and [`scorer`](Query::scorer),
    /// preserving the historical behavior. Queries whose matcher and
    /// scorer derive from the same expensive candidate computation
    /// (e.g. geo queries computing per-document distances) override
    /// this to run that computation once and build both from the
    /// shared result (#996).
    ///
    /// # Arguments
    ///
    /// * `reader` - The index reader to search.
    ///
    /// # Returns
    ///
    /// The `(matcher, scorer)` pair for this query.
    fn matcher_scorer(
        &self,
        reader: &dyn LexicalIndexReader,
    ) -> Result<(Box<dyn Matcher>, Box<dyn Scorer>)> {
        Ok((self.matcher(reader)?, self.scorer(reader)?))
    }

    /// Get the boost factor for this query.
    fn boost(&self) -> f32;

    /// Set the boost factor for this query.
    fn set_boost(&mut self, boost: f32);

    /// Get a human-readable description of this query.
    fn description(&self) -> String;

    /// Clone this query.
    fn clone_box(&self) -> Box<dyn Query>;

    /// Returns `true` if this query would match no documents in the given reader.
    ///
    /// Each implementor defines its own emptiness semantics. For example:
    /// - [`TermQuery`] checks whether
    ///   the term exists in the index via the reader.
    /// - [`BooleanQuery`] returns `true`
    ///   when it has no clauses or all of its clauses are empty.
    ///
    /// # Parameters
    ///
    /// - `reader` - The index reader used to check whether the query's terms exist.
    ///
    /// # Returns
    ///
    /// `Ok(true)` if this query would not match any documents, `Ok(false)` otherwise.
    /// Returns an error if the reader cannot be queried.
    fn is_empty(&self, reader: &dyn LexicalIndexReader) -> Result<bool>;

    /// Get the estimated cost of executing this query.
    fn cost(&self, reader: &dyn LexicalIndexReader) -> Result<u64>;

    /// Rewrite this query against the reader (Issue #613).
    ///
    /// Called once by the searcher at the top of query execution,
    /// against the top-level reader — before the per-segment fanout and
    /// BMW gates. Multi-term queries (prefix / wildcard / fuzzy /
    /// regexp) override this to lower themselves into a `BooleanQuery`
    /// of `TermQuery` clauses via one term-dictionary enumeration, so
    /// the subsequent `matcher` + `scorer` construction (and every
    /// per-segment execution) reuses the lowered form instead of
    /// re-enumerating. [`crate::lexical::query::boolean::BooleanQuery`]
    /// recurses into its scoring clauses.
    ///
    /// # Arguments
    ///
    /// * `reader` - The index reader the query will execute against.
    ///
    /// # Returns
    ///
    /// `Ok(Some(query))` with the rewritten query, or `Ok(None)` when
    /// this query has nothing to rewrite (the default — zero cost for
    /// term / phrase / range queries) or the reader does not support
    /// term enumeration (the caller keeps the original query, whose
    /// `matcher` / `scorer` fallback behavior is unchanged).
    fn rewrite(&self, reader: &dyn LexicalIndexReader) -> Result<Option<Box<dyn Query>>> {
        let _ = reader;
        Ok(None)
    }

    /// Get this query as Any for downcasting.
    fn as_any(&self) -> &dyn Any;

    /// Get the field name this query searches in, if applicable.
    /// Returns None for queries that don't target a specific field (e.g., BooleanQuery).
    fn field(&self) -> Option<&str> {
        None
    }

    /// Collect every field name referenced by this query and its sub-queries.
    ///
    /// The default implementation inserts `self.field()` into `out` if it
    /// returns `Some`. Composite queries that hold child queries (e.g.
    /// [`BooleanQuery`](crate::lexical::query::BooleanQuery), span queries)
    /// override this method to recurse into each child so every leaf field
    /// reference contributes.
    ///
    /// This is the authoritative way for callers (such as
    /// [`UnifiedQueryParser`](crate::engine::query::UnifiedQueryParser)) to
    /// validate that every field a query references is declared in the
    /// schema, replacing earlier regex-based heuristics.
    ///
    /// # Arguments
    ///
    /// * `out` - The set to populate with field names. Existing entries are
    ///   preserved; new names are inserted.
    fn collect_field_refs(&self, out: &mut HashSet<String>) {
        if let Some(f) = self.field() {
            out.insert(f.to_string());
        }
    }

    /// Apply field-level boosts to this query and its sub-queries.
    fn apply_field_boosts(&mut self, boosts: &HashMap<String, f32>) {
        if let Some(f) = self.field()
            && let Some(&b) = boosts.get(f)
        {
            self.set_boost(self.boost() * b);
        }
    }

    /// Return a stable, canonical cache key identifying the **set of documents**
    /// this query matches within a reader snapshot, or `None` if the query must
    /// not be cached.
    ///
    /// This key backs the snapshot-scoped filter/doc-id result cache
    /// (see [`QueryFilterCache`](crate::lexical::index::inverted::query_cache::QueryFilterCache)).
    /// It is **score-independent**: two queries that match the same document set
    /// must produce the same key, and the query's boost — which only scales
    /// scores, never changes membership — is deliberately excluded.
    ///
    /// # Correctness contract
    ///
    /// Returning `Some(key)` is a promise that the key is *canonical*: any two
    /// query instances that would match different document sets MUST yield
    /// different keys. Implementors that cannot guarantee this (e.g. a key built
    /// from a non-deterministic `HashMap` iteration order, or one that omits a
    /// membership-affecting parameter) MUST return `None` so the query bypasses
    /// the cache and is evaluated fresh. The default is `None` (not cacheable),
    /// which is always safe.
    ///
    /// Keys are namespaced with a per-type tag (e.g. `"term|…"`) so that two
    /// different query types can never collide in the shared cache map.
    ///
    /// # Returns
    ///
    /// `Some(canonical_key)` if this query may be cached, `None` otherwise.
    fn cache_key(&self) -> Option<String> {
        None
    }
}
