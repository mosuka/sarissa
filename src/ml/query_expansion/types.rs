//! Common types and enums for query expansion.

use serde::{Deserialize, Serialize};

use crate::query::{BooleanQuery, BooleanQueryBuilder, Query};

/// Types of query expansion.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExpansionType {
    /// Synonym-based expansion.
    Synonym,
    /// Semantic expansion using word embeddings.
    Semantic,
    /// Statistical co-occurrence expansion.
    Statistical,
    /// Morphological expansion (stems, variations).
    Morphological,
}

/// Query intent classification.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QueryIntent {
    /// Informational query (seeking knowledge).
    Informational,
    /// Navigational query (seeking specific resource).
    Navigational,
    /// Transactional query (seeking to perform action).
    Transactional,
    /// Unknown intent.
    Unknown,
}

/// Expanded query with original and expansion queries.
#[derive(Debug)]
pub struct ExpandedQuery {
    /// Original query as a Query object.
    pub original_query: Box<dyn Query>,
    /// Expanded queries with metadata.
    pub expanded_queries: Vec<ExpandedQueryClause>,
    /// Detected query intent.
    pub intent: QueryIntent,
    /// Overall expansion confidence.
    pub confidence: f64,
}

impl ExpandedQuery {
    /// Convert to a BooleanQuery for actual search.
    /// All clauses are combined with SHOULD (OR) semantics.
    pub fn to_boolean_query(&self) -> BooleanQuery {
        let mut builder = BooleanQueryBuilder::new();

        // Add original query with SHOULD
        builder = builder.should(self.original_query.clone_box());

        // Add expanded queries with SHOULD
        for expanded in &self.expanded_queries {
            builder = builder.should(expanded.query.clone_box());
        }

        builder.build()
    }

    /// Get only high-confidence expansion queries.
    pub fn get_high_confidence_expansions(&self, threshold: f64) -> Vec<&ExpandedQueryClause> {
        self.expanded_queries
            .iter()
            .filter(|e| e.confidence >= threshold)
            .collect()
    }
}

/// Expanded query clause with metadata.
#[derive(Debug)]
pub struct ExpandedQueryClause {
    /// The expanded query (TermQuery, PhraseQuery, etc.).
    pub query: Box<dyn Query>,
    /// Confidence score (0.0 - 1.0).
    pub confidence: f64,
    /// Type of expansion.
    pub expansion_type: ExpansionType,
    /// Source term that generated this expansion.
    pub source_term: String,
}
