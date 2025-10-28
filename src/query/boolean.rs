//! Boolean query implementation for combining multiple queries.

use crate::error::Result;
use crate::lexical::reader::IndexReader;
use crate::query::matcher::{
    AllMatcher, ConjunctionMatcher, ConjunctionNotMatcher, DisjunctionMatcher, EmptyMatcher,
    Matcher, NotMatcher,
};
use crate::query::query::Query;
use crate::query::scorer::{BM25Scorer, Scorer};

/// Occurrence requirements for boolean clauses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Occur {
    /// The clause must match (equivalent to AND).
    Must,
    /// The clause should match (equivalent to OR).
    Should,
    /// The clause must not match (equivalent to NOT).
    MustNot,
}

/// A clause in a boolean query.
#[derive(Debug)]
pub struct BooleanClause {
    /// The query for this clause.
    pub query: Box<dyn Query>,
    /// The occurrence requirement.
    pub occur: Occur,
}

impl Clone for BooleanClause {
    fn clone(&self) -> Self {
        BooleanClause {
            query: self.query.clone_box(),
            occur: self.occur,
        }
    }
}

impl BooleanClause {
    /// Create a new boolean clause.
    pub fn new(query: Box<dyn Query>, occur: Occur) -> Self {
        BooleanClause { query, occur }
    }

    /// Create a MUST clause.
    pub fn must(query: Box<dyn Query>) -> Self {
        BooleanClause::new(query, Occur::Must)
    }

    /// Create a SHOULD clause.
    pub fn should(query: Box<dyn Query>) -> Self {
        BooleanClause::new(query, Occur::Should)
    }

    /// Create a MUST_NOT clause.
    pub fn must_not(query: Box<dyn Query>) -> Self {
        BooleanClause::new(query, Occur::MustNot)
    }
}

/// A boolean query that combines multiple queries with boolean logic.
#[derive(Debug)]
pub struct BooleanQuery {
    /// The clauses in this boolean query.
    clauses: Vec<BooleanClause>,
    /// The boost factor for this query.
    boost: f32,
    /// Minimum number of should clauses that must match.
    minimum_should_match: usize,
}

impl BooleanQuery {
    /// Create a new empty boolean query.
    pub fn new() -> Self {
        BooleanQuery {
            clauses: Vec::new(),
            boost: 1.0,
            minimum_should_match: 0,
        }
    }

    /// Add a clause to this boolean query.
    pub fn add_clause(&mut self, clause: BooleanClause) {
        self.clauses.push(clause);
    }

    /// Add a MUST clause.
    pub fn add_must(&mut self, query: Box<dyn Query>) {
        self.add_clause(BooleanClause::must(query));
    }

    /// Add a SHOULD clause.
    pub fn add_should(&mut self, query: Box<dyn Query>) {
        self.add_clause(BooleanClause::should(query));
    }

    /// Add a MUST_NOT clause.
    pub fn add_must_not(&mut self, query: Box<dyn Query>) {
        self.add_clause(BooleanClause::must_not(query));
    }

    /// Set the boost factor.
    pub fn with_boost(mut self, boost: f32) -> Self {
        self.boost = boost;
        self
    }

    /// Set the minimum number of should clauses that must match.
    pub fn with_minimum_should_match(mut self, minimum: usize) -> Self {
        self.minimum_should_match = minimum;
        self
    }

    /// Get the clauses.
    pub fn clauses(&self) -> &[BooleanClause] {
        &self.clauses
    }

    /// Get the minimum should match value.
    pub fn minimum_should_match(&self) -> usize {
        self.minimum_should_match
    }

    /// Check if this query is empty.
    pub fn is_empty(&self) -> bool {
        self.clauses.is_empty()
    }

    /// Get clauses by occurrence type.
    pub fn clauses_by_occur(&self, occur: Occur) -> Vec<&BooleanClause> {
        self.clauses.iter().filter(|c| c.occur == occur).collect()
    }
}

impl Default for BooleanQuery {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for BooleanQuery {
    fn clone(&self) -> Self {
        BooleanQuery {
            clauses: self
                .clauses
                .iter()
                .map(|c| BooleanClause {
                    query: c.query.clone_box(),
                    occur: c.occur,
                })
                .collect(),
            boost: self.boost,
            minimum_should_match: self.minimum_should_match,
        }
    }
}

impl Query for BooleanQuery {
    fn matcher(&self, reader: &dyn IndexReader) -> Result<Box<dyn Matcher>> {
        if self.clauses.is_empty() {
            return Ok(Box::new(EmptyMatcher::new()));
        }

        let must_clauses = self.clauses_by_occur(Occur::Must);
        let should_clauses = self.clauses_by_occur(Occur::Should);
        let must_not_clauses = self.clauses_by_occur(Occur::MustNot);

        // Handle MUST and MUST_NOT clauses
        if !must_clauses.is_empty() || (!must_not_clauses.is_empty() && should_clauses.is_empty()) {
            // Create positive matcher from MUST clauses
            let mut positive_matcher = if !must_clauses.is_empty() {
                if must_clauses.len() == 1 {
                    // Single MUST clause
                    must_clauses[0].query.matcher(reader)?
                } else {
                    // Multiple MUST clauses - use ConjunctionMatcher
                    let mut matchers = Vec::new();
                    for clause in &must_clauses {
                        let matcher = clause.query.matcher(reader)?;
                        if matcher.is_exhausted() {
                            return Ok(Box::new(EmptyMatcher::new()));
                        }
                        matchers.push(matcher);
                    }
                    Box::new(ConjunctionMatcher::new(matchers))
                }
            } else {
                // No MUST clauses, but we have MUST_NOT clauses and no SHOULD clauses
                // Match all documents and exclude the ones matching MUST_NOT
                Box::new(AllMatcher::new(reader.max_doc()))
            };

            // If minimum_should_match is set and we have SHOULD clauses, combine them with MUST
            if self.minimum_should_match > 0 && !should_clauses.is_empty() {
                let mut should_matchers = Vec::new();
                for clause in &should_clauses {
                    let matcher = clause.query.matcher(reader)?;
                    if !matcher.is_exhausted() {
                        should_matchers.push(matcher);
                    }
                }

                if !should_matchers.is_empty() {
                    let should_matcher = if should_matchers.len() == 1 {
                        should_matchers.into_iter().next().unwrap()
                    } else {
                        Box::new(DisjunctionMatcher::new(should_matchers))
                    };

                    // Combine MUST and SHOULD with ConjunctionMatcher
                    positive_matcher = Box::new(ConjunctionMatcher::new(vec![
                        positive_matcher,
                        should_matcher,
                    ]));
                } else if !must_clauses.is_empty() {
                    // SHOULD clauses are exhausted but minimum_should_match requires them
                    // This means no documents can match
                    return Ok(Box::new(EmptyMatcher::new()));
                }
            }

            // Handle MUST_NOT clauses
            if !must_not_clauses.is_empty() {
                let mut negative_matchers = Vec::new();
                for clause in &must_not_clauses {
                    let matcher = clause.query.matcher(reader)?;
                    if !matcher.is_exhausted() {
                        negative_matchers.push(matcher);
                    }
                }

                if !negative_matchers.is_empty() {
                    if must_clauses.is_empty() {
                        // Only MUST_NOT clauses - use NotMatcher
                        if negative_matchers.len() == 1 {
                            Ok(Box::new(NotMatcher::new(
                                negative_matchers.into_iter().next().unwrap(),
                                reader.max_doc(),
                            )))
                        } else {
                            // Multiple MUST_NOT clauses - combine them with DisjunctionMatcher
                            let combined_negatives =
                                Box::new(DisjunctionMatcher::new(negative_matchers));
                            Ok(Box::new(NotMatcher::new(
                                combined_negatives,
                                reader.max_doc(),
                            )))
                        }
                    } else {
                        // Both MUST and MUST_NOT clauses - use ConjunctionNotMatcher
                        Ok(Box::new(ConjunctionNotMatcher::new(
                            positive_matcher,
                            negative_matchers,
                        )))
                    }
                } else {
                    // All negative matchers are exhausted, just return positive matcher
                    Ok(positive_matcher)
                }
            } else {
                // No MUST_NOT clauses, just return positive matcher
                Ok(positive_matcher)
            }
        } else if !should_clauses.is_empty() {
            // SHOULD clauses (possibly with MUST_NOT)
            let mut should_matchers = Vec::new();
            for clause in &should_clauses {
                let matcher = clause.query.matcher(reader)?;
                if !matcher.is_exhausted() {
                    should_matchers.push(matcher);
                }
            }

            if should_matchers.is_empty() {
                return Ok(Box::new(EmptyMatcher::new()));
            }

            let positive_matcher = if should_matchers.len() == 1 {
                should_matchers.into_iter().next().unwrap()
            } else {
                Box::new(DisjunctionMatcher::new(should_matchers))
            };

            // Handle MUST_NOT clauses with SHOULD
            if !must_not_clauses.is_empty() {
                let mut negative_matchers = Vec::new();
                for clause in &must_not_clauses {
                    let matcher = clause.query.matcher(reader)?;
                    if !matcher.is_exhausted() {
                        negative_matchers.push(matcher);
                    }
                }

                if !negative_matchers.is_empty() {
                    // Combine SHOULD (positive) with MUST_NOT (negative)
                    Ok(Box::new(ConjunctionNotMatcher::new(
                        positive_matcher,
                        negative_matchers,
                    )))
                } else {
                    Ok(positive_matcher)
                }
            } else {
                Ok(positive_matcher)
            }
        } else {
            Ok(Box::new(EmptyMatcher::new()))
        }
    }

    fn scorer(&self, reader: &dyn IndexReader) -> Result<Box<dyn Scorer>> {
        // For boolean queries, we combine the scores from child queries
        // For simplicity, we'll estimate based on the most restrictive clause
        let must_clauses = self.clauses_by_occur(Occur::Must);
        let should_clauses = self.clauses_by_occur(Occur::Should);

        // If we have MUST clauses, use the first one for scoring estimation
        if let Some(clause) = must_clauses.first() {
            let child_scorer = clause.query.scorer(reader)?;
            // Scale the boost to account for boolean combination
            let combined_boost = self.boost * child_scorer.boost();

            // Create a scorer with estimated parameters
            let scorer = BM25Scorer::new(
                reader.doc_count() / 4, // Estimate: 25% of docs might match
                reader.doc_count() / 2, // Estimate: 50% term frequency
                reader.doc_count(),
                10.0, // Estimated average field length
                reader.doc_count(),
                combined_boost,
            );
            return Ok(Box::new(scorer));
        }

        // If we have SHOULD clauses, use the first one
        if let Some(clause) = should_clauses.first() {
            let child_scorer = clause.query.scorer(reader)?;
            let combined_boost = self.boost * child_scorer.boost();

            let scorer = BM25Scorer::new(
                reader.doc_count() / 3, // More liberal estimate for SHOULD
                reader.doc_count() / 2,
                reader.doc_count(),
                10.0,
                reader.doc_count(),
                combined_boost,
            );
            return Ok(Box::new(scorer));
        }

        // Fallback for empty boolean query
        let scorer = BM25Scorer::new(
            1,
            1,
            reader.doc_count(),
            10.0,
            reader.doc_count(),
            self.boost,
        );
        Ok(Box::new(scorer))
    }

    fn boost(&self) -> f32 {
        self.boost
    }

    fn set_boost(&mut self, boost: f32) {
        self.boost = boost;
    }

    fn description(&self) -> String {
        if self.clauses.is_empty() {
            return "()".to_string();
        }

        let mut parts = Vec::new();

        for clause in &self.clauses {
            let clause_desc = match clause.occur {
                Occur::Must => format!("+{}", clause.query.description()),
                Occur::Should => clause.query.description(),
                Occur::MustNot => format!("-{}", clause.query.description()),
            };
            parts.push(clause_desc);
        }

        let result = format!("({})", parts.join(" "));

        if self.boost == 1.0 {
            result
        } else {
            format!("{}^{}", result, self.boost)
        }
    }

    fn clone_box(&self) -> Box<dyn Query> {
        Box::new(self.clone())
    }

    fn is_empty(&self, reader: &dyn IndexReader) -> Result<bool> {
        if self.clauses.is_empty() {
            return Ok(true);
        }

        // Check if any clause can match
        for clause in &self.clauses {
            if !clause.query.is_empty(reader)? {
                return Ok(false);
            }
        }

        Ok(true)
    }

    fn cost(&self, reader: &dyn IndexReader) -> Result<u64> {
        let mut total_cost = 0;

        for clause in &self.clauses {
            total_cost += clause.query.cost(reader)?;
        }

        Ok(total_cost)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Builder for creating boolean queries.
#[derive(Debug)]
pub struct BooleanQueryBuilder {
    query: BooleanQuery,
}

impl BooleanQueryBuilder {
    /// Create a new boolean query builder.
    pub fn new() -> Self {
        BooleanQueryBuilder {
            query: BooleanQuery::new(),
        }
    }

    /// Add a MUST clause.
    pub fn must(mut self, query: Box<dyn Query>) -> Self {
        self.query.add_must(query);
        self
    }

    /// Add a SHOULD clause.
    pub fn should(mut self, query: Box<dyn Query>) -> Self {
        self.query.add_should(query);
        self
    }

    /// Add a MUST_NOT clause.
    pub fn must_not(mut self, query: Box<dyn Query>) -> Self {
        self.query.add_must_not(query);
        self
    }

    /// Set the boost factor.
    pub fn boost(mut self, boost: f32) -> Self {
        self.query = self.query.with_boost(boost);
        self
    }

    /// Set the minimum should match.
    pub fn minimum_should_match(mut self, minimum: usize) -> Self {
        self.query = self.query.with_minimum_should_match(minimum);
        self
    }

    /// Build the boolean query.
    pub fn build(self) -> BooleanQuery {
        self.query
    }
}

impl Default for BooleanQueryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexical::index::reader::inverted::{InvertedIndexReader, InvertedIndexReaderConfig};
    use crate::query::term::TermQuery;

    use crate::storage::memory::MemoryStorage;
    use crate::storage::memory::MemoryStorageConfig;
    use std::sync::Arc;

    #[allow(dead_code)]
    #[test]
    fn test_boolean_query_creation() {
        let query = BooleanQuery::new();

        assert!(query.is_empty());
        assert_eq!(query.clauses().len(), 0);
        assert_eq!(query.boost(), 1.0);
        assert_eq!(query.minimum_should_match(), 0);
    }

    #[test]
    fn test_boolean_query_clauses() {
        let mut query = BooleanQuery::new();

        query.add_must(Box::new(TermQuery::new("title", "hello")));
        query.add_should(Box::new(TermQuery::new("body", "world")));
        query.add_must_not(Box::new(TermQuery::new("title", "spam")));

        assert_eq!(query.clauses().len(), 3);
        assert!(!query.is_empty());

        let must_clauses = query.clauses_by_occur(Occur::Must);
        let should_clauses = query.clauses_by_occur(Occur::Should);
        let must_not_clauses = query.clauses_by_occur(Occur::MustNot);

        assert_eq!(must_clauses.len(), 1);
        assert_eq!(should_clauses.len(), 1);
        assert_eq!(must_not_clauses.len(), 1);
    }

    #[test]
    fn test_boolean_query_builder() {
        let query = BooleanQueryBuilder::new()
            .must(Box::new(TermQuery::new("title", "hello")))
            .should(Box::new(TermQuery::new("body", "world")))
            .must_not(Box::new(TermQuery::new("title", "spam")))
            .boost(2.0)
            .minimum_should_match(1)
            .build();

        assert_eq!(query.clauses().len(), 3);
        assert_eq!(query.boost(), 2.0);
        assert_eq!(query.minimum_should_match(), 1);
    }

    #[test]
    fn test_boolean_query_description() {
        let query = BooleanQueryBuilder::new()
            .must(Box::new(TermQuery::new("title", "hello")))
            .should(Box::new(TermQuery::new("body", "world")))
            .must_not(Box::new(TermQuery::new("title", "spam")))
            .build();

        let desc = query.description();
        assert!(desc.contains("+title:hello"));
        assert!(desc.contains("body:world"));
        assert!(desc.contains("-title:spam"));
    }

    #[test]
    fn test_boolean_query_matcher() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let reader =
            InvertedIndexReader::new(vec![], storage, InvertedIndexReaderConfig::default())
                .unwrap();

        let query = BooleanQueryBuilder::new()
            .must(Box::new(TermQuery::new("title", "hello")))
            .build();

        let matcher = query.matcher(&reader).unwrap();
        // Should create a matcher without error
        assert!(matcher.is_exhausted() || matcher.doc_id() != u64::MAX);
    }

    #[test]
    fn test_boolean_query_scorer() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let reader =
            InvertedIndexReader::new(vec![], storage, InvertedIndexReaderConfig::default())
                .unwrap();

        let query = BooleanQueryBuilder::new()
            .must(Box::new(TermQuery::new("title", "hello")))
            .build();

        let scorer = query.scorer(&reader).unwrap();
        // Should create a scorer without error
        assert!(scorer.score(0, 1.0, None) >= 0.0);
    }

    #[test]
    fn test_boolean_clause_creation() {
        let query = Box::new(TermQuery::new("title", "hello"));

        let must_clause = BooleanClause::must(query.clone_box());
        assert_eq!(must_clause.occur, Occur::Must);

        let should_clause = BooleanClause::should(query.clone_box());
        assert_eq!(should_clause.occur, Occur::Should);

        let must_not_clause = BooleanClause::must_not(query.clone_box());
        assert_eq!(must_not_clause.occur, Occur::MustNot);
    }

    #[test]
    fn test_empty_boolean_query() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let reader =
            InvertedIndexReader::new(vec![], storage, InvertedIndexReaderConfig::default())
                .unwrap();

        let query = BooleanQuery::new();

        assert!(query.is_empty());
        assert_eq!(query.cost(&reader).unwrap(), 0);

        let matcher = query.matcher(&reader).unwrap();
        assert!(matcher.is_exhausted());
    }
}
