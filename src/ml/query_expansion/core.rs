//! Core QueryExpansion implementation.

use std::sync::Arc;

use crate::analysis::analyzer::Analyzer;
use crate::error::Result;
use crate::ml::MLContext;
use crate::query::{BooleanQuery, BooleanQueryBuilder, Query, TermQuery};

use super::builder::QueryExpansionBuilder;
use super::expander::QueryExpander;
use super::types::{ExpandedQuery, ExpandedQueryClause};

/// Query expansion system.
///
/// Combines multiple expansion strategies and manages the expansion pipeline.
pub struct QueryExpansion {
    pub(super) expanders: Vec<Box<dyn QueryExpander>>,
    pub(super) intent_classifier: Box<dyn crate::ml::intent_classifier::IntentClassifier>,
    pub(super) analyzer: Arc<dyn Analyzer>,
    pub(super) max_expansions: usize,
    pub(super) original_term_weight: f64,
}

impl QueryExpansion {
    /// Create a new query expansion system.
    ///
    /// This constructor is typically not called directly. Use `QueryExpansionBuilder` instead.
    pub fn new(
        expanders: Vec<Box<dyn QueryExpander>>,
        intent_classifier: Box<dyn crate::ml::intent_classifier::IntentClassifier>,
        analyzer: Arc<dyn Analyzer>,
        max_expansions: usize,
        original_term_weight: f64,
    ) -> Self {
        Self {
            expanders,
            intent_classifier,
            analyzer,
            max_expansions,
            original_term_weight,
        }
    }

    /// Create a new builder for query expansion.
    ///
    /// # Arguments
    /// * `analyzer` - The analyzer to use for tokenization
    ///
    /// # Example
    /// ```no_run
    /// use sarissa::analysis::StandardAnalyzer;
    /// use sarissa::ml::query_expansion::QueryExpansion;
    /// use std::sync::Arc;
    ///
    /// let analyzer = Arc::new(StandardAnalyzer::new().unwrap());
    /// let expander = QueryExpansion::builder(analyzer)
    ///     .with_synonyms(Some("synonyms.json"), 0.8).unwrap()
    ///     .max_expansions(5)
    ///     .build().unwrap();
    /// ```
    pub fn builder(analyzer: Arc<dyn Analyzer>) -> QueryExpansionBuilder {
        QueryExpansionBuilder::new(analyzer)
    }

    /// Expand a query using all configured expansion techniques.
    ///
    /// # Arguments
    /// * `original_query` - The original query string
    /// * `field` - The field name for the expanded queries
    /// * `context` - ML context containing user session and search history
    ///
    /// # Returns
    /// An `ExpandedQuery` containing the original query, expansions, and metadata
    pub fn expand_query(
        &self,
        original_query: &str,
        field: &str,
        context: &MLContext,
    ) -> Result<ExpandedQuery> {
        let tokens = self.tokenize_query(original_query)?;
        let mut expanded_queries = Vec::new();

        // Classify query intent
        let intent = self.intent_classifier.predict(original_query)?;

        // Apply all expanders
        for expander in &self.expanders {
            expanded_queries.extend(expander.expand(&tokens, field, context)?);
        }

        // Deduplicate by query description
        expanded_queries.sort_by(|a, b| {
            a.query
                .description()
                .cmp(&b.query.description())
                .then(b.confidence.partial_cmp(&a.confidence).unwrap())
        });
        expanded_queries.dedup_by(|a, b| a.query.description() == b.query.description());

        // Sort by confidence and limit
        expanded_queries.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        expanded_queries.truncate(self.max_expansions);

        // Build original query
        let original_query = self.build_query_from_tokens(&tokens, field);

        // Calculate overall confidence
        let confidence = self.calculate_expansion_confidence(&tokens, &expanded_queries);

        Ok(ExpandedQuery {
            original_query,
            expanded_queries,
            intent,
            confidence,
        })
    }

    fn tokenize_query(&self, query: &str) -> Result<Vec<String>> {
        let tokens = self.analyzer.analyze(query)?;
        Ok(tokens.into_iter().map(|t| t.text).collect())
    }

    fn build_query_from_tokens(&self, tokens: &[String], field: &str) -> Box<dyn Query> {
        if tokens.is_empty() {
            return Box::new(BooleanQuery::new());
        }

        if tokens.len() == 1 {
            let mut query = Box::new(TermQuery::new(field, tokens[0].clone())) as Box<dyn Query>;
            query.set_boost(self.original_term_weight as f32);
            return query;
        }

        // Multiple terms - create a boolean query with SHOULD clauses
        let mut builder = BooleanQueryBuilder::new();
        for token in tokens {
            let mut term_query = Box::new(TermQuery::new(field, token.clone())) as Box<dyn Query>;
            term_query.set_boost(self.original_term_weight as f32);
            builder = builder.should(term_query);
        }

        Box::new(builder.build())
    }

    fn calculate_expansion_confidence(
        &self,
        _original_terms: &[String],
        expansions: &[ExpandedQueryClause],
    ) -> f64 {
        if expansions.is_empty() {
            return 1.0;
        }

        let avg_expansion_confidence: f64 =
            expansions.iter().map(|e| e.confidence).sum::<f64>() / expansions.len() as f64;

        // Combine original query strength with expansion confidence
        0.7 * 1.0 + 0.3 * avg_expansion_confidence
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::analyzer::language::EnglishAnalyzer;
    use crate::ml::query_expansion::types::QueryIntent;

    #[test]
    fn test_builder_with_synonyms() {
        let analyzer = Arc::new(EnglishAnalyzer::new().unwrap());
        let expansion = QueryExpansionBuilder::new(analyzer)
            .with_synonyms(Some("resource/ml/synonyms.json"), 0.5)
            .unwrap()
            .max_expansions(5)
            .build()
            .unwrap();

        assert_eq!(expansion.expanders.len(), 1);
        assert_eq!(expansion.max_expansions, 5);
    }

    #[test]
    fn test_builder_multiple_expanders() {
        let analyzer = Arc::new(EnglishAnalyzer::new().unwrap());
        let expansion = QueryExpansionBuilder::new(analyzer)
            .with_synonyms(Some("resource/ml/synonyms.json"), 0.5)
            .unwrap()
            .with_statistical(0.3)
            .max_expansions(10)
            .build()
            .unwrap();

        assert_eq!(expansion.expanders.len(), 2);
    }

    #[test]
    fn test_expand_query() {
        let analyzer = Arc::new(EnglishAnalyzer::new().unwrap());
        let expansion = QueryExpansionBuilder::new(analyzer)
            .with_synonyms(Some("resource/ml/synonyms.json"), 0.5)
            .unwrap()
            .build()
            .unwrap();

        let ml_context = MLContext::default();
        let expanded = expansion
            .expand_query("ml python", "content", &ml_context)
            .unwrap();

        assert!(!expanded.expanded_queries.is_empty());
        assert_eq!(expanded.intent, QueryIntent::Unknown);
    }
}
