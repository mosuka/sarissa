//! Advanced query system with complex query composition and optimization.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::index::reader::IndexReader;
use crate::query::{BooleanQuery, Matcher, Occur, Query, QueryResult, Scorer};

/// Configuration for advanced query execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedQueryConfig {
    /// Enable query optimization.
    pub enable_optimization: bool,

    /// Maximum number of clauses to allow in boolean queries.
    pub max_clause_count: usize,

    /// Enable query caching.
    pub enable_caching: bool,

    /// Query timeout in milliseconds.
    pub timeout_ms: u64,

    /// Enable early termination for expensive queries.
    pub enable_early_termination: bool,

    /// Minimum score threshold for results.
    pub min_score: f32,
}

impl Default for AdvancedQueryConfig {
    fn default() -> Self {
        AdvancedQueryConfig {
            enable_optimization: true,
            max_clause_count: 1024,
            enable_caching: true,
            timeout_ms: 30000, // 30 seconds
            enable_early_termination: true,
            min_score: 0.0,
        }
    }
}

/// Advanced query with complex composition capabilities.
#[derive(Debug)]
pub struct AdvancedQuery {
    /// The core query.
    core_query: Box<dyn Query>,

    /// Field boosts for scoring.
    field_boosts: HashMap<String, f32>,

    /// Query-level boost factor.
    boost: f32,

    /// Minimum score threshold.
    min_score: f32,

    /// Filters to apply (must match).
    filters: Vec<Box<dyn Query>>,

    /// Negative filters (must not match).
    negative_filters: Vec<Box<dyn Query>>,

    /// Post filters (applied after scoring).
    post_filters: Vec<Box<dyn Query>>,

    /// Query configuration.
    config: AdvancedQueryConfig,
}

impl AdvancedQuery {
    /// Create a new advanced query.
    pub fn new(core_query: Box<dyn Query>) -> Self {
        AdvancedQuery {
            core_query,
            field_boosts: HashMap::new(),
            boost: 1.0,
            min_score: 0.0,
            filters: Vec::new(),
            negative_filters: Vec::new(),
            post_filters: Vec::new(),
            config: AdvancedQueryConfig::default(),
        }
    }

    /// Set field boost for scoring.
    pub fn with_field_boost(mut self, field: String, boost: f32) -> Self {
        self.field_boosts.insert(field, boost);
        self
    }

    /// Set query-level boost.
    pub fn with_boost(mut self, boost: f32) -> Self {
        self.boost = boost;
        self
    }

    /// Set minimum score threshold.
    pub fn with_min_score(mut self, min_score: f32) -> Self {
        self.min_score = min_score;
        self
    }

    /// Add a filter (must match).
    pub fn with_filter(mut self, filter: Box<dyn Query>) -> Self {
        self.filters.push(filter);
        self
    }

    /// Add a negative filter (must not match).
    pub fn with_negative_filter(mut self, filter: Box<dyn Query>) -> Self {
        self.negative_filters.push(filter);
        self
    }

    /// Add a post filter.
    pub fn with_post_filter(mut self, filter: Box<dyn Query>) -> Self {
        self.post_filters.push(filter);
        self
    }

    /// Set configuration.
    pub fn with_config(mut self, config: AdvancedQueryConfig) -> Self {
        self.config = config;
        self
    }

    /// Optimize the query for better performance.
    pub fn optimize(&mut self) -> Result<()> {
        if !self.config.enable_optimization {
            return Ok(());
        }

        // Combine filters into boolean query for efficiency
        if !self.filters.is_empty() || !self.negative_filters.is_empty() {
            let mut boolean_builder = BooleanQueryBuilder::new();

            // Add core query as must clause
            boolean_builder = boolean_builder.add_clause(self.core_query.clone_box(), Occur::Must);

            // Add filters as must clauses
            for filter in &self.filters {
                boolean_builder = boolean_builder.add_clause(filter.clone_box(), Occur::Must);
            }

            // Add negative filters as must_not clauses
            for neg_filter in &self.negative_filters {
                boolean_builder =
                    boolean_builder.add_clause(neg_filter.clone_box(), Occur::MustNot);
            }

            // Replace core query with optimized boolean query
            self.core_query = Box::new(boolean_builder.build());
            self.filters.clear();
            self.negative_filters.clear();
        }

        Ok(())
    }

    /// Execute the advanced query with optimization.
    pub fn execute(&mut self, reader: &dyn IndexReader) -> Result<Vec<QueryResult>> {
        // Optimize query first
        self.optimize()?;

        // Get matcher and scorer
        let matcher = self.core_query.matcher(reader)?;
        let scorer = self.core_query.scorer(reader)?;

        let mut results = Vec::new();
        let start_time = std::time::Instant::now();

        // Execute core query
        let mut matcher = matcher;
        while matcher.next()? {
            // Check timeout
            if self.config.timeout_ms > 0
                && start_time.elapsed().as_millis() > self.config.timeout_ms as u128
            {
                break;
            }

            let doc_id = matcher.doc_id() as u32;

            // Calculate score with field boosts
            let mut score = scorer.score(doc_id as u64, 1.0); // Use default term frequency
            score = self.apply_field_boosts(doc_id, score, reader)?;
            score *= self.boost;

            // Apply minimum score threshold
            if score < self.min_score.max(self.config.min_score) {
                continue;
            }

            // Apply post filters
            if !self.apply_post_filters(doc_id, reader)? {
                continue;
            }

            results.push(QueryResult { doc_id, score });

            // Early termination check
            if self.config.enable_early_termination && results.len() > 10000 {
                break;
            }
        }

        // Sort by score descending
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        Ok(results)
    }

    /// Apply field boosts to the score.
    fn apply_field_boosts(
        &self,
        _doc_id: u32,
        base_score: f32,
        _reader: &dyn IndexReader,
    ) -> Result<f32> {
        if self.field_boosts.is_empty() {
            return Ok(base_score);
        }

        // For now, return base score - field boost calculation would require
        // more detailed field-level scoring information
        Ok(base_score)
    }

    /// Apply post filters to a document.
    fn apply_post_filters(&self, doc_id: u32, reader: &dyn IndexReader) -> Result<bool> {
        for filter in &self.post_filters {
            let mut matcher = filter.matcher(reader)?;
            // Skip to the target document and check if it matches
            if !matcher.skip_to(doc_id as u64)? || matcher.doc_id() != doc_id as u64 {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

impl Query for AdvancedQuery {
    fn matcher(&self, reader: &dyn IndexReader) -> Result<Box<dyn Matcher>> {
        self.core_query.matcher(reader)
    }

    fn scorer(&self, reader: &dyn IndexReader) -> Result<Box<dyn Scorer>> {
        self.core_query.scorer(reader)
    }

    fn boost(&self) -> f32 {
        self.boost
    }

    fn set_boost(&mut self, boost: f32) {
        self.boost = boost;
    }

    fn description(&self) -> String {
        format!(
            "AdvancedQuery(core: {}, boost: {})",
            self.core_query.description(),
            self.boost
        )
    }

    fn is_empty(&self, reader: &dyn IndexReader) -> Result<bool> {
        self.core_query.is_empty(reader)
    }

    fn cost(&self, reader: &dyn IndexReader) -> Result<u64> {
        let base_cost = self.core_query.cost(reader)?;
        let filter_cost = self
            .filters
            .iter()
            .map(|f| f.cost(reader))
            .collect::<Result<Vec<_>>>()?
            .iter()
            .sum::<u64>();
        Ok(base_cost + filter_cost)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn Query> {
        Box::new(self.clone())
    }
}

impl Clone for AdvancedQuery {
    fn clone(&self) -> Self {
        AdvancedQuery {
            core_query: self.core_query.clone_box(),
            field_boosts: self.field_boosts.clone(),
            boost: self.boost,
            min_score: self.min_score,
            filters: self.filters.iter().map(|f| f.clone_box()).collect(),
            negative_filters: self
                .negative_filters
                .iter()
                .map(|f| f.clone_box())
                .collect(),
            post_filters: self.post_filters.iter().map(|f| f.clone_box()).collect(),
            config: self.config.clone(),
        }
    }
}

/// Builder for complex boolean queries with advanced features.
#[derive(Debug)]
pub struct BooleanQueryBuilder {
    /// Query clauses.
    clauses: Vec<(Box<dyn Query>, Occur)>,

    /// Minimum number of should clauses that must match.
    minimum_should_match: usize,

    /// Query boost.
    boost: f32,

    /// Configuration.
    config: AdvancedQueryConfig,
}

impl BooleanQueryBuilder {
    /// Create a new boolean query builder.
    pub fn new() -> Self {
        BooleanQueryBuilder {
            clauses: Vec::new(),
            minimum_should_match: 0,
            boost: 1.0,
            config: AdvancedQueryConfig::default(),
        }
    }

    /// Add a query clause.
    pub fn add_clause(mut self, query: Box<dyn Query>, occur: Occur) -> Self {
        self.clauses.push((query, occur));
        self
    }

    /// Set minimum should match.
    pub fn minimum_should_match(mut self, count: usize) -> Self {
        self.minimum_should_match = count;
        self
    }

    /// Set boost.
    pub fn boost(mut self, boost: f32) -> Self {
        self.boost = boost;
        self
    }

    /// Set configuration.
    pub fn config(mut self, config: AdvancedQueryConfig) -> Self {
        self.config = config;
        self
    }

    /// Build the boolean query.
    pub fn build(self) -> BooleanQuery {
        let mut boolean_query = BooleanQuery::new();

        for (query, occur) in self.clauses {
            match occur {
                Occur::Must => boolean_query.add_must(query),
                Occur::Should => boolean_query.add_should(query),
                Occur::MustNot => boolean_query.add_must_not(query),
            }
        }

        if self.minimum_should_match > 0 {
            boolean_query = boolean_query.with_minimum_should_match(self.minimum_should_match);
        }

        boolean_query.with_boost(self.boost)
    }
}

impl Default for BooleanQueryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Multi-field query that searches across multiple fields.
#[derive(Debug, Clone)]
pub struct MultiFieldQuery {
    /// Query text.
    query_text: String,

    /// Fields to search with their boosts.
    fields: HashMap<String, f32>,

    /// Query type for each field.
    query_type: MultiFieldQueryType,

    /// Cross-field matching strategy.
    tie_breaker: f32,
}

/// Type of multi-field query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MultiFieldQueryType {
    /// Best matching field.
    BestFields,

    /// Most matching fields.
    MostFields,

    /// Cross-field matching.
    CrossFields,

    /// Boolean combination.
    Boolean,
}

impl MultiFieldQuery {
    /// Create a new multi-field query.
    pub fn new(query_text: String) -> Self {
        MultiFieldQuery {
            query_text,
            fields: HashMap::new(),
            query_type: MultiFieldQueryType::BestFields,
            tie_breaker: 0.0,
        }
    }

    /// Add a field with boost.
    pub fn add_field(mut self, field: String, boost: f32) -> Self {
        self.fields.insert(field, boost);
        self
    }

    /// Set query type.
    pub fn query_type(mut self, query_type: MultiFieldQueryType) -> Self {
        self.query_type = query_type;
        self
    }

    /// Set tie breaker for best fields queries.
    pub fn tie_breaker(mut self, tie_breaker: f32) -> Self {
        self.tie_breaker = tie_breaker;
        self
    }
}

impl Query for MultiFieldQuery {
    fn matcher(&self, reader: &dyn IndexReader) -> Result<Box<dyn Matcher>> {
        // Create boolean query based on type
        let mut boolean_builder = BooleanQueryBuilder::new();

        match self.query_type {
            MultiFieldQueryType::BestFields | MultiFieldQueryType::Boolean => {
                // Add each field as a should clause
                for field in self.fields.keys() {
                    let term_query =
                        crate::query::TermQuery::new(field.clone(), self.query_text.clone());
                    boolean_builder =
                        boolean_builder.add_clause(Box::new(term_query), Occur::Should);
                }
            }
            MultiFieldQueryType::MostFields => {
                // All fields should match
                for field in self.fields.keys() {
                    let term_query =
                        crate::query::TermQuery::new(field.clone(), self.query_text.clone());
                    boolean_builder = boolean_builder.add_clause(Box::new(term_query), Occur::Must);
                }
            }
            MultiFieldQueryType::CrossFields => {
                // Create phrase query across fields (simplified)
                let mut combined_query = BooleanQuery::new();
                for field in self.fields.keys() {
                    let term_query =
                        crate::query::TermQuery::new(field.clone(), self.query_text.clone());
                    combined_query.add_should(Box::new(term_query));
                }
                return combined_query.matcher(reader);
            }
        }

        boolean_builder.build().matcher(reader)
    }

    fn scorer(&self, reader: &dyn IndexReader) -> Result<Box<dyn Scorer>> {
        // Create boolean query and use its scorer
        let mut boolean_builder = BooleanQueryBuilder::new();

        match self.query_type {
            MultiFieldQueryType::BestFields | MultiFieldQueryType::Boolean => {
                for field in self.fields.keys() {
                    let term_query =
                        crate::query::TermQuery::new(field.clone(), self.query_text.clone());
                    boolean_builder =
                        boolean_builder.add_clause(Box::new(term_query), Occur::Should);
                }
            }
            MultiFieldQueryType::MostFields => {
                for field in self.fields.keys() {
                    let term_query =
                        crate::query::TermQuery::new(field.clone(), self.query_text.clone());
                    boolean_builder = boolean_builder.add_clause(Box::new(term_query), Occur::Must);
                }
            }
            MultiFieldQueryType::CrossFields => {
                let mut combined_query = BooleanQuery::new();
                for field in self.fields.keys() {
                    let term_query =
                        crate::query::TermQuery::new(field.clone(), self.query_text.clone());
                    combined_query.add_should(Box::new(term_query));
                }
                return combined_query.scorer(reader);
            }
        }

        boolean_builder.build().scorer(reader)
    }

    fn boost(&self) -> f32 {
        1.0 // Default boost for multi-field queries
    }

    fn set_boost(&mut self, _boost: f32) {
        // Multi-field queries manage boosts per field
    }

    fn description(&self) -> String {
        format!(
            "MultiFieldQuery(text: {}, fields: {:?})",
            self.query_text,
            self.fields.keys().collect::<Vec<_>>()
        )
    }

    fn is_empty(&self, _reader: &dyn IndexReader) -> Result<bool> {
        Ok(self.query_text.is_empty() || self.fields.is_empty())
    }

    fn cost(&self, _reader: &dyn IndexReader) -> Result<u64> {
        // Estimate cost based on number of fields
        Ok(self.fields.len() as u64 * 100)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn Query> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::TermQuery;
    use crate::schema::{Schema, TextField};

    #[allow(dead_code)]
    fn create_test_schema() -> Schema {
        let mut schema = Schema::new().unwrap();
        schema
            .add_field("title", Box::new(TextField::new().stored(true)))
            .unwrap();
        schema
            .add_field("content", Box::new(TextField::new()))
            .unwrap();
        schema
    }

    #[test]
    fn test_advanced_query_creation() {
        let core_query = Box::new(TermQuery::new("title".to_string(), "test".to_string()));
        let advanced_query = AdvancedQuery::new(core_query)
            .with_boost(2.0)
            .with_min_score(0.5)
            .with_field_boost("title".to_string(), 1.5);

        assert_eq!(advanced_query.boost, 2.0);
        assert_eq!(advanced_query.min_score, 0.5);
        assert_eq!(advanced_query.field_boosts.get("title"), Some(&1.5));
    }

    #[test]
    fn test_boolean_query_builder() {
        let builder = BooleanQueryBuilder::new()
            .minimum_should_match(2)
            .boost(1.5);

        assert_eq!(builder.minimum_should_match, 2);
        assert_eq!(builder.boost, 1.5);
    }

    #[test]
    fn test_multi_field_query() {
        let query = MultiFieldQuery::new("test query".to_string())
            .add_field("title".to_string(), 2.0)
            .add_field("content".to_string(), 1.0)
            .query_type(MultiFieldQueryType::BestFields)
            .tie_breaker(0.3);

        assert_eq!(query.query_text, "test query");
        assert_eq!(query.fields.len(), 2);
        assert_eq!(query.tie_breaker, 0.3);
    }

    #[test]
    fn test_advanced_query_config() {
        let config = AdvancedQueryConfig {
            enable_optimization: false,
            max_clause_count: 500,
            timeout_ms: 10000,
            ..Default::default()
        };

        assert!(!config.enable_optimization);
        assert_eq!(config.max_clause_count, 500);
        assert_eq!(config.timeout_ms, 10000);
    }
}
