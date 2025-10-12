//! Query expansion system for improving search recall.
//!
//! This module provides automatic query expansion using various techniques:
//! - Synonym expansion
//! - Word embeddings for semantic expansion
//! - Statistical co-occurrence expansion
//!
//! # Architecture
//!
//! The query expansion system uses a Strategy pattern with the following components:
//! - `QueryExpander` trait: Defines the interface for expansion strategies
//! - `SynonymQueryExpander`: Dictionary-based synonym expansion
//! - `SemanticQueryExpander`: Embedding-based semantic expansion
//! - `StatisticalQueryExpander`: Co-occurrence based statistical expansion
//! - `QueryExpansionBuilder`: Fluent API for building expansion pipelines
//!
//! # Example
//!
//! ```rust,no_run
//! use sarissa::ml::query_expansion::QueryExpansionBuilder;
//! use sarissa::analysis::StandardAnalyzer;
//! use std::sync::Arc;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let analyzer = Arc::new(StandardAnalyzer::new()?);
//!
//! let expansion = QueryExpansionBuilder::new(analyzer)
//!     .with_synonyms(Some("synonyms.json"), 0.5)?
//!     .with_statistical(0.3)
//!     .max_expansions(10)
//!     .build()?;
//! # Ok(())
//! # }
//! ```

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::analysis::analyzer::Analyzer;
use crate::error::Result;
use crate::ml::MLContext;
use crate::query::{BooleanQuery, BooleanQueryBuilder, Query, TermQuery};

/// Query expansion strategy trait.
///
/// Implementations of this trait provide different methods for expanding
/// query terms to improve search recall.
pub trait QueryExpander: Send + Sync {
    /// Expand query tokens into additional search terms.
    ///
    /// # Arguments
    /// * `tokens` - The original query tokens to expand
    /// * `field` - The field name for the expanded queries
    /// * `context` - ML context containing user session and search history
    ///
    /// # Returns
    /// A vector of expanded query clauses with confidence scores
    fn expand(
        &self,
        tokens: &[String],
        field: &str,
        context: &MLContext,
    ) -> Result<Vec<ExpandedQueryClause>>;

    /// Get the name of this expander for debugging and logging.
    fn name(&self) -> &str;
}

/// Synonym-based query expander.
///
/// Expands query terms using a dictionary of synonyms.
pub struct SynonymQueryExpander {
    dictionary: SynonymDictionary,
    weight: f64,
}

impl SynonymQueryExpander {
    /// Create a new synonym expander.
    ///
    /// # Arguments
    /// * `dict_path` - Optional path to JSON synonym dictionary file
    /// * `weight` - Weight multiplier for expanded terms (0.0-1.0)
    pub fn new(dict_path: Option<&str>, weight: f64) -> Result<Self> {
        Ok(Self {
            dictionary: SynonymDictionary::new(dict_path)?,
            weight,
        })
    }
}

impl QueryExpander for SynonymQueryExpander {
    fn expand(
        &self,
        tokens: &[String],
        field: &str,
        _context: &MLContext,
    ) -> Result<Vec<ExpandedQueryClause>> {
        let mut expansions = Vec::new();

        for term in tokens {
            if let Some(synonyms) = self.dictionary.get_synonyms(term) {
                for synonym in synonyms {
                    let mut query = Box::new(TermQuery::new(field, synonym.clone())) as Box<dyn Query>;
                    let confidence = 0.8;
                    query.set_boost((confidence * self.weight) as f32);

                    expansions.push(ExpandedQueryClause {
                        query,
                        confidence,
                        expansion_type: ExpansionType::Synonym,
                        source_term: term.clone(),
                    });
                }
            }
        }

        Ok(expansions)
    }

    fn name(&self) -> &str {
        "synonym"
    }
}

/// Semantic query expander using word embeddings.
///
/// Expands query terms based on semantic similarity using word vectors.
pub struct SemanticQueryExpander {
    embeddings: WordEmbeddings,
    similarity_threshold: f64,
    weight: f64,
}

impl SemanticQueryExpander {
    /// Create a new semantic expander.
    ///
    /// # Arguments
    /// * `embeddings_path` - Optional path to word embeddings file
    /// * `similarity_threshold` - Minimum similarity score (0.0-1.0)
    /// * `weight` - Weight multiplier for expanded terms (0.0-1.0)
    pub fn new(
        embeddings_path: Option<&str>,
        similarity_threshold: f64,
        weight: f64,
    ) -> Result<Self> {
        let embeddings = if let Some(path) = embeddings_path {
            WordEmbeddings::load_from_file(path)?
        } else {
            WordEmbeddings::new()
        };

        Ok(Self {
            embeddings,
            similarity_threshold,
            weight,
        })
    }
}

impl QueryExpander for SemanticQueryExpander {
    fn expand(
        &self,
        tokens: &[String],
        field: &str,
        _context: &MLContext,
    ) -> Result<Vec<ExpandedQueryClause>> {
        let mut expansions = Vec::new();

        for term in tokens {
            if let Some(similar_terms) = self.embeddings.find_similar(term, self.similarity_threshold) {
                for (similar_term, similarity) in similar_terms {
                    let mut query = Box::new(TermQuery::new(field, similar_term.clone())) as Box<dyn Query>;
                    query.set_boost((similarity * self.weight) as f32);

                    expansions.push(ExpandedQueryClause {
                        query,
                        confidence: similarity,
                        expansion_type: ExpansionType::Semantic,
                        source_term: term.clone(),
                    });
                }
            }
        }

        Ok(expansions)
    }

    fn name(&self) -> &str {
        "semantic"
    }
}

/// Statistical co-occurrence based query expander.
///
/// Expands query terms based on statistical co-occurrence patterns learned from search history.
pub struct StatisticalQueryExpander {
    cooccurrence_model: CoOccurrenceModel,
    weight: f64,
}

impl StatisticalQueryExpander {
    /// Create a new statistical expander.
    ///
    /// # Arguments
    /// * `weight` - Weight multiplier for expanded terms (0.0-1.0)
    pub fn new(weight: f64) -> Self {
        Self {
            cooccurrence_model: CoOccurrenceModel::new(),
            weight,
        }
    }

    /// Update the co-occurrence model with user feedback.
    pub fn update_with_feedback(
        &mut self,
        original_query: &str,
        clicked_documents: &[String],
    ) -> Result<()> {
        self.cooccurrence_model
            .update_with_clicks(original_query, clicked_documents)
    }
}

impl QueryExpander for StatisticalQueryExpander {
    fn expand(
        &self,
        tokens: &[String],
        field: &str,
        context: &MLContext,
    ) -> Result<Vec<ExpandedQueryClause>> {
        let mut expansions = Vec::new();

        for term in tokens {
            if let Some(cooccurring_terms) = self.cooccurrence_model.get_cooccurring_terms(term, context) {
                for (coterm, score) in cooccurring_terms {
                    let mut query = Box::new(TermQuery::new(field, coterm.clone())) as Box<dyn Query>;
                    query.set_boost((score * self.weight) as f32);

                    expansions.push(ExpandedQueryClause {
                        query,
                        confidence: score,
                        expansion_type: ExpansionType::Statistical,
                        source_term: term.clone(),
                    });
                }
            }
        }

        Ok(expansions)
    }

    fn name(&self) -> &str {
        "statistical"
    }
}

/// Query expansion system.
///
/// Combines multiple expansion strategies and manages the expansion pipeline.
pub struct QueryExpansion {
    expanders: Vec<Box<dyn QueryExpander>>,
    intent_classifier: crate::ml::intent_classifier::IntentClassifier,
    analyzer: Arc<dyn Analyzer>,
    max_expansions: usize,
    original_term_weight: f64,
}

impl QueryExpansion {
    /// Create a new query expansion system.
    ///
    /// This constructor is typically not called directly. Use `QueryExpansionBuilder` instead.
    pub fn new(
        expanders: Vec<Box<dyn QueryExpander>>,
        intent_classifier: crate::ml::intent_classifier::IntentClassifier,
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

/// Builder for creating `QueryExpansion` instances.
///
/// Provides a fluent API for configuring query expansion pipelines.
///
/// # Example
///
/// ```rust,no_run
/// use sarissa::ml::query_expansion::QueryExpansionBuilder;
/// use sarissa::analysis::StandardAnalyzer;
/// use std::sync::Arc;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let analyzer = Arc::new(StandardAnalyzer::new()?);
///
/// let expansion = QueryExpansionBuilder::new(analyzer)
///     .with_synonyms(Some("resource/ml/synonyms.json"), 0.5)?
///     .with_semantic(Some("embeddings.bin"), 0.6, 0.4)?
///     .with_statistical(0.3)
///     .max_expansions(10)
///     .build()?;
/// # Ok(())
/// # }
/// ```
pub struct QueryExpansionBuilder {
    expanders: Vec<Box<dyn QueryExpander>>,
    analyzer: Arc<dyn Analyzer>,
    max_expansions: usize,
    original_term_weight: f64,
    use_ml_classifier: bool,
    ml_training_data_path: Option<String>,
}

impl QueryExpansionBuilder {
    /// Create a new builder with the specified analyzer.
    pub fn new(analyzer: Arc<dyn Analyzer>) -> Self {
        Self {
            expanders: Vec::new(),
            analyzer,
            max_expansions: 10,
            original_term_weight: 1.0,
            use_ml_classifier: false,
            ml_training_data_path: None,
        }
    }

    /// Add synonym-based expansion.
    ///
    /// # Arguments
    /// * `dict_path` - Optional path to JSON synonym dictionary file
    /// * `weight` - Weight multiplier for expanded terms (0.0-1.0)
    pub fn with_synonyms(mut self, dict_path: Option<&str>, weight: f64) -> Result<Self> {
        self.expanders
            .push(Box::new(SynonymQueryExpander::new(dict_path, weight)?));
        Ok(self)
    }

    /// Add semantic expansion using word embeddings.
    ///
    /// # Arguments
    /// * `embeddings_path` - Optional path to word embeddings file
    /// * `similarity_threshold` - Minimum similarity score (0.0-1.0)
    /// * `weight` - Weight multiplier for expanded terms (0.0-1.0)
    pub fn with_semantic(
        mut self,
        embeddings_path: Option<&str>,
        similarity_threshold: f64,
        weight: f64,
    ) -> Result<Self> {
        self.expanders.push(Box::new(SemanticQueryExpander::new(
            embeddings_path,
            similarity_threshold,
            weight,
        )?));
        Ok(self)
    }

    /// Add statistical co-occurrence expansion.
    ///
    /// # Arguments
    /// * `weight` - Weight multiplier for expanded terms (0.0-1.0)
    pub fn with_statistical(mut self, weight: f64) -> Self {
        self.expanders
            .push(Box::new(StatisticalQueryExpander::new(weight)));
        self
    }

    /// Add a custom expander.
    ///
    /// # Arguments
    /// * `expander` - A boxed implementation of the `QueryExpander` trait
    pub fn add_expander(mut self, expander: Box<dyn QueryExpander>) -> Self {
        self.expanders.push(expander);
        self
    }

    /// Set the maximum number of expansion terms.
    ///
    /// # Arguments
    /// * `max` - Maximum number of expansions (default: 10)
    pub fn max_expansions(mut self, max: usize) -> Self {
        self.max_expansions = max;
        self
    }

    /// Set the weight for original query terms.
    ///
    /// # Arguments
    /// * `weight` - Boost multiplier for original terms (default: 1.0)
    pub fn original_term_weight(mut self, weight: f64) -> Self {
        self.original_term_weight = weight;
        self
    }

    /// Enable ML-based intent classification.
    ///
    /// # Arguments
    /// * `training_data_path` - Path to training data JSON file
    pub fn with_ml_classifier(mut self, training_data_path: &str) -> Self {
        self.use_ml_classifier = true;
        self.ml_training_data_path = Some(training_data_path.to_string());
        self
    }

    /// Build the `QueryExpansion` instance.
    pub fn build(self) -> Result<QueryExpansion> {
        let intent_classifier = if self.use_ml_classifier {
            if let Some(ref path) = self.ml_training_data_path {
                let samples =
                    crate::ml::intent_classifier::IntentClassifier::load_training_data(path)?;
                crate::ml::intent_classifier::IntentClassifier::new_ml_based(
                    samples,
                    self.analyzer.clone(),
                )?
            } else {
                Self::create_default_keyword_classifier(self.analyzer.clone())
            }
        } else {
            Self::create_default_keyword_classifier(self.analyzer.clone())
        };

        Ok(QueryExpansion::new(
            self.expanders,
            intent_classifier,
            self.analyzer,
            self.max_expansions,
            self.original_term_weight,
        ))
    }

    fn create_default_keyword_classifier(
        analyzer: Arc<dyn Analyzer>,
    ) -> crate::ml::intent_classifier::IntentClassifier {
        let informational = HashSet::from([
            "what".to_string(),
            "how".to_string(),
            "why".to_string(),
            "when".to_string(),
            "where".to_string(),
            "who".to_string(),
            "definition".to_string(),
            "explain".to_string(),
        ]);
        let navigational = HashSet::from([
            "homepage".to_string(),
            "website".to_string(),
            "site".to_string(),
            "login".to_string(),
            "official".to_string(),
        ]);
        let transactional = HashSet::from([
            "buy".to_string(),
            "purchase".to_string(),
            "order".to_string(),
            "download".to_string(),
            "install".to_string(),
            "get".to_string(),
            "free".to_string(),
            "price".to_string(),
        ]);
        crate::ml::intent_classifier::IntentClassifier::new_keyword_based(
            informational,
            navigational,
            transactional,
            analyzer,
        )
    }
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

/// Synonym dictionary for term expansion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynonymDictionary {
    synonyms: HashMap<String, Vec<String>>,
}

impl Default for SynonymDictionary {
    fn default() -> Self {
        Self::new(None).unwrap()
    }
}

impl SynonymDictionary {
    /// Create a new synonym dictionary.
    /// If `path` is provided, loads synonyms from the specified JSON file.
    /// If `path` is `None`, creates an empty dictionary.
    pub fn new(path: Option<&str>) -> Result<Self> {
        match path {
            Some(file_path) => Self::load_from_file(file_path),
            None => Ok(Self {
                synonyms: HashMap::new(),
            }),
        }
    }

    /// Load synonym dictionary from a JSON file.
    /// The JSON file should contain an array of synonym groups, where each group
    /// is an array of terms that are synonyms of each other.
    ///
    /// Example format:
    /// ```json
    /// [
    ///   ["ml", "machine learning", "machine-learning"],
    ///   ["ai", "artificial intelligence"]
    /// ]
    /// ```
    pub fn load_from_file(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            crate::error::SarissaError::storage(format!(
                "Failed to read synonym dictionary file '{}': {}",
                path, e
            ))
        })?;

        let synonym_groups: Vec<Vec<String>> = serde_json::from_str(&content).map_err(|e| {
            crate::error::SarissaError::parse(format!(
                "Failed to parse synonym dictionary JSON from '{}': {}",
                path, e
            ))
        })?;

        let mut dict = Self::new(None)?;
        for group in synonym_groups {
            if !group.is_empty() {
                dict.add_synonym_group(group);
            }
        }

        Ok(dict)
    }

    pub fn get_synonyms(&self, term: &str) -> Option<&Vec<String>> {
        self.synonyms.get(term)
    }

    pub fn add_synonym_group(&mut self, terms: Vec<String>) {
        for (i, term) in terms.iter().enumerate() {
            let mut synonyms = Vec::new();
            for (j, other_term) in terms.iter().enumerate() {
                if i != j {
                    synonyms.push(other_term.clone());
                }
            }
            self.synonyms.insert(term.clone(), synonyms);
        }
    }
}

/// Word embeddings for semantic expansion.
#[derive(Debug)]
pub struct WordEmbeddings {
    #[allow(dead_code)]
    embeddings: HashMap<String, Vec<f32>>,
}

impl Default for WordEmbeddings {
    fn default() -> Self {
        Self::new()
    }
}

impl WordEmbeddings {
    pub fn new() -> Self {
        Self {
            embeddings: HashMap::new(),
        }
    }

    pub fn load_from_file(_path: &str) -> Result<Self> {
        // Placeholder implementation
        Ok(Self::new())
    }

    pub fn find_similar(&self, _term: &str, _threshold: f64) -> Option<Vec<(String, f64)>> {
        // Placeholder implementation
        None
    }
}

/// Statistical co-occurrence model.
#[derive(Debug)]
pub struct CoOccurrenceModel {
    #[allow(dead_code)]
    cooccurrences: HashMap<String, HashMap<String, f64>>,
}

impl Default for CoOccurrenceModel {
    fn default() -> Self {
        Self::new()
    }
}

impl CoOccurrenceModel {
    pub fn new() -> Self {
        Self {
            cooccurrences: HashMap::new(),
        }
    }

    pub fn get_cooccurring_terms(
        &self,
        _term: &str,
        _context: &MLContext,
    ) -> Option<Vec<(String, f64)>> {
        // Placeholder implementation
        None
    }

    pub fn update_with_clicks(
        &mut self,
        _query: &str,
        _clicked_documents: &[String],
    ) -> Result<()> {
        // Placeholder implementation
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::analyzer::language::EnglishAnalyzer;

    #[test]
    fn test_synonym_dictionary() {
        let mut dict = SynonymDictionary::new(None).unwrap();
        dict.add_synonym_group(vec![
            "big".to_string(),
            "large".to_string(),
            "huge".to_string(),
        ]);

        let synonyms = dict.get_synonyms("big").unwrap();
        assert!(synonyms.contains(&"large".to_string()));
        assert!(synonyms.contains(&"huge".to_string()));
        assert!(!synonyms.contains(&"big".to_string()));
    }

    #[test]
    fn test_synonym_dictionary_load_from_file() {
        let dict = SynonymDictionary::load_from_file("resource/ml/synonyms.json").unwrap();

        // Test English synonyms
        let ml_synonyms = dict.get_synonyms("ml");
        assert!(ml_synonyms.is_some());
        let ml_synonyms = ml_synonyms.unwrap();
        assert!(ml_synonyms.contains(&"machine learning".to_string()));
        assert!(ml_synonyms.contains(&"machine-learning".to_string()));

        // Test Japanese synonyms
        let learning_synonyms = dict.get_synonyms("学習");
        assert!(learning_synonyms.is_some());
        let learning_synonyms = learning_synonyms.unwrap();
        assert!(learning_synonyms.contains(&"勉強".to_string()));
        assert!(learning_synonyms.contains(&"習得".to_string()));
    }

    #[test]
    fn test_synonym_dictionary_with_path() {
        let dict = SynonymDictionary::new(Some("resource/ml/synonyms.json")).unwrap();

        // Verify that synonyms are loaded from file
        assert!(dict.get_synonyms("ml").is_some());
        assert!(dict.get_synonyms("ai").is_some());
    }

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

        assert!(expanded.expanded_queries.len() > 0);
        assert_eq!(expanded.intent, QueryIntent::Unknown);
    }
}
