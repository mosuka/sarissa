//! Query expansion system for improving search recall.
//!
//! This module provides automatic query expansion using various techniques:
//! - Synonym expansion
//! - Word embeddings for semantic expansion
//! - Query intent classification
//! - Statistical co-occurrence expansion

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::analysis::analyzer::Analyzer;
use crate::error::Result;
use crate::ml::MLContext;
use crate::query::{BooleanQuery, BooleanQueryBuilder, Query, TermQuery};

/// Configuration for query expansion system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryExpansionConfig {
    /// Enable query expansion.
    pub enabled: bool,
    /// Maximum number of expansion terms to add.
    pub max_expansions: usize,
    /// Minimum similarity threshold for semantic expansion.
    pub similarity_threshold: f64,
    /// Enable synonym expansion.
    pub enable_synonyms: bool,
    /// Enable semantic expansion using word embeddings.
    pub enable_semantic: bool,
    /// Enable statistical co-occurrence expansion.
    pub enable_statistical: bool,
    /// Weight for original query terms vs expanded terms.
    pub original_term_weight: f64,
    /// Weight for expanded terms.
    pub expansion_term_weight: f64,
    /// Path to synonym dictionary.
    pub synonym_dict_path: Option<String>,
    /// Path to word embeddings.
    pub embeddings_path: Option<String>,
    /// Use ML-based intent classification instead of keyword-based.
    pub use_ml_classifier: bool,
    /// Path to ML training data (JSON file with IntentSample array).
    pub ml_training_data_path: Option<String>,
    /// Language code for ML training data (e.g., "en", "ja").
    pub ml_training_language: Option<String>,
}

impl Default for QueryExpansionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_expansions: 10,
            similarity_threshold: 0.6,
            enable_synonyms: true,
            enable_semantic: true,
            enable_statistical: true,
            original_term_weight: 1.0,
            expansion_term_weight: 0.5,
            synonym_dict_path: None,
            embeddings_path: None,
            use_ml_classifier: false,
            ml_training_data_path: None,
            ml_training_language: None,
        }
    }
}

/// Query expansion system.
pub struct QueryExpansion {
    /// Configuration.
    config: QueryExpansionConfig,
    /// Synonym dictionary.
    synonym_dict: SynonymDictionary,
    /// Word embeddings for semantic expansion.
    embeddings: WordEmbeddings,
    /// Statistical co-occurrence model.
    cooccurrence_model: CoOccurrenceModel,
    /// Intent classifier (either keyword-based or ML-based).
    intent_classifier: crate::ml::intent_classifier::IntentClassifier,
    /// Analyzer for tokenization.
    analyzer: Arc<dyn Analyzer>,
}

impl QueryExpansion {
    /// Create a new query expansion system.
    pub fn new(config: QueryExpansionConfig, analyzer: Arc<dyn Analyzer>) -> Result<Self> {
        let synonym_dict = if let Some(ref path) = config.synonym_dict_path {
            SynonymDictionary::load_from_file(path)?
        } else {
            SynonymDictionary::new()
        };

        let embeddings = if let Some(ref path) = config.embeddings_path {
            WordEmbeddings::load_from_file(path)?
        } else {
            WordEmbeddings::new()
        };

        // Initialize intent classifier (ML-based or keyword-based)
        let intent_classifier = if config.use_ml_classifier {
            if let Some(ref path) = config.ml_training_data_path {
                let samples =
                    crate::ml::intent_classifier::IntentClassifier::load_training_data(path)?;
                crate::ml::intent_classifier::IntentClassifier::new_ml_based(
                    samples,
                    analyzer.clone(),
                )?
            } else {
                // Fallback to keyword-based if no training data provided
                Self::create_default_keyword_classifier(analyzer.clone())
            }
        } else {
            Self::create_default_keyword_classifier(analyzer.clone())
        };

        Ok(Self {
            config,
            synonym_dict,
            embeddings,
            cooccurrence_model: CoOccurrenceModel::new(),
            intent_classifier,
            analyzer,
        })
    }

    /// Create default keyword-based classifier.
    fn create_default_keyword_classifier(
        analyzer: Arc<dyn Analyzer>,
    ) -> crate::ml::intent_classifier::IntentClassifier {
        let informational = std::collections::HashSet::from([
            "what".to_string(),
            "how".to_string(),
            "why".to_string(),
            "when".to_string(),
            "where".to_string(),
            "who".to_string(),
            "definition".to_string(),
            "explain".to_string(),
        ]);
        let navigational = std::collections::HashSet::from([
            "homepage".to_string(),
            "website".to_string(),
            "site".to_string(),
            "login".to_string(),
            "official".to_string(),
        ]);
        let transactional = std::collections::HashSet::from([
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

    /// Expand a query using all enabled expansion techniques.
    /// Returns an ExpandedQuery that can be converted to a BooleanQuery for search.
    pub fn expand_query(
        &self,
        original_query: &str,
        field: &str,
        context: &MLContext,
    ) -> Result<ExpandedQuery> {
        if !self.config.enabled {
            // Create a simple term query for the original query
            let tokens = self.tokenize_query(original_query)?;
            let query = self.build_query_from_tokens(&tokens, field);

            return Ok(ExpandedQuery {
                original_query: query,
                expanded_queries: Vec::new(),
                intent: QueryIntent::Unknown,
                confidence: 1.0,
            });
        }

        let tokens = self.tokenize_query(original_query)?;
        let mut expanded_queries = Vec::new();

        // Classify query intent
        let intent = self.intent_classifier.predict(original_query)?;

        // Apply different expansion strategies based on intent
        match intent {
            QueryIntent::Informational => {
                if self.config.enable_synonyms {
                    expanded_queries.extend(self.expand_with_synonyms(&tokens, field)?);
                }
                if self.config.enable_semantic {
                    expanded_queries.extend(self.expand_with_semantics(&tokens, field)?);
                }
            }
            QueryIntent::Navigational => {
                // For navigational queries, be more conservative
                if self.config.enable_synonyms {
                    expanded_queries.extend(self.expand_with_synonyms(&tokens, field)?);
                }
            }
            QueryIntent::Transactional => {
                // Focus on related action terms
                if self.config.enable_statistical {
                    expanded_queries.extend(self.expand_with_statistics(&tokens, field, context)?);
                }
            }
            QueryIntent::Unknown => {
                // Apply all expansion methods
                if self.config.enable_synonyms {
                    expanded_queries.extend(self.expand_with_synonyms(&tokens, field)?);
                }
                if self.config.enable_semantic {
                    expanded_queries.extend(self.expand_with_semantics(&tokens, field)?);
                }
                if self.config.enable_statistical {
                    expanded_queries.extend(self.expand_with_statistics(&tokens, field, context)?);
                }
            }
        }

        // Deduplicate by term text (since Query doesn't implement Hash/Eq)
        expanded_queries.sort_by(|a, b| {
            a.query
                .description()
                .cmp(&b.query.description())
                .then(b.confidence.partial_cmp(&a.confidence).unwrap())
        });
        expanded_queries.dedup_by(|a, b| a.query.description() == b.query.description());

        // Filter by threshold and limit
        expanded_queries.retain(|clause| clause.confidence >= self.config.similarity_threshold);
        expanded_queries.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        expanded_queries.truncate(self.config.max_expansions);

        let confidence = self.calculate_expansion_confidence(&tokens, &expanded_queries);

        // Build original query
        let original_query = self.build_query_from_tokens(&tokens, field);

        Ok(ExpandedQuery {
            original_query,
            expanded_queries,
            intent,
            confidence,
        })
    }

    /// Update expansion models with user feedback.
    pub fn update_with_feedback(
        &mut self,
        original_query: &str,
        clicked_documents: &[String],
    ) -> Result<()> {
        // Update co-occurrence model with successful query-document pairs
        self.cooccurrence_model
            .update_with_clicks(original_query, clicked_documents)?;

        Ok(())
    }

    // Private expansion methods

    fn expand_with_synonyms(
        &self,
        terms: &[String],
        field: &str,
    ) -> Result<Vec<ExpandedQueryClause>> {
        let mut expansions = Vec::new();

        for term in terms {
            if let Some(synonyms) = self.synonym_dict.get_synonyms(term) {
                for synonym in synonyms {
                    let mut query = Box::new(TermQuery::new(field.to_string(), synonym.clone()))
                        as Box<dyn Query>;
                    let confidence = 0.8; // High confidence for dictionary synonyms
                    query.set_boost((confidence * self.config.expansion_term_weight) as f32);

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

    fn expand_with_semantics(
        &self,
        terms: &[String],
        field: &str,
    ) -> Result<Vec<ExpandedQueryClause>> {
        let mut expansions = Vec::new();

        for term in terms {
            if let Some(similar_terms) = self.embeddings.get_similar_words(term, 5)? {
                for (similar_term, similarity) in similar_terms {
                    if similarity >= self.config.similarity_threshold {
                        let mut query =
                            Box::new(TermQuery::new(field.to_string(), similar_term.clone()))
                                as Box<dyn Query>;
                        query.set_boost((similarity * self.config.expansion_term_weight) as f32);

                        expansions.push(ExpandedQueryClause {
                            query,
                            confidence: similarity,
                            expansion_type: ExpansionType::Semantic,
                            source_term: term.clone(),
                        });
                    }
                }
            }
        }

        Ok(expansions)
    }

    fn expand_with_statistics(
        &self,
        terms: &[String],
        field: &str,
        _context: &MLContext,
    ) -> Result<Vec<ExpandedQueryClause>> {
        let mut expansions = Vec::new();

        for term in terms {
            if let Some(cooccurrent_terms) =
                self.cooccurrence_model.get_cooccurrent_terms(term, 3)?
            {
                for (cooccurrent_term, score) in cooccurrent_terms {
                    if score >= self.config.similarity_threshold {
                        let mut query =
                            Box::new(TermQuery::new(field.to_string(), cooccurrent_term.clone()))
                                as Box<dyn Query>;
                        query.set_boost((score * self.config.expansion_term_weight) as f32);

                        expansions.push(ExpandedQueryClause {
                            query,
                            confidence: score,
                            expansion_type: ExpansionType::Statistical,
                            source_term: term.clone(),
                        });
                    }
                }
            }
        }

        Ok(expansions)
    }

    /// Tokenize query using the analyzer.
    fn tokenize_query(&self, query: &str) -> Result<Vec<String>> {
        let tokens: Vec<String> = self
            .analyzer
            .analyze(query)?
            .map(|token| token.text)
            .collect();
        Ok(tokens)
    }

    /// Build a Query object from tokens.
    fn build_query_from_tokens(&self, tokens: &[String], field: &str) -> Box<dyn Query> {
        if tokens.is_empty() {
            // Return an empty boolean query
            let builder = BooleanQueryBuilder::new();
            return Box::new(builder.build());
        }

        if tokens.len() == 1 {
            // Single term query
            let mut query =
                Box::new(TermQuery::new(field.to_string(), tokens[0].clone())) as Box<dyn Query>;
            query.set_boost(self.config.original_term_weight as f32);
            return query;
        }

        // Multiple terms - create a boolean query with SHOULD clauses
        let mut builder = BooleanQueryBuilder::new();
        for token in tokens {
            let mut term_query =
                Box::new(TermQuery::new(field.to_string(), token.clone())) as Box<dyn Query>;
            term_query.set_boost(self.config.original_term_weight as f32);
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
#[derive(Debug)]
pub struct SynonymDictionary {
    synonyms: HashMap<String, Vec<String>>,
}

impl Default for SynonymDictionary {
    fn default() -> Self {
        Self::new()
    }
}

impl SynonymDictionary {
    pub fn new() -> Self {
        let mut dict = Self {
            synonyms: HashMap::new(),
        };

        // デフォルトの同義語を追加（小文字で統一）
        dict.add_synonym_group(vec![
            "ml".to_string(),
            "machine learning".to_string(),
            "machine-learning".to_string(),
            "machinelearning".to_string(),
        ]);

        dict.add_synonym_group(vec![
            "ai".to_string(),
            "artificial intelligence".to_string(),
            "artificial-intelligence".to_string(),
            "artificialintelligence".to_string(),
        ]);

        dict.add_synonym_group(vec![
            "programming".to_string(),
            "coding".to_string(),
            "development".to_string(),
            "software engineering".to_string(),
        ]);

        dict.add_synonym_group(vec![
            "python".to_string(),
            "Python".to_string(),
            "py".to_string(),
        ]);

        dict.add_synonym_group(vec![
            "algorithm".to_string(),
            "algorithms".to_string(),
            "algo".to_string(),
        ]);

        dict.add_synonym_group(vec![
            "data".to_string(),
            "dataset".to_string(),
            "data science".to_string(),
            "data-science".to_string(),
        ]);

        // Japanese synonyms (tokenized forms)
        dict.add_synonym_group(vec!["機械".to_string(), "マシン".to_string()]);

        dict.add_synonym_group(vec![
            "学習".to_string(),
            "勉強".to_string(),
            "習得".to_string(),
            "ラーニング".to_string(),
        ]);

        dict.add_synonym_group(vec!["人工".to_string(), "artificial".to_string()]);

        dict.add_synonym_group(vec![
            "知能".to_string(),
            "intelligence".to_string(),
            "ai".to_string(),
        ]);

        dict.add_synonym_group(vec![
            "プログラミング".to_string(),
            "コーディング".to_string(),
            "開発".to_string(),
        ]);

        dict.add_synonym_group(vec![
            "購入".to_string(),
            "買う".to_string(),
            "注文".to_string(),
        ]);

        dict
    }

    pub fn load_from_file(_path: &str) -> Result<Self> {
        // Placeholder implementation
        Ok(Self::new())
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

    pub fn get_similar_words(
        &self,
        _word: &str,
        _count: usize,
    ) -> Result<Option<Vec<(String, f64)>>> {
        // Placeholder implementation - would compute cosine similarity
        Ok(None)
    }

    #[allow(dead_code)]
    fn cosine_similarity(&self, vec1: &[f32], vec2: &[f32]) -> f64 {
        if vec1.len() != vec2.len() {
            return 0.0;
        }

        let dot_product: f32 = vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f32 = vec1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = vec2.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm1 == 0.0 || norm2 == 0.0 {
            0.0
        } else {
            (dot_product / (norm1 * norm2)) as f64
        }
    }
}

/// Statistical co-occurrence model for expansion.
#[derive(Debug)]
pub struct CoOccurrenceModel {
    term_cooccurrences: HashMap<String, HashMap<String, f64>>,
}

impl Default for CoOccurrenceModel {
    fn default() -> Self {
        Self::new()
    }
}

impl CoOccurrenceModel {
    pub fn new() -> Self {
        Self {
            term_cooccurrences: HashMap::new(),
        }
    }

    pub fn update_with_clicks(&mut self, query: &str, clicked_documents: &[String]) -> Result<()> {
        let _query_terms: Vec<String> =
            query.split_whitespace().map(|s| s.to_lowercase()).collect();

        // Update co-occurrence statistics based on successful query-document pairs
        for _doc_id in clicked_documents {
            // In a real implementation, you'd analyze document content
            // and update co-occurrence counts between query terms and document terms
        }

        Ok(())
    }

    pub fn get_cooccurrent_terms(
        &self,
        term: &str,
        count: usize,
    ) -> Result<Option<Vec<(String, f64)>>> {
        if let Some(cooccurrences) = self.term_cooccurrences.get(term) {
            let mut terms: Vec<(String, f64)> = cooccurrences
                .iter()
                .map(|(t, &score)| (t.clone(), score))
                .collect();

            terms.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            terms.truncate(count);

            Ok(Some(terms))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::analyzer::language::EnglishAnalyzer;

    #[test]
    fn test_query_expansion_config_default() {
        let config = QueryExpansionConfig::default();
        assert!(config.enabled);
        assert_eq!(config.max_expansions, 10);
        assert_eq!(config.similarity_threshold, 0.6);
    }

    #[test]
    fn test_query_expansion_creation() {
        let config = QueryExpansionConfig::default();
        let analyzer = Arc::new(EnglishAnalyzer::new().unwrap());
        let expansion = QueryExpansion::new(config, analyzer).unwrap();
        assert!(expansion.config.enabled);
    }

    #[test]
    fn test_intent_classification() {
        let analyzer = Arc::new(EnglishAnalyzer::new().unwrap());
        let config = QueryExpansionConfig::default();
        let expansion = QueryExpansion::new(config, analyzer).unwrap();

        let intent = expansion.intent_classifier.predict("what is rust").unwrap();
        assert_eq!(intent, QueryIntent::Informational);

        let intent = expansion
            .intent_classifier
            .predict("rust official website")
            .unwrap();
        assert_eq!(intent, QueryIntent::Navigational);

        let intent = expansion
            .intent_classifier
            .predict("buy rust book")
            .unwrap();
        assert_eq!(intent, QueryIntent::Transactional);
    }

    #[test]
    fn test_synonym_dictionary() {
        let mut dict = SynonymDictionary::new();
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
    fn test_expanded_query_to_boolean() {
        let analyzer = Arc::new(EnglishAnalyzer::new().unwrap());
        let config = QueryExpansionConfig::default();
        let expansion = QueryExpansion::new(config, analyzer).unwrap();

        let ml_context = MLContext::default();
        let expanded = expansion
            .expand_query("rust programming", "content", &ml_context)
            .unwrap();

        let boolean_query = expanded.to_boolean_query();
        assert!(boolean_query.clauses().len() > 0);
    }
}
