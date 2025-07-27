//! Query expansion system for improving search recall.
//!
//! This module provides automatic query expansion using various techniques:
//! - Synonym expansion
//! - Word embeddings for semantic expansion
//! - Query intent classification
//! - Statistical co-occurrence expansion

use crate::error::Result;
use crate::ml::{MLContext, SearchHistoryItem};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

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
    /// Query intent classifier.
    intent_classifier: IntentClassifier,
}

impl QueryExpansion {
    /// Create a new query expansion system.
    pub fn new(config: QueryExpansionConfig) -> Result<Self> {
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

        Ok(Self {
            config,
            synonym_dict,
            embeddings,
            cooccurrence_model: CoOccurrenceModel::new(),
            intent_classifier: IntentClassifier::new(),
        })
    }

    /// Expand a query using all enabled expansion techniques.
    pub fn expand_query(&self, original_query: &str, context: &MLContext) -> Result<ExpandedQuery> {
        if !self.config.enabled {
            return Ok(ExpandedQuery {
                original_terms: self.tokenize_query(original_query),
                expanded_terms: Vec::new(),
                intent: QueryIntent::Unknown,
                confidence: 1.0,
            });
        }

        let original_terms = self.tokenize_query(original_query);
        let mut expansions = HashSet::new();

        // Classify query intent first
        let intent = self.intent_classifier.classify(&original_terms)?;

        // Apply different expansion strategies based on intent
        match intent {
            QueryIntent::Informational => {
                if self.config.enable_synonyms {
                    expansions.extend(self.expand_with_synonyms(&original_terms)?);
                }
                if self.config.enable_semantic {
                    expansions.extend(self.expand_with_semantics(&original_terms)?);
                }
            }
            QueryIntent::Navigational => {
                // For navigational queries, be more conservative
                if self.config.enable_synonyms {
                    expansions.extend(self.expand_with_synonyms(&original_terms)?);
                }
            }
            QueryIntent::Transactional => {
                // Focus on related action terms
                if self.config.enable_statistical {
                    expansions.extend(self.expand_with_statistics(&original_terms, context)?);
                }
            }
            QueryIntent::Unknown => {
                // Apply all expansion methods
                if self.config.enable_synonyms {
                    expansions.extend(self.expand_with_synonyms(&original_terms)?);
                }
                if self.config.enable_semantic {
                    expansions.extend(self.expand_with_semantics(&original_terms)?);
                }
                if self.config.enable_statistical {
                    expansions.extend(self.expand_with_statistics(&original_terms, context)?);
                }
            }
        }

        // Filter and limit expansions
        let mut expansion_terms: Vec<ExpansionTerm> = expansions
            .into_iter()
            .filter(|term| term.confidence >= self.config.similarity_threshold)
            .collect();

        // Sort by confidence and limit
        expansion_terms.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        expansion_terms.truncate(self.config.max_expansions);

        let confidence = self.calculate_expansion_confidence(&original_terms, &expansion_terms);

        Ok(ExpandedQuery {
            original_terms,
            expanded_terms: expansion_terms,
            intent,
            confidence,
        })
    }

    /// Update expansion models with user feedback.
    pub fn update_with_feedback(
        &mut self,
        original_query: &str,
        clicked_documents: &[String],
        search_history: &[SearchHistoryItem],
    ) -> Result<()> {
        // Update co-occurrence model with successful query-document pairs
        self.cooccurrence_model
            .update_with_clicks(original_query, clicked_documents)?;

        // Update intent classifier with search patterns
        self.intent_classifier.update_with_history(search_history)?;

        Ok(())
    }

    // Private expansion methods

    fn expand_with_synonyms(&self, terms: &[String]) -> Result<HashSet<ExpansionTerm>> {
        let mut expansions = HashSet::new();

        for term in terms {
            if let Some(synonyms) = self.synonym_dict.get_synonyms(term) {
                for synonym in synonyms {
                    expansions.insert(ExpansionTerm {
                        term: synonym.clone(),
                        confidence: 0.8, // High confidence for dictionary synonyms
                        expansion_type: ExpansionType::Synonym,
                        source_term: term.clone(),
                    });
                }
            }
        }

        Ok(expansions)
    }

    fn expand_with_semantics(&self, terms: &[String]) -> Result<HashSet<ExpansionTerm>> {
        let mut expansions = HashSet::new();

        for term in terms {
            if let Some(similar_terms) = self.embeddings.get_similar_words(term, 5)? {
                for (similar_term, similarity) in similar_terms {
                    if similarity >= self.config.similarity_threshold {
                        expansions.insert(ExpansionTerm {
                            term: similar_term,
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
        _context: &MLContext,
    ) -> Result<HashSet<ExpansionTerm>> {
        let mut expansions = HashSet::new();

        for term in terms {
            if let Some(cooccurrent_terms) =
                self.cooccurrence_model.get_cooccurrent_terms(term, 3)?
            {
                for (cooccurrent_term, score) in cooccurrent_terms {
                    if score >= self.config.similarity_threshold {
                        expansions.insert(ExpansionTerm {
                            term: cooccurrent_term,
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

    fn tokenize_query(&self, query: &str) -> Vec<String> {
        query
            .split_whitespace()
            .map(|s| {
                s.to_lowercase()
                    .trim_matches(|c: char| !c.is_alphanumeric())
                    .to_string()
            })
            .filter(|s| !s.is_empty())
            .collect()
    }

    fn calculate_expansion_confidence(
        &self,
        _original_terms: &[String],
        expansions: &[ExpansionTerm],
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

/// Expanded query with original and expansion terms.
#[derive(Debug, Clone)]
pub struct ExpandedQuery {
    /// Original query terms.
    pub original_terms: Vec<String>,
    /// Expanded terms with metadata.
    pub expanded_terms: Vec<ExpansionTerm>,
    /// Detected query intent.
    pub intent: QueryIntent,
    /// Overall expansion confidence.
    pub confidence: f64,
}

impl ExpandedQuery {
    /// Get all terms (original + expanded) for search.
    pub fn get_all_terms(&self) -> Vec<WeightedTerm> {
        let mut all_terms = Vec::new();

        // Add original terms with higher weight
        for term in &self.original_terms {
            all_terms.push(WeightedTerm {
                term: term.clone(),
                weight: 1.0, // Full weight for original terms
            });
        }

        // Add expanded terms with lower weight
        for expansion in &self.expanded_terms {
            all_terms.push(WeightedTerm {
                term: expansion.term.clone(),
                weight: expansion.confidence * 0.5, // Weighted by confidence
            });
        }

        all_terms
    }

    /// Get only high-confidence expansion terms.
    pub fn get_high_confidence_expansions(&self, threshold: f64) -> Vec<&ExpansionTerm> {
        self.expanded_terms
            .iter()
            .filter(|e| e.confidence >= threshold)
            .collect()
    }
}

/// Term with associated weight for search.
#[derive(Debug, Clone)]
pub struct WeightedTerm {
    pub term: String,
    pub weight: f64,
}

/// Expansion term with metadata.
#[derive(Debug, Clone)]
pub struct ExpansionTerm {
    /// The expanded term.
    pub term: String,
    /// Confidence score (0.0 - 1.0).
    pub confidence: f64,
    /// Type of expansion.
    pub expansion_type: ExpansionType,
    /// Original term that generated this expansion.
    pub source_term: String,
}

impl PartialEq for ExpansionTerm {
    fn eq(&self, other: &Self) -> bool {
        self.term == other.term && self.expansion_type == other.expansion_type
    }
}

impl Eq for ExpansionTerm {}

impl Hash for ExpansionTerm {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.term.hash(state);
        self.expansion_type.hash(state);
    }
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

/// Query intent classifier.
#[derive(Debug)]
pub struct IntentClassifier {
    /// Keywords that indicate informational intent.
    informational_keywords: HashSet<String>,
    /// Keywords that indicate navigational intent.
    navigational_keywords: HashSet<String>,
    /// Keywords that indicate transactional intent.
    transactional_keywords: HashSet<String>,
}

impl Default for IntentClassifier {
    fn default() -> Self {
        Self::new()
    }
}

impl IntentClassifier {
    pub fn new() -> Self {
        let mut informational = HashSet::new();
        informational.extend(vec![
            "what".to_string(),
            "how".to_string(),
            "why".to_string(),
            "when".to_string(),
            "where".to_string(),
            "who".to_string(),
            "definition".to_string(),
            "explain".to_string(),
        ]);

        let mut navigational = HashSet::new();
        navigational.extend(vec![
            "homepage".to_string(),
            "website".to_string(),
            "site".to_string(),
            "login".to_string(),
            "official".to_string(),
        ]);

        let mut transactional = HashSet::new();
        transactional.extend(vec![
            "buy".to_string(),
            "purchase".to_string(),
            "order".to_string(),
            "download".to_string(),
            "install".to_string(),
            "get".to_string(),
            "free".to_string(),
            "price".to_string(),
        ]);

        Self {
            informational_keywords: informational,
            navigational_keywords: navigational,
            transactional_keywords: transactional,
        }
    }

    pub fn classify(&self, query_terms: &[String]) -> Result<QueryIntent> {
        let mut informational_score = 0;
        let mut navigational_score = 0;
        let mut transactional_score = 0;

        for term in query_terms {
            if self.informational_keywords.contains(term) {
                informational_score += 1;
            }
            if self.navigational_keywords.contains(term) {
                navigational_score += 1;
            }
            if self.transactional_keywords.contains(term) {
                transactional_score += 1;
            }
        }

        let max_score = informational_score
            .max(navigational_score)
            .max(transactional_score);

        if max_score == 0 {
            Ok(QueryIntent::Unknown)
        } else if max_score == informational_score {
            Ok(QueryIntent::Informational)
        } else if max_score == navigational_score {
            Ok(QueryIntent::Navigational)
        } else {
            Ok(QueryIntent::Transactional)
        }
    }

    pub fn update_with_history(&mut self, _history: &[SearchHistoryItem]) -> Result<()> {
        // Learn from search patterns to improve intent classification
        // This would analyze successful queries and their outcomes
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let expansion = QueryExpansion::new(config).unwrap();
        assert!(expansion.config.enabled);
    }

    #[test]
    fn test_intent_classification() {
        let classifier = IntentClassifier::new();

        let info_query = vec!["what".to_string(), "is".to_string(), "rust".to_string()];
        let intent = classifier.classify(&info_query).unwrap();
        assert_eq!(intent, QueryIntent::Informational);

        let nav_query = vec![
            "rust".to_string(),
            "official".to_string(),
            "website".to_string(),
        ];
        let intent = classifier.classify(&nav_query).unwrap();
        assert_eq!(intent, QueryIntent::Navigational);

        let trans_query = vec!["buy".to_string(), "rust".to_string(), "book".to_string()];
        let intent = classifier.classify(&trans_query).unwrap();
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
    fn test_expanded_query_terms() {
        let expanded = ExpandedQuery {
            original_terms: vec!["rust".to_string(), "programming".to_string()],
            expanded_terms: vec![ExpansionTerm {
                term: "language".to_string(),
                confidence: 0.8,
                expansion_type: ExpansionType::Semantic,
                source_term: "programming".to_string(),
            }],
            intent: QueryIntent::Informational,
            confidence: 0.85,
        };

        let all_terms = expanded.get_all_terms();
        assert_eq!(all_terms.len(), 3);

        let high_conf = expanded.get_high_confidence_expansions(0.7);
        assert_eq!(high_conf.len(), 1);
    }

    #[test]
    fn test_expansion_term_equality() {
        let term1 = ExpansionTerm {
            term: "test".to_string(),
            confidence: 0.8,
            expansion_type: ExpansionType::Synonym,
            source_term: "exam".to_string(),
        };

        let term2 = ExpansionTerm {
            term: "test".to_string(),
            confidence: 0.9, // Different confidence
            expansion_type: ExpansionType::Synonym,
            source_term: "exam".to_string(),
        };

        // Terms should be equal even with different confidence for HashSet deduplication
        assert_eq!(term1, term2);
    }
}
