//! Advanced scoring and ranking systems for search relevance.

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::error::Result;

/// Configuration for scoring algorithms.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringConfig {
    /// K1 parameter for BM25 (term frequency saturation).
    pub k1: f32,

    /// B parameter for BM25 (field length normalization).
    pub b: f32,

    /// Boost factor for TF-IDF.
    pub tf_idf_boost: f32,

    /// Enable field length normalization.
    pub enable_field_norm: bool,

    /// Custom field boosts.
    pub field_boosts: HashMap<String, f32>,

    /// Enable query coordination factor.
    pub enable_coord: bool,
}

impl Default for ScoringConfig {
    fn default() -> Self {
        ScoringConfig {
            k1: 1.2,
            b: 0.75,
            tf_idf_boost: 1.0,
            enable_field_norm: true,
            field_boosts: HashMap::new(),
            enable_coord: true,
        }
    }
}

/// Document statistics for scoring.
#[derive(Debug, Clone)]
pub struct DocumentStats {
    /// Document ID.
    pub doc_id: u32,

    /// Document length (number of terms).
    pub doc_length: u64,

    /// Field lengths.
    pub field_lengths: HashMap<String, u64>,

    /// Term frequencies.
    pub term_frequencies: HashMap<String, u64>,

    /// Field term frequencies.
    pub field_term_frequencies: HashMap<String, HashMap<String, u64>>,
}

/// Collection statistics for scoring.
#[derive(Debug, Clone)]
pub struct CollectionStats {
    /// Total number of documents.
    pub total_docs: u64,

    /// Average document length.
    pub avg_doc_length: f64,

    /// Average field lengths.
    pub avg_field_lengths: HashMap<String, f64>,

    /// Document frequencies for terms.
    pub document_frequencies: HashMap<String, u64>,

    /// Field document frequencies.
    pub field_document_frequencies: HashMap<String, HashMap<String, u64>>,
}

/// Advanced scoring function trait.
///
/// Defines the interface for pluggable scoring algorithms.
pub trait ScoringFunction: Send + Sync + std::fmt::Debug {
    /// Calculate score for a document given query and statistics.
    ///
    /// # Arguments
    ///
    /// * `query_terms` - Terms from the query
    /// * `doc_stats` - Statistics for the document being scored
    /// * `collection_stats` - Collection-wide statistics
    /// * `config` - Scoring configuration parameters
    ///
    /// # Returns
    ///
    /// Relevance score for the document
    fn score(
        &self,
        query_terms: &[String],
        doc_stats: &DocumentStats,
        collection_stats: &CollectionStats,
        config: &ScoringConfig,
    ) -> Result<f32>;

    /// Get function name.
    ///
    /// # Returns
    ///
    /// String identifier for this scoring function
    fn name(&self) -> &str;

    /// Get function description.
    ///
    /// # Returns
    ///
    /// Human-readable description of the algorithm
    fn description(&self) -> &str;
}

/// BM25 scoring function implementation.
#[derive(Debug, Clone)]
pub struct BM25ScoringFunction;

impl ScoringFunction for BM25ScoringFunction {
    fn score(
        &self,
        query_terms: &[String],
        doc_stats: &DocumentStats,
        collection_stats: &CollectionStats,
        config: &ScoringConfig,
    ) -> Result<f32> {
        let mut total_score = 0.0;

        for term in query_terms {
            // Get term frequency in document
            let tf = *doc_stats.term_frequencies.get(term).unwrap_or(&0) as f32;
            if tf == 0.0 {
                continue;
            }

            // Get document frequency for IDF calculation
            let df = *collection_stats
                .document_frequencies
                .get(term)
                .unwrap_or(&1) as f32;
            let idf = ((collection_stats.total_docs as f32 - df + 0.5) / (df + 0.5)).ln();

            // BM25 formula
            let doc_len = doc_stats.doc_length as f32;
            let avg_len = collection_stats.avg_doc_length as f32;

            let tf_component = (tf * (config.k1 + 1.0))
                / (tf + config.k1 * (1.0 - config.b + config.b * (doc_len / avg_len)));

            total_score += idf * tf_component;
        }

        Ok(total_score)
    }

    fn name(&self) -> &str {
        "BM25"
    }

    fn description(&self) -> &str {
        "Best Matching 25 probabilistic ranking function"
    }
}

/// TF-IDF scoring function implementation.
#[derive(Debug, Clone)]
pub struct TfIdfScoringFunction;

impl ScoringFunction for TfIdfScoringFunction {
    fn score(
        &self,
        query_terms: &[String],
        doc_stats: &DocumentStats,
        collection_stats: &CollectionStats,
        config: &ScoringConfig,
    ) -> Result<f32> {
        let mut total_score = 0.0;

        for term in query_terms {
            // Get term frequency in document
            let tf = *doc_stats.term_frequencies.get(term).unwrap_or(&0) as f32;
            if tf == 0.0 {
                continue;
            }

            // Calculate TF component (log normalization)
            let tf_component = (1.0 + tf.ln()) * config.tf_idf_boost;

            // Calculate IDF component
            let df = *collection_stats
                .document_frequencies
                .get(term)
                .unwrap_or(&1) as f32;
            let idf = (collection_stats.total_docs as f32 / df).ln();

            // Apply field length normalization if enabled
            let norm_factor = if config.enable_field_norm {
                let doc_len = doc_stats.doc_length as f32;
                let avg_len = collection_stats.avg_doc_length as f32;
                (avg_len / doc_len).sqrt()
            } else {
                1.0
            };

            total_score += tf_component * idf * norm_factor;
        }

        Ok(total_score)
    }

    fn name(&self) -> &str {
        "TF-IDF"
    }

    fn description(&self) -> &str {
        "Term Frequency - Inverse Document Frequency"
    }
}

/// Vector Space Model scoring function.
#[derive(Debug, Clone)]
pub struct VectorSpaceScoringFunction;

impl ScoringFunction for VectorSpaceScoringFunction {
    fn score(
        &self,
        query_terms: &[String],
        doc_stats: &DocumentStats,
        collection_stats: &CollectionStats,
        _config: &ScoringConfig,
    ) -> Result<f32> {
        let mut query_vector = Vec::new();
        let mut doc_vector = Vec::new();

        for term in query_terms {
            // Query vector component (simplified: 1.0 for each term)
            query_vector.push(1.0);

            // Document vector component (TF-IDF)
            let tf = *doc_stats.term_frequencies.get(term).unwrap_or(&0) as f32;
            let df = *collection_stats
                .document_frequencies
                .get(term)
                .unwrap_or(&1) as f32;
            let idf = (collection_stats.total_docs as f32 / df).ln();

            doc_vector.push(tf * idf);
        }

        // Calculate cosine similarity
        let dot_product: f32 = query_vector
            .iter()
            .zip(doc_vector.iter())
            .map(|(q, d)| q * d)
            .sum();
        let query_magnitude: f32 = query_vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        let doc_magnitude: f32 = doc_vector.iter().map(|x| x * x).sum::<f32>().sqrt();

        if query_magnitude == 0.0 || doc_magnitude == 0.0 {
            Ok(0.0)
        } else {
            Ok(dot_product / (query_magnitude * doc_magnitude))
        }
    }

    fn name(&self) -> &str {
        "Vector Space"
    }

    fn description(&self) -> &str {
        "Vector Space Model with cosine similarity"
    }
}

/// Custom scoring function that allows user-defined scoring logic.
///
/// Wraps a user-provided closure to create a custom scoring algorithm.
pub struct CustomScoringFunction {
    /// Function name.
    name: String,

    /// Function description.
    description: String,

    /// Scoring function implementation.
    #[allow(clippy::type_complexity)]
    scorer: Arc<
        dyn Fn(&[String], &DocumentStats, &CollectionStats, &ScoringConfig) -> Result<f32>
            + Send
            + Sync,
    >,
}

impl std::fmt::Debug for CustomScoringFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CustomScoringFunction")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("scorer", &"<custom function>")
            .finish()
    }
}

impl CustomScoringFunction {
    /// Create a new custom scoring function.
    ///
    /// # Arguments
    ///
    /// * `name` - Identifier for this scoring function
    /// * `description` - Human-readable description
    /// * `scorer` - Closure implementing the scoring logic
    ///
    /// # Returns
    ///
    /// A new custom scoring function
    pub fn new<F>(name: String, description: String, scorer: F) -> Self
    where
        F: Fn(&[String], &DocumentStats, &CollectionStats, &ScoringConfig) -> Result<f32>
            + Send
            + Sync
            + 'static,
    {
        CustomScoringFunction {
            name,
            description,
            scorer: Arc::new(scorer),
        }
    }
}

impl ScoringFunction for CustomScoringFunction {
    fn score(
        &self,
        query_terms: &[String],
        doc_stats: &DocumentStats,
        collection_stats: &CollectionStats,
        config: &ScoringConfig,
    ) -> Result<f32> {
        (self.scorer)(query_terms, doc_stats, collection_stats, config)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }
}

/// Advanced scorer that uses pluggable scoring functions.
///
/// Applies configurable scoring algorithms with field boosts and
/// coordination factors.
#[derive(Debug)]
pub struct AdvancedScorer {
    /// Scoring function to use.
    scoring_function: Box<dyn ScoringFunction>,

    /// Scoring configuration.
    config: ScoringConfig,

    /// Collection statistics.
    collection_stats: CollectionStats,

    /// Query terms for scoring.
    query_terms: Vec<String>,
}

impl AdvancedScorer {
    /// Create a new advanced scorer.
    ///
    /// # Arguments
    ///
    /// * `scoring_function` - The scoring algorithm to use
    /// * `config` - Configuration for scoring behavior
    /// * `collection_stats` - Collection-wide statistics
    /// * `query_terms` - Terms from the search query
    ///
    /// # Returns
    ///
    /// A new advanced scorer instance
    pub fn new(
        scoring_function: Box<dyn ScoringFunction>,
        config: ScoringConfig,
        collection_stats: CollectionStats,
        query_terms: Vec<String>,
    ) -> Self {
        AdvancedScorer {
            scoring_function,
            config,
            collection_stats,
            query_terms,
        }
    }

    /// Score a document.
    ///
    /// Computes relevance score with optional field boosts and coordination factor.
    ///
    /// # Arguments
    ///
    /// * `doc_stats` - Statistics for the document to score
    ///
    /// # Returns
    ///
    /// Final relevance score for the document
    pub fn score_document(&self, doc_stats: &DocumentStats) -> Result<f32> {
        let mut base_score = self.scoring_function.score(
            &self.query_terms,
            doc_stats,
            &self.collection_stats,
            &self.config,
        )?;

        // Apply field boosts
        base_score = self.apply_field_boosts(base_score, doc_stats)?;

        // Apply coordination factor if enabled
        if self.config.enable_coord {
            base_score = self.apply_coordination_factor(base_score, doc_stats)?;
        }

        Ok(base_score)
    }

    /// Apply field boosts to the score.
    ///
    /// Adjusts score based on which fields contain matching terms.
    ///
    /// # Arguments
    ///
    /// * `base_score` - Base relevance score
    /// * `doc_stats` - Document statistics
    ///
    /// # Returns
    ///
    /// Boosted score
    fn apply_field_boosts(&self, base_score: f32, doc_stats: &DocumentStats) -> Result<f32> {
        if self.config.field_boosts.is_empty() {
            return Ok(base_score);
        }

        let mut boost_factor = 1.0;
        let mut total_weight = 0.0;

        for (field, boost) in &self.config.field_boosts {
            if let Some(field_term_freqs) = doc_stats.field_term_frequencies.get(field) {
                let field_score: u64 = field_term_freqs.values().sum();
                if field_score > 0 {
                    boost_factor += boost;
                    total_weight += boost;
                }
            }
        }

        if total_weight > 0.0 {
            boost_factor /= total_weight;
        }

        Ok(base_score * boost_factor)
    }

    /// Apply coordination factor (ratio of matched terms to total query terms).
    ///
    /// Rewards documents that match more query terms.
    ///
    /// # Arguments
    ///
    /// * `base_score` - Base relevance score
    /// * `doc_stats` - Document statistics
    ///
    /// # Returns
    ///
    /// Score adjusted by coordination factor
    fn apply_coordination_factor(&self, base_score: f32, doc_stats: &DocumentStats) -> Result<f32> {
        let matched_terms = self
            .query_terms
            .iter()
            .filter(|term| doc_stats.term_frequencies.contains_key(*term))
            .count();

        let coord_factor = if self.query_terms.is_empty() {
            1.0
        } else {
            matched_terms as f32 / self.query_terms.len() as f32
        };

        Ok(base_score * coord_factor)
    }
}

/// Scoring function registry for managing different scoring algorithms.
///
/// Provides a centralized registry of available scoring functions.
#[derive(Debug)]
pub struct ScoringRegistry {
    /// Registered scoring functions.
    functions: HashMap<String, Box<dyn ScoringFunction>>,
}

impl ScoringRegistry {
    /// Create a new scoring registry with default functions.
    ///
    /// Registers BM25, TF-IDF, and Vector Space scoring functions.
    ///
    /// # Returns
    ///
    /// Registry pre-populated with default scoring functions
    pub fn new() -> Self {
        let mut registry = ScoringRegistry {
            functions: HashMap::new(),
        };

        // Register default scoring functions
        registry.register("bm25", Box::new(BM25ScoringFunction));
        registry.register("tf_idf", Box::new(TfIdfScoringFunction));
        registry.register("vector_space", Box::new(VectorSpaceScoringFunction));

        registry
    }

    /// Register a scoring function.
    ///
    /// # Arguments
    ///
    /// * `name` - Identifier for the scoring function
    /// * `function` - Boxed scoring function implementation
    pub fn register(&mut self, name: &str, function: Box<dyn ScoringFunction>) {
        self.functions.insert(name.to_string(), function);
    }

    /// Get a scoring function by name.
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the scoring function to retrieve
    ///
    /// # Returns
    ///
    /// Reference to the scoring function, or None if not found
    pub fn get(&self, name: &str) -> Option<&dyn ScoringFunction> {
        self.functions.get(name).map(|f| f.as_ref())
    }

    /// List available scoring functions.
    ///
    /// # Returns
    ///
    /// Vector of (name, description) tuples for all registered functions
    pub fn list_functions(&self) -> Vec<(&str, &str)> {
        self.functions
            .iter()
            .map(|(name, func)| (name.as_str(), func.description()))
            .collect()
    }
}

impl Default for ScoringRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_doc_stats() -> DocumentStats {
        let mut term_frequencies = HashMap::new();
        term_frequencies.insert("test".to_string(), 3);
        term_frequencies.insert("query".to_string(), 2);

        DocumentStats {
            doc_id: 1,
            doc_length: 100,
            field_lengths: HashMap::new(),
            term_frequencies,
            field_term_frequencies: HashMap::new(),
        }
    }

    fn create_test_collection_stats() -> CollectionStats {
        let mut document_frequencies = HashMap::new();
        document_frequencies.insert("test".to_string(), 50);
        document_frequencies.insert("query".to_string(), 30);

        CollectionStats {
            total_docs: 1000,
            avg_doc_length: 120.0,
            avg_field_lengths: HashMap::new(),
            document_frequencies,
            field_document_frequencies: HashMap::new(),
        }
    }

    #[test]
    fn test_bm25_scoring() {
        let scorer = BM25ScoringFunction;
        let doc_stats = create_test_doc_stats();
        let collection_stats = create_test_collection_stats();
        let config = ScoringConfig::default();
        let query_terms = vec!["test".to_string(), "query".to_string()];

        let score = scorer
            .score(&query_terms, &doc_stats, &collection_stats, &config)
            .unwrap();

        assert!(score > 0.0);
        assert_eq!(scorer.name(), "BM25");
    }

    #[test]
    fn test_tf_idf_scoring() {
        let scorer = TfIdfScoringFunction;
        let doc_stats = create_test_doc_stats();
        let collection_stats = create_test_collection_stats();
        let config = ScoringConfig::default();
        let query_terms = vec!["test".to_string(), "query".to_string()];

        let score = scorer
            .score(&query_terms, &doc_stats, &collection_stats, &config)
            .unwrap();

        assert!(score > 0.0);
        assert_eq!(scorer.name(), "TF-IDF");
    }

    #[test]
    fn test_vector_space_scoring() {
        let scorer = VectorSpaceScoringFunction;
        let doc_stats = create_test_doc_stats();
        let collection_stats = create_test_collection_stats();
        let config = ScoringConfig::default();
        let query_terms = vec!["test".to_string(), "query".to_string()];

        let score = scorer
            .score(&query_terms, &doc_stats, &collection_stats, &config)
            .unwrap();

        assert!((0.0..=1.0).contains(&score)); // Cosine similarity range
        assert_eq!(scorer.name(), "Vector Space");
    }

    #[test]
    fn test_custom_scoring_function() {
        let custom_scorer = CustomScoringFunction::new(
            "custom".to_string(),
            "Custom test scorer".to_string(),
            |_terms, _doc_stats, _collection_stats, _config| Ok(42.0),
        );

        let doc_stats = create_test_doc_stats();
        let collection_stats = create_test_collection_stats();
        let config = ScoringConfig::default();
        let query_terms = vec!["test".to_string()];

        let score = custom_scorer
            .score(&query_terms, &doc_stats, &collection_stats, &config)
            .unwrap();

        assert_eq!(score, 42.0);
        assert_eq!(custom_scorer.name(), "custom");
    }

    #[test]
    fn test_scoring_registry() {
        let mut registry = ScoringRegistry::new();

        // Test default functions are registered
        assert!(registry.get("bm25").is_some());
        assert!(registry.get("tf_idf").is_some());
        assert!(registry.get("vector_space").is_some());

        // Test custom function registration
        let custom_scorer = Box::new(CustomScoringFunction::new(
            "test".to_string(),
            "Test scorer".to_string(),
            |_terms, _doc_stats, _collection_stats, _config| Ok(1.0),
        ));

        registry.register("test", custom_scorer);
        assert!(registry.get("test").is_some());

        // Test function listing
        let functions = registry.list_functions();
        assert!(functions.len() >= 4);
    }

    #[test]
    fn test_advanced_scorer() {
        let scoring_function = Box::new(BM25ScoringFunction);
        let config = ScoringConfig::default();
        let collection_stats = create_test_collection_stats();
        let query_terms = vec!["test".to_string(), "query".to_string()];

        let scorer = AdvancedScorer::new(scoring_function, config, collection_stats, query_terms);
        let doc_stats = create_test_doc_stats();

        let score = scorer.score_document(&doc_stats).unwrap();
        assert!(score > 0.0);
    }
}
