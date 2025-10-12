//! Builder for creating QueryExpansion instances.

use std::collections::HashSet;
use std::sync::Arc;

use crate::analysis::analyzer::Analyzer;
use crate::error::Result;

use super::core::QueryExpansion;
use super::expander::QueryExpander;
use super::semantic::SemanticQueryExpander;
use super::statistical::StatisticalQueryExpander;
use super::synonym::SynonymQueryExpander;

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
                let samples = crate::ml::intent_classifier::load_training_data(path)?;
                crate::ml::intent_classifier::new_ml_based(samples, self.analyzer.clone())?
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
    ) -> Box<dyn crate::ml::intent_classifier::IntentClassifier> {
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
        crate::ml::intent_classifier::new_keyword_based(
            informational,
            navigational,
            transactional,
            analyzer,
        )
    }
}
