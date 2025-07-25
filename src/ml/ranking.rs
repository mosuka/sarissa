//! Learning to Rank (LTR) system for improving search relevance.

use crate::error::Result;
use crate::ml::features::{FeatureExtractor, FeatureContext, QueryDocumentFeatures};
use crate::ml::models::{RankingModel, GBDTRanker, LabeledExample, TrainingStats};
use crate::ml::{MLError, FeedbackSignal, FeedbackType};
use crate::query::SearchResults;
use crate::schema::Document;
use crate::search::{SearchRequest, Search};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::sync::{Arc, RwLock};

/// Configuration for Learning to Rank system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankingConfig {
    /// Enable learning to rank.
    pub enabled: bool,
    /// Model type to use.
    pub model_type: ModelType,
    /// Path to training data.
    pub training_data_path: Option<String>,
    /// Interval for retraining the model (in hours).
    pub retrain_interval_hours: u64,
    /// Enable online learning from user feedback.
    pub online_learning: bool,
    /// Maximum number of feedback examples to keep for online learning.
    pub max_feedback_examples: usize,
    /// Minimum feedback examples before retraining.
    pub min_feedback_for_retrain: usize,
    /// Model hyperparameters.
    pub model_params: ModelParameters,
}

impl Default for RankingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            model_type: ModelType::GBDT,
            training_data_path: None,
            retrain_interval_hours: 24,
            online_learning: true,
            max_feedback_examples: 10000,
            min_feedback_for_retrain: 100,
            model_params: ModelParameters::default(),
        }
    }
}

/// Supported model types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    GBDT,
    LinearRegression,
    NeuralNetwork,
}

/// Model hyperparameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelParameters {
    pub learning_rate: f64,
    pub max_iterations: usize,
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub subsample_features: f64,
    pub subsample_rows: f64,
    pub regularization: f64,
}

impl Default for ModelParameters {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            max_iterations: 100,
            max_depth: 6,
            min_samples_split: 20,
            subsample_features: 0.8,
            subsample_rows: 0.8,
            regularization: 0.01,
        }
    }
}

/// Learning to Rank system.
pub struct LearningToRank {
    /// Configuration.
    config: RankingConfig,
    /// Feature extractor.
    feature_extractor: Arc<FeatureExtractor>,
    /// Ranking model.
    model: Arc<RwLock<Box<dyn RankingModel>>>,
    /// Feedback buffer for online learning.
    feedback_buffer: Arc<RwLock<VecDeque<FeedbackSignal>>>,
    /// Training data for batch learning.
    training_data: Arc<RwLock<Vec<LabeledExample<QueryDocumentFeatures, f64>>>>,
    /// Last retraining timestamp.
    last_retrain: Arc<RwLock<chrono::DateTime<chrono::Utc>>>,
    /// Performance metrics.
    metrics: Arc<RwLock<RankingMetrics>>,
}

/// Performance metrics for ranking system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankingMetrics {
    /// Mean Average Precision.
    pub map: f64,
    /// Normalized Discounted Cumulative Gain.
    pub ndcg: f64,
    /// Mean Reciprocal Rank.
    pub mrr: f64,
    /// Precision at K.
    pub precision_at_k: HashMap<usize, f64>,
    /// Recall at K.
    pub recall_at_k: HashMap<usize, f64>,
    /// Total queries processed.
    pub total_queries: u64,
    /// Total feedback signals received.
    pub total_feedback: u64,
    /// Last updated timestamp.
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

impl Default for RankingMetrics {
    fn default() -> Self {
        Self {
            map: 0.0,
            ndcg: 0.0,
            mrr: 0.0,
            precision_at_k: HashMap::new(),
            recall_at_k: HashMap::new(),
            total_queries: 0,
            total_feedback: 0,
            last_updated: chrono::Utc::now(),
        }
    }
}

impl LearningToRank {
    /// Create a new Learning to Rank system.
    pub fn new(config: RankingConfig) -> Result<Self> {
        let feature_extractor = Arc::new(FeatureExtractor::new());
        
        let model: Box<dyn RankingModel> = match config.model_type {
            ModelType::GBDT => Box::new(GBDTRanker::with_params(
                config.model_params.learning_rate,
                config.model_params.max_iterations,
                config.model_params.max_depth,
                config.model_params.min_samples_split,
                config.model_params.subsample_features,
                config.model_params.subsample_rows,
            )),
            ModelType::LinearRegression => {
                return Err(MLError::ModelNotTrained {
                    message: "Linear regression not implemented yet".to_string(),
                }.into());
            }
            ModelType::NeuralNetwork => {
                return Err(MLError::ModelNotTrained {
                    message: "Neural network not implemented yet".to_string(),
                }.into());
            }
        };
        
        Ok(Self {
            config,
            feature_extractor,
            model: Arc::new(RwLock::new(model)),
            feedback_buffer: Arc::new(RwLock::new(VecDeque::new())),
            training_data: Arc::new(RwLock::new(Vec::new())),
            last_retrain: Arc::new(RwLock::new(chrono::Utc::now())),
            metrics: Arc::new(RwLock::new(RankingMetrics::default())),
        })
    }
    
    /// Re-rank search results using the trained model.
    pub fn rerank_results(
        &self,
        query: &str,
        results: &SearchResults,
        documents: &[Document],
        context: &RerankingContext,
    ) -> Result<SearchResults> {
        if !self.config.enabled {
            return Ok(results.clone());
        }
        
        let model = self.model.read().unwrap();
        if !model.is_trained() {
            return Ok(results.clone());
        }
        
        let mut scored_results = Vec::new();
        
        for (i, hit) in results.hits.iter().enumerate() {
            if i < documents.len() {
                let feature_context = FeatureContext {
                    document_id: hit.doc_id.to_string(),
                    vector_similarity: context.vector_similarities.get(&hit.doc_id).copied(),
                    semantic_distance: context.semantic_distances.get(&hit.doc_id).copied(),
                    user_context_score: context.user_context_score,
                    timestamp: chrono::Utc::now(),
                };
                
                let features = self.feature_extractor.extract_features(
                    query,
                    &documents[i],
                    &feature_context,
                )?;
                
                let ml_score = model.predict(&features);
                
                // Combine original score with ML score
                let combined_score = self.combine_scores(hit.score, ml_score);
                
                scored_results.push((hit.clone(), combined_score));
            }
        }
        
        // Sort by combined score
        scored_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Update metrics
        {
            let mut metrics = self.metrics.write().unwrap();
            metrics.total_queries += 1;
        }
        
        Ok(SearchResults {
            hits: scored_results.into_iter().map(|(mut hit, score)| {
                hit.score = score;
                hit
            }).collect(),
            total_hits: results.total_hits,
            max_score: scored_results.first().map(|(_, score)| *score).unwrap_or(0.0),
        })
    }
    
    /// Process user feedback for online learning.
    pub fn process_feedback(&self, feedback: FeedbackSignal) -> Result<()> {
        if !self.config.online_learning {
            return Ok(());
        }
        
        // Add to feedback buffer
        {
            let mut buffer = self.feedback_buffer.write().unwrap();
            buffer.push_back(feedback);
            
            // Limit buffer size
            while buffer.len() > self.config.max_feedback_examples {
                buffer.pop_front();
            }
        }
        
        // Update metrics
        {
            let mut metrics = self.metrics.write().unwrap();
            metrics.total_feedback += 1;
        }
        
        // Check if we should retrain
        if self.should_retrain() {
            self.retrain_from_feedback()?;
        }
        
        Ok(())
    }
    
    /// Train the model with labeled data.
    pub fn train(&self, training_data: Vec<LabeledExample<QueryDocumentFeatures, f64>>) -> Result<()> {
        if training_data.is_empty() {
            return Err(MLError::InsufficientTrainingData {
                min_samples: 1,
                actual: 0,
            }.into());
        }
        
        // Update training data
        {
            let mut data = self.training_data.write().unwrap();
            *data = training_data;
        }
        
        // Train the model
        {
            let mut model = self.model.write().unwrap();
            let data = self.training_data.read().unwrap();
            model.train(&data)?;
        }
        
        // Update last retrain time
        {
            let mut last_retrain = self.last_retrain.write().unwrap();
            *last_retrain = chrono::Utc::now();
        }
        
        Ok(())
    }
    
    /// Load training data from file.
    pub fn load_training_data(&self, path: &Path) -> Result<Vec<LabeledExample<QueryDocumentFeatures, f64>>> {
        let content = std::fs::read_to_string(path)
            .map_err(|_| MLError::ModelLoadError { 
                path: path.display().to_string() 
            })?;
        
        let mut examples = Vec::new();
        
        for line in content.lines() {
            if line.trim().is_empty() {
                continue;
            }
            
            let example: LabeledExample<QueryDocumentFeatures, f64> = serde_json::from_str(line)
                .map_err(|_| MLError::FeatureExtractionError {
                    message: format!("Failed to parse training example: {}", line)
                })?;
            
            examples.push(example);
        }
        
        Ok(examples)
    }
    
    /// Save the trained model.
    pub fn save_model(&self, path: &Path) -> Result<()> {
        let model = self.model.read().unwrap();
        model.save(path)
    }
    
    /// Load a trained model.
    pub fn load_model(&self, path: &Path) -> Result<()> {
        let loaded_model: Box<dyn RankingModel> = match self.config.model_type {
            ModelType::GBDT => Box::new(GBDTRanker::load(path)?),
            _ => return Err(MLError::ModelLoadError {
                path: path.display().to_string(),
            }.into()),
        };
        
        let mut model = self.model.write().unwrap();
        *model = loaded_model;
        
        Ok(())
    }
    
    /// Get current ranking metrics.
    pub fn get_metrics(&self) -> RankingMetrics {
        self.metrics.read().unwrap().clone()
    }
    
    /// Get training statistics.
    pub fn get_training_stats(&self) -> TrainingStats {
        let model = self.model.read().unwrap();
        model.get_training_stats()
    }
    
    /// Check if the model is trained and ready.
    pub fn is_ready(&self) -> bool {
        if !self.config.enabled {
            return true; // If disabled, always ready (will pass through)
        }
        
        let model = self.model.read().unwrap();
        model.is_trained()
    }
    
    // Private helper methods
    
    fn combine_scores(&self, original_score: f32, ml_score: f64) -> f32 {
        // Simple weighted combination
        let alpha = 0.7; // Weight for ML score
        (alpha * ml_score + (1.0 - alpha) * original_score as f64) as f32
    }
    
    fn should_retrain(&self) -> bool {
        // Check time-based retraining
        let last_retrain = *self.last_retrain.read().unwrap();
        let hours_since_retrain = (chrono::Utc::now() - last_retrain).num_hours();
        
        if hours_since_retrain >= self.config.retrain_interval_hours as i64 {
            return true;
        }
        
        // Check feedback-based retraining
        let buffer = self.feedback_buffer.read().unwrap();
        buffer.len() >= self.config.min_feedback_for_retrain
    }
    
    fn retrain_from_feedback(&self) -> Result<()> {
        // Convert feedback to training examples
        let buffer = self.feedback_buffer.read().unwrap();
        let mut new_examples = Vec::new();
        
        for feedback in buffer.iter() {
            let relevance_score = self.feedback_to_relevance_score(feedback);
            
            // This is a simplified example - in practice, you'd need to
            // store the features that were used for this query-document pair
            // For now, we'll skip this implementation detail
        }
        
        if !new_examples.is_empty() {
            // Combine with existing training data
            let mut training_data = self.training_data.read().unwrap().clone();
            training_data.extend(new_examples);
            
            // Train the model
            let mut model = self.model.write().unwrap();
            model.train(&training_data)?;
            
            // Clear feedback buffer
            drop(buffer);
            self.feedback_buffer.write().unwrap().clear();
            
            // Update last retrain time
            *self.last_retrain.write().unwrap() = chrono::Utc::now();
        }
        
        Ok(())
    }
    
    fn feedback_to_relevance_score(&self, feedback: &FeedbackSignal) -> f64 {
        match feedback.feedback_type {
            FeedbackType::Click => 1.0,
            FeedbackType::DwellTime(duration) => {
                // Convert dwell time to relevance score
                let seconds = duration.as_secs() as f64;
                (seconds / 60.0).min(4.0) // Max relevance of 4.0 for 1+ minute
            }
            FeedbackType::Explicit(score) => score,
            FeedbackType::Skip => 0.0,
            FeedbackType::Bounce => -0.5,
        }
    }
}

/// Context for re-ranking operations.
#[derive(Debug, Clone)]
pub struct RerankingContext {
    /// Vector similarity scores for documents.
    pub vector_similarities: HashMap<String, f64>,
    /// Semantic distance scores for documents.
    pub semantic_distances: HashMap<String, f64>,
    /// User context score.
    pub user_context_score: Option<f64>,
    /// Additional context features.
    pub additional_features: HashMap<String, f64>,
}

impl Default for RerankingContext {
    fn default() -> Self {
        Self {
            vector_similarities: HashMap::new(),
            semantic_distances: HashMap::new(),
            user_context_score: None,
            additional_features: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::{SearchHit};
    use crate::schema::{Document, FieldValue};

    #[test]
    fn test_ranking_config_default() {
        let config = RankingConfig::default();
        assert!(config.enabled);
        assert_eq!(config.retrain_interval_hours, 24);
        assert!(config.online_learning);
    }

    #[test]
    fn test_learning_to_rank_creation() {
        let config = RankingConfig::default();
        let ltr = LearningToRank::new(config).unwrap();
        
        assert!(!ltr.is_ready()); // Not trained yet
        
        let metrics = ltr.get_metrics();
        assert_eq!(metrics.total_queries, 0);
        assert_eq!(metrics.total_feedback, 0);
    }

    #[test]
    fn test_reranking_context_default() {
        let context = RerankingContext::default();
        assert!(context.vector_similarities.is_empty());
        assert!(context.semantic_distances.is_empty());
        assert!(context.user_context_score.is_none());
    }

    #[test]
    fn test_feedback_processing() {
        let config = RankingConfig::default();
        let ltr = LearningToRank::new(config).unwrap();
        
        let feedback = FeedbackSignal {
            query: "test query".to_string(),
            document_id: "doc1".to_string(),
            feedback_type: FeedbackType::Click,
            relevance_score: 1.0,
            timestamp: chrono::Utc::now(),
        };
        
        let result = ltr.process_feedback(feedback);
        assert!(result.is_ok());
        
        let metrics = ltr.get_metrics();
        assert_eq!(metrics.total_feedback, 1);
    }

    #[test]
    fn test_model_parameters_default() {
        let params = ModelParameters::default();
        assert_eq!(params.learning_rate, 0.1);
        assert_eq!(params.max_iterations, 100);
        assert_eq!(params.max_depth, 6);
    }

    #[test]
    fn test_ranking_metrics_default() {
        let metrics = RankingMetrics::default();
        assert_eq!(metrics.map, 0.0);
        assert_eq!(metrics.ndcg, 0.0);
        assert_eq!(metrics.total_queries, 0);
    }
}