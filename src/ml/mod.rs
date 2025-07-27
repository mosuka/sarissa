//! Machine Learning integration for Sarissa search engine.
//!
//! This module provides machine learning capabilities to enhance search quality
//! through learning-to-rank, query expansion, recommendations, and anomaly detection.

pub mod anomaly;
pub mod features;
pub mod models;
pub mod optimization;
pub mod query_expansion;
pub mod ranking;
pub mod recommendation;

pub use anomaly::*;
pub use features::*;
pub use models::*;
pub use optimization::*;
pub use query_expansion::*;
pub use ranking::*;
pub use recommendation::*;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for machine learning features.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLConfig {
    /// Enable machine learning features.
    pub enabled: bool,
    /// Directory to store trained models.
    pub models_directory: String,
    /// Learning to rank configuration.
    pub ranking: RankingConfig,
    /// Query expansion configuration.
    pub query_expansion: QueryExpansionConfig,
    /// Recommendation system configuration.
    pub recommendation: RecommendationConfig,
    /// Anomaly detection configuration.
    pub anomaly_detection: AnomalyDetectionConfig,
    /// Auto optimization configuration.
    pub auto_optimization: AutoOptimizationConfig,
}

impl Default for MLConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            models_directory: "./models".to_string(),
            ranking: RankingConfig::default(),
            query_expansion: QueryExpansionConfig::default(),
            recommendation: RecommendationConfig::default(),
            anomaly_detection: AnomalyDetectionConfig::default(),
            auto_optimization: AutoOptimizationConfig::default(),
        }
    }
}

/// Machine learning context for search operations.
#[derive(Debug, Clone)]
pub struct MLContext {
    /// User session information.
    pub user_session: Option<UserSession>,
    /// Search history.
    pub search_history: Vec<SearchHistoryItem>,
    /// User preferences.
    pub user_preferences: HashMap<String, f64>,
    /// Current timestamp.
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl Default for MLContext {
    fn default() -> Self {
        Self {
            user_session: None,
            search_history: Vec::new(),
            user_preferences: HashMap::new(),
            timestamp: chrono::Utc::now(),
        }
    }
}

/// User session information for machine learning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserSession {
    /// Session identifier.
    pub session_id: String,
    /// User identifier (if authenticated).
    pub user_id: Option<String>,
    /// Session start time.
    pub start_time: chrono::DateTime<chrono::Utc>,
    /// User agent string.
    pub user_agent: Option<String>,
    /// IP address.
    pub ip_address: String,
}

/// Search history item for learning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHistoryItem {
    /// Original query.
    pub query: String,
    /// Clicked documents.
    pub clicked_documents: Vec<String>,
    /// Dwell times for each document.
    pub dwell_times: HashMap<String, std::time::Duration>,
    /// Search timestamp.
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Query result count.
    pub result_count: usize,
}

/// Feedback signal for machine learning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackSignal {
    /// Query that generated this feedback.
    pub query: String,
    /// Document ID that received the feedback.
    pub document_id: String,
    /// Type of feedback.
    pub feedback_type: FeedbackType,
    /// Relevance score (0.0 - 1.0).
    pub relevance_score: f64,
    /// Timestamp when feedback was generated.
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Types of user feedback signals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeedbackType {
    /// User clicked on the document.
    Click,
    /// User spent time reading the document.
    DwellTime(std::time::Duration),
    /// User explicitly rated the document.
    Explicit(f64),
    /// User ignored the document (negative signal).
    Skip,
    /// User bounced back quickly (negative signal).
    Bounce,
}

/// Machine learning error types.
#[derive(Debug, thiserror::Error)]
pub enum MLError {
    #[error("Model not trained: {message}")]
    ModelNotTrained { message: String },

    #[error("Invalid feature vector: {message}")]
    InvalidFeatureVector { message: String },

    #[error("Training data insufficient: need at least {min_samples} samples, got {actual}")]
    InsufficientTrainingData { min_samples: usize, actual: usize },

    #[error("Model loading failed: {path}")]
    ModelLoadError { path: String },

    #[error("Model saving failed: {path}")]
    ModelSaveError { path: String },

    #[error("Feature extraction failed: {message}")]
    FeatureExtractionError { message: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ml_config_default() {
        let config = MLConfig::default();
        assert!(config.enabled);
        assert_eq!(config.models_directory, "./models");
    }

    #[test]
    fn test_ml_context_creation() {
        let context = MLContext::default();
        assert!(context.user_session.is_none());
        assert!(context.search_history.is_empty());
        assert!(context.user_preferences.is_empty());
    }

    #[test]
    fn test_feedback_signal_creation() {
        let feedback = FeedbackSignal {
            query: "test query".to_string(),
            document_id: "doc1".to_string(),
            feedback_type: FeedbackType::Click,
            relevance_score: 0.8,
            timestamp: chrono::Utc::now(),
        };

        assert_eq!(feedback.query, "test query");
        assert_eq!(feedback.document_id, "doc1");
        assert_eq!(feedback.relevance_score, 0.8);
        assert!(matches!(feedback.feedback_type, FeedbackType::Click));
    }
}
