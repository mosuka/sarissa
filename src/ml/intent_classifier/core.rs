//! Helper functions for creating intent classifiers.

use std::sync::Arc;

use anyhow::Result;

use crate::analysis::analyzer::analyzer::Analyzer;
use crate::ml::intent_classifier::classifier::IntentClassifier;
use crate::ml::intent_classifier::keyword_classifier::KeywordBasedIntentClassifier;
use crate::ml::intent_classifier::ml_classifier::MLBasedIntentClassifier;
use crate::ml::intent_classifier::types::IntentSample;

/// Load training data from JSON file.
///
/// Reads a JSON file containing an array of `IntentSample` objects.
/// Each sample should have a "query" field (string) and an "intent" field (string).
///
/// # Arguments
/// * `path` - Path to the JSON file
///
/// # Returns
/// Vector of intent training samples
///
/// # Errors
/// Returns an error if the file cannot be read or the JSON is invalid
///
/// # Example JSON format
/// ```json
/// [
///   {"query": "what is rust", "intent": "Informational"},
///   {"query": "github homepage", "intent": "Navigational"}
/// ]
/// ```
pub fn load_training_data(path: &str) -> Result<Vec<IntentSample>> {
    let content = std::fs::read_to_string(path)?;
    let samples: Vec<IntentSample> = serde_json::from_str(&content)?;
    Ok(samples)
}

/// Create a new keyword-based intent classifier.
///
/// Creates a classifier that uses simple keyword matching to determine query intent.
/// Each keyword set represents terms strongly associated with a particular intent type.
///
/// # Arguments
/// * `informational_keywords` - Keywords indicating informational queries (e.g., "what", "how", "why")
/// * `navigational_keywords` - Keywords indicating navigational queries (e.g., "homepage", "login", "site")
/// * `transactional_keywords` - Keywords indicating transactional queries (e.g., "buy", "download", "purchase")
/// * `analyzer` - Text analyzer for tokenizing queries
///
/// # Returns
/// Boxed trait object implementing `IntentClassifier`
pub fn new_keyword_based(
    informational_keywords: std::collections::HashSet<String>,
    navigational_keywords: std::collections::HashSet<String>,
    transactional_keywords: std::collections::HashSet<String>,
    analyzer: Arc<dyn Analyzer>,
) -> Box<dyn IntentClassifier> {
    Box::new(KeywordBasedIntentClassifier::new(
        informational_keywords,
        navigational_keywords,
        transactional_keywords,
        analyzer,
    ))
}

/// Create a new ML-based intent classifier from training samples.
///
/// Creates a classifier that uses TF-IDF vectorization and cosine similarity
/// to determine query intent based on similarity to training examples.
///
/// # Arguments
/// * `samples` - Training samples with labeled query-intent pairs
/// * `analyzer` - Text analyzer for tokenizing queries
///
/// # Returns
/// Boxed trait object implementing `IntentClassifier`
///
/// # Errors
/// Returns an error if training samples are empty or if training fails
pub fn new_ml_based(
    samples: Vec<IntentSample>,
    analyzer: Arc<dyn Analyzer>,
) -> Result<Box<dyn IntentClassifier>> {
    Ok(Box::new(MLBasedIntentClassifier::new(samples, analyzer)?))
}
