//! Helper functions for creating intent classifiers.

use std::sync::Arc;

use anyhow::Result;

use crate::analysis::analyzer::analyzer::Analyzer;
use crate::ml::intent_classifier::classifier::IntentClassifier;
use crate::ml::intent_classifier::keyword_classifier::KeywordBasedIntentClassifier;
use crate::ml::intent_classifier::ml_classifier::MLBasedIntentClassifier;
use crate::ml::intent_classifier::types::IntentSample;

/// Load training data from JSON file.
pub fn load_training_data(path: &str) -> Result<Vec<IntentSample>> {
    let content = std::fs::read_to_string(path)?;
    let samples: Vec<IntentSample> = serde_json::from_str(&content)?;
    Ok(samples)
}

/// Create a new keyword-based intent classifier.
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
pub fn new_ml_based(
    samples: Vec<IntentSample>,
    analyzer: Arc<dyn Analyzer>,
) -> Result<Box<dyn IntentClassifier>> {
    Ok(Box::new(MLBasedIntentClassifier::new(samples, analyzer)?))
}
