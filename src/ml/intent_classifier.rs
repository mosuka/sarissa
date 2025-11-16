//! Machine learning-based intent classifier using TF-IDF and simple scoring.
//!
//! This module provides intent classification for search queries using either:
//! - Keyword-based classification: Simple rule-based matching
//! - ML-based classification: TF-IDF vectorization with cosine similarity
//!
//! # Architecture
//!
//! - `IntentClassifier` trait: Common interface for all classifiers
//! - `KeywordBasedIntentClassifier`: Keyword matching implementation
//! - `MLBasedIntentClassifier`: ML-based implementation using TF-IDF
//! - `TfIdfVectorizer`: Feature extraction using TF-IDF
//! - `IntentSample`: Training data structure
//!
//! # Example
//!
//! ```rust,no_run
//! use platypus::ml::intent_classifier::types::IntentSample;
//! use platypus::ml::intent_classifier::core::new_ml_based;
//! use platypus::analysis::analyzer::standard::StandardAnalyzer;
//! use std::sync::Arc;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let samples = vec![
//!     IntentSample {
//!         query: "what is rust".to_string(),
//!         intent: "Informational".to_string(),
//!     },
//! ];
//!
//! let analyzer = Arc::new(StandardAnalyzer::new()?);
//! let classifier = new_ml_based(samples, analyzer)?;
//!
//! let intent = classifier.predict("how to learn programming")?;
//! # Ok(())
//! # }
//! ```

pub mod classifier;
pub mod core;
pub mod keyword_classifier;
pub mod ml_classifier;
pub mod tfidf;
pub mod types;
