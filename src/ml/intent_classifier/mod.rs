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
//! use sarissa::ml::intent_classifier::{self, IntentSample};
//! use sarissa::analysis::StandardAnalyzer;
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
//! let classifier = intent_classifier::new_ml_based(samples, analyzer)?;
//!
//! let intent = classifier.predict("how to learn programming")?;
//! # Ok(())
//! # }
//! ```

mod classifier;
mod core;
mod keyword_classifier;
mod ml_classifier;
mod tfidf;
mod types;

// Public exports
pub use classifier::IntentClassifier;
pub use core::{load_training_data, new_keyword_based, new_ml_based};
pub use keyword_classifier::KeywordBasedIntentClassifier;
pub use ml_classifier::MLBasedIntentClassifier;
pub use tfidf::TfIdfVectorizer;
pub use types::IntentSample;
