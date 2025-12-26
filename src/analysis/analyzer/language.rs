//! Language-specific analyzers.
//!
//! This module provides analyzers optimized for specific languages, each with
//! appropriate tokenization and stop word filtering.
//!
//! # Available Languages
//!
//! - [`english`] - English text analysis with regex tokenization and English stop words
//! - [`japanese`] - Japanese text analysis with Lindera morphological analyzer
//!
//! # Examples
//!
//! ```
//! use sarissa::analysis::analyzer::analyzer::Analyzer;
//! use sarissa::analysis::analyzer::language::english::EnglishAnalyzer;
//!
//! let analyzer = EnglishAnalyzer::new().unwrap();
//! let tokens: Vec<_> = analyzer.analyze("Hello the world").unwrap().collect();
//!
//! // "the" is filtered as a stop word
//! assert_eq!(tokens.len(), 2);
//! ```

pub mod english;
pub mod japanese;
