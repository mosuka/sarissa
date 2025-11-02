//! Text analysis module for Yatagarasu.
//!
//! This module provides comprehensive text analysis functionality for processing
//! and transforming text before indexing or searching. It includes:
//!
//! - **Tokenizers**: Break text into individual tokens
//! - **Token Filters**: Transform, filter, or augment token streams
//! - **Analyzers**: Combine tokenizers and filters into analysis pipelines
//! - **Synonyms**: Support for synonym expansion during analysis
//!
//! # Architecture
//!
//! The analysis pipeline follows a simple flow:
//!
//! ```text
//! Text → Tokenizer → Token Stream → Token Filters → Analyzed Tokens
//! ```
//!
//! # Examples
//!
//! ```
//! use yatagarasu::analysis::analyzer::standard::StandardAnalyzer;
//! use yatagarasu::analysis::analyzer::analyzer::Analyzer;
//!
//! let analyzer = StandardAnalyzer::new().unwrap();
//! let tokens: Vec<_> = analyzer.analyze("Hello World!").unwrap().collect();
//! // Tokens: ["hello", "world"]
//! ```
//!
//! # Modules
//!
//! - [`analyzer`]: Pre-built and custom text analyzers
//! - [`tokenizer`]: Text tokenization strategies
//! - [`token_filter`]: Token transformation and filtering
//! - [`token`]: Token representation and manipulation
//! - [`synonym`]: Synonym dictionary and graph building

pub mod analyzer;
pub mod synonym;
pub mod token;
pub mod token_filter;
pub mod tokenizer;
