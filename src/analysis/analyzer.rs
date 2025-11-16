//! Analyzer implementations that combine tokenizers and filters.
//!
//! This module provides complete text analysis pipelines that combine tokenizers
//! and token filters to process text for indexing and searching. Analyzers are
//! the main entry point for text analysis in Platypus.
//!
//! # Available Analyzers
//!
//! - [`standard::StandardAnalyzer`] - General-purpose analyzer with whitespace tokenization
//! - [`simple::SimpleAnalyzer`] - Simple lowercase + letter tokenization
//! - [`keyword::KeywordAnalyzer`] - Treats entire input as single token (for IDs, tags)
//! - [`noop::NoopAnalyzer`] - No-op analyzer for testing
//! - [`pipeline::PipelineAnalyzer`] - Customizable analyzer with filter chain
//! - [`per_field::PerFieldAnalyzer`] - Different analyzers per field
//! - [`language`] - Language-specific analyzers (English, Japanese, etc.)
//!
//! # Architecture
//!
//! ```text
//! Text → Tokenizer → Token Filters → Analyzed Tokens
//! ```
//!
//! # Examples
//!
//! ```
//! use platypus::analysis::analyzer::analyzer::Analyzer;
//! use platypus::analysis::analyzer::standard::StandardAnalyzer;
//!
//! let analyzer = StandardAnalyzer::new().unwrap();
//! let tokens: Vec<_> = analyzer.analyze("Hello World!").unwrap().collect();
//!
//! // Tokens: ["hello", "world"]
//! assert_eq!(tokens.len(), 2);
//! ```

#[allow(clippy::module_inception)]
pub mod analyzer;
pub mod keyword;
pub mod language;
pub mod noop;
pub mod per_field;
pub mod pipeline;
pub mod simple;
pub mod standard;
