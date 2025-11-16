//! Core analyzer trait definition.
//!
//! This module defines the [`Analyzer`] trait, which is the main interface for
//! text analysis in Platypus. Analyzers combine tokenizers and filters to
//! transform raw text into indexed tokens.
//!
//! # Role in Analysis Pipeline
//!
//! Analyzers serve as the complete text processing pipeline:
//!
//! ```text
//! Raw Text → Analyzer → Token Stream → Index
//!             ↓
//!         Tokenizer
//!             ↓
//!         Filter 1
//!             ↓
//!         Filter 2
//!             ↓
//!         Filter N
//! ```
//!
//! # Available Implementations
//!
//! - [`StandardAnalyzer`](super::standard::StandardAnalyzer) - Good defaults for most use cases
//! - [`SimpleAnalyzer`](super::simple::SimpleAnalyzer) - Tokenization only, no filtering
//! - [`KeywordAnalyzer`](super::keyword::KeywordAnalyzer) - Treats entire input as one token
//! - [`PipelineAnalyzer`](super::pipeline::PipelineAnalyzer) - Custom tokenizer + filter chains
//! - [`EnglishAnalyzer`](super::language::english::EnglishAnalyzer) - English-optimized
//! - [`JapaneseAnalyzer`](super::language::japanese::JapaneseAnalyzer) - Japanese-optimized
//! - [`PerFieldAnalyzer`](super::per_field::PerFieldAnalyzer) - Different analyzers per field
//!
//! # Examples
//!
//! Using a built-in analyzer:
//!
//! ```
//! use platypus::analysis::analyzer::analyzer::Analyzer;
//! use platypus::analysis::analyzer::standard::StandardAnalyzer;
//!
//! let analyzer = StandardAnalyzer::new().unwrap();
//! let tokens: Vec<_> = analyzer.analyze("Hello World").unwrap().collect();
//!
//! assert_eq!(tokens[0].text, "hello");
//! assert_eq!(tokens[1].text, "world");
//! ```
//!
//! Implementing a custom analyzer:
//!
//! ```
//! use platypus::analysis::analyzer::analyzer::Analyzer;
//! use platypus::analysis::token::TokenStream;
//! use platypus::error::Result;
//!
//! struct MyAnalyzer;
//!
//! impl Analyzer for MyAnalyzer {
//!     fn analyze(&self, text: &str) -> Result<TokenStream> {
//!         // Custom analysis logic here
//!         Ok(Box::new(std::iter::empty()))
//!     }
//!
//!     fn name(&self) -> &'static str {
//!         "my_analyzer"
//!     }
//!
//!     fn as_any(&self) -> &dyn std::any::Any {
//!         self
//!     }
//! }
//! ```

use crate::analysis::token::TokenStream;
use crate::error::Result;

/// Trait for analyzers that convert text into processed tokens.
///
/// This is the core trait that all analyzers must implement. Analyzers are
/// responsible for the complete text processing pipeline, from raw text to
/// indexed tokens.
///
/// # Thread Safety
///
/// The trait requires `Send + Sync` to allow analyzers to be used safely
/// across thread boundaries, which is essential for concurrent indexing.
///
/// # Trait Methods
///
/// - [`analyze`](Self::analyze) - Process text into tokens
/// - [`name`](Self::name) - Get analyzer identifier
/// - [`as_any`](Self::as_any) - Enable downcasting to concrete types
pub trait Analyzer: Send + Sync {
    /// Analyze the given text and return a stream of tokens.
    ///
    /// This is the main method that performs the complete analysis pipeline,
    /// including tokenization and all configured filters.
    ///
    /// # Arguments
    ///
    /// * `text` - The raw input text to analyze
    ///
    /// # Returns
    ///
    /// A `TokenStream` (boxed iterator of tokens) that can be consumed by
    /// the indexer, or an error if analysis fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use platypus::analysis::analyzer::analyzer::Analyzer;
    /// use platypus::analysis::analyzer::standard::StandardAnalyzer;
    ///
    /// let analyzer = StandardAnalyzer::new().unwrap();
    /// let tokens: Vec<_> = analyzer.analyze("The quick brown fox").unwrap().collect();
    ///
    /// // "The" is removed as a stop word, others are lowercased
    /// assert_eq!(tokens.len(), 3);
    /// assert_eq!(tokens[0].text, "quick");
    /// ```
    fn analyze(&self, text: &str) -> Result<TokenStream>;

    /// Get the name of this analyzer (for debugging and configuration).
    ///
    /// The name is used to identify the analyzer in logs, error messages,
    /// and configuration files.
    ///
    /// # Returns
    ///
    /// A static string representing the analyzer's unique identifier.
    ///
    /// # Examples
    ///
    /// ```
    /// use platypus::analysis::analyzer::analyzer::Analyzer;
    /// use platypus::analysis::analyzer::standard::StandardAnalyzer;
    ///
    /// let analyzer = StandardAnalyzer::new().unwrap();
    /// assert_eq!(analyzer.name(), "standard");
    /// ```
    fn name(&self) -> &'static str;

    /// Provide access to the concrete type for downcasting.
    ///
    /// This method enables downcasting from `&dyn Analyzer` to a concrete
    /// analyzer type, which is useful when you need access to type-specific
    /// methods (e.g., `PerFieldAnalyzer::get_analyzer`).
    ///
    /// # Returns
    ///
    /// A reference to the analyzer as `&dyn Any` for downcasting.
    ///
    /// # Examples
    ///
    /// ```
    /// use platypus::analysis::analyzer::analyzer::Analyzer;
    /// use platypus::analysis::analyzer::per_field::PerFieldAnalyzer;
    /// use platypus::analysis::analyzer::standard::StandardAnalyzer;
    /// use std::sync::Arc;
    ///
    /// let per_field = PerFieldAnalyzer::new(Arc::new(StandardAnalyzer::new().unwrap()));
    /// let analyzer: &dyn Analyzer = &per_field;
    ///
    /// // Downcast to access PerFieldAnalyzer-specific methods
    /// if let Some(pf) = analyzer.as_any().downcast_ref::<PerFieldAnalyzer>() {
    ///     let field_analyzer = pf.get_analyzer("title");
    ///     println!("Got analyzer for 'title' field");
    /// }
    /// ```
    fn as_any(&self) -> &dyn std::any::Any;
}
