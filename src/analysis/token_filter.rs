//! Token filter implementations for token transformation.
//!
//! This module provides various filters that transform token streams produced
//! by tokenizers. Filters can modify, remove, or add tokens to implement
//! features like lowercasing, stemming, stop word removal, and synonym expansion.
//!
//! # Available Filters
//!
//! - [`lowercase::LowercaseFilter`] - Converts tokens to lowercase
//! - [`stop::StopFilter`] - Removes stop words
//! - [`stem::StemFilter`] - Reduces words to their stem form
//! - [`synonym_graph::SynonymGraphFilter`] - Expands synonyms
//! - [`limit::LimitFilter`] - Limits number of tokens
//! - [`boost::BoostFilter`] - Adjusts token scoring weights
//! - [`strip::StripFilter`] - Removes specific characters
//! - [`remove_empty::RemoveEmptyFilter`] - Removes empty tokens
//! - [`flatten_graph::FlattenGraphFilter`] - Flattens token graphs
//!
//! # Examples
//!
//! ```
//! use platypus::analysis::token_filter::Filter;
//! use platypus::analysis::token_filter::lowercase::LowercaseFilter;
//! use platypus::analysis::token::Token;
//!
//! let filter = LowercaseFilter::new();
//! let tokens = vec![Token::new("Hello", 0), Token::new("WORLD", 1)];
//! let filtered: Vec<_> = filter.filter(Box::new(tokens.into_iter()))
//!     .unwrap()
//!     .collect();
//!
//! assert_eq!(filtered[0].text, "hello");
//! assert_eq!(filtered[1].text, "world");
//! ```
//!
//! # Filter Chaining
//!
//! Filters can be chained together in an analyzer to create complex
//! text processing pipelines:
//!
//! ```text
//! Tokenizer → Lowercase → Stop Words → Stemmer → Index
//! ```

use crate::analysis::token::TokenStream;
use crate::error::Result;

/// Trait for filters that transform token streams.
///
/// All token filters must implement this trait to be used in the analysis
/// pipeline. Filters receive a stream of tokens and produce a new stream,
/// allowing them to modify, filter, or augment tokens.
///
/// The trait requires `Send + Sync` to allow use in concurrent contexts.
///
/// # Examples
///
/// Implementing a custom filter:
///
/// ```
/// use platypus::analysis::token::{Token, TokenStream};
/// use platypus::analysis::token_filter::Filter;
/// use platypus::error::Result;
///
/// struct ReverseFilter;
///
/// impl Filter for ReverseFilter {
///     fn filter(&self, tokens: TokenStream) -> Result<TokenStream> {
///         let reversed: Vec<Token> = tokens
///             .map(|mut t| {
///                 t.text = t.text.chars().rev().collect();
///                 t
///             })
///             .collect();
///         Ok(Box::new(reversed.into_iter()))
///     }
///
///     fn name(&self) -> &'static str {
///         "reverse"
///     }
/// }
/// ```
pub trait Filter: Send + Sync {
    /// Apply this filter to a token stream.
    ///
    /// # Arguments
    ///
    /// * `tokens` - The input token stream to filter
    ///
    /// # Returns
    ///
    /// A new `TokenStream` with the filter applied, or an error if filtering fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use platypus::analysis::token_filter::Filter;
    /// use platypus::analysis::token_filter::lowercase::LowercaseFilter;
    /// use platypus::analysis::token::Token;
    ///
    /// let filter = LowercaseFilter::new();
    /// let tokens = vec![Token::new("HELLO", 0)];
    /// let result: Vec<_> = filter.filter(Box::new(tokens.into_iter()))
    ///     .unwrap()
    ///     .collect();
    /// assert_eq!(result[0].text, "hello");
    /// ```
    fn filter(&self, tokens: TokenStream) -> Result<TokenStream>;

    /// Get the name of this filter (for debugging and configuration).
    ///
    /// # Returns
    ///
    /// A static string representing the filter's name.
    ///
    /// # Examples
    ///
    /// ```
    /// use platypus::analysis::token_filter::Filter;
    /// use platypus::analysis::token_filter::lowercase::LowercaseFilter;
    ///
    /// let filter = LowercaseFilter::new();
    /// assert_eq!(filter.name(), "lowercase");
    /// ```
    fn name(&self) -> &'static str;
}

// Individual filter modules
pub mod boost;
pub mod flatten_graph;
pub mod limit;
pub mod lowercase;
pub mod remove_empty;
pub mod stem;
pub mod stop;
pub mod strip;
pub mod synonym_graph;
