//! Tokenizer implementations for text analysis.
//!
//! This module provides various tokenization strategies for breaking text into tokens.
//! Tokenizers are the first step in the text analysis pipeline, responsible for
//! splitting input text into meaningful units (tokens).
//!
//! # Available Tokenizers
//!
//! - [`whitespace::WhitespaceTokenizer`] - Splits on whitespace characters
//! - [`unicode_word::UnicodeWordTokenizer`] - Uses Unicode word boundaries
//! - [`regex::RegexTokenizer`] - Custom regex-based tokenization
//! - [`ngram::NGramTokenizer`] - Character n-gram tokenization
//! - [`lindera::LinderaTokenizer`] - Japanese morphological analysis (requires `lindera` feature)
//! - [`whole::WholeTokenizer`] - Treats entire text as single token
//!
//! # Examples
//!
//! ```
//! use platypus::analysis::tokenizer::Tokenizer;
//! use platypus::analysis::tokenizer::whitespace::WhitespaceTokenizer;
//!
//! let tokenizer = WhitespaceTokenizer::new();
//! let tokens: Vec<_> = tokenizer.tokenize("Hello world").unwrap().collect();
//! assert_eq!(tokens.len(), 2);
//! ```

use crate::analysis::token::TokenStream;
use crate::error::Result;

/// Trait for tokenizers that convert text into tokens.
///
/// All tokenizers must implement this trait to be used in the analysis pipeline.
/// The trait requires `Send + Sync` to allow use in concurrent contexts.
///
/// # Examples
///
/// Implementing a custom tokenizer:
///
/// ```
/// use platypus::analysis::token::{Token, TokenStream};
/// use platypus::analysis::tokenizer::Tokenizer;
/// use platypus::error::Result;
///
/// struct CustomTokenizer;
///
/// impl Tokenizer for CustomTokenizer {
///     fn tokenize(&self, text: &str) -> Result<TokenStream> {
///         let tokens: Vec<Token> = text
///             .split(',')
///             .enumerate()
///             .map(|(i, s)| Token::new(s.trim(), i))
///             .collect();
///         Ok(Box::new(tokens.into_iter()))
///     }
///
///     fn name(&self) -> &'static str {
///         "custom"
///     }
/// }
/// ```
pub trait Tokenizer: Send + Sync {
    /// Tokenize the given text into a stream of tokens.
    ///
    /// # Arguments
    ///
    /// * `text` - The input text to tokenize
    ///
    /// # Returns
    ///
    /// A `TokenStream` (boxed iterator of tokens) on success, or an error if tokenization fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use platypus::analysis::tokenizer::Tokenizer;
    /// use platypus::analysis::tokenizer::whitespace::WhitespaceTokenizer;
    ///
    /// let tokenizer = WhitespaceTokenizer::new();
    /// let tokens: Vec<_> = tokenizer.tokenize("Hello world").unwrap().collect();
    /// assert_eq!(tokens[0].text, "Hello");
    /// assert_eq!(tokens[1].text, "world");
    /// ```
    fn tokenize(&self, text: &str) -> Result<TokenStream>;

    /// Get the name of this tokenizer (for debugging and configuration).
    ///
    /// # Returns
    ///
    /// A static string representing the tokenizer's name.
    ///
    /// # Examples
    ///
    /// ```
    /// use platypus::analysis::tokenizer::Tokenizer;
    /// use platypus::analysis::tokenizer::whitespace::WhitespaceTokenizer;
    ///
    /// let tokenizer = WhitespaceTokenizer::new();
    /// assert_eq!(tokenizer.name(), "whitespace");
    /// ```
    fn name(&self) -> &'static str;
}

// Individual tokenizer modules
pub mod lindera;
pub mod ngram;
pub mod regex;
pub mod unicode_word;
pub mod whitespace;
pub mod whole;
