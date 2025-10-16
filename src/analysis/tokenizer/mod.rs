//! Tokenizer implementations for text analysis.

use crate::analysis::token::TokenStream;
use crate::error::Result;

/// Trait for tokenizers that convert text into tokens.
pub trait Tokenizer: Send + Sync {
    /// Tokenize the given text into a stream of tokens.
    fn tokenize(&self, text: &str) -> Result<TokenStream>;

    /// Get the name of this tokenizer (for debugging and configuration).
    fn name(&self) -> &'static str;
}

// Individual tokenizer modules
pub mod lindera;
pub mod ngram;
pub mod regex;
pub mod unicode_word;
pub mod whitespace;
pub mod whole;
