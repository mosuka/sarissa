//! Core analyzer trait definition.

use crate::analysis::token::TokenStream;
use crate::error::Result;

/// Trait for analyzers that convert text into processed tokens.
pub trait Analyzer: Send + Sync {
    /// Analyze the given text and return a stream of tokens.
    fn analyze(&self, text: &str) -> Result<TokenStream>;

    /// Get the name of this analyzer (for debugging and configuration).
    fn name(&self) -> &'static str;
}