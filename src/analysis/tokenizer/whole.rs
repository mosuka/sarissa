//! Whole tokenizer implementation.
//!
//! This module provides a tokenizer that treats the entire input text as
//! a single token without any splitting. This is particularly useful for
//! ID fields, exact match fields, or other scenarios where text should be
//! indexed as-is.
//!
//! # Examples
//!
//! ```
//! use yatagarasu::analysis::tokenizer::Tokenizer;
//! use yatagarasu::analysis::tokenizer::whole::WholeTokenizer;
//!
//! let tokenizer = WholeTokenizer::new();
//! let tokens: Vec<_> = tokenizer.tokenize("user-id-12345").unwrap().collect();
//!
//! // The entire text is one token
//! assert_eq!(tokens.len(), 1);
//! assert_eq!(tokens[0].text, "user-id-12345");
//! ```

use crate::analysis::token::{Token, TokenStream};
use crate::analysis::tokenizer::Tokenizer;
use crate::error::Result;

/// A tokenizer that treats the entire input as a single token.
///
/// This tokenizer returns the complete input text as a single token,
/// preserving all characters, whitespace, and punctuation exactly as provided.
///
/// # Use Cases
///
/// - **ID fields**: Product IDs, user IDs, etc.
/// - **Exact match fields**: Email addresses, URLs, file paths
/// - **Keywords**: Tags, categories, enum values
/// - **Compound identifiers**: UUIDs, hash values
///
/// # Examples
///
/// ```
/// use yatagarasu::analysis::tokenizer::Tokenizer;
/// use yatagarasu::analysis::tokenizer::whole::WholeTokenizer;
///
/// let tokenizer = WholeTokenizer::new();
///
/// // ID field
/// let tokens: Vec<_> = tokenizer.tokenize("SKU-2024-001").unwrap().collect();
/// assert_eq!(tokens.len(), 1);
/// assert_eq!(tokens[0].text, "SKU-2024-001");
///
/// // Empty text produces no tokens
/// let tokens: Vec<_> = tokenizer.tokenize("").unwrap().collect();
/// assert_eq!(tokens.len(), 0);
/// ```
#[derive(Clone, Debug, Default)]
pub struct WholeTokenizer;

impl WholeTokenizer {
    /// Create a new whole tokenizer.
    pub fn new() -> Self {
        WholeTokenizer
    }
}

impl Tokenizer for WholeTokenizer {
    fn tokenize(&self, text: &str) -> Result<TokenStream> {
        if text.is_empty() {
            Ok(Box::new(std::iter::empty()))
        } else {
            let token = Token::with_offsets(text, 0, 0, text.len());
            Ok(Box::new(std::iter::once(token)))
        }
    }

    fn name(&self) -> &'static str {
        "whole"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_whole_tokenizer() {
        let tokenizer = WholeTokenizer::new();
        let tokens: Vec<Token> = tokenizer.tokenize("hello world").unwrap().collect();

        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].text, "hello world");
        assert_eq!(tokens[0].position, 0);
        assert_eq!(tokens[0].start_offset, 0);
        assert_eq!(tokens[0].end_offset, 11);
    }

    #[test]
    fn test_whole_tokenizer_empty() {
        let tokenizer = WholeTokenizer::new();
        let tokens: Vec<Token> = tokenizer.tokenize("").unwrap().collect();

        assert_eq!(tokens.len(), 0);
    }

    #[test]
    fn test_tokenizer_name() {
        assert_eq!(WholeTokenizer::new().name(), "whole");
    }
}
