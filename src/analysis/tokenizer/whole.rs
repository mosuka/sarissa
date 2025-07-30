//! Whole tokenizer implementation.

use super::Tokenizer;

use crate::analysis::token::{Token, TokenStream};
use crate::error::Result;

/// A tokenizer that treats the entire input as a single token.
///
/// This is useful for ID fields or other cases where you don't want to split the text.
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
