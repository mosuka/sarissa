//! Whitespace tokenizer implementation.

use super::Tokenizer;

use crate::analysis::token::{Token, TokenStream};
use crate::error::Result;
use crate::util::simd;

/// A tokenizer that splits text on whitespace.
#[derive(Clone, Debug, Default)]
pub struct WhitespaceTokenizer;

impl WhitespaceTokenizer {
    /// Create a new whitespace tokenizer.
    pub fn new() -> Self {
        WhitespaceTokenizer
    }
}

impl Tokenizer for WhitespaceTokenizer {
    fn tokenize(&self, text: &str) -> Result<TokenStream> {
        // Use SIMD-optimized whitespace detection for ASCII text
        if text.is_ascii() && text.len() >= 32 {
            self.tokenize_simd(text)
        } else {
            self.tokenize_fallback(text)
        }
    }

    fn name(&self) -> &'static str {
        "whitespace"
    }
}

impl WhitespaceTokenizer {
    /// SIMD-optimized tokenization for ASCII text.
    fn tokenize_simd(&self, text: &str) -> Result<TokenStream> {
        let bytes = text.as_bytes();
        let mut tokens = Vec::new();
        let mut position = 0;
        let mut start = 0;

        // Skip leading whitespace
        while start < bytes.len() && bytes[start].is_ascii_whitespace() {
            start += 1;
        }

        while start < bytes.len() {
            // Find the end of the current word using SIMD
            let word_end = match simd::ascii::find_whitespace_simd(&bytes[start..]) {
                Some(offset) => start + offset,
                None => bytes.len(),
            };

            if word_end > start {
                // Extract the word
                let word = &text[start..word_end];
                tokens.push(Token::with_offsets(word, position, start, word_end));
                position += 1;
            }

            // Skip whitespace to find the next word
            start = word_end;
            while start < bytes.len() && bytes[start].is_ascii_whitespace() {
                start += 1;
            }
        }

        Ok(Box::new(tokens.into_iter()))
    }

    /// Fallback implementation for non-ASCII or short text.
    fn tokenize_fallback(&self, text: &str) -> Result<TokenStream> {
        let tokens: Vec<Token> = text
            .split_whitespace()
            .enumerate()
            .map(|(position, word)| {
                // Find the actual position in the original text
                let start_offset = text.find(word).unwrap_or(0);
                let end_offset = start_offset + word.len();
                Token::with_offsets(word, position, start_offset, end_offset)
            })
            .collect();

        Ok(Box::new(tokens.into_iter()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_whitespace_tokenizer() {
        let tokenizer = WhitespaceTokenizer::new();
        let tokens: Vec<Token> = tokenizer.tokenize("hello  world\ttest").unwrap().collect();

        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].text, "hello");
        assert_eq!(tokens[1].text, "world");
        assert_eq!(tokens[2].text, "test");
    }

    #[test]
    fn test_tokenizer_name() {
        assert_eq!(WhitespaceTokenizer::new().name(), "whitespace");
    }
}
