//! Whitespace tokenizer implementation.

use super::Tokenizer;

use crate::analysis::token::{Token, TokenStream, TokenType};
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
    /// Detect token type based on the content of the word.
    fn detect_token_type(word: &str) -> TokenType {
        if word.is_empty() {
            return TokenType::Other;
        }

        // Check if all characters are numeric
        if word.chars().all(|c| c.is_ascii_digit()) {
            return TokenType::Num;
        }

        // Check if it contains CJK characters
        if word.chars().any(|c| {
            matches!(c,
                '\u{4E00}'..='\u{9FFF}' |  // CJK Unified Ideographs
                '\u{3400}'..='\u{4DBF}' |  // CJK Extension A
                '\u{20000}'..='\u{2A6DF}' | // CJK Extension B
                '\u{2A700}'..='\u{2B73F}' | // CJK Extension C
                '\u{2B740}'..='\u{2B81F}' | // CJK Extension D
                '\u{2B820}'..='\u{2CEAF}'   // CJK Extension E
            )
        }) {
            return TokenType::Cjk;
        }

        // Check if it's Katakana
        if word.chars().all(|c| matches!(c, '\u{30A0}'..='\u{30FF}')) {
            return TokenType::Katakana;
        }

        // Check if it's Hiragana
        if word.chars().all(|c| matches!(c, '\u{3040}'..='\u{309F}')) {
            return TokenType::Hiragana;
        }

        // Check if it's Hangul
        if word
            .chars()
            .any(|c| matches!(c, '\u{AC00}'..='\u{D7AF}' | '\u{1100}'..='\u{11FF}'))
        {
            return TokenType::Hangul;
        }

        // Check if it's alphanumeric (ASCII)
        if word
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        {
            return TokenType::Alphanum;
        }

        // Check if it's all punctuation
        if word.chars().all(|c| c.is_ascii_punctuation()) {
            return TokenType::Punctuation;
        }

        // Default to Other
        TokenType::Other
    }

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
                let token_type = Self::detect_token_type(word);
                let token = Token::with_offsets(word, position, start, word_end)
                    .with_token_type(token_type);
                tokens.push(token);
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
                let token_type = Self::detect_token_type(word);
                Token::with_offsets(word, position, start_offset, end_offset)
                    .with_token_type(token_type)
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
