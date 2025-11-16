//! Unicode word tokenizer implementation.
//!
//! This module provides a tokenizer that splits text using Unicode word boundary
//! rules (UAX #29). It properly handles international text and filters out non-word
//! segments like punctuation and whitespace.
//!
//! # Examples
//!
//! ```
//! use platypus::analysis::tokenizer::Tokenizer;
//! use platypus::analysis::tokenizer::unicode_word::UnicodeWordTokenizer;
//!
//! let tokenizer = UnicodeWordTokenizer::new();
//! let tokens: Vec<_> = tokenizer.tokenize("Hello, world! 你好世界").unwrap().collect();
//!
//! // Punctuation and whitespace are automatically filtered out
//! assert_eq!(tokens[0].text, "Hello");
//! assert_eq!(tokens[1].text, "world");
//! ```

use unicode_segmentation::UnicodeSegmentation;

use crate::analysis::token::{Token, TokenStream, TokenType};
use crate::analysis::tokenizer::Tokenizer;
use crate::error::Result;

/// A tokenizer that splits text on Unicode word boundaries.
///
/// This tokenizer uses the Unicode Text Segmentation algorithm (UAX #29) to
/// identify word boundaries. It automatically filters out non-word segments
/// like punctuation and whitespace, keeping only alphanumeric tokens.
///
/// # Features
///
/// - Proper handling of international text (CJK, Arabic, etc.)
/// - Automatic filtering of punctuation and whitespace
/// - Token type detection for different character scripts
/// - Compliant with Unicode Standard Annex #29
///
/// # Examples
///
/// ```
/// use platypus::analysis::tokenizer::Tokenizer;
/// use platypus::analysis::tokenizer::unicode_word::UnicodeWordTokenizer;
///
/// let tokenizer = UnicodeWordTokenizer::new();
/// let tokens: Vec<_> = tokenizer.tokenize("café résumé").unwrap().collect();
/// assert_eq!(tokens.len(), 2);
/// assert_eq!(tokens[0].text, "café");
/// assert_eq!(tokens[1].text, "résumé");
/// ```
#[derive(Clone, Debug, Default)]
pub struct UnicodeWordTokenizer;

impl UnicodeWordTokenizer {
    /// Create a new Unicode word tokenizer.
    pub fn new() -> Self {
        UnicodeWordTokenizer
    }

    /// Detect token type based on character content.
    ///
    /// Analyzes the word's characters to determine the appropriate token type:
    /// - All numeric → Num
    /// - All Hiragana → Hiragana
    /// - All Katakana → Katakana
    /// - Contains Hangul → Hangul
    /// - Contains CJK → Cjk
    /// - ASCII alphanumeric → Alphanum
    /// - All punctuation → Punctuation
    /// - Otherwise → Other
    fn detect_token_type(word: &str) -> TokenType {
        if word.is_empty() {
            return TokenType::Other;
        }

        // Check if all characters are numeric
        if word.chars().all(|c| c.is_numeric()) {
            return TokenType::Num;
        }

        // Check if it's Hiragana
        if word.chars().all(|c| matches!(c, '\u{3040}'..='\u{309F}')) {
            return TokenType::Hiragana;
        }

        // Check if it's Katakana
        if word.chars().all(|c| matches!(c, '\u{30A0}'..='\u{30FF}')) {
            return TokenType::Katakana;
        }

        // Check if it's Hangul
        if word
            .chars()
            .any(|c| matches!(c, '\u{AC00}'..='\u{D7AF}' | '\u{1100}'..='\u{11FF}'))
        {
            return TokenType::Hangul;
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

        // Check if it's alphanumeric (ASCII)
        if word
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        {
            return TokenType::Alphanum;
        }

        // Check if it's punctuation
        if word.chars().all(|c| c.is_ascii_punctuation()) {
            return TokenType::Punctuation;
        }

        TokenType::Other
    }
}

impl Tokenizer for UnicodeWordTokenizer {
    fn tokenize(&self, text: &str) -> Result<TokenStream> {
        let tokens: Vec<Token> = text
            .split_word_bounds()
            .enumerate()
            .filter_map(|(position, word)| {
                // Only keep actual words (not whitespace or punctuation)
                if word.chars().any(|c| c.is_alphanumeric()) {
                    // Find the actual position in the original text
                    let start_offset = text.find(word).unwrap_or(0);
                    let end_offset = start_offset + word.len();
                    let token_type = Self::detect_token_type(word);
                    Some(
                        Token::with_offsets(word, position, start_offset, end_offset)
                            .with_token_type(token_type),
                    )
                } else {
                    None
                }
            })
            .collect();

        Ok(Box::new(tokens.into_iter()))
    }

    fn name(&self) -> &'static str {
        "unicode_word"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unicode_word_tokenizer() {
        let tokenizer = UnicodeWordTokenizer::new();
        let tokens: Vec<Token> = tokenizer.tokenize("hello, world!").unwrap().collect();

        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "hello");
        assert_eq!(tokens[1].text, "world");
    }

    #[test]
    fn test_tokenizer_name() {
        assert_eq!(UnicodeWordTokenizer::new().name(), "unicode_word");
    }
}
