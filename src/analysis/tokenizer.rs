//! Tokenizer implementations for text analysis.

use crate::analysis::token::{Token, TokenStream};
use crate::error::{SarissaError, Result};
use crate::util::simd;
use regex::Regex;
use std::sync::Arc;
use unicode_segmentation::UnicodeSegmentation;

/// Trait for tokenizers that convert text into tokens.
pub trait Tokenizer: Send + Sync {
    /// Tokenize the given text into a stream of tokens.
    fn tokenize(&self, text: &str) -> Result<TokenStream>;

    /// Get the name of this tokenizer (for debugging and configuration).
    fn name(&self) -> &'static str;
}

/// A regex-based tokenizer that extracts tokens using regular expressions.
///
/// This is the default tokenizer and is equivalent to Whoosh's RegexTokenizer.
#[derive(Clone, Debug)]
pub struct RegexTokenizer {
    /// The regex pattern used to extract tokens
    pattern: Arc<Regex>,
    /// Whether to extract gaps (text between matches) instead of matches
    gaps: bool,
}

impl RegexTokenizer {
    /// Create a new regex tokenizer with the default pattern.
    ///
    /// The default pattern `r"\w+"` matches sequences of word characters.
    pub fn new() -> Result<Self> {
        Self::with_pattern(r"\w+")
    }

    /// Create a new regex tokenizer with a custom pattern.
    pub fn with_pattern(pattern: &str) -> Result<Self> {
        let regex = Regex::new(pattern)
            .map_err(|e| SarissaError::analysis(format!("Invalid regex pattern: {e}")))?;

        Ok(RegexTokenizer {
            pattern: Arc::new(regex),
            gaps: false,
        })
    }

    /// Create a tokenizer that extracts gaps (text between matches) instead of matches.
    pub fn with_gaps(pattern: &str) -> Result<Self> {
        let regex = Regex::new(pattern)
            .map_err(|e| SarissaError::analysis(format!("Invalid regex pattern: {e}")))?;

        Ok(RegexTokenizer {
            pattern: Arc::new(regex),
            gaps: true,
        })
    }

    /// Get the regex pattern used by this tokenizer.
    pub fn pattern(&self) -> &str {
        self.pattern.as_str()
    }

    /// Check if this tokenizer extracts gaps.
    pub fn gaps(&self) -> bool {
        self.gaps
    }
}

impl Default for RegexTokenizer {
    fn default() -> Self {
        Self::new().expect("Default regex pattern should be valid")
    }
}

impl Tokenizer for RegexTokenizer {
    fn tokenize(&self, text: &str) -> Result<TokenStream> {
        let pattern = Arc::clone(&self.pattern);
        let gaps = self.gaps;
        let text = text.to_owned();

        let tokens = if gaps {
            // Extract gaps between matches
            let mut tokens = Vec::new();
            let mut last_end = 0;
            let mut position = 0;

            for mat in pattern.find_iter(&text) {
                if mat.start() > last_end {
                    let gap_text = &text[last_end..mat.start()];
                    if !gap_text.is_empty() {
                        tokens.push(Token::with_offsets(
                            gap_text,
                            position,
                            last_end,
                            mat.start(),
                        ));
                        position += 1;
                    }
                }
                last_end = mat.end();
            }

            // Add final gap if any
            if last_end < text.len() {
                let gap_text = &text[last_end..];
                if !gap_text.is_empty() {
                    tokens.push(Token::with_offsets(
                        gap_text,
                        position,
                        last_end,
                        text.len(),
                    ));
                }
            }

            tokens
        } else {
            // Extract matches
            pattern
                .find_iter(&text)
                .enumerate()
                .map(|(position, mat)| {
                    Token::with_offsets(mat.as_str(), position, mat.start(), mat.end())
                })
                .collect()
        };

        Ok(Box::new(tokens.into_iter()))
    }

    fn name(&self) -> &'static str {
        "regex"
    }
}

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

/// A tokenizer that splits text on Unicode word boundaries.
#[derive(Clone, Debug, Default)]
pub struct UnicodeWordTokenizer;

impl UnicodeWordTokenizer {
    /// Create a new Unicode word tokenizer.
    pub fn new() -> Self {
        UnicodeWordTokenizer
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
                    Some(Token::with_offsets(
                        word,
                        position,
                        start_offset,
                        end_offset,
                    ))
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
    fn test_regex_tokenizer() {
        let tokenizer = RegexTokenizer::new().unwrap();
        let tokens: Vec<Token> = tokenizer.tokenize("hello world").unwrap().collect();

        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "hello");
        assert_eq!(tokens[0].position, 0);
        assert_eq!(tokens[0].start_offset, 0);
        assert_eq!(tokens[0].end_offset, 5);

        assert_eq!(tokens[1].text, "world");
        assert_eq!(tokens[1].position, 1);
        assert_eq!(tokens[1].start_offset, 6);
        assert_eq!(tokens[1].end_offset, 11);
    }

    #[test]
    fn test_regex_tokenizer_with_gaps() {
        let tokenizer = RegexTokenizer::with_gaps(r"\s+").unwrap();
        let tokens: Vec<Token> = tokenizer.tokenize("hello world").unwrap().collect();

        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "hello");
        assert_eq!(tokens[1].text, "world");
    }

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
    fn test_whitespace_tokenizer() {
        let tokenizer = WhitespaceTokenizer::new();
        let tokens: Vec<Token> = tokenizer.tokenize("hello  world\ttest").unwrap().collect();

        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].text, "hello");
        assert_eq!(tokens[1].text, "world");
        assert_eq!(tokens[2].text, "test");
    }

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
        assert_eq!(RegexTokenizer::new().unwrap().name(), "regex");
        assert_eq!(WholeTokenizer::new().name(), "whole");
        assert_eq!(WhitespaceTokenizer::new().name(), "whitespace");
        assert_eq!(UnicodeWordTokenizer::new().name(), "unicode_word");
    }
}
