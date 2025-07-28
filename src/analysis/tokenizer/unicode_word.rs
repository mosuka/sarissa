//! Unicode word tokenizer implementation.

use super::Tokenizer;
use crate::analysis::token::{Token, TokenStream};
use crate::error::Result;
use unicode_segmentation::UnicodeSegmentation;

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
