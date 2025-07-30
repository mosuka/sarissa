//! Lowercase filter implementation.

use super::Filter;

use crate::analysis::token::TokenStream;
use crate::error::Result;
use crate::util::simd;

/// A filter that converts tokens to lowercase.
#[derive(Clone, Debug, Default)]
pub struct LowercaseFilter;

impl LowercaseFilter {
    /// Create a new lowercase filter.
    pub fn new() -> Self {
        LowercaseFilter
    }
}

impl Filter for LowercaseFilter {
    fn filter(&self, tokens: TokenStream) -> Result<TokenStream> {
        let filtered_tokens = tokens
            .map(|token| {
                if token.is_stopped() {
                    token
                } else {
                    token.with_text(simd::ascii::to_lowercase(&token.text))
                }
            })
            .collect::<Vec<_>>();

        Ok(Box::new(filtered_tokens.into_iter()))
    }

    fn name(&self) -> &'static str {
        "lowercase"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::token::Token;

    #[test]
    fn test_lowercase_filter() {
        let filter = LowercaseFilter::new();
        let tokens = vec![
            Token::new("Hello", 0),
            Token::new("WORLD", 1),
            Token::new("Test", 2).stop(),
        ];
        let token_stream = Box::new(tokens.into_iter());

        let result: Vec<Token> = filter.filter(token_stream).unwrap().collect();

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].text, "hello");
        assert_eq!(result[1].text, "world");
        assert_eq!(result[2].text, "Test"); // Stopped tokens are not processed
        assert!(result[2].is_stopped());
    }

    #[test]
    fn test_filter_name() {
        assert_eq!(LowercaseFilter::new().name(), "lowercase");
    }
}
