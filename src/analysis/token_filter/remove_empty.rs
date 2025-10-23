//! Remove empty filter implementation.

use crate::analysis::token::{Token, TokenStream};
use crate::analysis::token_filter::Filter;
use crate::error::Result;

/// A filter that removes empty tokens from the stream.
#[derive(Clone, Debug, Default)]
pub struct RemoveEmptyFilter;

impl RemoveEmptyFilter {
    /// Create a new remove empty filter.
    pub fn new() -> Self {
        RemoveEmptyFilter
    }
}

impl Filter for RemoveEmptyFilter {
    fn filter(&self, tokens: TokenStream) -> Result<TokenStream> {
        let filtered_tokens: Vec<Token> = tokens
            .filter(|token| !token.is_stopped() && !token.text.is_empty())
            .collect();

        Ok(Box::new(filtered_tokens.into_iter()))
    }

    fn name(&self) -> &'static str {
        "remove_empty"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::token::Token;

    #[test]
    fn test_remove_empty_filter() {
        let filter = RemoveEmptyFilter::new();
        let tokens = vec![
            Token::new("hello", 0),
            Token::new("", 1),
            Token::new("world", 2),
            Token::new("test", 3).stop(),
        ];
        let token_stream = Box::new(tokens.into_iter());

        let result: Vec<Token> = filter.filter(token_stream).unwrap().collect();

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].text, "hello");
        assert_eq!(result[1].text, "world");
    }

    #[test]
    fn test_filter_name() {
        assert_eq!(RemoveEmptyFilter::new().name(), "remove_empty");
    }
}
