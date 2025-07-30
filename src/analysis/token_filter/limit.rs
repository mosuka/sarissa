//! Limit filter implementation.

use super::Filter;

use crate::analysis::token::{Token, TokenStream};
use crate::error::Result;

/// A filter that limits the number of tokens in the stream.
#[derive(Clone, Debug)]
pub struct LimitFilter {
    limit: usize,
}

impl LimitFilter {
    /// Create a new limit filter with the given limit.
    pub fn new(limit: usize) -> Self {
        LimitFilter { limit }
    }

    /// Get the limit.
    pub fn limit(&self) -> usize {
        self.limit
    }
}

impl Filter for LimitFilter {
    fn filter(&self, tokens: TokenStream) -> Result<TokenStream> {
        let limited_tokens: Vec<Token> = tokens.take(self.limit).collect();
        Ok(Box::new(limited_tokens.into_iter()))
    }

    fn name(&self) -> &'static str {
        "limit"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::token::Token;

    #[test]
    fn test_limit_filter() {
        let filter = LimitFilter::new(2);
        let tokens = vec![
            Token::new("hello", 0),
            Token::new("world", 1),
            Token::new("test", 2),
            Token::new("limit", 3),
        ];
        let token_stream = Box::new(tokens.into_iter());

        let result: Vec<Token> = filter.filter(token_stream).unwrap().collect();

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].text, "hello");
        assert_eq!(result[1].text, "world");
    }

    #[test]
    fn test_filter_name() {
        assert_eq!(LimitFilter::new(10).name(), "limit");
    }
}
