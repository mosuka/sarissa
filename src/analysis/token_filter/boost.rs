//! Boost filter implementation.

use super::Filter;
use crate::analysis::token::TokenStream;
use crate::error::Result;

/// A filter that applies a boost to all tokens.
#[derive(Clone, Debug)]
pub struct BoostFilter {
    boost: f32,
}

impl BoostFilter {
    /// Create a new boost filter with the given boost factor.
    pub fn new(boost: f32) -> Self {
        BoostFilter { boost }
    }

    /// Get the boost factor.
    pub fn boost(&self) -> f32 {
        self.boost
    }
}

impl Filter for BoostFilter {
    fn filter(&self, tokens: TokenStream) -> Result<TokenStream> {
        let filtered_tokens = tokens
            .map(|token| {
                if token.is_stopped() {
                    token
                } else {
                    let boost = token.boost;
                    token.with_boost(boost * self.boost)
                }
            })
            .collect::<Vec<_>>();

        Ok(Box::new(filtered_tokens.into_iter()))
    }

    fn name(&self) -> &'static str {
        "boost"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::token::Token;

    #[test]
    fn test_boost_filter() {
        let filter = BoostFilter::new(2.0);
        let tokens = vec![
            Token::new("hello", 0),
            Token::new("world", 1).with_boost(1.5),
            Token::new("test", 2).stop(),
        ];
        let token_stream = Box::new(tokens.into_iter());

        let result: Vec<Token> = filter.filter(token_stream).unwrap().collect();

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].boost, 2.0);
        assert_eq!(result[1].boost, 3.0);
        assert_eq!(result[2].boost, 1.0); // Stopped tokens are not processed
    }

    #[test]
    fn test_filter_name() {
        assert_eq!(BoostFilter::new(1.0).name(), "boost");
    }
}