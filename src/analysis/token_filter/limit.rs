//! Limit filter implementation.
//!
//! This module provides a filter that limits the maximum number of tokens
//! in a stream. This is useful for truncating long documents or controlling
//! indexing costs.
//!
//! # Examples
//!
//! ```
//! use sarissa::analysis::token_filter::Filter;
//! use sarissa::analysis::token_filter::limit::LimitFilter;
//! use sarissa::analysis::token::Token;
//!
//! let filter = LimitFilter::new(3);
//! let tokens = vec![
//!     Token::new("one", 0),
//!     Token::new("two", 1),
//!     Token::new("three", 2),
//!     Token::new("four", 3),
//!     Token::new("five", 4),
//! ];
//!
//! let result: Vec<_> = filter.filter(Box::new(tokens.into_iter()))
//!     .unwrap()
//!     .collect();
//!
//! // Only first 3 tokens are kept
//! assert_eq!(result.len(), 3);
//! ```

use crate::analysis::token::{Token, TokenStream};
use crate::analysis::token_filter::Filter;
use crate::error::Result;

/// A filter that limits the number of tokens in the stream.
///
/// This filter truncates the token stream after a specified number of tokens,
/// which is useful for:
///
/// - Controlling indexing costs for large documents
/// - Implementing "index first N tokens only" strategies
/// - Testing and development with truncated input
/// - Implementing document preview features
///
/// # Examples
///
/// ```
/// use sarissa::analysis::token_filter::Filter;
/// use sarissa::analysis::token_filter::limit::LimitFilter;
/// use sarissa::analysis::token::Token;
///
/// let filter = LimitFilter::new(2);
/// let tokens = vec![
///     Token::new("first", 0),
///     Token::new("second", 1),
///     Token::new("third", 2),
/// ];
///
/// let result: Vec<_> = filter.filter(Box::new(tokens.into_iter()))
///     .unwrap()
///     .collect();
///
/// assert_eq!(result.len(), 2);
/// assert_eq!(result[0].text, "first");
/// assert_eq!(result[1].text, "second");
/// ```
#[derive(Clone, Debug)]
pub struct LimitFilter {
    limit: usize,
}

impl LimitFilter {
    /// Create a new limit filter with the given limit.
    ///
    /// # Arguments
    ///
    /// * `limit` - The maximum number of tokens to pass through
    ///
    /// # Examples
    ///
    /// ```
    /// use sarissa::analysis::token_filter::limit::LimitFilter;
    ///
    /// let filter = LimitFilter::new(100);
    /// assert_eq!(filter.limit(), 100);
    /// ```
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
