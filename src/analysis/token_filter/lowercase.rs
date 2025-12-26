//! Lowercase filter implementation.
//!
//! This module provides a filter that converts all token text to lowercase,
//! which is essential for case-insensitive search. The filter uses SIMD
//! optimizations for ASCII text to provide better performance.
//!
//! # Examples
//!
//! ```
//! use sarissa::analysis::token_filter::Filter;
//! use sarissa::analysis::token_filter::lowercase::LowercaseFilter;
//! use sarissa::analysis::token::Token;
//!
//! let filter = LowercaseFilter::new();
//! let tokens = vec![Token::new("Hello", 0), Token::new("WORLD", 1)];
//! let filtered: Vec<_> = filter.filter(Box::new(tokens.into_iter()))
//!     .unwrap()
//!     .collect();
//!
//! assert_eq!(filtered[0].text, "hello");
//! assert_eq!(filtered[1].text, "world");
//! ```

use crate::analysis::token::TokenStream;
use crate::analysis::token_filter::Filter;
use crate::error::Result;
use crate::util::simd;

/// A filter that converts tokens to lowercase.
///
/// This filter normalizes text casing to enable case-insensitive matching.
/// It uses SIMD-accelerated lowercasing for ASCII text and falls back to
/// Unicode-aware lowercasing for other text.
///
/// # Behavior
///
/// - Converts all characters to lowercase
/// - Skips tokens marked as stopped
/// - Preserves token positions and offsets
/// - Uses SIMD optimization for ASCII text
///
/// # Examples
///
/// ```
/// use sarissa::analysis::token_filter::Filter;
/// use sarissa::analysis::token_filter::lowercase::LowercaseFilter;
/// use sarissa::analysis::token::Token;
///
/// let filter = LowercaseFilter::new();
/// let tokens = vec![
///     Token::new("The", 0),
///     Token::new("QUICK", 1),
///     Token::new("Brown", 2)
/// ];
///
/// let result: Vec<_> = filter.filter(Box::new(tokens.into_iter()))
///     .unwrap()
///     .collect();
///
/// assert_eq!(result[0].text, "the");
/// assert_eq!(result[1].text, "quick");
/// assert_eq!(result[2].text, "brown");
/// ```
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
