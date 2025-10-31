//! Strip filter implementation.
//!
//! This module provides a filter that removes leading and trailing whitespace
//! from token text. Tokens that become empty after stripping are marked as stopped.
//!
//! # Examples
//!
//! ```
//! use yatagarasu::analysis::token_filter::Filter;
//! use yatagarasu::analysis::token_filter::strip::StripFilter;
//! use yatagarasu::analysis::token::Token;
//!
//! let filter = StripFilter::new();
//! let tokens = vec![Token::new("  hello  ", 0), Token::new("world", 1)];
//! let result: Vec<_> = filter.filter(Box::new(tokens.into_iter()))
//!     .unwrap()
//!     .collect();
//!
//! assert_eq!(result[0].text, "hello");  // Whitespace stripped
//! assert_eq!(result[1].text, "world");
//! ```

use crate::analysis::token::TokenStream;
use crate::analysis::token_filter::Filter;
use crate::error::Result;

/// A filter that removes leading and trailing whitespace from tokens.
///
/// This filter uses Rust's `str::trim()` to remove whitespace from both ends
/// of token text. If trimming results in an empty string, the token is marked
/// as stopped.
///
/// # Behavior
///
/// - Trims leading and trailing whitespace
/// - Marks empty tokens as stopped
/// - Skips already-stopped tokens
///
/// # Examples
///
/// ```
/// use yatagarasu::analysis::token_filter::Filter;
/// use yatagarasu::analysis::token_filter::strip::StripFilter;
/// use yatagarasu::analysis::token::Token;
///
/// let filter = StripFilter::new();
/// let tokens = vec![
///     Token::new("  spaces  ", 0),
///     Token::new("\ttabs\t", 1),
///     Token::new("   ", 2)  // Will be marked as stopped
/// ];
///
/// let result: Vec<_> = filter.filter(Box::new(tokens.into_iter()))
///     .unwrap()
///     .collect();
///
/// assert_eq!(result[0].text, "spaces");
/// assert_eq!(result[1].text, "tabs");
/// assert!(result[2].is_stopped());
/// ```
#[derive(Clone, Debug, Default)]
pub struct StripFilter;

impl StripFilter {
    /// Create a new strip filter.
    pub fn new() -> Self {
        StripFilter
    }
}

impl Filter for StripFilter {
    fn filter(&self, tokens: TokenStream) -> Result<TokenStream> {
        let filtered_tokens = tokens
            .map(|token| {
                if token.is_stopped() {
                    token
                } else {
                    let trimmed = token.text.trim();
                    if trimmed.is_empty() {
                        token.stop()
                    } else {
                        token.with_text(trimmed)
                    }
                }
            })
            .collect::<Vec<_>>();

        Ok(Box::new(filtered_tokens.into_iter()))
    }

    fn name(&self) -> &'static str {
        "strip"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::token::Token;

    #[test]
    fn test_strip_filter() {
        let filter = StripFilter::new();
        let tokens = vec![
            Token::new("  hello  ", 0),
            Token::new("world", 1),
            Token::new("   ", 2),
        ];
        let token_stream = Box::new(tokens.into_iter());

        let result: Vec<Token> = filter.filter(token_stream).unwrap().collect();

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].text, "hello");
        assert_eq!(result[1].text, "world");
        assert_eq!(result[2].text, "   ");
        assert!(result[2].is_stopped());
    }

    #[test]
    fn test_filter_name() {
        assert_eq!(StripFilter::new().name(), "strip");
    }
}
