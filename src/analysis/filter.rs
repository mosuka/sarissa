//! Filter implementations for token transformation.

use crate::analysis::token::{Token, TokenStream};
use crate::error::Result;
use crate::util::simd;
use std::collections::HashSet;
use std::sync::Arc;

/// Trait for filters that transform token streams.
pub trait Filter: Send + Sync {
    /// Apply this filter to a token stream.
    fn filter(&self, tokens: TokenStream) -> Result<TokenStream>;

    /// Get the name of this filter (for debugging and configuration).
    fn name(&self) -> &'static str;
}

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

/// A filter that removes stop words from the token stream.
#[derive(Clone, Debug)]
pub struct StopFilter {
    /// The set of stop words to remove
    stop_words: Arc<HashSet<String>>,
    /// Whether to remove stopped tokens entirely or just mark them as stopped
    remove_stopped: bool,
}

impl StopFilter {
    /// Create a new stop filter with the default English stop words.
    pub fn new() -> Self {
        Self::with_stop_words(default_english_stop_words())
    }

    /// Create a new stop filter with custom stop words.
    pub fn with_stop_words(stop_words: HashSet<String>) -> Self {
        StopFilter {
            stop_words: Arc::new(stop_words),
            remove_stopped: true,
        }
    }

    /// Create a new stop filter from a list of stop words.
    pub fn from_words<I, S>(words: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let stop_words = words.into_iter().map(|s| s.into()).collect();
        Self::with_stop_words(stop_words)
    }

    /// Set whether to remove stopped tokens entirely or just mark them as stopped.
    pub fn remove_stopped(mut self, remove: bool) -> Self {
        self.remove_stopped = remove;
        self
    }

    /// Check if a word is a stop word.
    pub fn is_stop_word(&self, word: &str) -> bool {
        self.stop_words.contains(word)
    }

    /// Get the number of stop words.
    pub fn len(&self) -> usize {
        self.stop_words.len()
    }

    /// Check if the stop word set is empty.
    pub fn is_empty(&self) -> bool {
        self.stop_words.is_empty()
    }
}

impl Default for StopFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl Filter for StopFilter {
    fn filter(&self, tokens: TokenStream) -> Result<TokenStream> {
        let filtered_tokens: Vec<Token> = tokens
            .filter_map(|token| {
                if token.is_stopped() {
                    Some(token)
                } else if self.is_stop_word(&token.text) {
                    if self.remove_stopped {
                        None // Remove the token entirely
                    } else {
                        Some(token.stop()) // Mark as stopped but keep it
                    }
                } else {
                    Some(token)
                }
            })
            .collect();

        Ok(Box::new(filtered_tokens.into_iter()))
    }

    fn name(&self) -> &'static str {
        "stop"
    }
}

/// A filter that removes leading and trailing whitespace from tokens.
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

/// Default English stop words.
fn default_english_stop_words() -> HashSet<String> {
    let words = [
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in", "is",
        "it", "its", "of", "on", "that", "the", "to", "was", "were", "will", "with", "the", "this",
        "but", "they", "have", "had", "what", "said", "each", "which", "their", "time", "will",
        "about", "if", "up", "out", "many", "then", "them", "these", "so", "some", "her", "would",
        "make", "like", "into", "him", "two", "more", "go", "no", "way", "could", "my", "than",
        "first", "been", "call", "who", "oil", "sit", "now", "find", "down", "day", "did", "get",
        "come", "made", "may", "part",
    ];

    words.iter().map(|&s| s.to_string()).collect()
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
    fn test_stop_filter() {
        let filter = StopFilter::from_words(vec!["the", "and", "or"]);
        let tokens = vec![
            Token::new("hello", 0),
            Token::new("the", 1),
            Token::new("world", 2),
            Token::new("and", 3),
            Token::new("test", 4),
        ];
        let token_stream = Box::new(tokens.into_iter());

        let result: Vec<Token> = filter.filter(token_stream).unwrap().collect();

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].text, "hello");
        assert_eq!(result[1].text, "world");
        assert_eq!(result[2].text, "test");
    }

    #[test]
    fn test_stop_filter_preserve_stopped() {
        let filter = StopFilter::from_words(vec!["the", "and"]).remove_stopped(false);
        let tokens = vec![
            Token::new("hello", 0),
            Token::new("the", 1),
            Token::new("world", 2),
        ];
        let token_stream = Box::new(tokens.into_iter());

        let result: Vec<Token> = filter.filter(token_stream).unwrap().collect();

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].text, "hello");
        assert!(!result[0].is_stopped());
        assert_eq!(result[1].text, "the");
        assert!(result[1].is_stopped());
        assert_eq!(result[2].text, "world");
        assert!(!result[2].is_stopped());
    }

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
        assert_eq!(LowercaseFilter::new().name(), "lowercase");
        assert_eq!(StopFilter::new().name(), "stop");
        assert_eq!(StripFilter::new().name(), "strip");
        assert_eq!(RemoveEmptyFilter::new().name(), "remove_empty");
        assert_eq!(BoostFilter::new(1.0).name(), "boost");
        assert_eq!(LimitFilter::new(10).name(), "limit");
    }
}
