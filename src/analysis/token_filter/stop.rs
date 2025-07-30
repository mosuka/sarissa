//! Stop filter implementation.

use std::collections::HashSet;
use std::sync::Arc;

use super::Filter;

use crate::analysis::token::{Token, TokenStream};
use crate::error::Result;

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
    fn test_filter_name() {
        assert_eq!(StopFilter::new().name(), "stop");
    }
}
