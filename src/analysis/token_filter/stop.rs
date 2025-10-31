//! Stop filter implementation.
//!
//! This module provides a filter that removes common words (stop words) that
//! typically don't contribute to search relevance. Includes default stop word
//! lists for English and Japanese, with support for custom word lists.
//!
//! # Examples
//!
//! ```
//! use yatagarasu::analysis::token_filter::Filter;
//! use yatagarasu::analysis::token_filter::stop::StopFilter;
//! use yatagarasu::analysis::token::Token;
//!
//! let filter = StopFilter::new(); // Uses default English stop words
//! let tokens = vec![
//!     Token::new("the", 0),
//!     Token::new("quick", 1),
//!     Token::new("brown", 2)
//! ];
//!
//! let result: Vec<_> = filter.filter(Box::new(tokens.into_iter()))
//!     .unwrap()
//!     .collect();
//!
//! // "the" is removed as a stop word
//! assert_eq!(result.len(), 2);
//! assert_eq!(result[0].text, "quick");
//! assert_eq!(result[1].text, "brown");
//! ```

use std::collections::HashSet;
use std::sync::{Arc, LazyLock};

use crate::analysis::token::{Token, TokenStream};
use crate::analysis::token_filter::Filter;
use crate::error::Result;

/// Default English stop words list.
///
/// Common English words that are typically filtered out during indexing.
const DEFAULT_ENGLISH_STOP_WORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it",
    "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these",
    "they", "this", "to", "was", "will", "with",
];

const DEFAULT_JAPANESE_STOP_WORDS: &[&str] = &[
    "の",
    "に",
    "は",
    "を",
    "た",
    "が",
    "で",
    "て",
    "と",
    "し",
    "れ",
    "さ",
    "ある",
    "いる",
    "も",
    "する",
    "から",
    "な",
    "こと",
    "として",
    "い",
    "や",
    "れる",
    "など",
    "なっ",
    "ない",
    "この",
    "ため",
    "その",
    "あっ",
    "よう",
    "また",
    "もの",
    "という",
    "あり",
    "まで",
    "られ",
    "なる",
    "へ",
    "か",
    "だ",
    "これ",
    "によって",
    "により",
    "おり",
    "より",
    "による",
    "ず",
    "なり",
    "られる",
    "において",
    "ば",
    "なかっ",
    "なく",
    "しかし",
    "について",
    "せ",
    "だっ",
    "その後",
    "できる",
    "それ",
    "う",
    "ので",
    "なお",
    "のみ",
    "でき",
    "き",
    "つ",
    "における",
    "および",
    "いう",
    "さらに",
    "でも",
    "ら",
    "たり",
    "その他",
    "に関する",
    "たち",
    "ます",
    "ん",
    "なら",
    "に対して",
    "特に",
    "せる",
    "及び",
    "これら",
    "とき",
    "では",
    "にて",
    "ほか",
    "ながら",
    "うち",
    "そして",
    "とともに",
    "ただし",
    "かつて",
    "それぞれ",
    "または",
    "お",
    "ほど",
    "ものの",
    "に対する",
    "ほとんど",
    "と共に",
    "といった",
    "です",
    "とも",
    "ところ",
    "ここ",
];

/// Default English stop words as a HashSet.
pub static DEFAULT_ENGLISH_STOP_WORDS_SET: LazyLock<HashSet<String>> = LazyLock::new(|| {
    DEFAULT_ENGLISH_STOP_WORDS
        .iter()
        .map(|&s| s.to_string())
        .collect()
});

/// Default Japanese stop words as a HashSet.
pub static DEFAULT_JAPANESE_STOP_WORDS_SET: LazyLock<HashSet<String>> = LazyLock::new(|| {
    DEFAULT_JAPANESE_STOP_WORDS
        .iter()
        .map(|&s| s.to_string())
        .collect()
});

/// A filter that removes stop words from the token stream.
///
/// Stop words are common words (like "the", "is", "at") that are often
/// filtered out during text analysis because they typically don't contribute
/// to search relevance. This filter can either remove stop words entirely
/// or mark them as stopped while keeping them in the stream.
///
/// # Default Stop Word Lists
///
/// - English: 33 common words (articles, prepositions, conjunctions)
/// - Japanese: 127 common particles and auxiliary verbs
///
/// # Examples
///
/// ## Basic Usage
///
/// ```
/// use yatagarasu::analysis::token_filter::Filter;
/// use yatagarasu::analysis::token_filter::stop::StopFilter;
/// use yatagarasu::analysis::token::Token;
///
/// let filter = StopFilter::new();
/// let tokens = vec![
///     Token::new("this", 0),
///     Token::new("is", 1),
///     Token::new("test", 2)
/// ];
///
/// let result: Vec<_> = filter.filter(Box::new(tokens.into_iter()))
///     .unwrap()
///     .collect();
///
/// // Only "test" remains
/// assert_eq!(result.len(), 1);
/// assert_eq!(result[0].text, "test");
/// ```
///
/// ## Custom Stop Words
///
/// ```
/// use yatagarasu::analysis::token_filter::stop::StopFilter;
///
/// let filter = StopFilter::from_words(vec!["custom", "words", "list"]);
/// ```
///
/// ## Preserve Stopped Tokens
///
/// ```
/// use yatagarasu::analysis::token_filter::Filter;
/// use yatagarasu::analysis::token_filter::stop::StopFilter;
/// use yatagarasu::analysis::token::Token;
///
/// // Mark as stopped but don't remove
/// let filter = StopFilter::from_words(vec!["the"]).remove_stopped(false);
/// let tokens = vec![Token::new("the", 0), Token::new("quick", 1)];
///
/// let result: Vec<_> = filter.filter(Box::new(tokens.into_iter()))
///     .unwrap()
///     .collect();
///
/// assert_eq!(result.len(), 2);
/// assert!(result[0].is_stopped());  // Marked as stopped
/// assert!(!result[1].is_stopped());
/// ```
#[derive(Clone, Debug)]
pub struct StopFilter {
    /// The set of stop words to remove
    stop_words: Arc<HashSet<String>>,
    /// Whether to remove stopped tokens entirely or just mark them as stopped
    remove_stopped: bool,
}

impl StopFilter {
    /// Create a new stop filter with the default English stop words.
    ///
    /// # Examples
    ///
    /// ```
    /// use yatagarasu::analysis::token_filter::stop::StopFilter;
    ///
    /// let filter = StopFilter::new();
    /// assert!(filter.is_stop_word("the"));
    /// assert!(!filter.is_stop_word("hello"));
    /// ```
    pub fn new() -> Self {
        Self::with_stop_words(DEFAULT_ENGLISH_STOP_WORDS_SET.clone())
    }

    /// Create a new stop filter with custom stop words.
    ///
    /// # Arguments
    ///
    /// * `stop_words` - A set of words to filter out
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashSet;
    /// use yatagarasu::analysis::token_filter::stop::StopFilter;
    ///
    /// let mut words = HashSet::new();
    /// words.insert("custom".to_string());
    /// words.insert("stop".to_string());
    ///
    /// let filter = StopFilter::with_stop_words(words);
    /// assert!(filter.is_stop_word("custom"));
    /// ```
    pub fn with_stop_words(stop_words: HashSet<String>) -> Self {
        StopFilter {
            stop_words: Arc::new(stop_words),
            remove_stopped: true,
        }
    }

    /// Create a new stop filter from a list of stop words.
    ///
    /// # Arguments
    ///
    /// * `words` - An iterator of words to filter out
    ///
    /// # Examples
    ///
    /// ```
    /// use yatagarasu::analysis::token_filter::stop::StopFilter;
    ///
    /// let filter = StopFilter::from_words(vec!["foo", "bar", "baz"]);
    /// assert_eq!(filter.len(), 3);
    /// ```
    pub fn from_words<I, S>(words: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let stop_words = words.into_iter().map(|s| s.into()).collect();
        Self::with_stop_words(stop_words)
    }

    /// Set whether to remove stopped tokens entirely or just mark them as stopped.
    ///
    /// # Arguments
    ///
    /// * `remove` - If `true`, remove stopped tokens; if `false`, mark them as stopped
    ///
    /// # Examples
    ///
    /// ```
    /// use yatagarasu::analysis::token_filter::stop::StopFilter;
    ///
    /// // Keep stopped tokens but mark them
    /// let filter = StopFilter::new().remove_stopped(false);
    /// ```
    pub fn remove_stopped(mut self, remove: bool) -> Self {
        self.remove_stopped = remove;
        self
    }

    /// Check if a word is a stop word.
    ///
    /// # Arguments
    ///
    /// * `word` - The word to check
    ///
    /// # Returns
    ///
    /// `true` if the word is in the stop word set, `false` otherwise
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
