//! Token types and utilities for text analysis.

use serde::{Deserialize, Serialize};
use std::fmt;

/// A token represents a single unit of text after tokenization.
///
/// This is the fundamental unit that flows through the analysis pipeline.
/// It contains the text content, position information, and metadata.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Token {
    /// The text content of the token
    pub text: String,

    /// The position of the token in the original token stream (0-based)
    pub position: usize,

    /// The character offset where this token starts in the original text
    pub start_offset: usize,

    /// The character offset where this token ends in the original text
    pub end_offset: usize,

    /// Boost factor for this token (default: 1.0)
    pub boost: f32,

    /// Whether this token has been marked as stopped (removed) by a filter
    pub stopped: bool,

    /// Additional metadata that can be attached to tokens
    pub metadata: Option<TokenMetadata>,
}

/// Additional metadata that can be attached to tokens
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TokenMetadata {
    /// The original text before filtering (useful for highlighting)
    pub original_text: Option<String>,

    /// Token type (e.g., "word", "number", "punctuation")
    pub token_type: Option<String>,

    /// Language hint for language-specific processing
    pub language: Option<String>,

    /// Additional custom attributes
    pub attributes: std::collections::HashMap<String, String>,
}

impl Token {
    /// Create a new token with the given text and position.
    pub fn new<S: Into<String>>(text: S, position: usize) -> Self {
        Token {
            text: text.into(),
            position,
            start_offset: 0,
            end_offset: 0,
            boost: 1.0,
            stopped: false,
            metadata: None,
        }
    }

    /// Create a new token with text, position, and character offsets.
    pub fn with_offsets<S: Into<String>>(
        text: S,
        position: usize,
        start_offset: usize,
        end_offset: usize,
    ) -> Self {
        Token {
            text: text.into(),
            position,
            start_offset,
            end_offset,
            boost: 1.0,
            stopped: false,
            metadata: None,
        }
    }

    /// Get the length of the token text.
    pub fn len(&self) -> usize {
        self.text.len()
    }

    /// Check if the token is empty.
    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    /// Set the boost factor for this token.
    pub fn with_boost(mut self, boost: f32) -> Self {
        self.boost = boost;
        self
    }

    /// Mark this token as stopped.
    pub fn stop(mut self) -> Self {
        self.stopped = true;
        self
    }

    /// Check if this token is stopped.
    pub fn is_stopped(&self) -> bool {
        self.stopped
    }

    /// Set metadata for this token.
    pub fn with_metadata(mut self, metadata: TokenMetadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Get a reference to the metadata.
    pub fn metadata(&self) -> Option<&TokenMetadata> {
        self.metadata.as_ref()
    }

    /// Get a mutable reference to the metadata.
    pub fn metadata_mut(&mut self) -> Option<&mut TokenMetadata> {
        self.metadata.as_mut()
    }

    /// Set the original text in metadata.
    pub fn with_original_text<S: Into<String>>(mut self, original: S) -> Self {
        let metadata = self.metadata.get_or_insert_with(TokenMetadata::new);
        metadata.original_text = Some(original.into());
        self
    }

    /// Set the token type in metadata.
    pub fn with_token_type<S: Into<String>>(mut self, token_type: S) -> Self {
        let metadata = self.metadata.get_or_insert_with(TokenMetadata::new);
        metadata.token_type = Some(token_type.into());
        self
    }

    /// Clone this token with updated text.
    pub fn with_text<S: Into<String>>(&self, text: S) -> Self {
        let mut token = self.clone();
        token.text = text.into();
        token
    }

    /// Clone this token with updated position.
    pub fn with_position(&self, position: usize) -> Self {
        let mut token = self.clone();
        token.position = position;
        token
    }
}

impl TokenMetadata {
    /// Create a new empty metadata object.
    pub fn new() -> Self {
        TokenMetadata {
            original_text: None,
            token_type: None,
            language: None,
            attributes: std::collections::HashMap::new(),
        }
    }

    /// Set a custom attribute.
    pub fn set_attribute<K, V>(&mut self, key: K, value: V)
    where
        K: Into<String>,
        V: Into<String>,
    {
        self.attributes.insert(key.into(), value.into());
    }

    /// Get a custom attribute.
    pub fn get_attribute(&self, key: &str) -> Option<&str> {
        self.attributes.get(key).map(|s| s.as_str())
    }
}

impl Default for TokenMetadata {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.text)
    }
}

/// A token stream represents a sequence of tokens from the analysis pipeline.
pub type TokenStream = Box<dyn Iterator<Item = Token>>;

/// Trait for types that can produce a token stream.
pub trait IntoTokenStream {
    /// Convert this type into a token stream.
    fn into_token_stream(self) -> TokenStream;
}

impl IntoTokenStream for Vec<Token> {
    fn into_token_stream(self) -> TokenStream {
        Box::new(self.into_iter())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_creation() {
        let token = Token::new("hello", 0);
        assert_eq!(token.text, "hello");
        assert_eq!(token.position, 0);
        assert_eq!(token.start_offset, 0);
        assert_eq!(token.end_offset, 0);
        assert_eq!(token.boost, 1.0);
        assert!(!token.stopped);
        assert!(token.metadata.is_none());
    }

    #[test]
    fn test_token_with_offsets() {
        let token = Token::with_offsets("world", 1, 6, 11);
        assert_eq!(token.text, "world");
        assert_eq!(token.position, 1);
        assert_eq!(token.start_offset, 6);
        assert_eq!(token.end_offset, 11);
    }

    #[test]
    fn test_token_methods() {
        let token = Token::new("test", 0)
            .with_boost(2.0)
            .stop()
            .with_original_text("TEST")
            .with_token_type("word");

        assert_eq!(token.boost, 2.0);
        assert!(token.is_stopped());
        assert!(token.metadata.is_some());

        let metadata = token.metadata.as_ref().unwrap();
        assert_eq!(metadata.original_text.as_deref(), Some("TEST"));
        assert_eq!(metadata.token_type.as_deref(), Some("word"));
    }

    #[test]
    fn test_token_metadata() {
        let mut metadata = TokenMetadata::new();
        metadata.set_attribute("custom", "value");

        assert_eq!(metadata.get_attribute("custom"), Some("value"));
        assert_eq!(metadata.get_attribute("missing"), None);
    }

    #[test]
    fn test_token_display() {
        let token = Token::new("hello", 0);
        assert_eq!(format!("{token}"), "hello");
    }

    #[test]
    fn test_token_stream() {
        let tokens = vec![Token::new("hello", 0), Token::new("world", 1)];

        let stream = tokens.into_token_stream();
        let collected: Vec<_> = stream.collect();

        assert_eq!(collected.len(), 2);
        assert_eq!(collected[0].text, "hello");
        assert_eq!(collected[1].text, "world");
    }
}
