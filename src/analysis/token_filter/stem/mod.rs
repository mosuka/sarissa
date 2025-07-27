//! Stemming token filter and stemmer implementations.

use super::Filter;
use crate::analysis::token::TokenStream;
use crate::error::Result;

/// Trait for stemming algorithms.
pub trait Stemmer: Send + Sync {
    /// Stem a word to its root form.
    fn stem(&self, word: &str) -> String;

    /// Get the name of this stemmer.
    fn name(&self) -> &'static str;
}

// Stemmer implementations
pub mod identity;
pub mod porter;
pub mod simple;

// Re-export stemmers
pub use identity::IdentityStemmer;
pub use porter::PorterStemmer;
pub use simple::SimpleStemmer;

/// Filter that applies stemming to tokens.
pub struct StemFilter {
    /// The stemmer to use.
    stemmer: Box<dyn Stemmer>,
}

impl std::fmt::Debug for StemFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StemFilter")
            .field("stemmer", &"<stemmer>")
            .finish()
    }
}

impl StemFilter {
    /// Create a new stem filter with the Porter stemmer.
    pub fn new() -> Self {
        StemFilter {
            stemmer: Box::new(PorterStemmer::new()),
        }
    }

    /// Create a stem filter with a custom stemmer.
    pub fn with_stemmer(stemmer: Box<dyn Stemmer>) -> Self {
        StemFilter { stemmer }
    }

    /// Create a stem filter with the simple stemmer.
    pub fn simple() -> Self {
        StemFilter {
            stemmer: Box::new(SimpleStemmer::new()),
        }
    }
}

impl Default for StemFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl Filter for StemFilter {
    fn filter(&self, tokens: TokenStream) -> Result<TokenStream> {
        let filtered_tokens = tokens
            .map(|token| {
                if token.is_stopped() {
                    token
                } else {
                    let stemmed = self.stemmer.stem(&token.text);
                    token.with_text(stemmed)
                }
            })
            .collect::<Vec<_>>();

        Ok(Box::new(filtered_tokens.into_iter()))
    }

    fn name(&self) -> &'static str {
        "stem"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::token::Token;

    #[test]
    fn test_stem_filter() {
        let filter = StemFilter::new();
        let tokens = vec![
            Token::new("running", 0),
            Token::new("flies", 1),
            Token::new("test", 2).stop(),
        ];
        let token_stream = Box::new(tokens.into_iter());

        let result: Vec<Token> = filter.filter(token_stream).unwrap().collect();

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].text, "run");
        assert_eq!(result[1].text, "fli");
        assert_eq!(result[2].text, "test"); // Stopped tokens are not processed
        assert!(result[2].is_stopped());
    }

    #[test]
    fn test_simple_stem_filter() {
        let filter = StemFilter::simple();
        let tokens = vec![Token::new("running", 0), Token::new("beautiful", 1)];
        let token_stream = Box::new(tokens.into_iter());

        let result: Vec<Token> = filter.filter(token_stream).unwrap().collect();

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].text, "runn");
        assert_eq!(result[1].text, "beauti");
    }

    #[test]
    fn test_filter_name() {
        assert_eq!(StemFilter::new().name(), "stem");
    }
}