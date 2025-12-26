//! Identity stemmer implementation.
//!
//! This stemmer performs no stemming at all, returning words unchanged.
//! It's useful when you want to use the StemFilter infrastructure but
//! don't want any actual stemming to occur.
//!
//! # Use Cases
//!
//! - Testing and comparison purposes
//! - Languages or scenarios where stemming is not desired
//! - Placeholder when stem filter is required but not used
//!
//! # Examples
//!
//! ```
//! use sarissa::analysis::token_filter::stem::Stemmer;
//! use sarissa::analysis::token_filter::stem::identity::IdentityStemmer;
//!
//! let stemmer = IdentityStemmer::new();
//!
//! assert_eq!(stemmer.stem("running"), "running");
//! assert_eq!(stemmer.stem("flies"), "flies");
//! ```

use crate::analysis::token_filter::stem::Stemmer;

/// Identity stemmer that returns words unchanged.
#[derive(Debug, Clone, Default)]
pub struct IdentityStemmer;

impl IdentityStemmer {
    pub fn new() -> Self {
        IdentityStemmer
    }
}

impl Stemmer for IdentityStemmer {
    fn stem(&self, word: &str) -> String {
        word.to_string()
    }

    fn name(&self) -> &'static str {
        "identity"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_stemmer() {
        let stemmer = IdentityStemmer::new();

        assert_eq!(stemmer.stem("running"), "running");
        assert_eq!(stemmer.stem("flies"), "flies");
        assert_eq!(stemmer.stem("test"), "test");
    }
}
