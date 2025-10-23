//! Identity stemmer implementation.

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
