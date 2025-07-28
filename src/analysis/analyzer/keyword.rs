//! Keyword analyzer that treats the entire input as a single token.

use crate::analysis::analyzer::{Analyzer, SimpleAnalyzer};
use crate::analysis::token::TokenStream;
use crate::analysis::tokenizer::WholeTokenizer;
use crate::error::Result;
use std::sync::Arc;

/// A keyword analyzer that treats the entire input as a single token.
///
/// This is useful for ID fields or other cases where you don't want to split the text.
pub struct KeywordAnalyzer {
    inner: SimpleAnalyzer,
}

impl KeywordAnalyzer {
    /// Create a new keyword analyzer.
    pub fn new() -> Self {
        let tokenizer = Arc::new(WholeTokenizer::new());
        let analyzer = SimpleAnalyzer::new(tokenizer);

        KeywordAnalyzer { inner: analyzer }
    }

    /// Get the inner simple analyzer.
    pub fn inner(&self) -> &SimpleAnalyzer {
        &self.inner
    }
}

impl Default for KeywordAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl Analyzer for KeywordAnalyzer {
    fn analyze(&self, text: &str) -> Result<TokenStream> {
        self.inner.analyze(text)
    }

    fn name(&self) -> &'static str {
        "keyword"
    }
}

impl std::fmt::Debug for KeywordAnalyzer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KeywordAnalyzer")
            .field("inner", &self.inner)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::token::Token;

    #[test]
    fn test_keyword_analyzer() {
        let analyzer = KeywordAnalyzer::new();

        let tokens: Vec<Token> = analyzer.analyze("Hello World Test").unwrap().collect();

        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].text, "Hello World Test");
    }
}