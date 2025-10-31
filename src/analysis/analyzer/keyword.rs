//! Keyword analyzer that treats the entire input as a single token.
//!
//! This analyzer uses the WholeTokenizer to treat the entire input text as a
//! single token without any splitting or filtering. It's ideal for fields that
//! should be matched exactly as provided.
//!
//! # Use Cases
//!
//! - ID fields (user IDs, product codes, etc.)
//! - Tag fields where exact matching is required
//! - Category fields
//! - Email addresses or URLs that should be treated atomically
//! - Any field where you want exact string matching
//!
//! # Examples
//!
//! ```
//! use yatagarasu::analysis::analyzer::Analyzer;
//! use yatagarasu::analysis::analyzer::keyword::KeywordAnalyzer;
//!
//! let analyzer = KeywordAnalyzer::new();
//! let tokens: Vec<_> = analyzer.analyze("user-123-abc").unwrap().collect();
//!
//! // Entire input is a single token
//! assert_eq!(tokens.len(), 1);
//! assert_eq!(tokens[0].text, "user-123-abc");
//! ```

use std::sync::Arc;

use crate::analysis::analyzer::analyzer::Analyzer;
use crate::analysis::analyzer::simple::SimpleAnalyzer;
use crate::analysis::token::TokenStream;
use crate::analysis::tokenizer::whole::WholeTokenizer;
use crate::error::Result;

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

    fn as_any(&self) -> &dyn std::any::Any {
        self
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
