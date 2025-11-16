//! Simple analyzer that performs tokenization without filtering.
//!
//! This analyzer applies only tokenization without any token filtering.
//! It's useful when you want complete control over the tokenization process
//! or when you need to preserve all tokens without any modifications.
//!
//! # Use Cases
//!
//! - Custom analysis pipelines where you want to apply filters manually
//! - Testing and debugging tokenizers
//! - Cases where no filtering is desired (e.g., exact matching scenarios)
//!
//! # Examples
//!
//! ```
//! use platypus::analysis::analyzer::analyzer::Analyzer;
//! use platypus::analysis::analyzer::simple::SimpleAnalyzer;
//! use platypus::analysis::tokenizer::regex::RegexTokenizer;
//! use std::sync::Arc;
//!
//! let tokenizer = Arc::new(RegexTokenizer::new().unwrap());
//! let analyzer = SimpleAnalyzer::new(tokenizer);
//!
//! let tokens: Vec<_> = analyzer.analyze("Hello World").unwrap().collect();
//!
//! // No filtering applied - original case preserved
//! assert_eq!(tokens.len(), 2);
//! assert_eq!(tokens[0].text, "Hello");
//! assert_eq!(tokens[1].text, "World");
//! ```

use std::sync::Arc;

use crate::analysis::analyzer::analyzer::Analyzer;
use crate::analysis::token::TokenStream;
use crate::analysis::tokenizer::Tokenizer;
use crate::error::Result;

/// A simple analyzer that just tokenizes without any filtering.
#[derive(Clone)]
pub struct SimpleAnalyzer {
    tokenizer: Arc<dyn Tokenizer>,
}

impl SimpleAnalyzer {
    /// Create a new simple analyzer with the given tokenizer.
    pub fn new(tokenizer: Arc<dyn Tokenizer>) -> Self {
        SimpleAnalyzer { tokenizer }
    }

    /// Get the tokenizer used by this analyzer.
    pub fn tokenizer(&self) -> &Arc<dyn Tokenizer> {
        &self.tokenizer
    }
}

impl Analyzer for SimpleAnalyzer {
    fn analyze(&self, text: &str) -> Result<TokenStream> {
        self.tokenizer.tokenize(text)
    }

    fn name(&self) -> &'static str {
        "simple"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl std::fmt::Debug for SimpleAnalyzer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SimpleAnalyzer")
            .field("tokenizer", &self.tokenizer.name())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::token::Token;
    use crate::analysis::tokenizer::regex::RegexTokenizer;

    #[test]
    fn test_simple_analyzer() {
        let tokenizer = Arc::new(RegexTokenizer::new().unwrap());
        let analyzer = SimpleAnalyzer::new(tokenizer);

        let tokens: Vec<Token> = analyzer.analyze("Hello World").unwrap().collect();

        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "Hello");
        assert_eq!(tokens[1].text, "World");
    }
}
