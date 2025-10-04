//! Simple analyzer that performs tokenization without filtering.

use std::sync::Arc;

use crate::analysis::analyzer::Analyzer;
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
    use crate::analysis::tokenizer::RegexTokenizer;

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
