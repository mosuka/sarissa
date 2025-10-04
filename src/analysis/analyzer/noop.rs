//! No-op analyzer that performs no analysis.

use crate::analysis::analyzer::Analyzer;
use crate::analysis::token::TokenStream;
use crate::error::Result;

/// An analyzer that doesn't perform any analysis (no-op).
///
/// This is useful for stored-only fields or testing.
#[derive(Clone, Debug, Default)]
pub struct NoOpAnalyzer;

impl NoOpAnalyzer {
    /// Create a new no-op analyzer.
    pub fn new() -> Self {
        NoOpAnalyzer
    }
}

impl Analyzer for NoOpAnalyzer {
    fn analyze(&self, _text: &str) -> Result<TokenStream> {
        Ok(Box::new(std::iter::empty()))
    }

    fn name(&self) -> &'static str {
        "noop"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::token::Token;

    #[test]
    fn test_noop_analyzer() {
        let analyzer = NoOpAnalyzer::new();

        let tokens: Vec<Token> = analyzer.analyze("Hello World").unwrap().collect();

        assert_eq!(tokens.len(), 0);
    }
}
