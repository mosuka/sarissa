//! No-op analyzer that performs no analysis.
//!
//! This analyzer returns an empty token stream for any input, effectively
//! performing no analysis at all. It's useful for stored-only fields that
//! don't need to be indexed or searched.
//!
//! # Use Cases
//!
//! - Stored-only fields that should not be indexed
//! - Fields that are only used for display purposes
//! - Testing scenarios where you need an analyzer that does nothing
//! - Placeholder in configurations where an analyzer is required but not used
//!
//! # Examples
//!
//! ```
//! use yatagarasu::analysis::analyzer::analyzer::Analyzer;
//! use yatagarasu::analysis::analyzer::noop::NoOpAnalyzer;
//!
//! let analyzer = NoOpAnalyzer::new();
//! let tokens: Vec<_> = analyzer.analyze("any text here").unwrap().collect();
//!
//! // Always returns empty token stream
//! assert_eq!(tokens.len(), 0);
//! ```

use crate::analysis::analyzer::analyzer::Analyzer;
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
