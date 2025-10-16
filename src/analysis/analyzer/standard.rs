//! Standard analyzer that provides good defaults for most use cases.

use std::sync::Arc;

use crate::analysis::analyzer::analyzer::Analyzer;
use crate::analysis::analyzer::pipeline::PipelineAnalyzer;
use crate::analysis::token::TokenStream;
use crate::analysis::token_filter::lowercase::LowercaseFilter;
use crate::analysis::token_filter::stop::StopFilter;
use crate::analysis::tokenizer::regex::RegexTokenizer;
use crate::error::Result;

/// A standard analyzer that provides good defaults for most use cases.
///
/// This analyzer uses a regex tokenizer with lowercase and stop word filtering.
pub struct StandardAnalyzer {
    inner: PipelineAnalyzer,
}

impl StandardAnalyzer {
    /// Create a new standard analyzer with default settings.
    pub fn new() -> Result<Self> {
        let tokenizer = Arc::new(RegexTokenizer::new()?);
        let analyzer = PipelineAnalyzer::new(tokenizer)
            .add_filter(Arc::new(LowercaseFilter::new()))
            .add_filter(Arc::new(StopFilter::new()))
            .with_name("standard".to_string());

        Ok(StandardAnalyzer { inner: analyzer })
    }

    /// Create a new standard analyzer without stop word filtering.
    pub fn without_stop_words() -> Result<Self> {
        let tokenizer = Arc::new(RegexTokenizer::new()?);
        let analyzer = PipelineAnalyzer::new(tokenizer)
            .add_filter(Arc::new(LowercaseFilter::new()))
            .with_name("standard_no_stop".to_string());

        Ok(StandardAnalyzer { inner: analyzer })
    }

    /// Get the inner pipeline analyzer.
    pub fn inner(&self) -> &PipelineAnalyzer {
        &self.inner
    }
}

impl Default for StandardAnalyzer {
    fn default() -> Self {
        Self::new().expect("Standard analyzer should be creatable with default settings")
    }
}

impl Analyzer for StandardAnalyzer {
    fn analyze(&self, text: &str) -> Result<TokenStream> {
        self.inner.analyze(text)
    }

    fn name(&self) -> &'static str {
        "standard"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl std::fmt::Debug for StandardAnalyzer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StandardAnalyzer")
            .field("inner", &self.inner)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::token::Token;

    #[test]
    fn test_standard_analyzer() {
        let analyzer = StandardAnalyzer::new().unwrap();

        let tokens: Vec<Token> = analyzer
            .analyze("Hello the world and test")
            .unwrap()
            .collect();

        // "the" and "and" should be filtered out
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].text, "hello");
        assert_eq!(tokens[1].text, "world");
        assert_eq!(tokens[2].text, "test");
    }

    #[test]
    fn test_standard_analyzer_without_stop_words() {
        let analyzer = StandardAnalyzer::without_stop_words().unwrap();

        let tokens: Vec<Token> = analyzer.analyze("Hello the World").unwrap().collect();

        // "the" should not be filtered out
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].text, "hello");
        assert_eq!(tokens[1].text, "the");
        assert_eq!(tokens[2].text, "world");
    }
}
