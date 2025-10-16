use std::fmt::Debug;
use std::fmt::Formatter;
use std::sync::Arc;

use crate::analysis::analyzer::analyzer::Analyzer;
use crate::analysis::analyzer::pipeline::PipelineAnalyzer;
use crate::analysis::token::TokenStream;
use crate::analysis::token_filter::lowercase::LowercaseFilter;
use crate::analysis::token_filter::stop::StopFilter;
use crate::analysis::tokenizer::regex::RegexTokenizer;
use crate::error::Result;

pub struct EnglishAnalyzer {
    inner: PipelineAnalyzer,
}

impl EnglishAnalyzer {
    pub fn new() -> Result<Self> {
        let tokenizer = Arc::new(RegexTokenizer::new()?);
        let analyzer = PipelineAnalyzer::new(tokenizer)
            .add_filter(Arc::new(LowercaseFilter::new()))
            .add_filter(Arc::new(StopFilter::default()))
            .with_name("english".to_string());

        Ok(Self { inner: analyzer })
    }
}

impl Default for EnglishAnalyzer {
    fn default() -> Self {
        Self::new().expect("English analyzer should be creatable with default settings")
    }
}

impl Analyzer for EnglishAnalyzer {
    fn analyze(&self, text: &str) -> Result<TokenStream> {
        self.inner.analyze(text)
    }

    fn name(&self) -> &'static str {
        "english"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Debug for EnglishAnalyzer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EnglishAnalyzer")
            .field("inner", &self.inner)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::token::Token;

    #[test]
    fn test_english_analyzer() {
        let analyzer = EnglishAnalyzer::new().unwrap();

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
    fn test_english_analyzer_name() {
        let analyzer = EnglishAnalyzer::new().unwrap();

        assert_eq!(analyzer.name(), "english");
    }
}
