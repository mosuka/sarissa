use std::fmt::Debug;
use std::fmt::Formatter;
use std::sync::Arc;

use crate::analysis::LinderaTokenizer;
use crate::analysis::analyzer::Analyzer;
use crate::analysis::stop::DEFAULT_JAPANESE_STOP_WORDS_SET;
use crate::analysis::token::TokenStream;
use crate::analysis::{LowercaseFilter, PipelineAnalyzer, StopFilter};
use crate::error::Result;

pub struct JapaneseAnalyzer {
    inner: PipelineAnalyzer,
}
impl JapaneseAnalyzer {
    /// Create a new Japanese analyzer with default settings.
    pub fn new() -> Result<Self> {
        let tokenizer = Arc::new(LinderaTokenizer::new("normal", "embedded://unidic", None)?);
        let analyzer = PipelineAnalyzer::new(tokenizer)
            .add_filter(Arc::new(LowercaseFilter::new()))
            .add_filter(Arc::new(StopFilter::with_stop_words(
                DEFAULT_JAPANESE_STOP_WORDS_SET.clone(),
            )))
            .with_name("japanese".to_string());

        Ok(Self { inner: analyzer })
    }
}

impl Default for JapaneseAnalyzer {
    fn default() -> Self {
        Self::new().expect("Japanese analyzer should be creatable with default settings")
    }
}

impl Analyzer for JapaneseAnalyzer {
    fn analyze(&self, text: &str) -> Result<TokenStream> {
        self.inner.analyze(text)
    }

    fn name(&self) -> &'static str {
        "japanese"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Debug for JapaneseAnalyzer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JapaneseAnalyzer")
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
        let analyzer = JapaneseAnalyzer::new().unwrap();

        let text = "日本語の形態素解析を行うことができます。";

        let tokens: Vec<Token> = analyzer.analyze(text).unwrap().collect();

        assert_eq!(tokens.len(), 7);
        assert_eq!(tokens[0].text, "日本");
        assert_eq!(tokens[1].text, "語");
        assert_eq!(tokens[2].text, "形態");
        assert_eq!(tokens[3].text, "素");
        assert_eq!(tokens[4].text, "解析");
        assert_eq!(tokens[5].text, "行う");
        assert_eq!(tokens[6].text, "。");
    }

    #[test]
    fn test_japanese_analyzer_name() {
        let analyzer = JapaneseAnalyzer::new().unwrap();

        assert_eq!(analyzer.name(), "japanese");
    }
}
