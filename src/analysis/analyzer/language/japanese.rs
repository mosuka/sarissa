//! Japanese language analyzer implementation.
//!
//! This module provides a specialized analyzer for Japanese text that uses
//! Lindera for morphological analysis and includes Japanese-specific stop words.
//!
//! # Pipeline
//!
//! 1. Lindera tokenizer (UniDic dictionary)
//! 2. Lowercase filter
//! 3. Japanese stop word filter
//!
//! # Examples
//!
//! ```
//! use platypus::analysis::analyzer::analyzer::Analyzer;
//! use platypus::analysis::analyzer::language::japanese::JapaneseAnalyzer;
//!
//! let analyzer = JapaneseAnalyzer::new().unwrap();
//! let tokens: Vec<_> = analyzer.analyze("日本語のテキスト").unwrap().collect();
//!
//! // Properly segmented Japanese tokens
//! assert!(tokens.len() > 0);
//! ```

use std::fmt::Debug;
use std::fmt::Formatter;
use std::sync::Arc;

use crate::analysis::analyzer::analyzer::Analyzer;
use crate::analysis::analyzer::pipeline::PipelineAnalyzer;
use crate::analysis::token::TokenStream;
use crate::analysis::token_filter::lowercase::LowercaseFilter;
use crate::analysis::token_filter::stop::{DEFAULT_JAPANESE_STOP_WORDS_SET, StopFilter};
use crate::analysis::tokenizer::lindera::LinderaTokenizer;
use crate::error::Result;

/// Analyzer optimized for Japanese language text.
///
/// This analyzer uses Lindera for morphological analysis to properly segment
/// Japanese text (which doesn't use spaces between words) and applies
/// Japanese-specific stop word filtering.
///
/// # Components
///
/// - **Tokenizer**: LinderaTokenizer with UniDic dictionary
/// - **Filters**: Lowercase + Japanese stop words (127 common particles/auxiliaries)
///
/// # Examples
///
/// ```
/// use platypus::analysis::analyzer::analyzer::Analyzer;
/// use platypus::analysis::analyzer::language::japanese::JapaneseAnalyzer;
///
/// let analyzer = JapaneseAnalyzer::new().unwrap();
/// let tokens: Vec<_> = analyzer.analyze("日本語の形態素解析").unwrap().collect();
///
/// // Tokens are properly segmented
/// assert!(tokens.len() >= 3);
/// ```
pub struct JapaneseAnalyzer {
    inner: PipelineAnalyzer,
}
impl JapaneseAnalyzer {
    /// Create a new Japanese analyzer with default settings.
    ///
    /// # Returns
    ///
    /// A new `JapaneseAnalyzer` instance configured with:
    /// - LinderaTokenizer (UniDic dictionary)
    /// - LowercaseFilter
    /// - StopFilter with Japanese stop words
    ///
    /// # Errors
    ///
    /// Returns an error if the LinderaTokenizer cannot be initialized
    /// (e.g., dictionary loading fails).
    ///
    /// # Examples
    ///
    /// ```
    /// use platypus::analysis::analyzer::analyzer::Analyzer;
    /// use platypus::analysis::analyzer::language::japanese::JapaneseAnalyzer;
    ///
    /// let analyzer = JapaneseAnalyzer::new().unwrap();
    /// assert_eq!(analyzer.name(), "japanese");
    /// ```
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
