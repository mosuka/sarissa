//! Japanese language analyzer implementation.
//!
//! This module provides a specialized analyzer for Japanese text that uses
//! Lindera for morphological analysis and includes Japanese-specific stop words.
//!
//! # Pipeline
//!
//! 1. UnicodeNormalizationCharFilter (NFKC normalization)
//! 2. JapaneseIterationMarkCharFilter (iteration mark normalization)
//! 3. LinderaTokenizer (IPADIC dictionary)
//! 4. LowercaseFilter
//! 5. StopFilter (Japanese stop words — 127 common particles/auxiliaries)
//!
//! # Examples
//!
//! ```ignore
//! use laurus::analysis::analyzer::analyzer::Analyzer;
//! use laurus::analysis::analyzer::language::japanese::JapaneseAnalyzer;
//!
//! let analyzer = JapaneseAnalyzer::new(
//!     "normal",
//!     "/var/lib/lindera/ipadic",
//!     None,
//! ).unwrap();
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
use crate::analysis::char_filter::japanese_iteration_mark::JapaneseIterationMarkCharFilter;
use crate::analysis::char_filter::unicode_normalize::NormalizationForm;
use crate::analysis::char_filter::unicode_normalize::UnicodeNormalizationCharFilter;
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
/// - **Char filters**: UnicodeNormalizationCharFilter (NFKC) + JapaneseIterationMarkCharFilter
/// - **Tokenizer**: LinderaTokenizer (caller-provided dictionary path)
/// - **Token filters**: LowercaseFilter + StopFilter (Japanese stop words — 127 common particles/auxiliaries)
///
/// # Examples
///
/// ```
/// use laurus::analysis::analyzer::analyzer::Analyzer;
/// use laurus::analysis::analyzer::language::japanese::JapaneseAnalyzer;
///
/// // In tests the embedded ipadic is available via the dev-dependency feature.
/// let analyzer = JapaneseAnalyzer::new("normal", "embedded://ipadic", None).unwrap();
/// let tokens: Vec<_> = analyzer.analyze("日本語の形態素解析").unwrap().collect();
///
/// // Tokens are properly segmented
/// assert!(tokens.len() >= 3);
/// ```
pub struct JapaneseAnalyzer {
    inner: PipelineAnalyzer,
}
impl JapaneseAnalyzer {
    /// Create a new Japanese analyzer with the given Lindera configuration.
    ///
    /// # Arguments
    ///
    /// * `mode_str` - Lindera segmentation mode: `"normal"`, `"search"`, or
    ///   `"decompose"`.
    /// * `dict_uri` - Lindera dictionary URI. In production builds, supply
    ///   a filesystem path to a Lindera dictionary directory (typically
    ///   IPADIC). `embedded://*` URIs only resolve when the matching
    ///   `embed-*` Lindera feature is enabled, which `laurus` does not
    ///   enable by default.
    /// * `user_dict_uri` - Optional user dictionary path.
    ///
    /// # Returns
    ///
    /// A new `JapaneseAnalyzer` instance configured with:
    /// - UnicodeNormalizationCharFilter (NFKC)
    /// - JapaneseIterationMarkCharFilter
    /// - LinderaTokenizer (caller-provided dictionary)
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
    /// use laurus::analysis::analyzer::analyzer::Analyzer;
    /// use laurus::analysis::analyzer::language::japanese::JapaneseAnalyzer;
    ///
    /// let analyzer = JapaneseAnalyzer::new("normal", "embedded://ipadic", None).unwrap();
    /// assert_eq!(analyzer.name(), "japanese");
    /// ```
    pub fn new(mode_str: &str, dict_uri: &str, user_dict_uri: Option<&str>) -> Result<Self> {
        let tokenizer = Arc::new(LinderaTokenizer::new(mode_str, dict_uri, user_dict_uri)?);
        let analyzer = PipelineAnalyzer::new(tokenizer)
            .add_char_filter(Arc::new(UnicodeNormalizationCharFilter::new(
                NormalizationForm::NFKC,
            )))
            .add_char_filter(Arc::new(JapaneseIterationMarkCharFilter::new(true, true)))
            .add_filter(Arc::new(LowercaseFilter::new()))
            .add_filter(Arc::new(StopFilter::with_stop_words(
                DEFAULT_JAPANESE_STOP_WORDS_SET.clone(),
            )))
            .with_name("japanese".to_string());

        Ok(Self { inner: analyzer })
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
    fn test_japanese_analyzer_segmentation() {
        let analyzer = JapaneseAnalyzer::new("normal", "embedded://ipadic", None).unwrap();

        let text = "日本語の形態素解析を行うことができます。";

        let tokens: Vec<Token> = analyzer.analyze(text).unwrap().collect();

        assert_eq!(tokens.len(), 5);
        assert_eq!(tokens[0].text, "日本語");
        assert_eq!(tokens[1].text, "形態素");
        assert_eq!(tokens[2].text, "解析");
        assert_eq!(tokens[3].text, "行う");
        assert_eq!(tokens[4].text, "。");
    }

    #[test]
    fn test_japanese_analyzer_name() {
        let analyzer = JapaneseAnalyzer::new("normal", "embedded://ipadic", None).unwrap();

        assert_eq!(analyzer.name(), "japanese");
    }
}
