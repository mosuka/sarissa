//! Pipeline analyzer that combines tokenizers and filters.
//!
//! This is the main building block for custom analyzers. It allows you to
//! combine a tokenizer with any number of token filters to create a custom
//! analysis pipeline.
//!
//! # Architecture
//!
//! The PipelineAnalyzer applies processing in this order:
//! 1. Char Filters: Normalizes raw text
//! 2. Tokenizer: Splits text into tokens
//! 3. Token Filters: Applied sequentially in the order they were added
//!
//! # Examples
//!
//! ```
//! use sarissa::analysis::analyzer::analyzer::Analyzer;
//! use sarissa::analysis::analyzer::pipeline::PipelineAnalyzer;
//! use sarissa::analysis::tokenizer::regex::RegexTokenizer;
//! use sarissa::analysis::token_filter::lowercase::LowercaseFilter;
//! use sarissa::analysis::token_filter::stop::StopFilter;
//! use std::sync::Arc;
//!
//! // Create a custom analyzer with tokenizer + filters
//! let tokenizer = Arc::new(RegexTokenizer::new().unwrap());
//! let analyzer = PipelineAnalyzer::new(tokenizer)
//!     .add_filter(Arc::new(LowercaseFilter::new()))
//!     .add_filter(Arc::new(StopFilter::from_words(vec!["the", "and"])))
//!     .with_name("my_custom_analyzer".to_string());
//!
//! let tokens: Vec<_> = analyzer.analyze("Hello THE world AND test").unwrap().collect();
//!
//! assert_eq!(tokens.len(), 3);
//! assert_eq!(tokens[0].text, "hello");
//! assert_eq!(tokens[1].text, "world");
//! assert_eq!(tokens[2].text, "test");
//! ```

use std::sync::Arc;

use crate::analysis::analyzer::analyzer::Analyzer;
use crate::analysis::char_filter::CharFilter;
use crate::analysis::token::TokenStream;
use crate::analysis::token_filter::Filter;
use crate::analysis::tokenizer::Tokenizer;
use crate::error::Result;

/// A configurable analyzer that combines a tokenizer with a chain of filters.
///
/// This is the main analyzer type that allows building analysis pipelines
/// by combining different tokenizers and filters.
#[derive(Clone)]
pub struct PipelineAnalyzer {
    tokenizer: Arc<dyn Tokenizer>,
    char_filters: Vec<Arc<dyn CharFilter>>,
    filters: Vec<Arc<dyn Filter>>,
    name: String,
}

impl PipelineAnalyzer {
    /// Create a new pipeline analyzer with the given tokenizer.
    pub fn new(tokenizer: Arc<dyn Tokenizer>) -> Self {
        PipelineAnalyzer {
            name: format!("pipeline_{}", tokenizer.name()),
            tokenizer,
            char_filters: Vec::new(),
            filters: Vec::new(),
        }
    }

    /// Add a char filter to the pipeline.
    pub fn add_char_filter(mut self, char_filter: Arc<dyn CharFilter>) -> Self {
        self.char_filters.push(char_filter);
        self
    }

    /// Add a filter to the pipeline.
    pub fn add_filter(mut self, filter: Arc<dyn Filter>) -> Self {
        self.filters.push(filter);
        self
    }

    /// Set a custom name for this analyzer.
    pub fn with_name<S: Into<String>>(mut self, name: S) -> Self {
        self.name = name.into();
        self
    }

    /// Get the tokenizer used by this analyzer.
    pub fn tokenizer(&self) -> &Arc<dyn Tokenizer> {
        &self.tokenizer
    }

    /// Get the char filters used by this analyzer.
    pub fn char_filters(&self) -> &[Arc<dyn CharFilter>] {
        &self.char_filters
    }

    /// Get the filters used by this analyzer.
    pub fn filters(&self) -> &[Arc<dyn Filter>] {
        &self.filters
    }
}

impl Analyzer for PipelineAnalyzer {
    fn analyze(&self, text: &str) -> Result<TokenStream> {
        // Apply char filters
        let mut filtered_text = text.to_string();
        let mut filter_transformations = Vec::with_capacity(self.char_filters.len());

        for char_filter in &self.char_filters {
            let (new_text, transformations) = char_filter.filter(&filtered_text);
            filtered_text = new_text;
            filter_transformations.push(transformations);
        }

        // Start with tokenization
        let mut tokens = self.tokenizer.tokenize(&filtered_text)?;

        // Apply filters in sequence
        for filter in &self.filters {
            tokens = filter.filter(tokens)?;
        }

        // If we have char filters, we need to correct offsets
        if !self.char_filters.is_empty() {
            let collected: Vec<_> = tokens
                .map(|mut token| {
                    // Correct offsets by applying char filters in reverse order
                    // We map from Final -> Filter N -> ... -> Filter 1 -> Original
                    for transformations in filter_transformations.iter().rev() {
                        token.start_offset =
                            Self::correct_offset(token.start_offset, transformations);
                        token.end_offset = Self::correct_offset(token.end_offset, transformations);
                    }
                    token
                })
                .collect();
            return Ok(Box::new(collected.into_iter()));
        }

        Ok(tokens)
    }

    fn name(&self) -> &'static str {
        // We can't return a reference to self.name because it's not static
        // Instead, we'll use a default name
        "pipeline"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl PipelineAnalyzer {
    /// Maps an offset in the filtered text back to the original text using transformations.
    fn correct_offset(
        offset: usize,
        transformations: &[crate::analysis::char_filter::Transformation],
    ) -> usize {
        let mut corrected = offset;
        // Transformations are ordered by position
        for t in transformations {
            if offset >= t.new_end {
                // The point is after this transformation.
                // We need to adjust for the length difference caused by this transformation.
                let original_len = t.original_end - t.original_start;
                let new_len = t.new_end - t.new_start;

                // If original was longer (orig: 5, new: 3), we added 2 to get to original.
                // If original was shorter (orig: 3, new: 5), we subtract 2.
                // corrected = corrected - new_len + original_len
                // Note: using isize to avoid underflow during calculation, though final result must be usize
                corrected =
                    (corrected as isize - new_len as isize + original_len as isize) as usize;
            } else if offset >= t.new_start {
                // The point falls STRICTLY inside distinct transformation (new_start <= offset < new_end).
                // Or if offset == new_end? Captured by first branch if >=.
                // So this branch is new_start <= offset < new_end.

                // We map relative position.
                let offset_in_new = offset - t.new_start;
                let new_len = t.new_end - t.new_start;
                let original_len = t.original_end - t.original_start;

                if new_len == 0 {
                    // Inserted text has 0 length? No, new_len=0 means deletion in original produced nothing?
                    // Then offset cannot be "inside" (start=end).
                    // So new_len > 0.
                    return t.original_start;
                }

                // Linear interpolation: original_start + (offset_in_new * original_len / new_len)
                // This is an approximation.
                let offset_in_original = (offset_in_new * original_len) / new_len;
                return t.original_start + offset_in_original;
            }
            // If offset < t.new_start, this transformation hasn't happened yet (relative to this point), so it doesn't affect the offset.
        }
        corrected
    }
}

impl std::fmt::Debug for PipelineAnalyzer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PipelineAnalyzer")
            .field("name", &self.name)
            .field("tokenizer", &self.tokenizer.name())
            .field(
                "char_filters",
                &self
                    .char_filters
                    .iter()
                    .map(|f| f.name())
                    .collect::<Vec<_>>(),
            )
            .field(
                "filters",
                &self.filters.iter().map(|f| f.name()).collect::<Vec<_>>(),
            )
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::char_filter::pattern_replace::PatternReplaceCharFilter;
    use crate::analysis::char_filter::unicode_normalize::{
        NormalizationForm, UnicodeNormalizationCharFilter,
    };
    use crate::analysis::token::Token;
    use crate::analysis::token_filter::lowercase::LowercaseFilter;
    use crate::analysis::token_filter::stop::StopFilter;
    use crate::analysis::tokenizer::regex::RegexTokenizer;
    use crate::analysis::tokenizer::whitespace::WhitespaceTokenizer;

    #[test]
    fn test_pipeline_analyzer() {
        let tokenizer = Arc::new(RegexTokenizer::new().unwrap());
        let analyzer = PipelineAnalyzer::new(tokenizer)
            .add_filter(Arc::new(LowercaseFilter::new()))
            .add_filter(Arc::new(StopFilter::from_words(vec!["the", "and"])));

        let tokens: Vec<Token> = analyzer
            .analyze("Hello THE world AND test")
            .unwrap()
            .collect();

        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].text, "hello");
        assert_eq!(tokens[1].text, "world");
        assert_eq!(tokens[2].text, "test");
    }

    #[test]
    fn test_pipeline_with_char_filter() {
        let tokenizer = Arc::new(WhitespaceTokenizer::new());
        let analyzer = PipelineAnalyzer::new(tokenizer)
            .add_char_filter(Arc::new(UnicodeNormalizationCharFilter::new(
                NormalizationForm::NFKC,
            )))
            .add_filter(Arc::new(LowercaseFilter::new()));

        // U+FF21 is Fullwidth Latin Capital Letter A
        let tokens: Vec<Token> = analyzer.analyze("\u{ff21}BC DEF").unwrap().collect();

        assert_eq!(tokens.len(), 2);
        // Should be normalized to ASCII "A" then lowercased to "a" -> "abc"
        assert_eq!(tokens[0].text, "abc");
        assert_eq!(tokens[1].text, "def");
    }

    #[test]
    fn test_pipeline_with_pattern_replace() {
        let tokenizer = Arc::new(WhitespaceTokenizer::new());
        let analyzer = PipelineAnalyzer::new(tokenizer)
            .add_char_filter(Arc::new(PatternReplaceCharFilter::new(r"-", "").unwrap()));

        let tokens: Vec<Token> = analyzer.analyze("123-456 789").unwrap().collect();

        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "123456");
        assert_eq!(tokens[1].text, "789");
    }

    #[test]
    fn test_offset_correction_normalization() {
        let tokenizer = Arc::new(WhitespaceTokenizer::new());
        let analyzer = PipelineAnalyzer::new(tokenizer).add_char_filter(Arc::new(
            UnicodeNormalizationCharFilter::new(NormalizationForm::NFKC),
        ));

        // "㌂" (U+3302, 3 bytes) -> "アンペア" (12 bytes)
        // Offset in original: 0..3
        // Offset in filtered: 0..12
        // Corrected offset should be 0..3
        let tokens: Vec<Token> = analyzer.analyze("㌂").unwrap().collect();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].text, "アンペア");
        assert_eq!(tokens[0].start_offset, 0);
        assert_eq!(tokens[0].end_offset, 3);
    }

    #[test]
    fn test_offset_correction_pattern_replace() {
        let tokenizer = Arc::new(WhitespaceTokenizer::new());
        let analyzer = PipelineAnalyzer::new(tokenizer)
            .add_char_filter(Arc::new(PatternReplaceCharFilter::new(r"-", "").unwrap()));

        // "foo-bar" (7 bytes) -> "foobar" (6 bytes)
        // "-" is removed.
        // "foobar" token.
        // Filtered offset: 0..6
        // Original offset: 0..7
        let tokens: Vec<Token> = analyzer.analyze("foo-bar").unwrap().collect();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].text, "foobar");
        assert_eq!(tokens[0].start_offset, 0);
        assert_eq!(tokens[0].end_offset, 7);
    }
}
