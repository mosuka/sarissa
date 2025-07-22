//! Analyzer implementations that combine tokenizers and filters.

use crate::analysis::filter::Filter;
use crate::analysis::token::TokenStream;
use crate::analysis::tokenizer::Tokenizer;
use crate::error::Result;
use std::sync::Arc;

/// Trait for analyzers that convert text into processed tokens.
pub trait Analyzer: Send + Sync {
    /// Analyze the given text and return a stream of tokens.
    fn analyze(&self, text: &str) -> Result<TokenStream>;

    /// Get the name of this analyzer (for debugging and configuration).
    fn name(&self) -> &'static str;
}

/// A configurable analyzer that combines a tokenizer with a chain of filters.
///
/// This is the main analyzer type that allows building analysis pipelines
/// by combining different tokenizers and filters.
#[derive(Clone)]
pub struct PipelineAnalyzer {
    tokenizer: Arc<dyn Tokenizer>,
    filters: Vec<Arc<dyn Filter>>,
    name: String,
}

impl PipelineAnalyzer {
    /// Create a new pipeline analyzer with the given tokenizer.
    pub fn new(tokenizer: Arc<dyn Tokenizer>) -> Self {
        PipelineAnalyzer {
            name: format!("pipeline_{}", tokenizer.name()),
            tokenizer,
            filters: Vec::new(),
        }
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

    /// Get the filters used by this analyzer.
    pub fn filters(&self) -> &[Arc<dyn Filter>] {
        &self.filters
    }
}

impl Analyzer for PipelineAnalyzer {
    fn analyze(&self, text: &str) -> Result<TokenStream> {
        // Start with tokenization
        let mut tokens = self.tokenizer.tokenize(text)?;

        // Apply filters in sequence
        for filter in &self.filters {
            tokens = filter.filter(tokens)?;
        }

        Ok(tokens)
    }

    fn name(&self) -> &'static str {
        // We can't return a reference to self.name because it's not static
        // Instead, we'll use a default name
        "pipeline"
    }
}

impl std::fmt::Debug for PipelineAnalyzer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PipelineAnalyzer")
            .field("name", &self.name)
            .field("tokenizer", &self.tokenizer.name())
            .field(
                "filters",
                &self.filters.iter().map(|f| f.name()).collect::<Vec<_>>(),
            )
            .finish()
    }
}

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
}

impl std::fmt::Debug for SimpleAnalyzer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SimpleAnalyzer")
            .field("tokenizer", &self.tokenizer.name())
            .finish()
    }
}

/// A standard analyzer that provides good defaults for most use cases.
///
/// This analyzer uses a regex tokenizer with lowercase and stop word filtering.
pub struct StandardAnalyzer {
    inner: PipelineAnalyzer,
}

impl StandardAnalyzer {
    /// Create a new standard analyzer with default settings.
    pub fn new() -> Result<Self> {
        use crate::analysis::filter::{LowercaseFilter, StopFilter};
        use crate::analysis::tokenizer::RegexTokenizer;

        let tokenizer = Arc::new(RegexTokenizer::new()?);
        let analyzer = PipelineAnalyzer::new(tokenizer)
            .add_filter(Arc::new(LowercaseFilter::new()))
            .add_filter(Arc::new(StopFilter::new()))
            .with_name("standard".to_string());

        Ok(StandardAnalyzer { inner: analyzer })
    }

    /// Create a new standard analyzer without stop word filtering.
    pub fn without_stop_words() -> Result<Self> {
        use crate::analysis::filter::LowercaseFilter;
        use crate::analysis::tokenizer::RegexTokenizer;

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
}

impl std::fmt::Debug for StandardAnalyzer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StandardAnalyzer")
            .field("inner", &self.inner)
            .finish()
    }
}

/// A keyword analyzer that treats the entire input as a single token.
///
/// This is useful for ID fields or other cases where you don't want to split the text.
pub struct KeywordAnalyzer {
    inner: SimpleAnalyzer,
}

impl KeywordAnalyzer {
    /// Create a new keyword analyzer.
    pub fn new() -> Self {
        use crate::analysis::tokenizer::WholeTokenizer;

        let tokenizer = Arc::new(WholeTokenizer::new());
        let analyzer = SimpleAnalyzer::new(tokenizer);

        KeywordAnalyzer { inner: analyzer }
    }

    /// Get the inner simple analyzer.
    pub fn inner(&self) -> &SimpleAnalyzer {
        &self.inner
    }
}

impl Default for KeywordAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl Analyzer for KeywordAnalyzer {
    fn analyze(&self, text: &str) -> Result<TokenStream> {
        self.inner.analyze(text)
    }

    fn name(&self) -> &'static str {
        "keyword"
    }
}

impl std::fmt::Debug for KeywordAnalyzer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KeywordAnalyzer")
            .field("inner", &self.inner)
            .finish()
    }
}

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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::filter::{LowercaseFilter, StopFilter};
    use crate::analysis::token::Token;
    use crate::analysis::tokenizer::RegexTokenizer;

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
    fn test_simple_analyzer() {
        let tokenizer = Arc::new(RegexTokenizer::new().unwrap());
        let analyzer = SimpleAnalyzer::new(tokenizer);

        let tokens: Vec<Token> = analyzer.analyze("Hello World").unwrap().collect();

        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "Hello");
        assert_eq!(tokens[1].text, "World");
    }

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

    #[test]
    fn test_keyword_analyzer() {
        let analyzer = KeywordAnalyzer::new();

        let tokens: Vec<Token> = analyzer.analyze("Hello World Test").unwrap().collect();

        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].text, "Hello World Test");
    }

    #[test]
    fn test_noop_analyzer() {
        let analyzer = NoOpAnalyzer::new();

        let tokens: Vec<Token> = analyzer.analyze("Hello World").unwrap().collect();

        assert_eq!(tokens.len(), 0);
    }

    #[test]
    fn test_analyzer_names() {
        let tokenizer = Arc::new(RegexTokenizer::new().unwrap());
        let pipeline = PipelineAnalyzer::new(tokenizer.clone());
        let simple = SimpleAnalyzer::new(tokenizer);
        let standard = StandardAnalyzer::new().unwrap();
        let keyword = KeywordAnalyzer::new();
        let noop = NoOpAnalyzer::new();

        assert_eq!(pipeline.name(), "pipeline");
        assert_eq!(simple.name(), "simple");
        assert_eq!(standard.name(), "standard");
        assert_eq!(keyword.name(), "keyword");
        assert_eq!(noop.name(), "noop");
    }
}
