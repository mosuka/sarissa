//! Pipeline analyzer that combines tokenizers and filters.

use std::sync::Arc;

use crate::analysis::analyzer::Analyzer;
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

    fn as_any(&self) -> &dyn std::any::Any {
        self
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::token::Token;
    use crate::analysis::token_filter::{LowercaseFilter, StopFilter};
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
}
