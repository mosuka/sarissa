//! Per-field analyzer (Lucene-compatible).

use crate::analysis::TokenStream;
use crate::analysis::analyzer::Analyzer;
use crate::error::Result;
use ahash::AHashMap;
use std::sync::Arc;

/// A per-field analyzer that applies different analyzers to different fields.
///
/// This is similar to Lucene's PerFieldAnalyzerWrapper. It allows you to specify
/// a different analyzer for each field, with a default analyzer for fields not
/// explicitly configured.
///
/// # Memory Efficiency
///
/// When using the same analyzer for multiple fields, reuse a single instance
/// with `Arc::clone` to save memory. This is especially important for analyzers
/// with large dictionaries (e.g., Lindera for Japanese).
///
/// # Example
///
/// ```
/// use sage::analysis::{Analyzer, PerFieldAnalyzer, StandardAnalyzer, KeywordAnalyzer};
/// use std::sync::Arc;
///
/// // Reuse analyzer instances to save memory
/// let keyword_analyzer: Arc<dyn Analyzer> = Arc::new(KeywordAnalyzer::new());
/// let mut analyzer = PerFieldAnalyzer::new(Arc::new(StandardAnalyzer::new().unwrap()));
/// analyzer.add_analyzer("id", Arc::clone(&keyword_analyzer));
/// analyzer.add_analyzer("category", Arc::clone(&keyword_analyzer));
/// // "title" and "body" will use StandardAnalyzer
/// // "id" and "category" will use the same KeywordAnalyzer instance
/// ```
#[derive(Clone)]
pub struct PerFieldAnalyzer {
    /// Default analyzer for fields not in the map.
    default_analyzer: Arc<dyn Analyzer>,

    /// Map of field names to their specific analyzers.
    field_analyzers: AHashMap<String, Arc<dyn Analyzer>>,
}

impl PerFieldAnalyzer {
    /// Create a new per-field analyzer with a default analyzer.
    pub fn new(default_analyzer: Arc<dyn Analyzer>) -> Self {
        Self {
            default_analyzer,
            field_analyzers: AHashMap::new(),
        }
    }

    /// Add a field-specific analyzer.
    pub fn add_analyzer(&mut self, field: impl Into<String>, analyzer: Arc<dyn Analyzer>) {
        self.field_analyzers.insert(field.into(), analyzer);
    }

    /// Get the analyzer for a specific field.
    pub fn get_analyzer(&self, field: &str) -> &Arc<dyn Analyzer> {
        self.field_analyzers
            .get(field)
            .unwrap_or(&self.default_analyzer)
    }

    /// Get the default analyzer.
    pub fn default_analyzer(&self) -> &Arc<dyn Analyzer> {
        &self.default_analyzer
    }

    /// Analyze text with the analyzer for the given field.
    pub fn analyze_field(&self, field: &str, text: &str) -> Result<TokenStream> {
        self.get_analyzer(field).analyze(text)
    }
}

impl Analyzer for PerFieldAnalyzer {
    fn analyze(&self, text: &str) -> Result<TokenStream> {
        // When used as a regular Analyzer, use the default analyzer
        self.default_analyzer.analyze(text)
    }

    fn name(&self) -> &'static str {
        "PerFieldAnalyzer"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::{KeywordAnalyzer, StandardAnalyzer};

    #[test]
    fn test_per_field_analyzer() {
        let mut analyzer = PerFieldAnalyzer::new(Arc::new(StandardAnalyzer::new().unwrap()));
        analyzer.add_analyzer("id", Arc::new(KeywordAnalyzer::new()));
        analyzer.add_analyzer("category", Arc::new(KeywordAnalyzer::new()));

        // Test that different fields use different analyzers
        let text = "Hello World";

        // Default analyzer (StandardAnalyzer) lowercases and tokenizes
        let tokens: Vec<_> = analyzer.analyze_field("title", text).unwrap().collect();
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "hello");
        assert_eq!(tokens[1].text, "world");

        // KeywordAnalyzer keeps as single token (not lowercased by default)
        let tokens: Vec<_> = analyzer.analyze_field("id", text).unwrap().collect();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].text, "Hello World");

        // Another field with KeywordAnalyzer
        let tokens: Vec<_> = analyzer.analyze_field("category", text).unwrap().collect();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].text, "Hello World");
    }

    #[test]
    fn test_default_analyzer_when_field_not_configured() {
        let analyzer = PerFieldAnalyzer::new(Arc::new(StandardAnalyzer::new().unwrap()));

        let text = "Hello World";
        let tokens: Vec<_> = analyzer
            .analyze_field("unknown_field", text)
            .unwrap()
            .collect();

        // Should use default StandardAnalyzer
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "hello");
        assert_eq!(tokens[1].text, "world");
    }

    #[test]
    fn test_as_analyzer_trait() {
        let mut analyzer = PerFieldAnalyzer::new(Arc::new(StandardAnalyzer::new().unwrap()));
        analyzer.add_analyzer("id", Arc::new(KeywordAnalyzer::new()));

        // When used as Analyzer trait, should use default analyzer
        let text = "Hello World";
        let tokens: Vec<_> = analyzer.analyze(text).unwrap().collect();

        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "hello");
        assert_eq!(tokens[1].text, "world");
    }
}
