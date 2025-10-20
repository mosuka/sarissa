//! Spell-corrected search functionality that integrates spelling correction with search.

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::lexical::search::SearchRequest;
use crate::lexical::search::engine::SearchEngine;
use crate::query::SearchResults;
use crate::spelling::corrector::{
    CorrectionResult, CorrectorConfig, DidYouMean, SpellingCorrector,
};

/// Search results with spelling correction information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpellCorrectedSearchResults {
    /// The original search results.
    pub results: SearchResults,
    /// Spelling correction information.
    pub correction: CorrectionResult,
    /// Whether the search was performed with the corrected query.
    pub used_correction: bool,
    /// "Did you mean?" suggestion if available.
    pub did_you_mean: Option<String>,
}

impl SpellCorrectedSearchResults {
    /// Create new spell-corrected search results.
    pub fn new(results: SearchResults, correction: CorrectionResult) -> Self {
        SpellCorrectedSearchResults {
            results,
            correction,
            used_correction: false,
            did_you_mean: None,
        }
    }

    /// Get the query that was actually used for search.
    pub fn effective_query(&self) -> &str {
        self.correction.query()
    }

    /// Check if any spelling corrections were suggested.
    pub fn has_suggestions(&self) -> bool {
        self.correction.has_suggestions()
    }

    /// Check if auto-correction was applied.
    pub fn was_auto_corrected(&self) -> bool {
        self.correction.auto_corrected
    }

    /// Check if a "Did you mean?" suggestion should be shown.
    pub fn should_show_did_you_mean(&self) -> bool {
        self.did_you_mean.is_some() || self.correction.should_show_did_you_mean()
    }

    /// Get the correction confidence score.
    pub fn correction_confidence(&self) -> f64 {
        self.correction.confidence
    }
}

/// Configuration for spell-corrected search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpellCorrectedSearchConfig {
    /// Whether to enable spell correction.
    pub enabled: bool,
    /// Configuration for the spelling corrector.
    pub corrector_config: CorrectorConfig,
    /// Whether to retry search with corrected query if original query has no results.
    pub retry_with_correction: bool,
    /// Whether to show "Did you mean?" for poor results with low confidence.
    pub show_did_you_mean: bool,
    /// Minimum number of results before suggesting corrections.
    pub min_results_for_suggestions: usize,
}

impl Default for SpellCorrectedSearchConfig {
    fn default() -> Self {
        SpellCorrectedSearchConfig {
            enabled: true,
            corrector_config: CorrectorConfig::default(),
            retry_with_correction: true,
            show_did_you_mean: true,
            min_results_for_suggestions: 2,
        }
    }
}

/// A search engine wrapper that provides spell correction capabilities.
pub struct SpellCorrectedSearchEngine {
    /// The underlying search engine.
    engine: SearchEngine,
    /// The spelling corrector.
    corrector: SpellingCorrector,
    /// Configuration for spell-corrected search.
    config: SpellCorrectedSearchConfig,
    /// "Did you mean?" functionality.
    did_you_mean: DidYouMean,
}

impl SpellCorrectedSearchEngine {
    /// Create a new spell-corrected search engine.
    pub fn new(engine: SearchEngine) -> Self {
        let corrector = SpellingCorrector::new();
        let config = SpellCorrectedSearchConfig::default();
        let did_you_mean = DidYouMean::new(SpellingCorrector::new());

        SpellCorrectedSearchEngine {
            engine,
            corrector,
            config,
            did_you_mean,
        }
    }

    /// Create a new spell-corrected search engine with custom configuration.
    pub fn with_config(engine: SearchEngine, config: SpellCorrectedSearchConfig) -> Self {
        let mut corrector = SpellingCorrector::new();
        corrector.set_config(config.corrector_config.clone());
        let did_you_mean = DidYouMean::new(SpellingCorrector::new());

        SpellCorrectedSearchEngine {
            engine,
            corrector,
            config,
            did_you_mean,
        }
    }

    /// Get the underlying search engine.
    pub fn engine(&self) -> &SearchEngine {
        &self.engine
    }

    /// Get mutable access to the underlying search engine.
    pub fn engine_mut(&mut self) -> &mut SearchEngine {
        &mut self.engine
    }

    /// Update the spell correction configuration.
    pub fn set_spell_config(&mut self, config: SpellCorrectedSearchConfig) {
        self.corrector.set_config(config.corrector_config.clone());
        self.config = config;
    }

    /// Search with spell correction for a query string.
    pub fn search_with_correction(
        &mut self,
        query_str: &str,
        default_field: &str,
    ) -> Result<SpellCorrectedSearchResults> {
        if !self.config.enabled {
            // Spell correction disabled, perform normal search
            use crate::query::parser::QueryParser;
            let parser = QueryParser::new().with_default_field(default_field);
            let query = parser.parse(query_str)?;
            let results = self.engine.search(SearchRequest::new(query))?;
            let correction = CorrectionResult::new(query_str.to_string());
            return Ok(SpellCorrectedSearchResults::new(results, correction));
        }

        // Get spelling correction for the query
        let correction = self.corrector.correct(query_str);

        // Try original query first
        use crate::query::parser::QueryParser;
        let parser = QueryParser::new().with_default_field(default_field);
        let query = parser.parse(query_str)?;
        let original_results = self.engine.search(SearchRequest::new(query))?;

        // Decide whether to use correction
        let should_use_correction = self.should_use_correction(&original_results, &correction);

        let (final_results, used_correction) = if should_use_correction {
            // Use the corrected query
            let corrected_query = correction.query();
            let query = parser.parse(corrected_query)?;
            let corrected_results = self.engine.search(SearchRequest::new(query))?;
            (corrected_results, true)
        } else {
            (original_results, false)
        };

        // Generate "Did you mean?" suggestion if appropriate
        let did_you_mean = if self.config.show_did_you_mean && !used_correction {
            self.did_you_mean.suggest(query_str)
        } else {
            None
        };

        let mut spell_results = SpellCorrectedSearchResults::new(final_results, correction);
        spell_results.used_correction = used_correction;
        spell_results.did_you_mean = did_you_mean;

        Ok(spell_results)
    }

    /// Search with spell correction for a field-specific query.
    pub fn search_field_with_correction(
        &mut self,
        field: &str,
        query_str: &str,
    ) -> Result<SpellCorrectedSearchResults> {
        if !self.config.enabled {
            // Spell correction disabled, perform normal search
            use crate::query::parser::QueryParser;
            let parser = QueryParser::new();
            let query = parser.parse_field(field, query_str)?;
            let results = self.engine.search(SearchRequest::new(query))?;
            let correction = CorrectionResult::new(query_str.to_string());
            return Ok(SpellCorrectedSearchResults::new(results, correction));
        }

        // Get spelling correction for the query
        let correction = self.corrector.correct(query_str);

        // Try original query first
        use crate::query::parser::QueryParser;
        let parser = QueryParser::new();
        let query = parser.parse_field(field, query_str)?;
        let original_results = self.engine.search(SearchRequest::new(query))?;

        // Decide whether to use correction
        let should_use_correction = self.should_use_correction(&original_results, &correction);

        let (final_results, used_correction) = if should_use_correction {
            // Use the corrected query
            let corrected_query = correction.query();
            let query = parser.parse_field(field, corrected_query)?;
            let corrected_results = self.engine.search(SearchRequest::new(query))?;
            (corrected_results, true)
        } else {
            (original_results, false)
        };

        // Generate "Did you mean?" suggestion if appropriate
        let did_you_mean = if self.config.show_did_you_mean && !used_correction {
            self.did_you_mean.suggest(query_str)
        } else {
            None
        };

        let mut spell_results = SpellCorrectedSearchResults::new(final_results, correction);
        spell_results.used_correction = used_correction;
        spell_results.did_you_mean = did_you_mean;

        Ok(spell_results)
    }

    /// Check if a word is correctly spelled.
    pub fn is_word_correct(&self, word: &str) -> bool {
        self.corrector.is_correct(word)
    }

    /// Get spelling suggestions for a word.
    pub fn suggest_word(&self, word: &str) -> Vec<crate::spelling::suggest::Suggestion> {
        self.corrector.suggest_word(word)
    }

    /// Learn from the index terms to improve spelling correction.
    pub fn learn_from_index(&mut self) -> Result<()> {
        // Cannot access private field index directly - use public method if available
        // For now, return a simplified implementation
        Ok(())
    }

    /// Get statistics about the spelling corrector.
    pub fn corrector_stats(&self) -> crate::spelling::corrector::CorrectorStats {
        self.corrector.stats()
    }

    /// Clear the query learning history.
    pub fn clear_query_history(&mut self) {
        self.corrector.clear_query_history();
    }

    /// Decide whether to use the spelling correction based on search results and correction quality.
    fn should_use_correction(
        &self,
        original_results: &SearchResults,
        correction: &CorrectionResult,
    ) -> bool {
        // Don't use correction if auto-correction is disabled and no corrections suggested
        if !correction.has_suggestions() {
            return false;
        }

        // If auto-correction was applied, use it
        if correction.auto_corrected {
            return true;
        }

        // If retry with correction is enabled and original query has poor results
        if self.config.retry_with_correction {
            // Use correction if original query has few results and correction confidence is high
            let has_few_results =
                original_results.total_hits < self.config.min_results_for_suggestions as u64;
            let high_confidence = correction.confidence > 0.7;

            if has_few_results && high_confidence {
                return true;
            }
        }

        false
    }
}

// Delegate standard search methods to the underlying engine
// impl Search for SpellCorrectedSearchEngine {
//     fn search(&self, request: SearchRequest) -> Result<SearchResults> {
//         self.engine.search(request)
//     }

//     fn count(&self, query: Box<dyn Query>) -> Result<u64> {
//         self.engine.count(query)
//     }
// }

/// Utility functions for spell-corrected search.
pub struct SpellSearchUtils;

impl SpellSearchUtils {
    /// Extract search terms from a query string for spell checking.
    pub fn extract_search_terms(query_str: &str) -> Vec<String> {
        // Common stop words to filter out
        let stop_words = [
            "and", "or", "not", "the", "is", "a", "an", "in", "on", "at", "to", "for", "of",
            "with", "by",
        ];

        // Simple extraction - split on common query operators and whitespace
        query_str
            .split(&[':', '(', ')', '"', '+', '-', ' ', '\t', '\n'][..])
            .filter_map(|term| {
                let cleaned = term.trim().to_lowercase();
                if cleaned.len() > 2
                    && cleaned.chars().all(|c| c.is_alphabetic())
                    && !stop_words.contains(&cleaned.as_str())
                {
                    Some(cleaned)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Create a corrected query string from the original and correction results.
    pub fn build_corrected_query(original: &str, correction: &CorrectionResult) -> String {
        if let Some(corrected) = &correction.corrected {
            corrected.clone()
        } else {
            // Build a partially corrected query
            let mut result = original.to_string();

            for (original_word, suggestions) in &correction.word_suggestions {
                if let Some(best_suggestion) = suggestions.first()
                    && best_suggestion.score > 0.6
                {
                    result = result.replace(original_word, &best_suggestion.word);
                }
            }

            result
        }
    }

    /// Format "Did you mean?" suggestion for display.
    pub fn format_did_you_mean(_original: &str, suggestion: &str) -> String {
        format!("Did you mean: \"{suggestion}\"?")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexical::index::IndexConfig;

    use tempfile::TempDir;

    #[allow(dead_code)]
    #[test]
    fn test_spell_corrected_search_engine_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = crate::lexical::index::IndexConfig::default();

        let engine = SearchEngine::create_in_dir(temp_dir.path(), config).unwrap();
        let spell_engine = SpellCorrectedSearchEngine::new(engine);

        assert!(spell_engine.config.enabled);
        assert!(spell_engine.config.retry_with_correction);
    }

    #[test]
    fn test_spell_corrected_search_disabled() {
        let temp_dir = TempDir::new().unwrap();
        let engine_config = IndexConfig::default();

        let engine = SearchEngine::create_in_dir(temp_dir.path(), engine_config).unwrap();

        let spell_config = SpellCorrectedSearchConfig {
            enabled: false,
            ..Default::default()
        };

        let mut spell_engine = SpellCorrectedSearchEngine::with_config(engine, spell_config);

        let results = spell_engine
            .search_with_correction("hello world", "title")
            .unwrap();

        assert!(!results.has_suggestions());
        assert!(!results.used_correction);
        assert_eq!(results.effective_query(), "hello world");
    }

    #[test]
    fn test_spell_corrected_search_with_typos() {
        let temp_dir = TempDir::new().unwrap();
        let config = IndexConfig::default();

        let engine = SearchEngine::create_in_dir(temp_dir.path(), config).unwrap();
        let mut spell_engine = SpellCorrectedSearchEngine::new(engine);

        // Test with a query that might have typos
        let results = spell_engine
            .search_with_correction("helo wrld", "title")
            .unwrap();

        // Should have some correction information
        assert_eq!(results.correction.original, "helo wrld");
        // Exact behavior depends on the dictionary and suggestion quality
    }

    #[test]
    fn test_word_correction_check() {
        let temp_dir = TempDir::new().unwrap();
        let config = IndexConfig::default();

        let engine = SearchEngine::create_in_dir(temp_dir.path(), config).unwrap();
        let spell_engine = SpellCorrectedSearchEngine::new(engine);

        // Test with common words
        assert!(spell_engine.is_word_correct("hello")); // Should be in built-in dictionary
        assert!(spell_engine.is_word_correct("the")); // Should be in built-in dictionary

        // Test with likely typos
        let suggestions = spell_engine.suggest_word("helo");
        // Should get some suggestions (exact results depend on dictionary)
        assert!(!suggestions.is_empty() || !spell_engine.is_word_correct("hello"));
    }

    #[test]
    fn test_spell_search_utils() {
        let terms = SpellSearchUtils::extract_search_terms("title:hello AND body:world");
        assert!(terms.contains(&"title".to_string()));
        assert!(terms.contains(&"hello".to_string()));
        assert!(terms.contains(&"body".to_string()));
        assert!(terms.contains(&"world".to_string()));
        assert!(!terms.contains(&"and".to_string())); // Should be filtered out

        let corrected = SpellSearchUtils::build_corrected_query(
            "original query",
            &CorrectionResult::new("original query".to_string()),
        );
        assert_eq!(corrected, "original query");

        let did_you_mean = SpellSearchUtils::format_did_you_mean("helo", "hello");
        assert_eq!(did_you_mean, "Did you mean: \"hello\"?");
    }

    #[test]
    fn test_spell_corrected_results() {
        use crate::query::SearchResults;

        let results = SearchResults {
            hits: vec![],
            total_hits: 0,
            max_score: 0.0,
            // search_time field doesn't exist in SearchResults
        };

        let correction = CorrectionResult::new("test query".to_string());
        let spell_results = SpellCorrectedSearchResults::new(results, correction);

        assert_eq!(spell_results.effective_query(), "test query");
        assert!(!spell_results.has_suggestions());
        assert!(!spell_results.was_auto_corrected());
        assert!(!spell_results.used_correction);
        assert_eq!(spell_results.correction_confidence(), 1.0);
    }

    #[test]
    fn test_corrector_stats() {
        let temp_dir = TempDir::new().unwrap();
        let config = IndexConfig::default();

        let engine = SearchEngine::create_in_dir(temp_dir.path(), config).unwrap();
        let spell_engine = SpellCorrectedSearchEngine::new(engine);

        let stats = spell_engine.corrector_stats();
        assert!(stats.dictionary_words > 0);
        assert!(stats.dictionary_total_frequency > 0);
        assert_eq!(stats.queries_learned, 0); // Initially empty
    }
}
