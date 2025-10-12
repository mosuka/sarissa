//! Keyword-based intent classifier.

use std::collections::HashSet;
use std::sync::Arc;

use anyhow::Result;

use crate::analysis::analyzer::Analyzer;
use crate::ml::query_expansion::QueryIntent;

use super::classifier::IntentClassifier;

/// Keyword-based intent classifier.
///
/// Uses simple keyword matching to determine query intent.
pub struct KeywordBasedIntentClassifier {
    informational_keywords: HashSet<String>,
    navigational_keywords: HashSet<String>,
    transactional_keywords: HashSet<String>,
    analyzer: Arc<dyn Analyzer>,
}

impl std::fmt::Debug for KeywordBasedIntentClassifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KeywordBasedIntentClassifier")
            .field("informational_keywords", &self.informational_keywords)
            .field("navigational_keywords", &self.navigational_keywords)
            .field("transactional_keywords", &self.transactional_keywords)
            .field("analyzer", &self.analyzer.name())
            .finish()
    }
}

impl KeywordBasedIntentClassifier {
    /// Create a new keyword-based intent classifier.
    pub fn new(
        informational_keywords: HashSet<String>,
        navigational_keywords: HashSet<String>,
        transactional_keywords: HashSet<String>,
        analyzer: Arc<dyn Analyzer>,
    ) -> Self {
        Self {
            informational_keywords,
            navigational_keywords,
            transactional_keywords,
            analyzer,
        }
    }
}

impl IntentClassifier for KeywordBasedIntentClassifier {
    fn predict(&self, query: &str) -> Result<QueryIntent> {
        // Use analyzer to tokenize the query
        let query_terms: Vec<String> = self
            .analyzer
            .analyze(query)?
            .map(|token| token.text)
            .collect();

        let mut informational_score = 0;
        let mut navigational_score = 0;
        let mut transactional_score = 0;

        for term in &query_terms {
            if self.informational_keywords.contains(term) {
                informational_score += 1;
            }
            if self.navigational_keywords.contains(term) {
                navigational_score += 1;
            }
            if self.transactional_keywords.contains(term) {
                transactional_score += 1;
            }
        }

        let max_score = informational_score
            .max(navigational_score)
            .max(transactional_score);

        if max_score == 0 {
            Ok(QueryIntent::Unknown)
        } else if max_score == informational_score {
            Ok(QueryIntent::Informational)
        } else if max_score == navigational_score {
            Ok(QueryIntent::Navigational)
        } else {
            Ok(QueryIntent::Transactional)
        }
    }

    fn name(&self) -> &str {
        "keyword"
    }
}
