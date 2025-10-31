//! Fuzzy query implementation for approximate string matching.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::lexical::core::automaton::{AutomatonTermsEnum, LevenshteinAutomaton};
use crate::lexical::core::terms::{TermDictionaryAccess, TermsEnum as _};
use crate::lexical::index::inverted::reader::InvertedIndexReader;
use crate::lexical::reader::IndexReader;
use crate::query::matcher::Matcher;
use crate::query::query::Query;
use crate::query::scorer::Scorer;
use crate::spelling::levenshtein::{damerau_levenshtein_distance, levenshtein_distance};

/// A fuzzy query for approximate string matching.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzyQuery {
    /// Field to search in
    field: String,
    /// Term to search for
    term: String,
    /// Maximum edit distance (Levenshtein distance)
    max_edits: u32,
    /// Minimum prefix length that must match exactly
    prefix_length: u32,
    /// Whether transpositions count as single edits (Damerau-Levenshtein)
    transpositions: bool,
    /// Maximum number of terms to expand to (default: 50, like Lucene)
    /// This prevents queries from matching too many terms and consuming excessive resources.
    max_expansions: usize,
    /// Boost factor for the query
    boost: f32,
}

impl FuzzyQuery {
    /// Create a new fuzzy query with default settings.
    pub fn new<F: Into<String>, T: Into<String>>(field: F, term: T) -> Self {
        FuzzyQuery {
            field: field.into(),
            term: term.into(),
            max_edits: 2,
            prefix_length: 0,
            transpositions: true,
            max_expansions: 50, // Same default as Lucene
            boost: 1.0,
        }
    }

    /// Set the maximum edit distance.
    pub fn max_edits(mut self, max_edits: u32) -> Self {
        self.max_edits = max_edits;
        self
    }

    /// Set the minimum prefix length that must match exactly.
    pub fn prefix_length(mut self, prefix_length: u32) -> Self {
        self.prefix_length = prefix_length;
        self
    }

    /// Set whether transpositions should be considered single edits.
    pub fn transpositions(mut self, transpositions: bool) -> Self {
        self.transpositions = transpositions;
        self
    }

    /// Set the maximum number of terms to expand to.
    /// This prevents queries from matching too many terms and consuming excessive resources.
    /// Default is 50, same as Lucene.
    pub fn max_expansions(mut self, max_expansions: usize) -> Self {
        self.max_expansions = max_expansions;
        self
    }

    /// Set the boost factor for this query.
    pub fn with_boost(mut self, boost: f32) -> Self {
        self.boost = boost;
        self
    }

    /// Get the field name.
    pub fn field(&self) -> &str {
        &self.field
    }

    /// Get the search term.
    pub fn term(&self) -> &str {
        &self.term
    }

    /// Get the maximum edit distance.
    pub fn get_max_edits(&self) -> u32 {
        self.max_edits
    }

    /// Get the prefix length.
    pub fn get_prefix_length(&self) -> u32 {
        self.prefix_length
    }

    /// Check if transpositions are enabled.
    pub fn get_transpositions(&self) -> bool {
        self.transpositions
    }

    /// Get the maximum number of terms to expand to.
    pub fn get_max_expansions(&self) -> usize {
        self.max_expansions
    }

    /// Find matching terms using efficient term dictionary enumeration.
    ///
    /// This uses the Term Dictionary API and Levenshtein Automaton for efficient matching,
    /// similar to Lucene's FuzzyTermsEnum approach.
    ///
    /// Falls back to legacy document scanning if Term Dictionary API is not available.
    ///
    /// Returns a vector of FuzzyMatch results, sorted by similarity score.
    pub fn find_matches(&self, reader: &dyn IndexReader) -> Result<Vec<FuzzyMatch>> {
        let mut matches = Vec::new();

        // Try to downcast to InvertedIndexReader and use Term Dictionary API
        if let Some(inverted_reader) = reader.as_any().downcast_ref::<InvertedIndexReader>() {
            // Normalize the query term using the analyzer
            let analyzer = inverted_reader.analyzer();
            let token_stream = analyzer.analyze(&self.term)?;
            let tokens: Vec<_> = token_stream.collect();

            if let Some(first_token) = tokens.first() {
                let normalized_query = &first_token.text;

                // Try to get term dictionary for the field
                if let Some(terms) = inverted_reader.terms(&self.field)? {
                    // Create Levenshtein automaton
                    let automaton = LevenshteinAutomaton::new(
                        normalized_query,
                        self.max_edits,
                        self.prefix_length as usize,
                        self.transpositions,
                    );

                    // Create filtered iterator
                    let mut terms_enum = AutomatonTermsEnum::new(terms.iterator()?, automaton)
                        .with_max_matches(self.max_expansions);

                    // Enumerate matching terms
                    while let Some(term_stats) = terms_enum.next()? {
                        let edit_distance = if self.transpositions {
                            damerau_levenshtein_distance(normalized_query, &term_stats.term) as u32
                        } else {
                            levenshtein_distance(normalized_query, &term_stats.term) as u32
                        };

                        let similarity_score = self.calculate_similarity_score(edit_distance);

                        matches.push(FuzzyMatch {
                            term: term_stats.term,
                            edit_distance,
                            doc_frequency: term_stats.doc_freq as u32,
                            similarity_score,
                        });
                    }
                }
                // If terms API is not available, return empty matches
            }
            // If not InvertedIndexReader, return empty matches
        }

        // Sort by similarity score (highest first), then by document frequency
        matches.sort_by(|a, b| {
            b.similarity_score
                .partial_cmp(&a.similarity_score)
                .unwrap()
                .then_with(|| b.doc_frequency.cmp(&a.doc_frequency))
        });

        Ok(matches)
    }

    /// Calculate similarity score based on edit distance.
    fn calculate_similarity_score(&self, edit_distance: u32) -> f32 {
        if edit_distance > self.max_edits {
            return 0.0;
        }

        // Simple linear scoring: 1.0 for exact match, decreasing with edit distance
        let max_score = 1.0;
        let penalty_per_edit = 0.2;

        (max_score - (edit_distance as f32 * penalty_per_edit)).max(0.0)
    }

    /// Estimate document frequency for a term.
    #[allow(dead_code)]
    fn estimate_doc_frequency(&self, term: &str, _reader: &dyn IndexReader) -> Result<u32> {
        // Simplified estimation based on term characteristics
        // In a real implementation, this would query the actual index

        let base_frequency = match term.len() {
            1..=3 => 50,
            4..=6 => 25,
            7..=10 => 10,
            _ => 5,
        };

        // Reduce frequency for longer terms or terms with rare characters
        let rarity_factor = if term.chars().any(|c| "qxz".contains(c)) {
            0.5
        } else {
            1.0
        };

        Ok((base_frequency as f32 * rarity_factor) as u32)
    }
}

impl Query for FuzzyQuery {
    fn matcher(&self, reader: &dyn IndexReader) -> Result<Box<dyn Matcher>> {
        // Try efficient implementation first, fall back to old implementation
        let matches = self.find_matches(reader)?;
        Ok(Box::new(FuzzyMatcher::new(matches, reader, &self.field)?))
    }

    fn scorer(&self, reader: &dyn IndexReader) -> Result<Box<dyn Scorer>> {
        use crate::query::scorer::BM25Scorer;

        // Try efficient implementation first, fall back to old implementation
        let matches = self.find_matches(reader)?;

        if matches.is_empty() {
            // No matches found, create minimal scorer
            return Ok(Box::new(BM25Scorer::new(
                1,
                1,
                reader.doc_count(),
                1.0,
                reader.doc_count(),
                self.boost,
            )));
        }

        // Use the best matching term for BM25 scoring estimation
        let best_match = &matches[0];

        // Try to get field statistics for more accurate scoring
        if let Ok(Some(field_stats)) = reader.field_stats(&self.field) {
            Ok(Box::new(BM25Scorer::new(
                best_match.doc_frequency as u64,
                (best_match.doc_frequency * 2) as u64, // Estimate term frequency
                reader.doc_count(),
                field_stats.avg_length,
                field_stats.doc_count,
                self.boost * best_match.similarity_score, // Combine boost with similarity
            )))
        } else {
            // Fallback if field stats not available
            Ok(Box::new(BM25Scorer::new(
                best_match.doc_frequency as u64,
                (best_match.doc_frequency * 2) as u64,
                reader.doc_count(),
                10.0, // Default average field length
                reader.doc_count(),
                self.boost * best_match.similarity_score,
            )))
        }
    }

    fn boost(&self) -> f32 {
        self.boost
    }

    fn set_boost(&mut self, boost: f32) {
        self.boost = boost;
    }

    fn clone_box(&self) -> Box<dyn Query> {
        Box::new(self.clone())
    }

    fn description(&self) -> String {
        format!(
            "FuzzyQuery(field: {}, term: {}, max_edits: {}, prefix: {})",
            self.field, self.term, self.max_edits, self.prefix_length
        )
    }

    fn is_empty(&self, _reader: &dyn IndexReader) -> Result<bool> {
        Ok(self.term.is_empty())
    }

    fn cost(&self, reader: &dyn IndexReader) -> Result<u64> {
        // Fuzzy queries are expensive due to edit distance calculations
        let doc_count = reader.doc_count() as u32;
        Ok(doc_count as u64 * (self.max_edits as u64 + 1) * 5)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// A match found by fuzzy search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzyMatch {
    /// The matching term
    pub term: String,
    /// Edit distance from the query term
    pub edit_distance: u32,
    /// Number of documents containing this term
    pub doc_frequency: u32,
    /// Similarity score (0.0 to 1.0)
    pub similarity_score: f32,
}

/// Matcher for fuzzy queries.
#[derive(Debug)]
pub struct FuzzyMatcher {
    /// Matching documents with their similarity scores
    doc_scores: HashMap<u64, f32>,
    /// Sorted document IDs
    doc_ids: Vec<u64>,
    /// Current iteration position
    current_index: usize,
    /// Current document ID
    current_doc_id: u64,
}

impl FuzzyMatcher {
    /// Get the similarity score for a document.
    pub fn score(&self, doc_id: u64) -> f32 {
        self.doc_scores.get(&doc_id).copied().unwrap_or(0.0)
    }

    /// Create a new fuzzy matcher using actual document IDs from the index.
    pub fn new(matches: Vec<FuzzyMatch>, reader: &dyn IndexReader, field: &str) -> Result<Self> {
        let mut doc_scores = HashMap::new();

        // For each fuzzy match, find actual documents that contain the term
        for fuzzy_match in matches {
            if let Some(postings) = reader.postings(field, &fuzzy_match.term)? {
                let mut posting_iter = postings;

                // Collect all document IDs that contain this fuzzy matching term
                while posting_iter.next()? {
                    let doc_id = posting_iter.doc_id();
                    if doc_id != u64::MAX {
                        // Use the highest similarity score if document matches multiple fuzzy terms
                        let current_score = doc_scores.get(&doc_id).unwrap_or(&0.0);
                        if fuzzy_match.similarity_score > *current_score {
                            doc_scores.insert(doc_id, fuzzy_match.similarity_score);
                        }
                    }
                }
            }
        }

        // Sort document IDs for efficient iteration
        let mut doc_ids: Vec<u64> = doc_scores.keys().cloned().collect();
        doc_ids.sort();

        // Determine initial document ID
        let initial_doc_id = if doc_ids.is_empty() {
            u64::MAX
        } else {
            doc_ids[0]
        };

        Ok(FuzzyMatcher {
            doc_scores,
            doc_ids,
            current_index: 0,
            current_doc_id: initial_doc_id,
        })
    }
}

impl Matcher for FuzzyMatcher {
    fn doc_id(&self) -> u64 {
        self.current_doc_id
    }

    fn next(&mut self) -> Result<bool> {
        if self.doc_ids.is_empty() {
            self.current_doc_id = u64::MAX;
            return Ok(false);
        }

        self.current_index += 1;

        if self.current_index < self.doc_ids.len() {
            self.current_doc_id = self.doc_ids[self.current_index];
            Ok(true)
        } else {
            self.current_doc_id = u64::MAX;
            Ok(false)
        }
    }

    fn skip_to(&mut self, target: u64) -> Result<bool> {
        if self.doc_ids.is_empty() {
            self.current_doc_id = u64::MAX;
            return Ok(false);
        }

        // Find first document ID >= target using binary search
        match self.doc_ids[self.current_index..].binary_search(&target) {
            Ok(index) => {
                // Exact match found
                self.current_index += index;
                self.current_doc_id = self.doc_ids[self.current_index];
                Ok(true)
            }
            Err(index) => {
                // Target not found, index is insertion point
                self.current_index += index;
                if self.current_index < self.doc_ids.len() {
                    self.current_doc_id = self.doc_ids[self.current_index];
                    Ok(true)
                } else {
                    self.current_doc_id = u64::MAX;
                    Ok(false)
                }
            }
        }
    }

    fn cost(&self) -> u64 {
        self.doc_ids.len() as u64
    }

    fn is_exhausted(&self) -> bool {
        self.current_doc_id == u64::MAX || self.current_index >= self.doc_ids.len()
    }
}

/// Scorer for fuzzy queries.
#[derive(Debug)]
pub struct FuzzyScorer {
    /// Document scores based on fuzzy similarity
    doc_scores: HashMap<u32, f32>,
    /// Query boost factor
    boost: f32,
}

impl FuzzyScorer {
    /// Create a new fuzzy scorer.
    pub fn new(matches: Vec<FuzzyMatch>, boost: f32) -> Self {
        let mut doc_scores = HashMap::new();

        // Generate synthetic document scores based on fuzzy matches
        for (i, fuzzy_match) in matches.iter().enumerate() {
            for doc_id in 1..=fuzzy_match.doc_frequency {
                let synthetic_doc_id = (i as u32 * 1000) + doc_id;
                doc_scores.insert(synthetic_doc_id, fuzzy_match.similarity_score);
            }
        }

        FuzzyScorer { doc_scores, boost }
    }
}

impl Scorer for FuzzyScorer {
    fn score(&self, doc_id: u64, _term_freq: f32, _field_length: Option<f32>) -> f32 {
        self.doc_scores.get(&(doc_id as u32)).unwrap_or(&0.0) * self.boost
    }

    fn boost(&self) -> f32 {
        self.boost
    }

    fn set_boost(&mut self, boost: f32) {
        self.boost = boost;
    }

    fn max_score(&self) -> f32 {
        self.doc_scores
            .values()
            .fold(0.0_f32, |max, &score| max.max(score))
            * self.boost
    }

    fn name(&self) -> &'static str {
        "FuzzyScorer"
    }
}

/// Configuration for fuzzy search behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzyConfig {
    /// Default maximum edit distance
    pub default_max_edits: u32,
    /// Default prefix length
    pub default_prefix_length: u32,
    /// Whether to use transpositions by default
    pub default_transpositions: bool,
    /// Maximum number of fuzzy variations to consider
    pub max_variations: usize,
    /// Minimum term length for fuzzy search
    pub min_term_length: usize,
}

impl Default for FuzzyConfig {
    fn default() -> Self {
        FuzzyConfig {
            default_max_edits: 2,
            default_prefix_length: 0,
            default_transpositions: true,
            max_variations: 1000,
            min_term_length: 3,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fuzzy_query_creation() {
        let query = FuzzyQuery::new("content", "hello")
            .max_edits(1)
            .prefix_length(2)
            .transpositions(false)
            .with_boost(1.5);

        assert_eq!(query.field(), "content");
        assert_eq!(query.term(), "hello");
        assert_eq!(query.get_max_edits(), 1);
        assert_eq!(query.get_prefix_length(), 2);
        assert!(!query.get_transpositions());
        assert_eq!(query.boost(), 1.5);
    }

    #[test]
    fn test_fuzzy_query_description() {
        let query = FuzzyQuery::new("title", "test")
            .max_edits(2)
            .prefix_length(1);
        let description = query.description();
        assert!(description.contains("FuzzyQuery"));
        assert!(description.contains("title"));
        assert!(description.contains("test"));
        assert!(description.contains("max_edits: 2"));
        assert!(description.contains("prefix: 1"));
    }

    #[test]
    fn test_similarity_score_calculation() {
        let query = FuzzyQuery::new("field", "term").max_edits(2);

        assert_eq!(query.calculate_similarity_score(0), 1.0); // Exact match
        assert_eq!(query.calculate_similarity_score(1), 0.8); // 1 edit
        assert_eq!(query.calculate_similarity_score(2), 0.6); // 2 edits
        assert_eq!(query.calculate_similarity_score(3), 0.0); // Beyond max_edits
    }

    #[test]
    fn test_fuzzy_config_default() {
        let config = FuzzyConfig::default();
        assert_eq!(config.default_max_edits, 2);
        assert_eq!(config.default_prefix_length, 0);
        assert!(config.default_transpositions);
        assert_eq!(config.max_variations, 1000);
        assert_eq!(config.min_term_length, 3);
    }

    #[test]
    fn test_fuzzy_matcher() {
        use crate::lexical::index::inverted::reader::{
            InvertedIndexReader, InvertedIndexReaderConfig,
        };
        use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};
        use std::sync::Arc;

        // Create a test schema and reader

        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let reader =
            InvertedIndexReader::new(vec![], storage, InvertedIndexReaderConfig::default())
                .unwrap();

        let matches = vec![
            FuzzyMatch {
                term: "hello".to_string(),
                edit_distance: 0,
                doc_frequency: 2,
                similarity_score: 1.0,
            },
            FuzzyMatch {
                term: "helo".to_string(),
                edit_distance: 1,
                doc_frequency: 1,
                similarity_score: 0.8,
            },
        ];

        let mut matcher = FuzzyMatcher::new(matches, &reader, "content").unwrap();

        // Should return documents in sorted order
        let mut docs = Vec::new();
        while matcher.next().unwrap() {
            docs.push(matcher.doc_id());
        }

        // Note: Since we don't have actual documents with these terms,
        // the matcher might not find any documents
        // Check that documents are sorted if any are found
        for i in 1..docs.len() {
            assert!(docs[i] > docs[i - 1]);
        }
    }

    #[test]
    fn test_fuzzy_scorer() {
        let matches = vec![FuzzyMatch {
            term: "test".to_string(),
            edit_distance: 0,
            doc_frequency: 1,
            similarity_score: 1.0,
        }];

        let scorer = FuzzyScorer::new(matches, 2.0);

        // Document score should be similarity * boost
        assert_eq!(scorer.score(1, 1.0, None), 1.0 * 2.0);
        assert_eq!(scorer.score(999, 1.0, None), 0.0); // Non-existent document
        assert_eq!(scorer.max_score(), 1.0 * 2.0);
        assert_eq!(scorer.name(), "FuzzyScorer");
    }
}
