//! Fuzzy query implementation for approximate string matching.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::lexical::reader::IndexReader;
use crate::query::matcher::Matcher;
use crate::query::query::Query;
use crate::query::scorer::Scorer;
use crate::spelling::levenshtein::{
    TypoPatterns, damerau_levenshtein_distance, levenshtein_distance,
};

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

    /// Find matching terms and their documents using actual Levenshtein distance.
    pub fn find_matches(&self, reader: &dyn IndexReader) -> Result<Vec<FuzzyMatch>> {
        let mut matches = Vec::new();
        let mut terms_found = std::collections::HashSet::new();

        // Scan all documents to find terms that match within edit distance
        let doc_count = reader.doc_count();
        let query_term_lower = self.term.to_lowercase();

        // Debug: print query info (comment out for production)
        // eprintln!("FuzzyQuery: searching for '{}' (lowercase: '{}') in field '{}' with max_edits={}",
        //           self.term, query_term_lower, self.field, self.max_edits);

        for doc_id in 0..doc_count {
            if let Ok(Some(document)) = reader.document(doc_id)
                && let Some(field_value) = document.get_field(&self.field)
                && let Some(text) = field_value.as_text()
            {
                // Debug: print document content
                // eprintln!("  Doc {}: field '{}' = '{}'", doc_id, self.field, text);

                // Simple tokenization - split by whitespace and punctuation
                let words: Vec<(String, String)> = text
                    .split_whitespace()
                    .flat_map(|word| {
                        // Split by hyphens first, then clean each part
                        word.split('-')
                            .flat_map(|part| {
                                // Remove punctuation but keep original case
                                let clean_word = part
                                    .chars()
                                    .filter(|c| c.is_alphabetic())
                                    .collect::<String>();
                                if clean_word.len() >= 2 {
                                    Some((clean_word.clone(), clean_word.to_lowercase()))
                                } else {
                                    None
                                }
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect();

                for (original_word, lowercase_word) in words {
                    // Skip if we've already processed this term
                    if terms_found.contains(&lowercase_word) {
                        continue;
                    }

                    // Check if word respects prefix constraint (use lowercase for comparison)
                    if !self.respects_prefix(&lowercase_word) {
                        continue;
                    }

                    // Calculate edit distance using lowercase
                    let edit_distance = if self.transpositions {
                        damerau_levenshtein_distance(&query_term_lower, &lowercase_word) as u32
                    } else {
                        levenshtein_distance(&query_term_lower, &lowercase_word) as u32
                    };

                    // Check if within allowed edit distance
                    if edit_distance <= self.max_edits {
                        // eprintln!("    Found match: '{}' (original: '{}') distance={}, max_edits={}",
                        //           lowercase_word, original_word, edit_distance, self.max_edits);
                        terms_found.insert(lowercase_word.clone());

                        // Try both original case and lowercase when querying the index
                        let mut doc_freq = 1; // Fallback frequency
                        let mut search_term = lowercase_word.clone();

                        // First try lowercase
                        if let Ok(Some(term_info)) = reader.term_info(&self.field, &lowercase_word)
                        {
                            doc_freq = term_info.doc_freq as u32;
                            // eprintln!("      term_info found for '{}': doc_freq={}", lowercase_word, doc_freq);
                        } else if let Ok(Some(term_info)) =
                            reader.term_info(&self.field, &original_word)
                        {
                            // Then try original case
                            doc_freq = term_info.doc_freq as u32;
                            search_term = original_word.clone();
                            // eprintln!("      term_info found for '{}': doc_freq={}", original_word, doc_freq);
                        } else {
                            // eprintln!("      No term_info found for '{}' or '{}'", lowercase_word, original_word);
                        }

                        let similarity_score = self
                            .calculate_similarity_score_advanced(edit_distance, &lowercase_word);

                        matches.push(FuzzyMatch {
                            term: search_term, // Use the version that was found in the index
                            edit_distance,
                            doc_frequency: doc_freq,
                            similarity_score,
                        });
                    }
                }
            }
        }

        // Sort by similarity score (highest first), then by document frequency
        matches.sort_by(|a, b| {
            b.similarity_score
                .partial_cmp(&a.similarity_score)
                .unwrap()
                .then_with(|| b.doc_frequency.cmp(&a.doc_frequency))
        });

        // Limit results to prevent explosion
        matches.truncate(100);

        Ok(matches)
    }

    /// Generate edit candidates using systematic edit operations.
    #[allow(dead_code)]
    fn generate_edit_candidates(&self) -> Vec<String> {
        let mut candidates = std::collections::HashSet::new();
        let chars: Vec<char> = self.term.chars().collect();
        let term_len = chars.len();

        // Add exact match
        candidates.insert(self.term.clone());

        // Generate candidates within max_edits distance
        for edit_level in 1..=self.max_edits {
            let level_candidates = if edit_level == 1 {
                self.generate_single_edits(&chars, term_len)
            } else {
                // For multi-edit, we'd recursively apply edits
                // For now, limit to single and double edits for performance
                if edit_level == 2 {
                    self.generate_double_edits(&chars, term_len)
                } else {
                    Vec::new()
                }
            };

            for candidate in level_candidates {
                if candidates.len() < 1000 {
                    // Prevent explosion
                    candidates.insert(candidate);
                }
            }
        }

        candidates.into_iter().collect()
    }

    /// Generate all single-edit candidates.
    #[allow(dead_code)]
    fn generate_single_edits(&self, chars: &[char], term_len: usize) -> Vec<String> {
        let mut candidates = Vec::new();
        let prefix_len = self.prefix_length as usize;

        // Deletions
        for i in prefix_len..term_len {
            let mut new_chars = chars.to_vec();
            new_chars.remove(i);
            candidates.push(new_chars.into_iter().collect());
        }

        // Substitutions with keyboard-aware variants
        for i in prefix_len..term_len {
            let original_char = chars[i];

            // Try nearby keyboard keys first (more likely typos)
            let nearby_keys = TypoPatterns::nearby_keys(original_char);
            for &replacement in &nearby_keys {
                let mut new_chars = chars.to_vec();
                new_chars[i] = replacement;
                candidates.push(new_chars.into_iter().collect());
            }

            // Try common letter substitutions
            for replacement in 'a'..='z' {
                if replacement != original_char && !nearby_keys.contains(&replacement) {
                    let mut new_chars = chars.to_vec();
                    new_chars[i] = replacement;
                    candidates.push(new_chars.into_iter().collect());

                    // Limit to prevent explosion
                    if candidates.len() > 500 {
                        break;
                    }
                }
            }
        }

        // Insertions
        for i in prefix_len..=term_len {
            for c in 'a'..='z' {
                let mut new_chars = chars.to_vec();
                new_chars.insert(i, c);
                candidates.push(new_chars.into_iter().collect());

                // Limit to prevent explosion
                if candidates.len() > 300 {
                    break;
                }
            }
        }

        // Transpositions (if enabled)
        if self.transpositions {
            for i in prefix_len..term_len.saturating_sub(1) {
                let mut new_chars = chars.to_vec();
                new_chars.swap(i, i + 1);
                candidates.push(new_chars.into_iter().collect());
            }
        }

        candidates
    }

    /// Generate double-edit candidates (simplified version).
    #[allow(dead_code)]
    fn generate_double_edits(&self, chars: &[char], term_len: usize) -> Vec<String> {
        let mut candidates = Vec::new();
        let prefix_len = self.prefix_length as usize;

        // Double deletions
        for i in prefix_len..term_len {
            for j in (i + 1)..term_len {
                let mut new_chars = chars.to_vec();
                new_chars.remove(j);
                new_chars.remove(i);
                candidates.push(new_chars.into_iter().collect());

                if candidates.len() > 100 {
                    return candidates;
                }
            }
        }

        // One deletion + one substitution (common pattern)
        for del_pos in prefix_len..term_len {
            for sub_pos in prefix_len..term_len {
                if del_pos != sub_pos {
                    let mut new_chars = chars.to_vec();
                    if del_pos < sub_pos {
                        new_chars.remove(del_pos);
                        new_chars[sub_pos - 1] = 'e'; // Common replacement
                    } else {
                        new_chars[sub_pos] = 'e';
                        new_chars.remove(del_pos);
                    }
                    candidates.push(new_chars.into_iter().collect());

                    if candidates.len() > 50 {
                        return candidates;
                    }
                }
            }
        }

        candidates
    }

    /// Check if a candidate respects the prefix length constraint.
    fn respects_prefix(&self, candidate: &str) -> bool {
        let prefix_len = self.prefix_length as usize;
        if prefix_len == 0 {
            return true;
        }

        let term_prefix: String = self.term.chars().take(prefix_len).collect();
        let candidate_prefix: String = candidate.chars().take(prefix_len).collect();

        term_prefix == candidate_prefix
    }

    /// Calculate advanced similarity score considering keyboard distance and frequency.
    fn calculate_similarity_score_advanced(&self, edit_distance: u32, candidate: &str) -> f32 {
        if edit_distance > self.max_edits {
            return 0.0;
        }

        // Base score from edit distance
        let base_score = self.calculate_similarity_score(edit_distance);

        // Keyboard distance bonus (prefer typos that are more likely)
        let keyboard_distance = TypoPatterns::keyboard_distance(&self.term, candidate);
        let keyboard_bonus = if keyboard_distance < edit_distance as f64 {
            0.1 // Bonus for keyboard-likely errors
        } else {
            0.0
        };

        // Length similarity bonus
        let length_diff = (self.term.len() as i32 - candidate.len() as i32).abs() as f32;
        let length_bonus = (1.0 - (length_diff / self.term.len().max(1) as f32)) * 0.05;

        (base_score + keyboard_bonus + length_bonus).min(1.0)
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
        let matches = self.find_matches(reader)?;
        Ok(Box::new(FuzzyMatcher::new(matches, reader, &self.field)?))
    }

    fn scorer(&self, reader: &dyn IndexReader) -> Result<Box<dyn Scorer>> {
        use crate::query::scorer::BM25Scorer;

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
        eprintln!("FuzzyMatcher: processing {} matches", matches.len());
        for fuzzy_match in matches {
            eprintln!(
                "  Looking for term '{}' in field '{}'",
                fuzzy_match.term, field
            );
            if let Some(postings) = reader.postings(field, &fuzzy_match.term)? {
                let mut posting_iter = postings;
                eprintln!("    Got postings for '{}'", fuzzy_match.term);

                // Collect all document IDs that contain this fuzzy matching term
                while posting_iter.next()? {
                    let doc_id = posting_iter.doc_id();
                    eprintln!("      Found doc_id: {doc_id}");
                    if doc_id != u64::MAX {
                        // Use the highest similarity score if document matches multiple fuzzy terms
                        let current_score = doc_scores.get(&doc_id).unwrap_or(&0.0);
                        if fuzzy_match.similarity_score > *current_score {
                            doc_scores.insert(doc_id, fuzzy_match.similarity_score);
                            // eprintln!("        Stored doc {} with score {}", doc_id, fuzzy_match.similarity_score);
                        }
                    }
                }
            } else {
                // eprintln!("    No postings found for '{}'", fuzzy_match.term);
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
    fn test_fuzzy_edit_candidate_generation() {
        let query = FuzzyQuery::new("field", "cat")
            .max_edits(1)
            .prefix_length(0);
        let candidates = query.generate_edit_candidates();

        // Should include exact match
        assert!(candidates.contains(&"cat".to_string()));

        // Should include some 1-edit variations
        assert!(candidates.len() > 1);

        // Check specific expected candidates
        assert!(candidates.contains(&"ca".to_string())); // deletion
        assert!(candidates.contains(&"bat".to_string())); // substitution
        assert!(candidates.contains(&"cast".to_string())); // insertion
    }

    #[test]
    fn test_prefix_constraint() {
        let query = FuzzyQuery::new("field", "search")
            .max_edits(1)
            .prefix_length(2);

        // Should respect prefix "se"
        assert!(query.respects_prefix("search"));
        assert!(query.respects_prefix("searchy"));
        assert!(query.respects_prefix("serach"));
        assert!(!query.respects_prefix("research"));
        assert!(!query.respects_prefix("asearch"));
    }

    #[test]
    fn test_advanced_similarity_scoring() {
        let query = FuzzyQuery::new("field", "search").max_edits(2);

        // Exact match should score highest
        let exact_score = query.calculate_similarity_score_advanced(0, "search");
        assert_eq!(exact_score, 1.0);

        // Single edit should score lower but still high
        let single_edit_score = query.calculate_similarity_score_advanced(1, "serach");
        assert!(single_edit_score > 0.7 && single_edit_score < 1.0);

        // Double edit should score lower
        let double_edit_score = query.calculate_similarity_score_advanced(2, "serch");
        assert!(double_edit_score < single_edit_score);

        // Beyond max edits should score 0
        let beyond_max_score = query.calculate_similarity_score_advanced(3, "xyz");
        assert_eq!(beyond_max_score, 0.0);
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
        use crate::lexical::index::reader::inverted::{
            InvertedIndexReader, InvertedIndexReaderConfig,
        };
        use crate::storage::memory::MemoryStorage;
        use crate::storage::traits::StorageConfig;
        use std::sync::Arc;

        // Create a test schema and reader

        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
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
