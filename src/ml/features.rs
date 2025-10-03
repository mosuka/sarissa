//! Feature extraction for machine learning models.

use std::collections::HashMap;

use chrono::{Datelike, Timelike};
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::document::Document;

/// Features extracted from query-document pairs for ranking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryDocumentFeatures {
    // Text-based features
    pub bm25_score: f64,
    pub tf_idf_score: f64,
    pub edit_distance: f64,
    pub query_term_coverage: f64,
    pub exact_match_count: usize,
    pub partial_match_count: usize,

    // Vector-based features
    pub vector_similarity: f64,
    pub semantic_distance: f64,

    // Statistical features
    pub document_length: usize,
    pub query_length: usize,
    pub term_frequency_variance: f64,
    pub inverse_document_frequency_sum: f64,

    // Structural features
    pub title_match_score: f64,
    pub field_match_scores: HashMap<String, f64>,
    pub position_features: PositionFeatures,

    // Popularity features
    pub click_through_rate: f64,
    pub document_age_days: u32,
    pub document_popularity: f64,
    pub query_frequency: u64,

    // Context features
    pub time_of_day: f64,
    pub day_of_week: u8,
    pub user_context_score: f64,
}

/// Position-based features for ranking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionFeatures {
    /// Average position of query terms in document.
    pub avg_term_position: f64,
    /// Minimum distance between query terms.
    pub min_term_distance: usize,
    /// Whether query terms appear in same sentence.
    pub same_sentence: bool,
    /// Whether query terms appear in same paragraph.
    pub same_paragraph: bool,
    /// First occurrence position of any query term.
    pub first_occurrence: usize,
}

impl Default for PositionFeatures {
    fn default() -> Self {
        Self {
            avg_term_position: 0.0,
            min_term_distance: usize::MAX,
            same_sentence: false,
            same_paragraph: false,
            first_occurrence: usize::MAX,
        }
    }
}

/// Feature extractor for query-document pairs.
pub struct FeatureExtractor {
    /// Term statistics for IDF calculation.
    term_stats: TermStatistics,
    /// Click-through rate statistics.
    click_stats: ClickStatistics,
    /// Document popularity statistics.
    popularity_stats: PopularityStatistics,
}

impl Default for FeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureExtractor {
    /// Create a new feature extractor.
    pub fn new() -> Self {
        Self {
            term_stats: TermStatistics::with_demo_data(),
            click_stats: ClickStatistics::with_demo_data(),
            popularity_stats: PopularityStatistics::with_demo_data(),
        }
    }

    /// Extract features from a query-document pair.
    pub fn extract_features(
        &self,
        query: &str,
        document: &Document,
        context: &FeatureContext,
    ) -> Result<QueryDocumentFeatures> {
        let query_terms = self.tokenize_query(query);
        let doc_text = self.extract_document_text(document);
        let doc_terms = self.tokenize_document(&doc_text);

        Ok(QueryDocumentFeatures {
            // Text features
            bm25_score: self.calculate_bm25_score(&query_terms, &doc_terms, &doc_text)?,
            tf_idf_score: self.calculate_tf_idf_score(&query_terms, &doc_terms)?,
            edit_distance: self.calculate_edit_distance(query, &doc_text),
            query_term_coverage: self.calculate_term_coverage(&query_terms, &doc_terms),
            exact_match_count: self.count_exact_matches(&query_terms, &doc_terms),
            partial_match_count: self.count_partial_matches(&query_terms, &doc_terms),

            // Vector features
            vector_similarity: context.vector_similarity.unwrap_or(0.0),
            semantic_distance: context.semantic_distance.unwrap_or(1.0),

            // Statistical features
            document_length: doc_terms.len(),
            query_length: query_terms.len(),
            term_frequency_variance: self.calculate_tf_variance(&doc_terms),
            inverse_document_frequency_sum: self.calculate_idf_sum(&query_terms)?,

            // Structural features
            title_match_score: self.calculate_title_match_score(&query_terms, document)?,
            field_match_scores: self.calculate_field_match_scores(&query_terms, document)?,
            position_features: self.extract_position_features(&query_terms, &doc_text)?,

            // Popularity features
            click_through_rate: self.get_click_through_rate(&context.document_id)?,
            document_age_days: self.calculate_document_age_days(document)?,
            document_popularity: self.get_document_popularity(&context.document_id)?,
            query_frequency: self.get_query_frequency(query)?,

            // Context features
            time_of_day: self.normalize_time_of_day(context.timestamp),
            day_of_week: context.timestamp.weekday().num_days_from_monday() as u8,
            user_context_score: context.user_context_score.unwrap_or(0.0),
        })
    }

    /// Calculate BM25 score.
    fn calculate_bm25_score(
        &self,
        query_terms: &[String],
        doc_terms: &[String],
        _doc_text: &str,
    ) -> Result<f64> {
        const K1: f64 = 1.2;
        const B: f64 = 0.75;

        let doc_length = doc_terms.len() as f64;
        let avg_doc_length = self.term_stats.average_document_length();

        let mut score = 0.0;

        for term in query_terms {
            let tf = self.term_frequency(term, doc_terms) as f64;
            let idf = self.term_stats.inverse_document_frequency(term)?;

            let numerator = tf * (K1 + 1.0);
            let denominator = tf + K1 * (1.0 - B + B * (doc_length / avg_doc_length));

            score += idf * (numerator / denominator);
        }

        Ok(score)
    }

    /// Calculate TF-IDF score.
    fn calculate_tf_idf_score(&self, query_terms: &[String], doc_terms: &[String]) -> Result<f64> {
        let mut score = 0.0;

        for term in query_terms {
            let tf = self.term_frequency(term, doc_terms) as f64;
            let idf = self.term_stats.inverse_document_frequency(term)?;
            score += tf * idf;
        }

        Ok(score)
    }

    /// Calculate edit distance between query and document text.
    fn calculate_edit_distance(&self, query: &str, doc_text: &str) -> f64 {
        let query_chars: Vec<char> = query.chars().collect();
        let doc_chars: Vec<char> = doc_text.chars().take(query_chars.len() * 2).collect();

        if query_chars.is_empty() || doc_chars.is_empty() {
            return 1.0;
        }

        let distance = edit_distance(&query_chars, &doc_chars);
        1.0 - (distance as f64 / query_chars.len().max(doc_chars.len()) as f64)
    }

    /// Calculate term coverage ratio.
    fn calculate_term_coverage(&self, query_terms: &[String], doc_terms: &[String]) -> f64 {
        if query_terms.is_empty() {
            return 0.0;
        }

        let doc_term_set: std::collections::HashSet<&String> = doc_terms.iter().collect();
        let covered_terms = query_terms
            .iter()
            .filter(|term| doc_term_set.contains(term))
            .count();

        covered_terms as f64 / query_terms.len() as f64
    }

    /// Count exact matches between query and document terms.
    fn count_exact_matches(&self, query_terms: &[String], doc_terms: &[String]) -> usize {
        let doc_term_set: std::collections::HashSet<&String> = doc_terms.iter().collect();
        query_terms
            .iter()
            .filter(|term| doc_term_set.contains(term))
            .count()
    }

    /// Count partial matches (prefix, suffix, substring).
    fn count_partial_matches(&self, query_terms: &[String], doc_terms: &[String]) -> usize {
        let mut matches = 0;

        for query_term in query_terms {
            for doc_term in doc_terms {
                if query_term != doc_term
                    && (doc_term.contains(query_term)
                        || query_term.contains(doc_term)
                        || self.has_common_prefix(query_term, doc_term, 3))
                {
                    matches += 1;
                }
            }
        }

        matches
    }

    /// Extract position-based features.
    fn extract_position_features(
        &self,
        query_terms: &[String],
        doc_text: &str,
    ) -> Result<PositionFeatures> {
        let doc_words: Vec<&str> = doc_text.split_whitespace().collect();
        let mut positions = Vec::new();

        // Find positions of query terms
        for (i, word) in doc_words.iter().enumerate() {
            if query_terms
                .iter()
                .any(|term| term.eq_ignore_ascii_case(word))
            {
                positions.push(i);
            }
        }

        if positions.is_empty() {
            return Ok(PositionFeatures::default());
        }

        let avg_position = positions.iter().sum::<usize>() as f64 / positions.len() as f64;
        let min_distance = if positions.len() > 1 {
            positions
                .windows(2)
                .map(|w| w[1] - w[0])
                .min()
                .unwrap_or(usize::MAX)
        } else {
            usize::MAX
        };

        Ok(PositionFeatures {
            avg_term_position: avg_position,
            min_term_distance: min_distance,
            same_sentence: self.terms_in_same_sentence(&positions, doc_text),
            same_paragraph: self.terms_in_same_paragraph(&positions, doc_text),
            first_occurrence: positions.into_iter().min().unwrap_or(usize::MAX),
        })
    }

    /// Helper methods
    fn tokenize_query(&self, query: &str) -> Vec<String> {
        query.split_whitespace().map(|s| s.to_lowercase()).collect()
    }

    fn tokenize_document(&self, text: &str) -> Vec<String> {
        text.split_whitespace().map(|s| s.to_lowercase()).collect()
    }

    fn extract_document_text(&self, document: &Document) -> String {
        // Extract text from all text fields
        let mut text_parts = Vec::new();

        for field_name in document.field_names() {
            if let Some(field_value) = document.get_field(field_name) {
                if let Some(text) = field_value.as_text() {
                    text_parts.push(text);
                }
            }
        }

        text_parts.join(" ")
    }

    fn term_frequency(&self, term: &str, doc_terms: &[String]) -> usize {
        doc_terms.iter().filter(|t| *t == term).count()
    }

    fn calculate_tf_variance(&self, doc_terms: &[String]) -> f64 {
        let term_counts: HashMap<&String, usize> =
            doc_terms.iter().fold(HashMap::new(), |mut acc, term| {
                *acc.entry(term).or_insert(0) += 1;
                acc
            });

        if term_counts.is_empty() {
            return 0.0;
        }

        let mean = term_counts.values().sum::<usize>() as f64 / term_counts.len() as f64;
        term_counts
            .values()
            .map(|&count| (count as f64 - mean).powi(2))
            .sum::<f64>()
            / term_counts.len() as f64
    }

    fn has_common_prefix(&self, s1: &str, s2: &str, min_length: usize) -> bool {
        if s1.len() < min_length || s2.len() < min_length {
            return false;
        }

        s1.chars()
            .zip(s2.chars())
            .take(min_length)
            .all(|(c1, c2)| c1 == c2)
    }

    fn terms_in_same_sentence(&self, positions: &[usize], _doc_text: &str) -> bool {
        // Simple heuristic: check if all positions are within 20 words
        if positions.len() < 2 {
            return true;
        }

        let min_pos = *positions.iter().min().unwrap();
        let max_pos = *positions.iter().max().unwrap();

        max_pos - min_pos <= 20
    }

    fn terms_in_same_paragraph(&self, positions: &[usize], _doc_text: &str) -> bool {
        // Simple heuristic: check if all positions are within 100 words
        if positions.len() < 2 {
            return true;
        }

        let min_pos = *positions.iter().min().unwrap();
        let max_pos = *positions.iter().max().unwrap();

        max_pos - min_pos <= 100
    }

    fn normalize_time_of_day(&self, timestamp: chrono::DateTime<chrono::Utc>) -> f64 {
        let hour = timestamp.hour() as f64;
        let minute = timestamp.minute() as f64;
        (hour + minute / 60.0) / 24.0
    }

    // Placeholder implementations for statistics methods
    fn calculate_title_match_score(
        &self,
        _query_terms: &[String],
        _document: &Document,
    ) -> Result<f64> {
        // Implementation would check title field specifically
        Ok(0.0)
    }

    fn calculate_field_match_scores(
        &self,
        _query_terms: &[String],
        _document: &Document,
    ) -> Result<HashMap<String, f64>> {
        // Implementation would calculate scores for each field
        Ok(HashMap::new())
    }

    fn calculate_idf_sum(&self, query_terms: &[String]) -> Result<f64> {
        let mut sum = 0.0;
        for term in query_terms {
            sum += self.term_stats.inverse_document_frequency(term)?;
        }
        Ok(sum)
    }

    fn get_click_through_rate(&self, document_id: &str) -> Result<f64> {
        Ok(self.click_stats.get_ctr(document_id))
    }

    fn calculate_document_age_days(&self, _document: &Document) -> Result<u32> {
        // Implementation would extract creation date from document
        Ok(0)
    }

    fn get_document_popularity(&self, document_id: &str) -> Result<f64> {
        Ok(self.popularity_stats.get_popularity(document_id))
    }

    fn get_query_frequency(&self, query: &str) -> Result<u64> {
        Ok(self.term_stats.query_frequency(query))
    }
}

/// Context information for feature extraction.
#[derive(Debug, Clone)]
pub struct FeatureContext {
    pub document_id: String,
    pub vector_similarity: Option<f64>,
    pub semantic_distance: Option<f64>,
    pub user_context_score: Option<f64>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Term statistics for IDF calculation.
#[derive(Debug)]
pub struct TermStatistics {
    term_document_counts: HashMap<String, u64>,
    total_documents: u64,
    average_doc_length: f64,
    query_frequencies: HashMap<String, u64>,
}

impl Default for TermStatistics {
    fn default() -> Self {
        Self::new()
    }
}

impl TermStatistics {
    pub fn new() -> Self {
        Self {
            term_document_counts: HashMap::new(),
            total_documents: 0,
            average_doc_length: 0.0,
            query_frequencies: HashMap::new(),
        }
    }

    pub fn with_demo_data() -> Self {
        let mut term_counts = HashMap::new();

        // Add common programming terms with document frequencies
        term_counts.insert("machine".to_string(), 3);
        term_counts.insert("learning".to_string(), 4);
        term_counts.insert("python".to_string(), 2);
        term_counts.insert("rust".to_string(), 2);
        term_counts.insert("programming".to_string(), 3);
        term_counts.insert("data".to_string(), 3);
        term_counts.insert("science".to_string(), 2);
        term_counts.insert("algorithm".to_string(), 2);
        term_counts.insert("deep".to_string(), 2);
        term_counts.insert("neural".to_string(), 2);
        term_counts.insert("network".to_string(), 2);
        term_counts.insert("language".to_string(), 3);
        term_counts.insert("natural".to_string(), 1);
        term_counts.insert("processing".to_string(), 1);
        term_counts.insert("javascript".to_string(), 1);
        term_counts.insert("web".to_string(), 1);
        term_counts.insert("development".to_string(), 2);
        term_counts.insert("computer".to_string(), 1);
        term_counts.insert("structures".to_string(), 1);
        term_counts.insert("artificial".to_string(), 2);
        term_counts.insert("intelligence".to_string(), 2);

        let mut query_freqs = HashMap::new();
        query_freqs.insert("machine learning".to_string(), 150);
        query_freqs.insert("python programming".to_string(), 85);
        query_freqs.insert("data science".to_string(), 120);
        query_freqs.insert("deep learning".to_string(), 95);
        query_freqs.insert("artificial intelligence".to_string(), 75);
        query_freqs.insert("web development".to_string(), 60);
        query_freqs.insert("rust programming".to_string(), 25);
        query_freqs.insert("javascript".to_string(), 40);

        Self {
            term_document_counts: term_counts,
            total_documents: 100,      // Assume 100 total documents in corpus
            average_doc_length: 150.0, // Average 150 words per document
            query_frequencies: query_freqs,
        }
    }

    pub fn inverse_document_frequency(&self, term: &str) -> Result<f64> {
        let doc_count = self.term_document_counts.get(term).copied().unwrap_or(1);
        Ok((self.total_documents as f64 / doc_count as f64).ln())
    }

    pub fn average_document_length(&self) -> f64 {
        self.average_doc_length
    }

    pub fn query_frequency(&self, query: &str) -> u64 {
        self.query_frequencies.get(query).copied().unwrap_or(0)
    }
}

/// Click-through rate statistics.
#[derive(Debug)]
pub struct ClickStatistics {
    document_clicks: HashMap<String, u64>,
    document_impressions: HashMap<String, u64>,
}

impl Default for ClickStatistics {
    fn default() -> Self {
        Self::new()
    }
}

impl ClickStatistics {
    pub fn new() -> Self {
        Self {
            document_clicks: HashMap::new(),
            document_impressions: HashMap::new(),
        }
    }

    pub fn with_demo_data() -> Self {
        let mut clicks = HashMap::new();
        let mut impressions = HashMap::new();

        // Document CTR data for demo
        clicks.insert("doc1".to_string(), 45);
        impressions.insert("doc1".to_string(), 200);
        clicks.insert("doc2".to_string(), 38);
        impressions.insert("doc2".to_string(), 180);
        clicks.insert("doc3".to_string(), 22);
        impressions.insert("doc3".to_string(), 150);
        clicks.insert("doc4".to_string(), 15);
        impressions.insert("doc4".to_string(), 120);
        clicks.insert("doc5".to_string(), 8);
        impressions.insert("doc5".to_string(), 100);
        clicks.insert("doc6".to_string(), 5);
        impressions.insert("doc6".to_string(), 80);

        Self {
            document_clicks: clicks,
            document_impressions: impressions,
        }
    }

    pub fn get_ctr(&self, document_id: &str) -> f64 {
        let clicks = self.document_clicks.get(document_id).copied().unwrap_or(0);
        let impressions = self
            .document_impressions
            .get(document_id)
            .copied()
            .unwrap_or(0);

        if impressions > 0 {
            clicks as f64 / impressions as f64
        } else {
            0.0
        }
    }
}

/// Document popularity statistics.
#[derive(Debug)]
pub struct PopularityStatistics {
    popularity_scores: HashMap<String, f64>,
}

impl Default for PopularityStatistics {
    fn default() -> Self {
        Self::new()
    }
}

impl PopularityStatistics {
    pub fn new() -> Self {
        Self {
            popularity_scores: HashMap::new(),
        }
    }

    pub fn with_demo_data() -> Self {
        let mut scores = HashMap::new();

        // Demo popularity scores based on content types
        scores.insert("doc1".to_string(), 0.85); // High popularity for ML content
        scores.insert("doc2".to_string(), 0.75); // Good popularity for AI content
        scores.insert("doc3".to_string(), 0.65); // Medium popularity for general programming
        scores.insert("doc4".to_string(), 0.55); // Lower popularity for web development
        scores.insert("doc5".to_string(), 0.45); // Low popularity for algorithms
        scores.insert("doc6".to_string(), 0.35); // Lowest popularity for NLP

        Self {
            popularity_scores: scores,
        }
    }

    pub fn get_popularity(&self, document_id: &str) -> f64 {
        self.popularity_scores
            .get(document_id)
            .copied()
            .unwrap_or(0.0)
    }
}

/// Calculate edit distance between two character sequences.
fn edit_distance(s1: &[char], s2: &[char]) -> usize {
    let len1 = s1.len();
    let len2 = s2.len();

    if len1 == 0 {
        return len2;
    }
    if len2 == 0 {
        return len1;
    }

    let mut dp = vec![vec![0; len2 + 1]; len1 + 1];

    // Initialize base cases
    for (i, row) in dp.iter_mut().enumerate().take(len1 + 1) {
        row[0] = i;
    }
    for j in 0..=len2 {
        dp[0][j] = j;
    }

    // Fill DP table
    for i in 1..=len1 {
        for j in 1..=len2 {
            if s1[i - 1] == s2[j - 1] {
                dp[i][j] = dp[i - 1][j - 1];
            } else {
                dp[i][j] = 1 + dp[i - 1][j].min(dp[i][j - 1]).min(dp[i - 1][j - 1]);
            }
        }
    }

    dp[len1][len2]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_extractor_creation() {
        let extractor = FeatureExtractor::new();
        // FeatureExtractor::new() はデモデータで初期化される
        assert_eq!(extractor.term_stats.total_documents, 100);
        assert!(extractor.term_stats.average_document_length() > 0.0);

        // デモデータに含まれる主要な用語の確認
        assert!(
            extractor
                .term_stats
                .inverse_document_frequency("machine")
                .is_ok()
        );
        assert!(
            extractor
                .term_stats
                .inverse_document_frequency("learning")
                .is_ok()
        );
        assert!(
            extractor
                .term_stats
                .inverse_document_frequency("python")
                .is_ok()
        );
    }

    #[test]
    fn test_query_document_features_default() {
        let features = QueryDocumentFeatures {
            bm25_score: 1.0,
            tf_idf_score: 0.5,
            edit_distance: 0.8,
            query_term_coverage: 0.9,
            exact_match_count: 2,
            partial_match_count: 1,
            vector_similarity: 0.7,
            semantic_distance: 0.3,
            document_length: 100,
            query_length: 3,
            term_frequency_variance: 0.2,
            inverse_document_frequency_sum: 5.0,
            title_match_score: 0.6,
            field_match_scores: HashMap::new(),
            position_features: PositionFeatures::default(),
            click_through_rate: 0.1,
            document_age_days: 30,
            document_popularity: 0.4,
            query_frequency: 10,
            time_of_day: 0.5,
            day_of_week: 1,
            user_context_score: 0.3,
        };

        assert_eq!(features.bm25_score, 1.0);
        assert_eq!(features.exact_match_count, 2);
        assert_eq!(features.query_length, 3);
    }

    #[test]
    fn test_edit_distance_calculation() {
        let s1: Vec<char> = "hello".chars().collect();
        let s2: Vec<char> = "hallo".chars().collect();
        let distance = edit_distance(&s1, &s2);
        assert_eq!(distance, 1);

        let s3: Vec<char> = "".chars().collect();
        let s4: Vec<char> = "test".chars().collect();
        let distance2 = edit_distance(&s3, &s4);
        assert_eq!(distance2, 4);
    }

    #[test]
    fn test_feature_context_creation() {
        let context = FeatureContext {
            document_id: "doc1".to_string(),
            vector_similarity: Some(0.8),
            semantic_distance: Some(0.2),
            user_context_score: Some(0.5),
            timestamp: chrono::Utc::now(),
        };

        assert_eq!(context.document_id, "doc1");
        assert_eq!(context.vector_similarity, Some(0.8));
    }
}
