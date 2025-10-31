//! Similarity search functionality for finding similar documents.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::analysis::analyzer::analyzer::Analyzer;
use crate::analysis::analyzer::standard::StandardAnalyzer;
use crate::document::field_value::FieldValue;
use crate::error::Result;
use crate::lexical::index::inverted::query::Query;
use crate::lexical::index::inverted::query::matcher::Matcher;
use crate::lexical::index::inverted::query::scorer::Scorer;
use crate::lexical::reader::IndexReader;

/// Configuration for similarity search.
#[derive(Debug, Clone)]
pub struct SimilarityConfig {
    /// Algorithm to use for similarity calculation.
    pub algorithm: SimilarityAlgorithm,
    /// Minimum similarity score to include in results.
    pub min_similarity: f32,
    /// Maximum number of similar documents to return.
    pub max_results: usize,
    /// Fields to consider for similarity calculation.
    pub similarity_fields: Vec<String>,
    /// Whether to normalize document vectors.
    pub normalize_vectors: bool,
    /// Boost factor for exact term matches.
    pub exact_match_boost: f32,
}

impl Default for SimilarityConfig {
    fn default() -> Self {
        SimilarityConfig {
            algorithm: SimilarityAlgorithm::Cosine,
            min_similarity: 0.1,
            max_results: 20,
            similarity_fields: vec!["content".to_string()],
            normalize_vectors: true,
            exact_match_boost: 1.5,
        }
    }
}

/// Similarity algorithms available for document comparison.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SimilarityAlgorithm {
    /// Cosine similarity (default).
    Cosine,
    /// Jaccard similarity (intersection over union).
    Jaccard,
    /// Euclidean distance-based similarity.
    Euclidean,
    /// Manhattan distance-based similarity.
    Manhattan,
    /// Term frequency-based similarity.
    TermFrequency,
    /// BM25-based similarity.
    BM25,
}

/// Document vector representation for similarity calculations.
#[derive(Debug, Clone)]
pub struct DocumentVector {
    /// Document ID.
    pub doc_id: u32,
    /// Term frequencies or weights.
    pub features: HashMap<String, f32>,
    /// Normalized magnitude for cosine similarity.
    pub magnitude: f32,
}

impl DocumentVector {
    /// Create a new document vector.
    pub fn new(doc_id: u32) -> Self {
        DocumentVector {
            doc_id,
            features: HashMap::new(),
            magnitude: 0.0,
        }
    }

    /// Add or update a feature weight.
    pub fn set_feature(&mut self, term: String, weight: f32) {
        self.features.insert(term, weight);
    }

    /// Calculate and cache the vector magnitude.
    pub fn calculate_magnitude(&mut self) {
        self.magnitude = self.features.values().map(|w| w * w).sum::<f32>().sqrt();
    }

    /// Normalize the vector to unit length.
    pub fn normalize(&mut self) {
        if self.magnitude == 0.0 {
            self.calculate_magnitude();
        }

        if self.magnitude > 0.0 {
            for weight in self.features.values_mut() {
                *weight /= self.magnitude;
            }
            self.magnitude = 1.0;
        }
    }

    /// Get the set of terms in this vector.
    pub fn terms(&self) -> HashSet<String> {
        self.features.keys().cloned().collect()
    }
}

/// Result of similarity search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityResult {
    /// Similar document ID.
    pub doc_id: u32,
    /// Similarity score (0.0 to 1.0).
    pub similarity: f32,
    /// Explanation of how similarity was calculated.
    pub explanation: Option<String>,
}

impl SimilarityResult {
    /// Create a new similarity result.
    pub fn new(doc_id: u32, similarity: f32) -> Self {
        SimilarityResult {
            doc_id,
            similarity,
            explanation: None,
        }
    }

    /// Add explanation for the similarity score.
    pub fn with_explanation(mut self, explanation: String) -> Self {
        self.explanation = Some(explanation);
        self
    }
}

/// Similarity search engine for finding similar documents.
pub struct SimilaritySearchEngine {
    /// Configuration for similarity search.
    config: SimilarityConfig,
    /// Text analyzer for extracting features.
    analyzer: Box<dyn Analyzer>,
    /// Cached document vectors.
    document_vectors: HashMap<u32, DocumentVector>,
}

impl std::fmt::Debug for SimilaritySearchEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SimilaritySearchEngine")
            .field("config", &self.config)
            .field("analyzer", &"<dyn Analyzer>")
            .field("document_vectors", &self.document_vectors)
            .finish()
    }
}

impl SimilaritySearchEngine {
    /// Create a new similarity search engine.
    pub fn new(config: SimilarityConfig) -> Self {
        SimilaritySearchEngine {
            config,
            analyzer: Box::new(StandardAnalyzer::new().unwrap()),
            document_vectors: HashMap::new(),
        }
    }

    /// Create with custom analyzer.
    pub fn with_analyzer(config: SimilarityConfig, analyzer: Box<dyn Analyzer>) -> Self {
        SimilaritySearchEngine {
            config,
            analyzer,
            document_vectors: HashMap::new(),
        }
    }

    /// Find documents similar to a given document.
    pub fn find_similar(
        &mut self,
        target_doc_id: u32,
        reader: &dyn IndexReader,
    ) -> Result<Vec<SimilarityResult>> {
        // Get or create vector for target document
        let target_vector = self.get_or_create_document_vector(target_doc_id, reader)?;

        // Find all candidate documents
        let candidate_docs = self.get_candidate_documents(reader)?;

        let mut results = Vec::new();

        for candidate_doc_id in candidate_docs {
            if candidate_doc_id == target_doc_id {
                continue; // Skip the target document itself
            }

            let candidate_vector = self.get_or_create_document_vector(candidate_doc_id, reader)?;
            let similarity = self.calculate_similarity(&target_vector, &candidate_vector)?;

            if similarity >= self.config.min_similarity {
                results.push(SimilarityResult::new(candidate_doc_id, similarity));
            }
        }

        // Sort by similarity (highest first)
        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(Ordering::Equal)
        });

        // Limit results
        results.truncate(self.config.max_results);

        Ok(results)
    }

    /// Find documents similar to a text query.
    pub fn find_similar_to_text(
        &mut self,
        text: &str,
        reader: &dyn IndexReader,
    ) -> Result<Vec<SimilarityResult>> {
        // Create vector from input text
        let query_vector = self.create_vector_from_text(text, 0)?; // Use doc_id 0 for query

        // Find candidate documents
        let candidate_docs = self.get_candidate_documents(reader)?;

        let mut results = Vec::new();

        for candidate_doc_id in candidate_docs {
            let candidate_vector = self.get_or_create_document_vector(candidate_doc_id, reader)?;
            let similarity = self.calculate_similarity(&query_vector, &candidate_vector)?;

            if similarity >= self.config.min_similarity {
                results.push(SimilarityResult::new(candidate_doc_id, similarity));
            }
        }

        // Sort by similarity (highest first)
        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(Ordering::Equal)
        });

        // Limit results
        results.truncate(self.config.max_results);

        Ok(results)
    }

    /// Get or create document vector for a document.
    fn get_or_create_document_vector(
        &mut self,
        doc_id: u32,
        reader: &dyn IndexReader,
    ) -> Result<DocumentVector> {
        if let Some(vector) = self.document_vectors.get(&doc_id) {
            return Ok(vector.clone());
        }

        // Create vector from document
        let vector = self.create_document_vector(doc_id, reader)?;
        self.document_vectors.insert(doc_id, vector.clone());

        Ok(vector)
    }

    /// Create document vector from stored document fields.
    fn create_document_vector(
        &self,
        doc_id: u32,
        reader: &dyn IndexReader,
    ) -> Result<DocumentVector> {
        let mut vector = DocumentVector::new(doc_id);

        // For each similarity field, extract and analyze text
        for field_name in &self.config.similarity_fields {
            if let Ok(field_text) = self.get_document_field_text(doc_id, field_name, reader) {
                self.add_text_to_vector(&mut vector, &field_text)?;
            }
        }

        // Calculate magnitude and normalize if configured
        vector.calculate_magnitude();
        if self.config.normalize_vectors {
            vector.normalize();
        }

        Ok(vector)
    }

    /// Create vector from text input.
    fn create_vector_from_text(&self, text: &str, doc_id: u32) -> Result<DocumentVector> {
        let mut vector = DocumentVector::new(doc_id);
        self.add_text_to_vector(&mut vector, text)?;

        vector.calculate_magnitude();
        if self.config.normalize_vectors {
            vector.normalize();
        }

        Ok(vector)
    }

    /// Add text content to a document vector.
    fn add_text_to_vector(&self, vector: &mut DocumentVector, text: &str) -> Result<()> {
        let tokens = self.analyzer.analyze(text)?;
        let mut term_counts: HashMap<String, f32> = HashMap::new();

        // Count term frequencies
        for token in tokens {
            *term_counts.entry(token.text.to_lowercase()).or_insert(0.0) += 1.0;
        }

        // Add terms to vector (simple TF weighting for now)
        for (term, count) in term_counts {
            let current_weight = vector.features.get(&term).unwrap_or(&0.0);
            vector.set_feature(term, current_weight + count);
        }

        Ok(())
    }

    /// Calculate TF-IDF weighted vector from document.
    #[allow(dead_code)]
    fn create_tfidf_vector(&self, doc_id: u32, reader: &dyn IndexReader) -> Result<DocumentVector> {
        let mut vector = DocumentVector::new(doc_id);

        // Get document content
        for field_name in &self.config.similarity_fields {
            if let Ok(field_text) = self.get_document_field_text(doc_id, field_name, reader) {
                self.add_tfidf_terms_to_vector(&mut vector, &field_text, reader)?;
            }
        }

        vector.calculate_magnitude();
        if self.config.normalize_vectors {
            vector.normalize();
        }

        Ok(vector)
    }

    /// Add TF-IDF weighted terms to vector.
    #[allow(dead_code)]
    fn add_tfidf_terms_to_vector(
        &self,
        vector: &mut DocumentVector,
        text: &str,
        reader: &dyn IndexReader,
    ) -> Result<()> {
        let tokens = self.analyzer.analyze(text)?;
        let mut term_counts: HashMap<String, f32> = HashMap::new();

        // Count term frequencies in this document
        for token in tokens {
            *term_counts.entry(token.text.to_lowercase()).or_insert(0.0) += 1.0;
        }

        let total_doc_count = reader.doc_count() as f32;
        let doc_length = term_counts.len() as f32;

        // Calculate TF-IDF for each term
        for (term, tf) in term_counts {
            // Calculate TF (term frequency)
            let tf_normalized = tf / doc_length; // Normalize by document length

            // Calculate IDF (inverse document frequency)
            // For now, use a simplified IDF calculation
            // In a full implementation, we would query the index for document frequency
            let df = self
                .estimate_document_frequency(&term, reader)
                .unwrap_or(1.0);
            let idf = (total_doc_count / df).ln() + 1.0;

            // TF-IDF weight
            let tfidf_weight = tf_normalized * idf;

            let current_weight = vector.features.get(&term).unwrap_or(&0.0);
            vector.set_feature(term, current_weight + tfidf_weight);
        }

        Ok(())
    }

    /// Estimate document frequency for a term (simplified implementation).
    #[allow(dead_code)]
    fn estimate_document_frequency(&self, term: &str, _reader: &dyn IndexReader) -> Result<f32> {
        // This is a simplified estimation
        // In a real implementation, we would query the index for exact document frequency

        // Use a heuristic based on term length and characteristics
        let df = match term.len() {
            1..=3 => 100.0, // Very common short words
            4..=6 => 50.0,  // Common words
            7..=10 => 20.0, // Less common words
            _ => 5.0,       // Rare long words
        };

        // Adjust for stop words (very rough heuristic)
        let stop_words = [
            "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
        ];
        if stop_words.contains(&term) {
            Ok(500.0) // Very high document frequency for stop words
        } else {
            Ok(df)
        }
    }

    /// Get text content for a document field.
    fn get_document_field_text(
        &self,
        doc_id: u32,
        field_name: &str,
        reader: &dyn IndexReader,
    ) -> Result<String> {
        // Try to get the stored document from the index
        match reader.document(doc_id as u64) {
            Ok(Some(document)) => {
                // Extract the field value
                if let Some(field_value) = document.get_field(field_name) {
                    match field_value {
                        FieldValue::Text(text) => Ok(text.clone()),
                        FieldValue::Integer(value) => Ok(value.to_string()),
                        FieldValue::Float(value) => Ok(value.to_string()),
                        FieldValue::Boolean(value) => Ok(value.to_string()),
                        FieldValue::Binary(_) => Ok(String::new()), // Binary data can't be converted to meaningful text
                        FieldValue::DateTime(dt) => Ok(dt.to_rfc3339()),
                        FieldValue::Geo(point) => Ok(format!("{},{}", point.lat, point.lon)),
                        FieldValue::Null => Ok(String::new()),
                    }
                } else {
                    Ok(String::new())
                }
            }
            Ok(None) => {
                // Document not found
                Ok(String::new())
            }
            Err(_) => {
                // Fallback for compatibility - some readers might not implement get_stored_document
                Ok(format!("Document {doc_id} field {field_name}"))
            }
        }
    }

    /// Get list of candidate documents for similarity search.
    fn get_candidate_documents(&self, reader: &dyn IndexReader) -> Result<Vec<u32>> {
        // Try to get the actual document count from the reader
        let doc_count = reader.doc_count() as u32;

        // Return all available document IDs (up to a reasonable limit for performance)
        let max_candidates = std::cmp::min(doc_count, 10000); // Limit to 10k docs for performance
        Ok((1..=max_candidates).collect())
    }

    /// Calculate similarity between two document vectors.
    fn calculate_similarity(
        &self,
        vector1: &DocumentVector,
        vector2: &DocumentVector,
    ) -> Result<f32> {
        match self.config.algorithm {
            SimilarityAlgorithm::Cosine => self.cosine_similarity(vector1, vector2),
            SimilarityAlgorithm::Jaccard => self.jaccard_similarity(vector1, vector2),
            SimilarityAlgorithm::Euclidean => self.euclidean_similarity(vector1, vector2),
            SimilarityAlgorithm::Manhattan => self.manhattan_similarity(vector1, vector2),
            SimilarityAlgorithm::TermFrequency => self.term_frequency_similarity(vector1, vector2),
            SimilarityAlgorithm::BM25 => self.bm25_similarity(vector1, vector2),
        }
    }

    /// Calculate cosine similarity between two vectors.
    fn cosine_similarity(&self, vector1: &DocumentVector, vector2: &DocumentVector) -> Result<f32> {
        if vector1.features.is_empty() || vector2.features.is_empty() {
            return Ok(0.0);
        }

        let mut dot_product = 0.0;
        let mut magnitude1 = 0.0;
        let mut magnitude2 = 0.0;

        // Get all unique terms
        let all_terms: HashSet<_> = vector1
            .features
            .keys()
            .chain(vector2.features.keys())
            .collect();

        for term in all_terms {
            let weight1 = vector1.features.get(term).unwrap_or(&0.0);
            let weight2 = vector2.features.get(term).unwrap_or(&0.0);

            dot_product += weight1 * weight2;
            magnitude1 += weight1 * weight1;
            magnitude2 += weight2 * weight2;
        }

        if magnitude1 == 0.0 || magnitude2 == 0.0 {
            return Ok(0.0);
        }

        Ok(dot_product / (magnitude1.sqrt() * magnitude2.sqrt()))
    }

    /// Calculate Jaccard similarity between two vectors.
    fn jaccard_similarity(
        &self,
        vector1: &DocumentVector,
        vector2: &DocumentVector,
    ) -> Result<f32> {
        let terms1 = vector1.terms();
        let terms2 = vector2.terms();

        let intersection = terms1.intersection(&terms2).count();
        let union = terms1.union(&terms2).count();

        if union == 0 {
            Ok(0.0)
        } else {
            Ok(intersection as f32 / union as f32)
        }
    }

    /// Calculate Euclidean distance-based similarity.
    fn euclidean_similarity(
        &self,
        vector1: &DocumentVector,
        vector2: &DocumentVector,
    ) -> Result<f32> {
        let all_terms: HashSet<_> = vector1
            .features
            .keys()
            .chain(vector2.features.keys())
            .collect();

        let mut sum_squared_diff = 0.0;

        for term in all_terms {
            let weight1 = vector1.features.get(term).unwrap_or(&0.0);
            let weight2 = vector2.features.get(term).unwrap_or(&0.0);
            let diff = weight1 - weight2;
            sum_squared_diff += diff * diff;
        }

        let distance = sum_squared_diff.sqrt();

        // Convert distance to similarity (1 / (1 + distance))
        Ok(1.0 / (1.0 + distance))
    }

    /// Calculate Manhattan distance-based similarity.
    fn manhattan_similarity(
        &self,
        vector1: &DocumentVector,
        vector2: &DocumentVector,
    ) -> Result<f32> {
        let all_terms: HashSet<_> = vector1
            .features
            .keys()
            .chain(vector2.features.keys())
            .collect();

        let mut sum_abs_diff = 0.0;

        for term in all_terms {
            let weight1 = vector1.features.get(term).unwrap_or(&0.0);
            let weight2 = vector2.features.get(term).unwrap_or(&0.0);
            sum_abs_diff += (weight1 - weight2).abs();
        }

        // Convert distance to similarity
        Ok(1.0 / (1.0 + sum_abs_diff))
    }

    /// Calculate term frequency-based similarity.
    fn term_frequency_similarity(
        &self,
        vector1: &DocumentVector,
        vector2: &DocumentVector,
    ) -> Result<f32> {
        let mut shared_weight = 0.0;
        let mut total_weight = 0.0;

        let all_terms: HashSet<_> = vector1
            .features
            .keys()
            .chain(vector2.features.keys())
            .collect();

        for term in all_terms {
            let weight1 = vector1.features.get(term).unwrap_or(&0.0);
            let weight2 = vector2.features.get(term).unwrap_or(&0.0);

            shared_weight += weight1.min(*weight2);
            total_weight += weight1.max(*weight2);
        }

        if total_weight == 0.0 {
            Ok(0.0)
        } else {
            Ok(shared_weight / total_weight)
        }
    }

    /// Calculate BM25-based similarity (simplified).
    fn bm25_similarity(&self, vector1: &DocumentVector, vector2: &DocumentVector) -> Result<f32> {
        // This is a simplified BM25-like similarity calculation
        // In a real implementation, we would need document frequencies and other corpus statistics

        let k1 = 1.2;
        let b = 0.75;

        let mut score = 0.0;
        let doc_len1 = vector1.features.values().sum::<f32>();
        let doc_len2 = vector2.features.values().sum::<f32>();
        let avg_doc_len = (doc_len1 + doc_len2) / 2.0;

        for (term, &tf1) in &vector1.features {
            if let Some(&tf2) = vector2.features.get(term) {
                let norm_tf1 = tf1 / (tf1 + k1 * (1.0 - b + b * doc_len1 / avg_doc_len));
                let norm_tf2 = tf2 / (tf2 + k1 * (1.0 - b + b * doc_len2 / avg_doc_len));

                score += norm_tf1 * norm_tf2;
            }
        }

        Ok(score)
    }

    /// Clear cached document vectors.
    pub fn clear_cache(&mut self) {
        self.document_vectors.clear();
    }

    /// Get cache statistics.
    pub fn cache_stats(&self) -> (usize, usize) {
        let cached_docs = self.document_vectors.len();
        let total_features: usize = self
            .document_vectors
            .values()
            .map(|v| v.features.len())
            .sum();
        (cached_docs, total_features)
    }
}

/// More-Like-This query implementation.
#[derive(Debug)]
pub struct MoreLikeThisQuery {
    /// Configuration for similarity search.
    config: SimilarityConfig,
    /// Input text or document to find similar documents for.
    input: MoreLikeThisInput,
}

/// Input for More-Like-This queries.
#[derive(Debug, Clone)]
pub enum MoreLikeThisInput {
    /// Text input.
    Text(String),
    /// Document ID input.
    DocumentId(u32),
    /// Multiple document IDs.
    DocumentIds(Vec<u32>),
}

impl MoreLikeThisQuery {
    /// Create a new More-Like-This query from text.
    pub fn from_text(text: String, config: SimilarityConfig) -> Self {
        MoreLikeThisQuery {
            config,
            input: MoreLikeThisInput::Text(text),
        }
    }

    /// Create a new More-Like-This query from a document ID.
    pub fn from_document(doc_id: u32, config: SimilarityConfig) -> Self {
        MoreLikeThisQuery {
            config,
            input: MoreLikeThisInput::DocumentId(doc_id),
        }
    }

    /// Create a new More-Like-This query from multiple document IDs.
    pub fn from_documents(doc_ids: Vec<u32>, config: SimilarityConfig) -> Self {
        MoreLikeThisQuery {
            config,
            input: MoreLikeThisInput::DocumentIds(doc_ids),
        }
    }

    /// Execute the More-Like-This query.
    pub fn execute(&self, reader: &dyn IndexReader) -> Result<Vec<SimilarityResult>> {
        let mut engine = SimilaritySearchEngine::new(self.config.clone());

        match &self.input {
            MoreLikeThisInput::Text(text) => engine.find_similar_to_text(text, reader),
            MoreLikeThisInput::DocumentId(doc_id) => engine.find_similar(*doc_id, reader),
            MoreLikeThisInput::DocumentIds(doc_ids) => {
                // For multiple documents, we could average their vectors or find documents similar to any of them
                // For simplicity, let's use the first document
                if let Some(&first_doc) = doc_ids.first() {
                    engine.find_similar(first_doc, reader)
                } else {
                    Ok(Vec::new())
                }
            }
        }
    }

    /// Set boost factor for the query.
    pub fn with_boost(mut self, boost: f32) -> Self {
        self.config.exact_match_boost = boost;
        self
    }

    /// Set minimum similarity threshold.
    pub fn min_similarity(mut self, min_similarity: f32) -> Self {
        self.config.min_similarity = min_similarity;
        self
    }

    /// Set maximum number of results.
    pub fn max_results(mut self, max_results: usize) -> Self {
        self.config.max_results = max_results;
        self
    }

    /// Set similarity fields.
    pub fn similarity_fields(mut self, fields: Vec<String>) -> Self {
        self.config.similarity_fields = fields;
        self
    }

    /// Set similarity algorithm.
    pub fn algorithm(mut self, algorithm: SimilarityAlgorithm) -> Self {
        self.config.algorithm = algorithm;
        self
    }
}

impl Query for MoreLikeThisQuery {
    fn matcher(&self, reader: &dyn IndexReader) -> Result<Box<dyn Matcher>> {
        // Create a custom matcher for More-Like-This
        let results = self.execute(reader)?;
        Ok(Box::new(MoreLikeThisMatcher::new(results)))
    }

    fn scorer(&self, reader: &dyn IndexReader) -> Result<Box<dyn Scorer>> {
        // Create a custom scorer for More-Like-This
        let results = self.execute(reader)?;
        Ok(Box::new(MoreLikeThisScorer::new(results)))
    }

    fn boost(&self) -> f32 {
        self.config.exact_match_boost
    }

    fn set_boost(&mut self, boost: f32) {
        self.config.exact_match_boost = boost;
    }

    fn clone_box(&self) -> Box<dyn Query> {
        Box::new(MoreLikeThisQuery {
            config: self.config.clone(),
            input: self.input.clone(),
        })
    }

    fn description(&self) -> String {
        match &self.input {
            MoreLikeThisInput::Text(text) => format!("MoreLikeThis(text: \"{text}\"))"),
            MoreLikeThisInput::DocumentId(doc_id) => format!("MoreLikeThis(doc: {doc_id}))"),
            MoreLikeThisInput::DocumentIds(doc_ids) => format!("MoreLikeThis(docs: {doc_ids:?})"),
        }
    }

    fn is_empty(&self, _reader: &dyn IndexReader) -> Result<bool> {
        match &self.input {
            MoreLikeThisInput::Text(text) => Ok(text.trim().is_empty()),
            MoreLikeThisInput::DocumentId(_) => Ok(false),
            MoreLikeThisInput::DocumentIds(doc_ids) => Ok(doc_ids.is_empty()),
        }
    }

    fn cost(&self, reader: &dyn IndexReader) -> Result<u64> {
        // Cost is proportional to the number of documents to compare
        let doc_count = reader.doc_count();
        Ok(doc_count * 10) // Similarity comparison is expensive
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Matcher for More-Like-This queries.
#[derive(Debug)]
pub struct MoreLikeThisMatcher {
    results: Vec<SimilarityResult>,
    current_index: usize,
    current_doc_id: u64,
}

impl MoreLikeThisMatcher {
    /// Create a new More-Like-This matcher.
    pub fn new(mut results: Vec<SimilarityResult>) -> Self {
        // Sort results by document ID for proper iteration
        results.sort_by_key(|r| r.doc_id);
        MoreLikeThisMatcher {
            results,
            current_index: 0,
            current_doc_id: 0,
        }
    }
}

impl Matcher for MoreLikeThisMatcher {
    fn doc_id(&self) -> u64 {
        self.current_doc_id
    }

    fn next(&mut self) -> Result<bool> {
        if self.current_index < self.results.len() {
            self.current_doc_id = self.results[self.current_index].doc_id as u64;
            self.current_index += 1;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn skip_to(&mut self, target: u64) -> Result<bool> {
        // Find the first document ID >= target
        while self.current_index < self.results.len() {
            let doc_id = self.results[self.current_index].doc_id as u64;
            if doc_id >= target {
                self.current_doc_id = doc_id;
                self.current_index += 1;
                return Ok(true);
            }
            self.current_index += 1;
        }
        Ok(false)
    }

    fn cost(&self) -> u64 {
        self.results.len() as u64
    }

    fn is_exhausted(&self) -> bool {
        self.current_index >= self.results.len()
    }
}

/// Scorer for More-Like-This queries.
#[derive(Debug)]
pub struct MoreLikeThisScorer {
    similarity_scores: HashMap<u32, f32>,
    boost: f32,
}

impl MoreLikeThisScorer {
    /// Create a new More-Like-This scorer.
    pub fn new(results: Vec<SimilarityResult>) -> Self {
        let mut similarity_scores = HashMap::new();
        for result in results {
            similarity_scores.insert(result.doc_id, result.similarity);
        }

        MoreLikeThisScorer {
            similarity_scores,
            boost: 1.0,
        }
    }

    /// Set boost factor.
    pub fn set_boost(&mut self, boost: f32) {
        self.boost = boost;
    }
}

impl Scorer for MoreLikeThisScorer {
    fn score(&self, doc_id: u64, _term_freq: f32, _field_length: Option<f32>) -> f32 {
        self.similarity_scores.get(&(doc_id as u32)).unwrap_or(&0.0) * self.boost
    }

    fn boost(&self) -> f32 {
        self.boost
    }

    fn set_boost(&mut self, boost: f32) {
        self.boost = boost;
    }

    fn max_score(&self) -> f32 {
        self.similarity_scores
            .values()
            .fold(0.0_f32, |max, &score| max.max(score))
            * self.boost
    }

    fn name(&self) -> &'static str {
        "MoreLikeThisScorer"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_similarity_config() {
        let config = SimilarityConfig::default();
        assert_eq!(config.algorithm, SimilarityAlgorithm::Cosine);
        assert_eq!(config.min_similarity, 0.1);
        assert_eq!(config.max_results, 20);
        assert!(config.normalize_vectors);
    }

    #[test]
    fn test_document_vector() {
        let mut vector = DocumentVector::new(1);
        vector.set_feature("term1".to_string(), 2.0);
        vector.set_feature("term2".to_string(), 3.0);

        vector.calculate_magnitude();
        assert!((vector.magnitude - (4.0 + 9.0_f32).sqrt()).abs() < 1e-6);

        vector.normalize();
        assert!((vector.magnitude - 1.0).abs() < 1e-6);

        let terms = vector.terms();
        assert_eq!(terms.len(), 2);
        assert!(terms.contains("term1"));
        assert!(terms.contains("term2"));
    }

    #[test]
    fn test_similarity_result() {
        let result =
            SimilarityResult::new(123, 0.85).with_explanation("High cosine similarity".to_string());

        assert_eq!(result.doc_id, 123);
        assert_eq!(result.similarity, 0.85);
        assert_eq!(
            result.explanation,
            Some("High cosine similarity".to_string())
        );
    }

    #[test]
    fn test_cosine_similarity() {
        let config = SimilarityConfig::default();
        let engine = SimilaritySearchEngine::new(config);

        let mut vector1 = DocumentVector::new(1);
        vector1.set_feature("term1".to_string(), 1.0);
        vector1.set_feature("term2".to_string(), 0.0);

        let mut vector2 = DocumentVector::new(2);
        vector2.set_feature("term1".to_string(), 1.0);
        vector2.set_feature("term2".to_string(), 0.0);

        let similarity = engine.cosine_similarity(&vector1, &vector2).unwrap();
        assert!((similarity - 1.0).abs() < 1e-6); // Identical vectors should have similarity 1.0

        vector2.set_feature("term1".to_string(), 0.0);
        vector2.set_feature("term2".to_string(), 1.0);

        let similarity = engine.cosine_similarity(&vector1, &vector2).unwrap();
        assert!((similarity - 0.0).abs() < 1e-6); // Orthogonal vectors should have similarity 0.0
    }

    #[test]
    fn test_jaccard_similarity() {
        let config = SimilarityConfig::default();
        let engine = SimilaritySearchEngine::new(config);

        let mut vector1 = DocumentVector::new(1);
        vector1.set_feature("term1".to_string(), 1.0);
        vector1.set_feature("term2".to_string(), 1.0);

        let mut vector2 = DocumentVector::new(2);
        vector2.set_feature("term1".to_string(), 1.0);
        vector2.set_feature("term3".to_string(), 1.0);

        let similarity = engine.jaccard_similarity(&vector1, &vector2).unwrap();
        // Intersection: {term1} = 1, Union: {term1, term2, term3} = 3
        assert!((similarity - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_more_like_this_query() {
        let config = SimilarityConfig::default();

        let text_query = MoreLikeThisQuery::from_text("sample text".to_string(), config.clone());
        match text_query.input {
            MoreLikeThisInput::Text(ref text) => assert_eq!(text, "sample text"),
            _ => panic!("Expected text input"),
        }

        let doc_query = MoreLikeThisQuery::from_document(42, config.clone());
        match doc_query.input {
            MoreLikeThisInput::DocumentId(doc_id) => assert_eq!(doc_id, 42),
            _ => panic!("Expected document ID input"),
        }

        let docs_query = MoreLikeThisQuery::from_documents(vec![1, 2, 3], config);
        match docs_query.input {
            MoreLikeThisInput::DocumentIds(ref doc_ids) => assert_eq!(doc_ids, &vec![1, 2, 3]),
            _ => panic!("Expected document IDs input"),
        }
    }

    #[test]
    fn test_more_like_this_query_builder() {
        let config = SimilarityConfig::default();

        let query = MoreLikeThisQuery::from_text("test text".to_string(), config)
            .with_boost(2.0)
            .min_similarity(0.3)
            .max_results(10)
            .algorithm(SimilarityAlgorithm::Jaccard);

        assert_eq!(query.boost(), 2.0);
        assert_eq!(query.config.min_similarity, 0.3);
        assert_eq!(query.config.max_results, 10);
        assert_eq!(query.config.algorithm, SimilarityAlgorithm::Jaccard);
    }

    #[test]
    fn test_more_like_this_matcher() {
        let results = vec![
            SimilarityResult::new(3, 0.8),
            SimilarityResult::new(1, 0.9),
            SimilarityResult::new(5, 0.7),
        ];

        let mut matcher = MoreLikeThisMatcher::new(results);

        // Should return documents in sorted order
        assert!(matcher.next().unwrap());
        assert_eq!(matcher.doc_id(), 1);
        assert!(matcher.next().unwrap());
        assert_eq!(matcher.doc_id(), 3);
        assert!(matcher.next().unwrap());
        assert_eq!(matcher.doc_id(), 5);
        assert!(!matcher.next().unwrap());
    }

    #[test]
    fn test_more_like_this_scorer() {
        let results = vec![
            SimilarityResult::new(1, 0.9),
            SimilarityResult::new(2, 0.8),
            SimilarityResult::new(3, 0.7),
        ];

        let mut scorer = MoreLikeThisScorer::new(results);
        scorer.set_boost(2.0);

        assert_eq!(scorer.score(1, 1.0, None), 0.9 * 2.0);
        assert_eq!(scorer.score(2, 1.0, None), 0.8 * 2.0);
        assert_eq!(scorer.score(3, 1.0, None), 0.7 * 2.0);
        assert_eq!(scorer.score(999, 1.0, None), 0.0); // Non-existent document

        assert_eq!(scorer.max_score(), 0.9 * 2.0);
    }

    #[test]
    fn test_tfidf_estimation() {
        let config = SimilarityConfig::default();
        let engine = SimilaritySearchEngine::new(config);

        // Test document frequency estimation
        assert!(
            engine
                .estimate_document_frequency("the", &MockIndexReader)
                .unwrap()
                > 100.0
        );
        assert!(
            engine
                .estimate_document_frequency("specialized", &MockIndexReader)
                .unwrap()
                < 50.0
        );
        assert!(
            engine
                .estimate_document_frequency("antidisestablishmentarianism", &MockIndexReader)
                .unwrap()
                < 10.0
        );
    }

    // Mock IndexReader for testing
    #[derive(Debug)]
    struct MockIndexReader;

    impl IndexReader for MockIndexReader {
        fn doc_count(&self) -> u64 {
            1000
        }
        fn max_doc(&self) -> u64 {
            1000
        }
        fn is_deleted(&self, _doc_id: u64) -> bool {
            false
        }
        fn document(&self, _doc_id: u64) -> Result<Option<crate::document::document::Document>> {
            Ok(None)
        }
        fn term_info(
            &self,
            _field: &str,
            _term: &str,
        ) -> Result<Option<crate::lexical::reader::ReaderTermInfo>> {
            Ok(None)
        }
        fn postings(
            &self,
            _field: &str,
            _term: &str,
        ) -> Result<Option<Box<dyn crate::lexical::reader::PostingIterator>>> {
            Ok(None)
        }
        fn field_stats(&self, _field: &str) -> Result<Option<crate::lexical::reader::FieldStats>> {
            Ok(None)
        }
        fn is_closed(&self) -> bool {
            false
        }
        fn close(&mut self) -> Result<()> {
            Ok(())
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }
}
