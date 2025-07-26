//! Hybrid search combining vector similarity and keyword search.
//!
//! This module provides the ability to combine traditional keyword-based search
//! with vector-based semantic search, offering the best of both approaches:
//! - Precise keyword matching for exact terms
//! - Semantic understanding through vector embeddings
//! - Configurable weighting between the two approaches

use crate::error::Result;
use crate::query::{Query, SearchResults};
use crate::search::{Search, SearchRequest};
use crate::vector::{
    Vector,
    types::{VectorSearchConfig, VectorSearchResults},
};
use crate::vector_index::embeddings::{EmbeddingConfig, EmbeddingEngine};
use crate::vector_search::VectorSearchEngine;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Configuration for hybrid search combining keyword and vector search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridSearchConfig {
    /// Weight for keyword search results (0.0-1.0).
    pub keyword_weight: f32,
    /// Weight for vector search results (0.0-1.0).
    pub vector_weight: f32,
    /// Minimum keyword score threshold.
    pub min_keyword_score: f32,
    /// Minimum vector similarity threshold.
    pub min_vector_similarity: f32,
    /// Maximum number of results to return.
    pub max_results: usize,
    /// Whether to require both keyword and vector matches.
    pub require_both: bool,
    /// Normalization strategy for combining scores.
    pub normalization: ScoreNormalization,
    /// Vector search configuration.
    pub vector_config: VectorSearchConfig,
    /// Embedding configuration for text processing.
    pub embedding_config: EmbeddingConfig,
}

impl Default for HybridSearchConfig {
    fn default() -> Self {
        Self {
            keyword_weight: 0.6,
            vector_weight: 0.4,
            min_keyword_score: 0.0,
            min_vector_similarity: 0.3,
            max_results: 50,
            require_both: false,
            normalization: ScoreNormalization::MinMax,
            vector_config: VectorSearchConfig::default(),
            embedding_config: EmbeddingConfig::default(),
        }
    }
}

/// Score normalization strategies for combining keyword and vector scores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScoreNormalization {
    /// No normalization - use raw scores.
    None,
    /// Min-max normalization to [0, 1] range.
    MinMax,
    /// Z-score normalization.
    ZScore,
    /// Rank-based normalization.
    Rank,
}

/// A single result from hybrid search containing both keyword and vector scores.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridSearchResult {
    /// Document ID.
    pub doc_id: u64,
    /// Combined hybrid score.
    pub hybrid_score: f32,
    /// Keyword search score (if available).
    pub keyword_score: Option<f32>,
    /// Vector similarity score (if available).
    pub vector_similarity: Option<f32>,
    /// Document content (if requested).
    pub document: Option<HashMap<String, String>>,
    /// Vector data (if requested).
    pub vector: Option<Vector>,
    /// Additional metadata.
    pub metadata: HashMap<String, String>,
}

impl HybridSearchResult {
    /// Create a new hybrid search result.
    pub fn new(doc_id: u64, hybrid_score: f32) -> Self {
        Self {
            doc_id,
            hybrid_score,
            keyword_score: None,
            vector_similarity: None,
            document: None,
            vector: None,
            metadata: HashMap::new(),
        }
    }

    /// Set keyword search score.
    pub fn with_keyword_score(mut self, score: f32) -> Self {
        self.keyword_score = Some(score);
        self
    }

    /// Set vector similarity score.
    pub fn with_vector_similarity(mut self, similarity: f32) -> Self {
        self.vector_similarity = Some(similarity);
        self
    }

    /// Add document content.
    pub fn with_document(mut self, document: HashMap<String, String>) -> Self {
        self.document = Some(document);
        self
    }

    /// Add vector data.
    pub fn with_vector(mut self, vector: Vector) -> Self {
        self.vector = Some(vector);
        self
    }

    /// Add metadata.
    pub fn with_metadata(mut self, metadata: HashMap<String, String>) -> Self {
        self.metadata = metadata;
        self
    }
}

/// Collection of hybrid search results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridSearchResults {
    /// List of results, sorted by hybrid score (descending).
    pub results: Vec<HybridSearchResult>,
    /// Total number of documents searched.
    pub total_searched: usize,
    /// Number of keyword matches.
    pub keyword_matches: usize,
    /// Number of vector matches.
    pub vector_matches: usize,
    /// Query processing time in milliseconds.
    pub query_time_ms: u64,
    /// Query text used for search.
    pub query_text: String,
}

impl HybridSearchResults {
    /// Create new empty hybrid search results.
    pub fn empty() -> Self {
        Self {
            results: Vec::new(),
            total_searched: 0,
            keyword_matches: 0,
            vector_matches: 0,
            query_time_ms: 0,
            query_text: String::new(),
        }
    }

    /// Create new hybrid search results.
    pub fn new(
        results: Vec<HybridSearchResult>,
        total_searched: usize,
        keyword_matches: usize,
        vector_matches: usize,
        query_time_ms: u64,
        query_text: String,
    ) -> Self {
        Self {
            results,
            total_searched,
            keyword_matches,
            vector_matches,
            query_time_ms,
            query_text,
        }
    }

    /// Get the number of results.
    pub fn len(&self) -> usize {
        self.results.len()
    }

    /// Check if results are empty.
    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }

    /// Get the best result.
    pub fn best_result(&self) -> Option<&HybridSearchResult> {
        self.results.first()
    }

    /// Filter results by minimum hybrid score.
    pub fn filter_by_score(&mut self, min_score: f32) {
        self.results
            .retain(|result| result.hybrid_score >= min_score);
    }

    /// Sort results by hybrid score (descending).
    pub fn sort_by_score(&mut self) {
        self.results.sort_by(|a, b| {
            b.hybrid_score
                .partial_cmp(&a.hybrid_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Limit the number of results.
    pub fn limit(&mut self, max_results: usize) {
        if self.results.len() > max_results {
            self.results.truncate(max_results);
        }
    }
}

/// Hybrid search engine that combines keyword and vector search.
pub struct HybridSearchEngine {
    /// Configuration for hybrid search.
    config: HybridSearchConfig,
    /// Vector search engine for semantic search.
    _vector_search_engine: VectorSearchEngine,
    /// Text embedder for converting queries to vectors.
    embedder: Arc<RwLock<EmbeddingEngine>>,
    /// Document storage for retrieving full documents.
    document_store: Arc<RwLock<HashMap<u64, HashMap<String, String>>>>,
}

impl HybridSearchEngine {
    /// Create a new hybrid search engine.
    pub fn new(config: HybridSearchConfig) -> Result<Self> {
        let vector_search_engine = VectorSearchEngine::new(Default::default())?;
        let embedder = EmbeddingEngine::new(config.embedding_config.clone())?;

        Ok(Self {
            config,
            _vector_search_engine: vector_search_engine,
            embedder: Arc::new(RwLock::new(embedder)),
            document_store: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Add a document to both keyword and vector indexes.
    pub async fn add_document(
        &mut self,
        doc_id: u64,
        fields: HashMap<String, String>,
    ) -> Result<()> {
        // Store the document
        {
            let mut store = self.document_store.write().await;
            store.insert(doc_id, fields.clone());
        }

        // Create text content for embedding
        let text_content = self.extract_text_content(&fields);

        // Generate embedding (vector indexing would be handled separately)
        {
            let embedder = self.embedder.read().await;
            if embedder.is_trained() {
                let _vector = embedder.embed(&text_content)?;
                // TODO: Add to vector index through vector_index module
            }
        }

        Ok(())
    }

    /// Train the embedder on a collection of documents.
    pub async fn train_embedder(&mut self, documents: &[&str]) -> Result<()> {
        let mut embedder = self.embedder.write().await;
        embedder.train(documents).await?;
        Ok(())
    }

    /// Check if the embedder is trained.
    pub async fn is_embedder_trained(&self) -> bool {
        let embedder = self.embedder.read().await;
        embedder.is_trained()
    }

    /// Remove a document from both indexes.
    pub async fn remove_document(&mut self, doc_id: u64) -> Result<bool> {
        // Remove from document store
        let existed = {
            let mut store = self.document_store.write().await;
            store.remove(&doc_id).is_some()
        };

        // TODO: Remove from vector index through vector_index module

        Ok(existed)
    }

    /// Perform hybrid search combining keyword and vector search.
    pub async fn search<S: Search>(
        &self,
        query_text: &str,
        keyword_searcher: &S,
        keyword_query: Box<dyn Query>,
    ) -> Result<HybridSearchResults> {
        let start_time = std::time::Instant::now();

        // Perform keyword search
        let keyword_request = SearchRequest::new(keyword_query)
            .max_docs(self.config.max_results * 2) // Get more to allow for merging
            .min_score(self.config.min_keyword_score);

        let keyword_results = keyword_searcher.search(keyword_request)?;

        // Perform vector search if embedder is trained
        let vector_results = if self.is_embedder_trained().await {
            let embedder = self.embedder.read().await;
            let _query_vector = embedder.embed(query_text)?;
            drop(embedder);

            let mut vector_config = self.config.vector_config.clone();
            vector_config.top_k = self.config.max_results * 2;
            vector_config.min_similarity = self.config.min_vector_similarity;

            // TODO: Use vector search engine
            // Some(self.vector_search_engine.search(&query_vector, &vector_config).await?)
            None
        } else {
            None
        };

        // Merge and rank results
        let hybrid_results = self
            .merge_results(
                keyword_results,
                vector_results,
                query_text.to_string(),
                start_time.elapsed().as_millis() as u64,
            )
            .await?;

        Ok(hybrid_results)
    }

    /// Extract text content from document fields for embedding.
    fn extract_text_content(&self, fields: &HashMap<String, String>) -> String {
        // Combine all text fields with space separation
        // You can customize this based on field importance
        fields.values().cloned().collect::<Vec<String>>().join(" ")
    }

    /// Merge keyword and vector search results into hybrid results.
    async fn merge_results(
        &self,
        keyword_results: SearchResults,
        vector_results: Option<VectorSearchResults>,
        query_text: String,
        query_time_ms: u64,
    ) -> Result<HybridSearchResults> {
        let mut result_map: HashMap<u64, HybridSearchResult> = HashMap::new();
        let mut keyword_scores = Vec::new();
        let mut vector_similarities = Vec::new();

        // Process keyword results
        for hit in &keyword_results.hits {
            keyword_scores.push(hit.score);
            let result = HybridSearchResult::new(hit.doc_id, 0.0).with_keyword_score(hit.score);
            result_map.insert(hit.doc_id, result);
        }

        // Process vector results
        if let Some(ref vector_results) = vector_results {
            for result in &vector_results.results {
                vector_similarities.push(result.similarity);

                if let Some(existing) = result_map.get_mut(&result.doc_id) {
                    existing.vector_similarity = Some(result.similarity);
                } else {
                    let hybrid_result = HybridSearchResult::new(result.doc_id, 0.0)
                        .with_vector_similarity(result.similarity);
                    result_map.insert(result.doc_id, hybrid_result);
                }
            }
        }

        // Filter results based on requirements
        if self.config.require_both {
            result_map.retain(|_, result| {
                result.keyword_score.is_some() && result.vector_similarity.is_some()
            });
        }

        // Normalize scores
        self.normalize_scores(&mut result_map, &keyword_scores, &vector_similarities)?;

        // Calculate hybrid scores
        for result in result_map.values_mut() {
            let keyword_component =
                result.keyword_score.unwrap_or(0.0) * self.config.keyword_weight;
            let vector_component =
                result.vector_similarity.unwrap_or(0.0) * self.config.vector_weight;
            result.hybrid_score = keyword_component + vector_component;
        }

        // Add document content if available
        self.add_document_content(&mut result_map).await?;

        // Convert to vector and sort
        let mut results: Vec<HybridSearchResult> = result_map.into_values().collect();
        results.sort_by(|a, b| {
            b.hybrid_score
                .partial_cmp(&a.hybrid_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Limit results
        if results.len() > self.config.max_results {
            results.truncate(self.config.max_results);
        }

        let total_searched = self.document_store.read().await.len();
        let keyword_matches = keyword_results.hits.len();
        let vector_matches = vector_results.map(|vr| vr.results.len()).unwrap_or(0);

        Ok(HybridSearchResults::new(
            results,
            total_searched,
            keyword_matches,
            vector_matches,
            query_time_ms,
            query_text,
        ))
    }

    /// Normalize scores based on the configured normalization strategy.
    fn normalize_scores(
        &self,
        results: &mut HashMap<u64, HybridSearchResult>,
        keyword_scores: &[f32],
        vector_similarities: &[f32],
    ) -> Result<()> {
        match self.config.normalization {
            ScoreNormalization::None => {
                // No normalization needed
            }
            ScoreNormalization::MinMax => {
                self.normalize_min_max(results, keyword_scores, vector_similarities)?;
            }
            ScoreNormalization::ZScore => {
                self.normalize_z_score(results, keyword_scores, vector_similarities)?;
            }
            ScoreNormalization::Rank => {
                self.normalize_rank(results, keyword_scores, vector_similarities)?;
            }
        }

        Ok(())
    }

    /// Min-max normalization to [0, 1] range.
    fn normalize_min_max(
        &self,
        results: &mut HashMap<u64, HybridSearchResult>,
        keyword_scores: &[f32],
        vector_similarities: &[f32],
    ) -> Result<()> {
        // Normalize keyword scores
        if !keyword_scores.is_empty() {
            let min_keyword = keyword_scores.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max_keyword = keyword_scores
                .iter()
                .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let keyword_range = max_keyword - min_keyword;

            if keyword_range > 0.0 {
                for result in results.values_mut() {
                    if let Some(score) = result.keyword_score {
                        result.keyword_score = Some((score - min_keyword) / keyword_range);
                    }
                }
            }
        }

        // Normalize vector similarities
        if !vector_similarities.is_empty() {
            let min_vector = vector_similarities
                .iter()
                .fold(f32::INFINITY, |a, &b| a.min(b));
            let max_vector = vector_similarities
                .iter()
                .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let vector_range = max_vector - min_vector;

            if vector_range > 0.0 {
                for result in results.values_mut() {
                    if let Some(similarity) = result.vector_similarity {
                        result.vector_similarity = Some((similarity - min_vector) / vector_range);
                    }
                }
            }
        }

        Ok(())
    }

    /// Z-score normalization.
    fn normalize_z_score(
        &self,
        results: &mut HashMap<u64, HybridSearchResult>,
        keyword_scores: &[f32],
        vector_similarities: &[f32],
    ) -> Result<()> {
        // Calculate keyword statistics
        if !keyword_scores.is_empty() {
            let keyword_mean = keyword_scores.iter().sum::<f32>() / keyword_scores.len() as f32;
            let keyword_variance = keyword_scores
                .iter()
                .map(|&score| (score - keyword_mean).powi(2))
                .sum::<f32>()
                / keyword_scores.len() as f32;
            let keyword_std = keyword_variance.sqrt();

            if keyword_std > 0.0 {
                for result in results.values_mut() {
                    if let Some(score) = result.keyword_score {
                        let z_score = (score - keyword_mean) / keyword_std;
                        result.keyword_score = Some((z_score + 3.0) / 6.0); // Normalize to [0, 1] approximately
                    }
                }
            }
        }

        // Calculate vector statistics
        if !vector_similarities.is_empty() {
            let vector_mean =
                vector_similarities.iter().sum::<f32>() / vector_similarities.len() as f32;
            let vector_variance = vector_similarities
                .iter()
                .map(|&sim| (sim - vector_mean).powi(2))
                .sum::<f32>()
                / vector_similarities.len() as f32;
            let vector_std = vector_variance.sqrt();

            if vector_std > 0.0 {
                for result in results.values_mut() {
                    if let Some(similarity) = result.vector_similarity {
                        let z_score = (similarity - vector_mean) / vector_std;
                        result.vector_similarity = Some((z_score + 3.0) / 6.0); // Normalize to [0, 1] approximately
                    }
                }
            }
        }

        Ok(())
    }

    /// Rank-based normalization.
    fn normalize_rank(
        &self,
        results: &mut HashMap<u64, HybridSearchResult>,
        keyword_scores: &[f32],
        vector_similarities: &[f32],
    ) -> Result<()> {
        // Create rank mappings
        let keyword_ranks = self.create_rank_mapping(keyword_scores);
        let vector_ranks = self.create_rank_mapping(vector_similarities);

        // Apply keyword ranks
        for result in results.values_mut() {
            if let Some(score) = result.keyword_score {
                if let Some(&rank) = keyword_ranks.get(&((score * 1000000.0) as i32)) {
                    result.keyword_score = Some(rank);
                }
            }
        }

        // Apply vector ranks
        for result in results.values_mut() {
            if let Some(similarity) = result.vector_similarity {
                if let Some(&rank) = vector_ranks.get(&((similarity * 1000000.0) as i32)) {
                    result.vector_similarity = Some(rank);
                }
            }
        }

        Ok(())
    }

    /// Create rank mapping for scores.
    fn create_rank_mapping(&self, scores: &[f32]) -> HashMap<i32, f32> {
        let mut unique_scores: Vec<f32> = scores.to_vec();
        unique_scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        unique_scores.dedup();

        let mut rank_map = HashMap::new();
        for (rank, &score) in unique_scores.iter().enumerate() {
            let normalized_rank = 1.0 - (rank as f32 / unique_scores.len() as f32);
            rank_map.insert((score * 1000000.0) as i32, normalized_rank);
        }

        rank_map
    }

    /// Add document content to results.
    async fn add_document_content(
        &self,
        results: &mut HashMap<u64, HybridSearchResult>,
    ) -> Result<()> {
        let store = self.document_store.read().await;

        for (doc_id, result) in results.iter_mut() {
            if let Some(document) = store.get(doc_id) {
                result.document = Some(document.clone());
            }
        }

        Ok(())
    }

    /// Get statistics about the hybrid search engine.
    pub async fn stats(&self) -> HybridSearchStats {
        let document_count = self.document_store.read().await.len();
        let embedder_trained = self.is_embedder_trained().await;

        HybridSearchStats {
            total_documents: document_count,
            vector_index_size: 0, // TODO: Get from vector search engine
            embedder_trained,
            embedding_dimension: self.config.embedding_config.dimension,
            vector_memory_usage: 0, // TODO: Get from vector search engine
        }
    }

    /// Clear all indexed data.
    pub async fn clear(&mut self) -> Result<()> {
        // TODO: Clear vector search engine
        self.document_store.write().await.clear();
        Ok(())
    }
}

/// Statistics for hybrid search engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridSearchStats {
    /// Total number of documents indexed.
    pub total_documents: usize,
    /// Number of vectors in the vector index.
    pub vector_index_size: usize,
    /// Whether the embedder is trained.
    pub embedder_trained: bool,
    /// Embedding dimension.
    pub embedding_dimension: usize,
    /// Memory usage of vector index in bytes.
    pub vector_memory_usage: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::{SearchHit, SearchResults, TermQuery};
    use crate::search::{Search, SearchRequest};

    struct MockSearch {
        results: Vec<SearchHit>,
    }

    impl MockSearch {
        fn new(results: Vec<SearchHit>) -> Self {
            Self { results }
        }
    }

    impl Search for MockSearch {
        fn search(&self, _request: SearchRequest) -> Result<SearchResults> {
            Ok(SearchResults {
                hits: self.results.clone(),
                total_hits: self.results.len() as u64,
                max_score: self.results.first().map(|r| r.score).unwrap_or(0.0),
            })
        }

        fn count(&self, _query: Box<dyn Query>) -> Result<u64> {
            Ok(self.results.len() as u64)
        }
    }

    #[test]
    fn test_hybrid_search_config() {
        let config = HybridSearchConfig::default();
        assert_eq!(config.keyword_weight, 0.6);
        assert_eq!(config.vector_weight, 0.4);
        assert_eq!(config.max_results, 50);
        assert!(!config.require_both);
    }

    #[test]
    fn test_hybrid_search_result() {
        let result = HybridSearchResult::new(1, 0.8)
            .with_keyword_score(0.7)
            .with_vector_similarity(0.9);

        assert_eq!(result.doc_id, 1);
        assert_eq!(result.hybrid_score, 0.8);
        assert_eq!(result.keyword_score, Some(0.7));
        assert_eq!(result.vector_similarity, Some(0.9));
    }

    #[test]
    fn test_hybrid_search_results() {
        let mut results = HybridSearchResults::empty();
        assert!(results.is_empty());

        results.results.push(HybridSearchResult::new(1, 0.8));
        results.results.push(HybridSearchResult::new(2, 0.6));

        assert_eq!(results.len(), 2);
        assert!(!results.is_empty());
        assert_eq!(results.best_result().unwrap().doc_id, 1);

        results.filter_by_score(0.7);
        assert_eq!(results.len(), 1);

        results.limit(0);
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_hybrid_search_engine_creation() {
        let config = HybridSearchConfig::default();
        let engine = HybridSearchEngine::new(config);
        assert!(engine.is_ok());
    }

    #[tokio::test]
    async fn test_add_and_remove_document() {
        let config = HybridSearchConfig::default();
        let mut engine = HybridSearchEngine::new(config).unwrap();

        let mut fields = HashMap::new();
        fields.insert("title".to_string(), "Test Document".to_string());
        fields.insert("content".to_string(), "This is test content".to_string());

        assert!(engine.add_document(1, fields).await.is_ok());

        let stats = engine.stats().await;
        assert_eq!(stats.total_documents, 1);

        assert!(engine.remove_document(1).await.unwrap());
        assert!(!engine.remove_document(1).await.unwrap()); // Already removed

        let stats = engine.stats().await;
        assert_eq!(stats.total_documents, 0);
    }

    #[tokio::test]
    async fn test_embedder_training() {
        let config = HybridSearchConfig::default();
        let mut engine = HybridSearchEngine::new(config).unwrap();

        assert!(!engine.is_embedder_trained().await);

        let documents = vec!["This is a test document", "Another test document"];

        assert!(engine.train_embedder(&documents).await.is_ok());
        assert!(engine.is_embedder_trained().await);
    }

    #[test]
    fn test_score_normalization_strategies() {
        let config = HybridSearchConfig {
            normalization: ScoreNormalization::MinMax,
            ..Default::default()
        };
        let engine = HybridSearchEngine::new(config).unwrap();

        let mut results = HashMap::new();
        results.insert(1, HybridSearchResult::new(1, 0.0).with_keyword_score(0.8));
        results.insert(2, HybridSearchResult::new(2, 0.0).with_keyword_score(0.4));

        let keyword_scores = vec![0.8, 0.4];
        let vector_similarities = vec![];

        assert!(
            engine
                .normalize_scores(&mut results, &keyword_scores, &vector_similarities)
                .is_ok()
        );

        // After min-max normalization, scores should be in [0, 1] range
        assert_eq!(results.get(&1).unwrap().keyword_score, Some(1.0));
        assert_eq!(results.get(&2).unwrap().keyword_score, Some(0.0));
    }

    #[tokio::test]
    async fn test_hybrid_search_with_mock() {
        let config = HybridSearchConfig::default();
        let engine = HybridSearchEngine::new(config).unwrap();

        let keyword_results = vec![
            SearchHit {
                doc_id: 1,
                score: 0.8,
                document: None,
            },
            SearchHit {
                doc_id: 2,
                score: 0.6,
                document: None,
            },
        ];

        let mock_searcher = MockSearch::new(keyword_results);
        let query = Box::new(TermQuery::new("title", "test"));

        let results = engine.search("test query", &mock_searcher, query).await;
        assert!(results.is_ok());

        let results = results.unwrap();
        assert!(!results.is_empty());
        assert_eq!(results.keyword_matches, 2);
        assert_eq!(results.vector_matches, 0); // No trained embedder
    }

    #[test]
    fn test_extract_text_content() {
        let config = HybridSearchConfig::default();
        let engine = HybridSearchEngine::new(config).unwrap();

        let mut fields = HashMap::new();
        fields.insert("title".to_string(), "Test Title".to_string());
        fields.insert("content".to_string(), "Test Content".to_string());

        let text = engine.extract_text_content(&fields);
        assert!(text.contains("Test Title"));
        assert!(text.contains("Test Content"));
    }

    #[tokio::test]
    async fn test_hybrid_search_stats() {
        let config = HybridSearchConfig::default();
        let engine = HybridSearchEngine::new(config).unwrap();

        let stats = engine.stats().await;
        assert_eq!(stats.total_documents, 0);
        assert_eq!(stats.vector_index_size, 0);
        assert!(!stats.embedder_trained);
        assert_eq!(stats.embedding_dimension, 128); // Default dimension
    }

    #[tokio::test]
    async fn test_clear_engine() {
        let config = HybridSearchConfig::default();
        let mut engine = HybridSearchEngine::new(config).unwrap();

        let mut fields = HashMap::new();
        fields.insert("title".to_string(), "Test".to_string());
        engine.add_document(1, fields).await.unwrap();

        assert_eq!(engine.stats().await.total_documents, 1);

        engine.clear().await.unwrap();
        assert_eq!(engine.stats().await.total_documents, 0);
    }
}
