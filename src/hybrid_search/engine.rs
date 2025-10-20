//! Hybrid search engine implementation.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::RwLock;

use super::config::HybridSearchConfig;
use super::merger::ResultMerger;
use super::stats::HybridSearchStats;
use super::types::HybridSearchResults;
use crate::embeding::engine::EmbeddingEngine;
use crate::error::Result;
use crate::lexical::search::SearchRequest;
use crate::query::SearchResults;
use crate::query::query::Query;
use crate::vector::types::{VectorSearchResult, VectorSearchResults};
use crate::vector::{DistanceMetric, Vector};

/// Trait for searchable types in hybrid search.
pub trait Searchable: Send {
    fn search(&self, request: SearchRequest) -> Result<SearchResults>;
}

/// Hybrid search engine that combines keyword and vector search.
pub struct HybridSearchEngine {
    /// Configuration for hybrid search.
    config: HybridSearchConfig,
    /// Text embedder for converting queries to vectors.
    embedder: Arc<RwLock<EmbeddingEngine>>,
    /// Document storage for retrieving full documents.
    document_store: Arc<RwLock<HashMap<u64, HashMap<String, String>>>>,
    /// Vector storage for similarity search.
    vector_store: Arc<RwLock<HashMap<u64, Vector>>>,
    /// Result merger for combining search results.
    merger: ResultMerger,
}

impl HybridSearchEngine {
    /// Create a new hybrid search engine.
    pub fn new(config: HybridSearchConfig) -> Result<Self> {
        let embedder = EmbeddingEngine::new(config.embedding_config.clone())?;
        let merger = ResultMerger::new(config.clone());

        Ok(Self {
            config,
            embedder: Arc::new(RwLock::new(embedder)),
            document_store: Arc::new(RwLock::new(HashMap::new())),
            vector_store: Arc::new(RwLock::new(HashMap::new())),
            merger,
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

        // Generate embedding and store vector
        {
            let embedder = self.embedder.read().await;
            if embedder.is_trained() {
                let vector = embedder.embed(&text_content)?;
                drop(embedder);

                // Store the vector for similarity search
                let mut vector_store = self.vector_store.write().await;
                vector_store.insert(doc_id, vector);
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

        // Remove from vector store
        {
            let mut vector_store = self.vector_store.write().await;
            vector_store.remove(&doc_id);
        }

        Ok(existed)
    }

    /// Perform hybrid search combining keyword and vector search.
    pub async fn search<S: Searchable>(
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
            let query_vector = embedder.embed(query_text)?;
            drop(embedder);

            let vector_config = self.config.vector_config.clone();

            // Perform similarity search on stored vectors
            Some(
                self.vector_similarity_search(&query_vector, &vector_config)
                    .await?,
            )
        } else {
            None
        };

        // Merge and rank results
        let document_store = self.document_store.read().await;
        let hybrid_results = self
            .merger
            .merge_results(
                keyword_results,
                vector_results,
                query_text.to_string(),
                start_time.elapsed().as_millis() as u64,
                &document_store,
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

    /// Perform vector similarity search on stored vectors.
    async fn vector_similarity_search(
        &self,
        query_vector: &Vector,
        config: &crate::vector::types::VectorSearchConfig,
    ) -> Result<VectorSearchResults> {
        let vector_store = self.vector_store.read().await;
        let mut similarities = Vec::new();

        // Calculate similarity for each stored vector
        for (doc_id, doc_vector) in vector_store.iter() {
            let similarity =
                DistanceMetric::Cosine.similarity(&query_vector.data, &doc_vector.data)?;
            let distance = DistanceMetric::Cosine.distance(&query_vector.data, &doc_vector.data)?;

            if similarity >= config.min_similarity {
                similarities.push(VectorSearchResult {
                    doc_id: *doc_id,
                    similarity,
                    distance,
                    vector: if config.include_vectors {
                        Some(doc_vector.clone())
                    } else {
                        None
                    },
                    metadata: HashMap::new(),
                });
            }
        }

        // Sort by similarity (descending)
        similarities.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Limit results
        similarities.truncate(config.top_k);

        Ok(VectorSearchResults {
            results: similarities,
            candidates_examined: vector_store.len(),
            search_time_ms: 0.0, // We could measure this if needed
            query_metadata: HashMap::new(),
        })
    }

    /// Get statistics about the hybrid search engine.
    pub async fn stats(&self) -> HybridSearchStats {
        let document_count = self.document_store.read().await.len();
        let embedder_trained = self.is_embedder_trained().await;

        let vector_count = self.vector_store.read().await.len();

        HybridSearchStats {
            total_documents: document_count,
            vector_index_size: vector_count,
            embedder_trained,
            embedding_dimension: self.config.embedding_config.dimension,
            vector_memory_usage: vector_count * self.config.embedding_config.dimension * 4, // Approximate bytes
        }
    }

    /// Clear all indexed data.
    pub async fn clear(&mut self) -> Result<()> {
        self.document_store.write().await.clear();
        self.vector_store.write().await.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexical::search::SearchRequest;
    use crate::query::term::TermQuery;
    use crate::query::{SearchHit, SearchResults};

    struct MockSearch {
        results: Vec<SearchHit>,
    }

    impl MockSearch {
        fn new(results: Vec<SearchHit>) -> Self {
            Self { results }
        }
    }

    impl Searchable for MockSearch {
        fn search(&self, _request: SearchRequest) -> Result<SearchResults> {
            Ok(SearchResults {
                hits: self.results.clone(),
                total_hits: self.results.len() as u64,
                max_score: self.results.first().map(|r| r.score).unwrap_or(0.0),
            })
        }
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

    #[tokio::test]
    async fn test_search_without_embedder() {
        let config = HybridSearchConfig::default();
        let engine = HybridSearchEngine::new(config).unwrap();

        let mock_searcher = MockSearch::new(vec![SearchHit {
            doc_id: 1,
            score: 0.8,
            document: None,
        }]);

        let query = Box::new(TermQuery::new("content", "test"));
        let results = engine.search("test", &mock_searcher, query).await.unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results.keyword_matches, 1);
        assert_eq!(results.vector_matches, 0);
        assert!(!engine.is_embedder_trained().await);
    }
}
