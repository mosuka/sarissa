//! Parallel hybrid search engine implementation.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::RwLock;

use crate::embedding::engine::EmbeddingEngine;
use crate::error::{Result, SageError};
use crate::lexical::reader::IndexReader;
use crate::parallel_hybrid_search::config::{LoadBalancingStrategy, ParallelHybridSearchConfig};
use crate::parallel_hybrid_search::executor::ParallelHybridSearchExecutor;
use crate::parallel_hybrid_search::types::{
    CacheStats, ParallelHybridSearchResults, SearchTimeBreakdown,
};
use crate::query::query::Query;
use crate::vector::reader::VectorIndexReader;

/// Index handle containing both keyword and vector readers.
pub struct HybridIndexHandle {
    /// Index identifier.
    pub id: String,

    /// Keyword index reader.
    pub keyword_reader: Arc<dyn IndexReader>,

    /// Vector index reader (optional).
    pub vector_reader: Option<Arc<dyn VectorIndexReader>>,

    /// Index weight for result ranking.
    pub weight: f32,

    /// Whether this index is active.
    pub active: bool,
}

/// Statistics for tracking index load.
#[derive(Debug, Clone)]
struct IndexLoadStats {
    /// Number of active searches.
    active_searches: usize,
    /// Average search time in milliseconds.
    avg_search_time_ms: f64,
}

impl Default for IndexLoadStats {
    fn default() -> Self {
        Self {
            active_searches: 0,
            avg_search_time_ms: 0.0,
        }
    }
}

/// Parallel hybrid search engine combining keyword and vector search.
pub struct ParallelHybridSearchEngine {
    /// Configuration.
    config: ParallelHybridSearchConfig,

    /// Executor for parallel search tasks.
    executor: Arc<ParallelHybridSearchExecutor>,

    /// Index handles.
    indices: Arc<RwLock<HashMap<String, HybridIndexHandle>>>,

    /// Text embedder for query vectorization.
    embedder: Arc<RwLock<EmbeddingEngine>>,

    /// Document store for retrieving full documents.
    document_store: Arc<RwLock<HashMap<u64, HashMap<String, String>>>>,

    /// Result cache.
    result_cache: Arc<RwLock<HashMap<String, ParallelHybridSearchResults>>>,

    /// Cache statistics.
    cache_stats: Arc<RwLock<CacheStats>>,

    /// Round-robin index counter.
    next_index: Arc<RwLock<usize>>,

    /// Load statistics for each index.
    index_load_stats: Arc<RwLock<HashMap<String, IndexLoadStats>>>,
}

impl ParallelHybridSearchEngine {
    /// Create a new parallel hybrid search engine.
    pub fn new(config: ParallelHybridSearchConfig) -> Result<Self> {
        config.validate()?;

        let executor = Arc::new(ParallelHybridSearchExecutor::new(config.clone())?);
        let embedder = Arc::new(RwLock::new(EmbeddingEngine::new(Default::default())?));

        Ok(Self {
            config,
            executor,
            indices: Arc::new(RwLock::new(HashMap::new())),
            embedder,
            document_store: Arc::new(RwLock::new(HashMap::new())),
            result_cache: Arc::new(RwLock::new(HashMap::new())),
            cache_stats: Arc::new(RwLock::new(CacheStats {
                keyword_hits: 0,
                keyword_misses: 0,
                vector_hits: 0,
                vector_misses: 0,
            })),
            next_index: Arc::new(RwLock::new(0)),
            index_load_stats: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Add an index to the engine.
    pub async fn add_index(
        &self,
        id: String,
        keyword_reader: Arc<dyn IndexReader>,
        vector_reader: Option<Arc<dyn VectorIndexReader>>,
        weight: f32,
    ) -> Result<()> {
        let handle = HybridIndexHandle {
            id: id.clone(),
            keyword_reader,
            vector_reader,
            weight,
            active: true,
        };

        let mut indices = self.indices.write().await;
        indices.insert(id, handle);
        Ok(())
    }

    /// Remove an index from the engine.
    pub async fn remove_index(&self, id: &str) -> Result<()> {
        let mut indices = self.indices.write().await;
        indices
            .remove(id)
            .ok_or_else(|| SageError::internal(format!("Index '{id}' not found")))?;
        Ok(())
    }

    /// Train the embedder on a collection of documents.
    pub async fn train_embedder(&self, documents: &[&str]) -> Result<()> {
        let mut embedder = self.embedder.write().await;
        embedder.train(documents).await?;
        Ok(())
    }

    /// Check if the embedder is trained.
    pub async fn is_embedder_trained(&self) -> bool {
        let embedder = self.embedder.read().await;
        embedder.is_trained()
    }

    /// Add a document to the document store.
    pub async fn add_document(&self, doc_id: u64, fields: HashMap<String, String>) -> Result<()> {
        let mut store = self.document_store.write().await;
        store.insert(doc_id, fields);
        Ok(())
    }

    /// Execute a parallel hybrid search.
    pub async fn search(
        &self,
        query_text: &str,
        keyword_query: Box<dyn Query>,
    ) -> Result<ParallelHybridSearchResults> {
        let start_time = Instant::now();

        // Check cache first
        if self.config.enable_result_caching {
            let cache_key = self.compute_cache_key(query_text);
            let cache = self.result_cache.read().await;
            if let Some(cached_result) = cache.get(&cache_key) {
                self.update_cache_stats(true, true).await;
                return Ok(cached_result.clone());
            }
        }

        // Query expansion timing
        let expansion_start = Instant::now();

        // TODO: Implement query expansion here
        // For now, we just use the original query
        let expanded_keyword_query = keyword_query;

        let expansion_time_ms = expansion_start.elapsed().as_secs_f64() * 1000.0;

        // Generate query vector if embedder is trained
        let query_vector = if self.is_embedder_trained().await {
            let embedder = self.embedder.read().await;
            Some(embedder.embed(query_text)?)
        } else {
            None
        };

        // Get active indices
        let indices = self.get_active_indices().await?;
        if indices.is_empty() {
            return Ok(ParallelHybridSearchResults {
                results: Vec::new(),
                total_keyword_matches: 0,
                total_vector_matches: 0,
                indices_searched: 0,
                search_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
                time_breakdown: SearchTimeBreakdown {
                    keyword_search_ms: 0.0,
                    vector_search_ms: 0.0,
                    merge_ms: 0.0,
                    expansion_ms: 0.0,
                    ranking_ms: 0.0,
                },
                cache_stats: self.get_cache_stats().await,
                index_stats: Vec::new(),
            });
        }

        // Execute parallel search
        let document_store = self.document_store.read().await;

        // Set indices in executor
        self.executor.set_indices(indices.clone());

        let mut results = self
            .executor
            .execute_parallel_search(
                query_text,
                expanded_keyword_query,
                query_vector,
                indices,
                &document_store,
            )
            .await?;

        // Add expansion time to the results
        results.time_breakdown.expansion_ms = expansion_time_ms;

        // Cache the results
        if self.config.enable_result_caching {
            let cache_key = self.compute_cache_key(query_text);
            let mut cache = self.result_cache.write().await;
            if cache.len() < self.config.cache_size_limit {
                cache.insert(cache_key, results.clone());
            }
        }

        Ok(results)
    }

    /// Batch search for multiple queries.
    pub async fn batch_search(
        &self,
        queries: Vec<(&str, Box<dyn Query>)>,
    ) -> Result<Vec<ParallelHybridSearchResults>> {
        let mut results = Vec::with_capacity(queries.len());

        // Process queries in batches
        for batch in queries.chunks(self.config.batch_size) {
            let batch_futures: Vec<_> = batch
                .iter()
                .map(|(query_text, keyword_query)| {
                    self.search(query_text, keyword_query.clone_box())
                })
                .collect();

            let batch_results = futures::future::join_all(batch_futures).await;

            for result in batch_results {
                results.push(result?);
            }
        }

        Ok(results)
    }

    /// Clear all caches.
    pub async fn clear_caches(&self) {
        self.result_cache.write().await.clear();
        *self.cache_stats.write().await = CacheStats {
            keyword_hits: 0,
            keyword_misses: 0,
            vector_hits: 0,
            vector_misses: 0,
        };
    }

    /// Get current cache statistics.
    pub async fn get_cache_stats(&self) -> CacheStats {
        self.cache_stats.read().await.clone()
    }

    /// Get active indices based on load balancing strategy.
    async fn get_active_indices(&self) -> Result<Vec<HybridIndexHandle>> {
        let indices = self.indices.read().await;
        let active_indices: Vec<_> = indices
            .values()
            .filter(|h| h.active)
            .cloned()
            .map(|h| HybridIndexHandle {
                id: h.id,
                keyword_reader: h.keyword_reader.clone(),
                vector_reader: h.vector_reader.clone(),
                weight: h.weight,
                active: h.active,
            })
            .collect();

        if active_indices.is_empty() {
            return Ok(Vec::new());
        }

        // Apply load balancing strategy
        match self.config.load_balancing_strategy {
            LoadBalancingStrategy::RoundRobin => {
                let mut next_index = self.next_index.write().await;
                let start_idx = *next_index % active_indices.len();
                *next_index = (*next_index + 1) % active_indices.len();

                let mut selected = Vec::new();
                for i in 0..self.config.max_concurrent_indices.min(active_indices.len()) {
                    let idx = (start_idx + i) % active_indices.len();
                    selected.push(active_indices[idx].clone());
                }
                Ok(selected)
            }
            LoadBalancingStrategy::Random => {
                use rand::seq::SliceRandom;
                let mut rng = rand::rng();
                let mut selected = active_indices;
                selected.shuffle(&mut rng);
                selected.truncate(self.config.max_concurrent_indices);
                Ok(selected)
            }
            LoadBalancingStrategy::LeastLoaded => {
                // Select indices with the least active searches
                let load_stats = self.index_load_stats.read().await;
                let mut scored_indices: Vec<_> = active_indices
                    .into_iter()
                    .map(|handle| {
                        let stats = load_stats.get(&handle.id).cloned().unwrap_or_default();
                        let load_score =
                            stats.active_searches as f64 + (stats.avg_search_time_ms / 1000.0); // Normalize to seconds
                        (handle, load_score)
                    })
                    .collect();

                // Sort by load score (ascending - less loaded first)
                scored_indices
                    .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

                let selected = scored_indices
                    .into_iter()
                    .take(self.config.max_concurrent_indices)
                    .map(|(handle, _)| handle)
                    .collect();
                Ok(selected)
            }
            LoadBalancingStrategy::IndexAware => {
                // Select indices based on their characteristics (doc count, weight)
                let mut scored_indices: Vec<_> = active_indices
                    .into_iter()
                    .map(|handle| {
                        // Calculate a score based on:
                        // - Index weight (higher weight = prefer this index)
                        // - Document count (more docs = more comprehensive)
                        let doc_count = handle.keyword_reader.doc_count();
                        let has_vector = handle.vector_reader.is_some();

                        // Composite score: weight is primary, doc count and vector presence are secondary
                        let score = handle.weight
                            + (doc_count as f32 / 10000.0) // Normalize doc count
                            + (if has_vector { 0.2 } else { 0.0 }); // Bonus for vector support

                        (handle, score)
                    })
                    .collect();

                // Sort by score (descending - higher score first)
                scored_indices
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                let selected = scored_indices
                    .into_iter()
                    .take(self.config.max_concurrent_indices)
                    .map(|(handle, _)| handle)
                    .collect();
                Ok(selected)
            }
        }
    }

    /// Compute cache key for a query.
    fn compute_cache_key(&self, query_text: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        query_text.hash(&mut hasher);
        self.config.keyword_weight.to_bits().hash(&mut hasher);
        self.config.vector_weight.to_bits().hash(&mut hasher);
        self.config.merge_strategy.hash(&mut hasher);

        format!("{:x}", hasher.finish())
    }

    /// Update cache statistics.
    async fn update_cache_stats(&self, keyword_hit: bool, vector_hit: bool) {
        let mut stats = self.cache_stats.write().await;
        if keyword_hit {
            stats.keyword_hits += 1;
        } else {
            stats.keyword_misses += 1;
        }
        if vector_hit {
            stats.vector_hits += 1;
        } else {
            stats.vector_misses += 1;
        }
    }
}

impl Clone for HybridIndexHandle {
    fn clone(&self) -> Self {
        Self {
            id: self.id.clone(),
            keyword_reader: self.keyword_reader.clone(),
            vector_reader: self.vector_reader.clone(),
            weight: self.weight,
            active: self.active,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexical::index::reader::inverted_index::{InvertedIndexReader, InvertedIndexReaderConfig};
    use crate::query::term::TermQuery;
    use crate::storage::memory::MemoryStorage;
    use crate::storage::traits::StorageConfig;

    fn create_test_keyword_reader() -> Arc<dyn IndexReader> {
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        Arc::new(
            InvertedIndexReader::new(vec![], storage, InvertedIndexReaderConfig::default()).unwrap(),
        )
    }

    #[tokio::test]
    async fn test_engine_creation() {
        let config = ParallelHybridSearchConfig::default();
        let engine = ParallelHybridSearchEngine::new(config);
        assert!(engine.is_ok());
    }

    #[tokio::test]
    async fn test_add_remove_index() {
        let config = ParallelHybridSearchConfig::default();
        let engine = ParallelHybridSearchEngine::new(config).unwrap();

        let reader = create_test_keyword_reader();
        assert!(
            engine
                .add_index("index1".to_string(), reader, None, 1.0)
                .await
                .is_ok()
        );

        assert!(engine.remove_index("index1").await.is_ok());
        assert!(engine.remove_index("index1").await.is_err());
    }

    #[tokio::test]
    async fn test_empty_search() {
        let config = ParallelHybridSearchConfig::default();
        let engine = ParallelHybridSearchEngine::new(config).unwrap();

        let query = Box::new(TermQuery::new("text", "test"));
        let results = engine.search("test query", query).await.unwrap();

        assert!(results.is_empty());
        assert_eq!(results.indices_searched, 0);
    }

    #[tokio::test]
    async fn test_cache_key_generation() {
        let config = ParallelHybridSearchConfig::default();
        let engine = ParallelHybridSearchEngine::new(config).unwrap();

        let key1 = engine.compute_cache_key("test query");
        let key2 = engine.compute_cache_key("test query");
        let key3 = engine.compute_cache_key("different query");

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }

    #[tokio::test]
    async fn test_load_balancing_strategies() {
        use super::LoadBalancingStrategy;

        // Test RoundRobin
        let config = ParallelHybridSearchConfig {
            load_balancing_strategy: LoadBalancingStrategy::RoundRobin,
            max_concurrent_indices: 2,
            ..Default::default()
        };
        let engine = ParallelHybridSearchEngine::new(config).unwrap();

        // Add multiple indices
        for i in 0..4 {
            let reader = create_test_keyword_reader();
            engine
                .add_index(format!("index_{}", i), reader, None, 1.0)
                .await
                .unwrap();
        }

        let indices1 = engine.get_active_indices().await.unwrap();
        let indices2 = engine.get_active_indices().await.unwrap();

        assert_eq!(indices1.len(), 2);
        assert_eq!(indices2.len(), 2);
        // Round-robin should rotate
        assert_ne!(indices1[0].id, indices2[0].id);
    }

    #[tokio::test]
    async fn test_hybrid_search_with_multiple_indices() {
        let config = ParallelHybridSearchConfig::default();
        let engine = ParallelHybridSearchEngine::new(config).unwrap();

        // Add multiple indices
        let reader1 = create_test_keyword_reader();
        let reader2 = create_test_keyword_reader();

        engine
            .add_index("index1".to_string(), reader1, None, 1.0)
            .await
            .unwrap();
        engine
            .add_index("index2".to_string(), reader2, None, 0.8)
            .await
            .unwrap();

        // Perform search
        let query = Box::new(TermQuery::new("content", "test"));
        let results = engine.search("test query", query).await.unwrap();

        // Should have searched across indices (even if no results)
        assert!(results.indices_searched > 0);
        assert!(results.search_time_ms >= 0.0);
        assert!(results.time_breakdown.keyword_search_ms >= 0.0);
        assert!(results.time_breakdown.expansion_ms >= 0.0);
        assert!(results.time_breakdown.ranking_ms >= 0.0);
    }

    #[tokio::test]
    async fn test_result_caching() {
        let config = ParallelHybridSearchConfig {
            enable_result_caching: true,
            ..Default::default()
        };
        let engine = ParallelHybridSearchEngine::new(config).unwrap();

        let reader = create_test_keyword_reader();
        engine
            .add_index("index1".to_string(), reader, None, 1.0)
            .await
            .unwrap();

        // First search
        let query1 = Box::new(TermQuery::new("content", "test"));
        let results1 = engine.search("test query", query1).await.unwrap();

        // Second search with same query (should hit cache)
        let query2 = Box::new(TermQuery::new("content", "test"));
        let results2 = engine.search("test query", query2).await.unwrap();

        // Results should be identical
        assert_eq!(
            results1.total_keyword_matches,
            results2.total_keyword_matches
        );
        assert_eq!(results1.indices_searched, results2.indices_searched);

        // Cache stats should show hits
        let cache_stats = engine.get_cache_stats().await;
        assert!(cache_stats.keyword_hits > 0 || cache_stats.vector_hits > 0);
    }

    #[tokio::test]
    async fn test_batch_search() {
        let config = ParallelHybridSearchConfig::default();
        let engine = ParallelHybridSearchEngine::new(config).unwrap();

        let reader = create_test_keyword_reader();
        engine
            .add_index("index1".to_string(), reader, None, 1.0)
            .await
            .unwrap();

        // Prepare multiple queries
        let queries = vec![
            (
                "query1",
                Box::new(TermQuery::new("content", "rust")) as Box<dyn Query>,
            ),
            (
                "query2",
                Box::new(TermQuery::new("content", "python")) as Box<dyn Query>,
            ),
            (
                "query3",
                Box::new(TermQuery::new("content", "java")) as Box<dyn Query>,
            ),
        ];

        let results = engine.batch_search(queries).await.unwrap();

        assert_eq!(results.len(), 3);
        for result in results {
            assert!(result.search_time_ms >= 0.0);
        }
    }

    #[tokio::test]
    async fn test_index_aware_load_balancing() {
        use super::LoadBalancingStrategy;

        let config = ParallelHybridSearchConfig {
            load_balancing_strategy: LoadBalancingStrategy::IndexAware,
            max_concurrent_indices: 2,
            ..Default::default()
        };
        let engine = ParallelHybridSearchEngine::new(config).unwrap();

        // Add indices with different weights
        let reader1 = create_test_keyword_reader();
        let reader2 = create_test_keyword_reader();
        let reader3 = create_test_keyword_reader();

        engine
            .add_index("high_weight".to_string(), reader1, None, 2.0)
            .await
            .unwrap();
        engine
            .add_index("medium_weight".to_string(), reader2, None, 1.0)
            .await
            .unwrap();
        engine
            .add_index("low_weight".to_string(), reader3, None, 0.5)
            .await
            .unwrap();

        let indices = engine.get_active_indices().await.unwrap();

        assert_eq!(indices.len(), 2);
        // Should prefer higher-weighted indices
        assert!(indices.iter().any(|i| i.id == "high_weight"));
    }
}
