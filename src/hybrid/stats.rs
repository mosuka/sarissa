//! Statistics and monitoring for hybrid search.
//!
//! This module provides structures for tracking and reporting statistics
//! about hybrid search engine state and performance.

use serde::{Deserialize, Serialize};

/// Statistics for hybrid search engine.
///
/// Contains information about the current state of the hybrid search engine,
/// including document counts, index sizes, and memory usage.
///
/// # Examples
///
/// ```
/// use sarissa::hybrid::stats::HybridSearchStats;
///
/// let stats = HybridSearchStats {
///     total_documents: 1000,
///     vector_index_size: 500,
///     embedder_trained: true,
///     embedding_dimension: 384,
///     vector_memory_usage: 1024 * 1024,  // 1MB
/// };
///
/// assert_eq!(stats.total_documents, 1000);
/// assert_eq!(stats.embedding_dimension, 384);
/// ```
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

impl Default for HybridSearchStats {
    fn default() -> Self {
        Self {
            total_documents: 0,
            vector_index_size: 0,
            embedder_trained: false,
            embedding_dimension: 128,
            vector_memory_usage: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hybrid_search_stats_default() {
        let stats = HybridSearchStats::default();
        assert_eq!(stats.total_documents, 0);
        assert_eq!(stats.vector_index_size, 0);
        assert!(!stats.embedder_trained);
        assert_eq!(stats.embedding_dimension, 128);
        assert_eq!(stats.vector_memory_usage, 0);
    }

    #[test]
    fn test_hybrid_search_stats_creation() {
        let stats = HybridSearchStats {
            total_documents: 1000,
            vector_index_size: 500,
            embedder_trained: true,
            embedding_dimension: 256,
            vector_memory_usage: 1024 * 1024, // 1MB
        };

        assert_eq!(stats.total_documents, 1000);
        assert_eq!(stats.vector_index_size, 500);
        assert!(stats.embedder_trained);
        assert_eq!(stats.embedding_dimension, 256);
        assert_eq!(stats.vector_memory_usage, 1024 * 1024);
    }

    #[test]
    fn test_hybrid_search_stats_clone() {
        let stats = HybridSearchStats {
            total_documents: 100,
            vector_index_size: 50,
            embedder_trained: true,
            embedding_dimension: 128,
            vector_memory_usage: 512,
        };

        let cloned = stats.clone();
        assert_eq!(stats.total_documents, cloned.total_documents);
        assert_eq!(stats.vector_index_size, cloned.vector_index_size);
        assert_eq!(stats.embedder_trained, cloned.embedder_trained);
        assert_eq!(stats.embedding_dimension, cloned.embedding_dimension);
        assert_eq!(stats.vector_memory_usage, cloned.vector_memory_usage);
    }
}
