//! Core hybrid index implementation.
//!
//! This module provides the core data structure that manages both lexical and vector
//! indexes for hybrid search.

use std::sync::Arc;

use crate::error::Result;
use crate::lexical::reader::IndexReader;
use crate::vector::reader::VectorIndexReader;

/// Hybrid index that combines lexical and vector indexes.
///
/// This structure manages both a lexical (inverted) index for keyword search
/// and a vector index for semantic search, providing a unified interface
/// for hybrid search operations.
pub struct HybridIndex {
    /// Lexical index reader for keyword-based search
    pub lexical_index: Arc<dyn IndexReader>,
    /// Vector index reader for semantic search
    pub vector_index: Arc<dyn VectorIndexReader>,
}

impl HybridIndex {
    /// Create a new hybrid index from existing lexical and vector index readers.
    ///
    /// # Arguments
    ///
    /// * `lexical_index` - The lexical index reader for keyword search
    /// * `vector_index` - The vector index reader for semantic search
    ///
    /// # Returns
    ///
    /// A new `HybridIndex` instance
    pub fn new(
        lexical_index: Arc<dyn IndexReader>,
        vector_index: Arc<dyn VectorIndexReader>,
    ) -> Self {
        Self {
            lexical_index,
            vector_index,
        }
    }

    /// Get a reference to the lexical index reader.
    pub fn lexical_index(&self) -> &dyn IndexReader {
        self.lexical_index.as_ref()
    }

    /// Get a reference to the vector index reader.
    pub fn vector_index(&self) -> &dyn VectorIndexReader {
        self.vector_index.as_ref()
    }

    /// Get statistics about the hybrid index.
    pub fn stats(&self) -> Result<HybridIndexStats> {
        Ok(HybridIndexStats {
            lexical_doc_count: self.lexical_index.doc_count(),
            vector_doc_count: self.vector_index.vector_count() as u64,
        })
    }
}

/// Statistics about a hybrid index.
#[derive(Debug, Clone)]
pub struct HybridIndexStats {
    /// Number of documents in the lexical index
    pub lexical_doc_count: u64,
    /// Number of vectors in the vector index
    pub vector_doc_count: u64,
}
