//! Hybrid search engine implementation.
//!
//! This module provides the `HybridEngine` that combines lexical and vector search
//! engines to provide unified hybrid search functionality.

use crate::error::Result;

/// High-level hybrid search engine combining lexical and vector search.
///
/// This engine wraps both `LexicalEngine` and `VectorEngine` to provide
/// unified hybrid search functionality. It follows the same pattern as the
/// individual engines but coordinates searches across both indexes.
///
/// # Examples
///
/// ```no_run
/// use yatagarasu::hybrid::engine::HybridEngine;
/// use yatagarasu::hybrid::search::searcher::HybridSearchRequest;
/// use yatagarasu::lexical::engine::LexicalEngine;
/// use yatagarasu::vector::engine::VectorEngine;
/// use yatagarasu::vector::Vector;
///
/// # async fn example(lexical_engine: LexicalEngine, vector_engine: VectorEngine) -> yatagarasu::error::Result<()> {
/// // Create hybrid engine from existing engines
/// let engine = HybridEngine::new(lexical_engine, vector_engine)?;
///
/// // Text-only search
/// let request = HybridSearchRequest::new("rust programming");
/// let results = engine.search(request).await?;
///
/// // Hybrid search with vector
/// let vector = Vector::new(vec![1.0, 2.0, 3.0]);
/// let request = HybridSearchRequest::new("machine learning")
///     .with_vector(vector)
///     .keyword_weight(0.7)
///     .vector_weight(0.3);
/// let results = engine.search(request).await?;
/// # Ok(())
/// # }
/// ```
pub struct HybridEngine {
    /// Lexical search engine for keyword-based search.
    lexical_engine: crate::lexical::engine::LexicalEngine,
    /// Vector search engine for semantic search.
    vector_engine: crate::vector::engine::VectorEngine,
    /// Next document ID counter for synchronized ID assignment.
    next_doc_id: u64,
}

impl HybridEngine {
    /// Create a new hybrid search engine.
    ///
    /// # Arguments
    ///
    /// * `lexical_engine` - The lexical search engine
    /// * `vector_engine` - The vector search engine
    ///
    /// # Returns
    ///
    /// A new `HybridEngine` instance
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use yatagarasu::hybrid::engine::HybridEngine;
    /// use yatagarasu::lexical::engine::LexicalEngine;
    /// use yatagarasu::vector::engine::VectorEngine;
    ///
    /// # fn example(lexical_engine: LexicalEngine, vector_engine: VectorEngine) -> yatagarasu::error::Result<()> {
    /// let engine = HybridEngine::new(lexical_engine, vector_engine)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(
        lexical_engine: crate::lexical::engine::LexicalEngine,
        vector_engine: crate::vector::engine::VectorEngine,
    ) -> Result<Self> {
        Ok(Self {
            lexical_engine,
            vector_engine,
            next_doc_id: 0,
        })
    }

    /// Add a document to both lexical and vector indexes.
    /// Returns the assigned document ID.
    ///
    /// This method ensures that the same document ID is used in both indexes.
    /// The document should contain both text fields (for lexical indexing) and
    /// vector fields (for vector indexing).
    ///
    /// # Arguments
    ///
    /// * `doc` - The document to add (containing both text and vector fields)
    ///
    /// # Returns
    ///
    /// The assigned document ID
    pub async fn add_document(&mut self, doc: crate::document::document::Document) -> Result<u64> {
        let doc_id = self.next_doc_id;
        self.add_document_with_id(doc_id, doc).await?;
        Ok(doc_id)
    }

    /// Add a document using a specific document ID.
    ///
    /// # Arguments
    ///
    /// * `doc_id` - The document ID to use
    /// * `doc` - The document to add (containing both text and vector fields)
    pub async fn add_document_with_id(
        &mut self,
        doc_id: u64,
        doc: crate::document::document::Document,
    ) -> Result<()> {
        // Clone the document for both indexes since they'll process different fields
        self.lexical_engine
            .add_document_with_id(doc_id, doc.clone())?;
        self.vector_engine.add_document_with_id(doc_id, doc).await?;

        // Update next_doc_id if necessary
        if doc_id >= self.next_doc_id {
            self.next_doc_id = doc_id + 1;
        }

        Ok(())
    }

    /// Commit changes to both lexical and vector indexes.
    pub fn commit(&mut self) -> Result<()> {
        self.lexical_engine.commit()?;
        self.vector_engine.commit()?;
        Ok(())
    }

    /// Optimize both indexes.
    pub fn optimize(&mut self) -> Result<()> {
        self.lexical_engine.optimize()?;
        self.vector_engine.optimize()?;
        Ok(())
    }

    /// Execute a hybrid search combining keyword and semantic search.
    ///
    /// This is an async method that performs lexical and vector searches,
    /// then merges the results using the configured fusion strategy.
    ///
    /// # Arguments
    ///
    /// * `request` - The hybrid search request containing query and parameters
    ///
    /// # Returns
    ///
    /// Combined search results from both engines
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use yatagarasu::hybrid::engine::HybridEngine;
    /// # use yatagarasu::hybrid::search::searcher::HybridSearchRequest;
    /// # async fn example(engine: HybridEngine) -> yatagarasu::error::Result<()> {
    /// let request = HybridSearchRequest::new("rust programming");
    /// let results = engine.search(request).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn search(
        &self,
        request: crate::hybrid::search::searcher::HybridSearchRequest,
    ) -> Result<crate::hybrid::search::searcher::HybridSearchResults> {
        use std::collections::HashMap;
        use std::time::Instant;

        let start = Instant::now();

        // Prepare lexical search request
        let lexical_request =
            crate::lexical::search::searcher::LexicalSearchRequest::new(request.text_query.clone())
                .max_docs(request.lexical_params.max_docs)
                .min_score(request.lexical_params.min_score)
                .load_documents(request.lexical_params.load_documents);

        // Prepare vector search request if vector query provided
        let vector_request_opt = request.vector_query.as_ref().map(|vector| {
            crate::vector::search::searcher::VectorSearchRequest::new(vector.clone())
                .top_k(request.vector_params.top_k)
                .min_similarity(request.params.min_vector_similarity)
        });

        // Execute both searches sequentially (engines are not Send, so can't use spawn_blocking)
        // However, we can still execute them efficiently using tokio's runtime
        let keyword_results = self.lexical_engine.search(lexical_request)?;

        let vector_results = if let Some(vector_request) = vector_request_opt {
            Some(self.vector_engine.search(vector_request)?)
        } else {
            None
        };

        // Merge results
        let merger = crate::hybrid::search::merger::ResultMerger::new(request.params.clone());
        let query_time_ms = start.elapsed().as_millis() as u64;

        // TODO: Implement proper document store
        let document_store = HashMap::new();

        // merge_results is async, use .await directly
        merger
            .merge_results(
                keyword_results,
                vector_results,
                request.text_query,
                query_time_ms,
                &document_store,
            )
            .await
    }

    /// Get a reference to the lexical engine.
    pub fn lexical_engine(&self) -> &crate::lexical::engine::LexicalEngine {
        &self.lexical_engine
    }

    /// Get a reference to the vector engine.
    pub fn vector_engine(&self) -> &crate::vector::engine::VectorEngine {
        &self.vector_engine
    }
}

#[cfg(test)]
mod tests {
    // Note: Full integration tests would require setting up both engines
    // These are placeholder tests for the basic structure
    #[test]
    fn test_hybrid_engine_structure() {
        // This test just verifies the struct can be constructed
        // Real tests would need actual engine instances
    }
}
