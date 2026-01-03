//! Core hybrid index implementation.
//!
//! This module provides the core data structure that manages both lexical and vector
//! indexes for hybrid search.

use std::sync::Arc;

use crate::error::Result;
use crate::hybrid::search::searcher::{HybridSearchRequest, HybridSearchResults};
use crate::lexical::reader::LexicalIndexReader;
use crate::vector::reader::VectorIndexReader;

/// Hybrid index that combines lexical and vector indexes.
///
/// This structure manages both a lexical (inverted) index for keyword search
/// and a vector index for semantic search, providing a unified interface
/// for hybrid search operations.
pub struct HybridIndex {
    /// Lexical index reader for keyword-based search
    pub lexical_index: Arc<dyn LexicalIndexReader>,
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
        lexical_index: Arc<dyn LexicalIndexReader>,
        vector_index: Arc<dyn VectorIndexReader>,
    ) -> Self {
        Self {
            lexical_index,
            vector_index,
        }
    }

    /// Get a reference to the lexical index reader.
    ///
    /// # Returns
    ///
    /// A reference to the underlying lexical (inverted) index reader for keyword search
    pub fn lexical_index(&self) -> &dyn LexicalIndexReader {
        self.lexical_index.as_ref()
    }

    /// Get a reference to the vector index reader.
    ///
    /// # Returns
    ///
    /// A reference to the underlying vector index reader for semantic search
    pub fn vector_index(&self) -> &dyn VectorIndexReader {
        self.vector_index.as_ref()
    }

    /// Get statistics about the hybrid index.
    ///
    /// Returns document counts from both the lexical and vector indexes.
    ///
    /// # Returns
    ///
    /// Statistics including counts from both indexes
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use sarissa::hybrid::index::HybridIndex;
    /// # use sarissa::lexical::reader::LexicalIndexReader;
    /// # use sarissa::vector::reader::VectorIndexReader;
    /// # use std::sync::Arc;
    /// # fn example(lexical: Arc<dyn LexicalIndexReader>, vector: Arc<dyn VectorIndexReader>) -> sarissa::error::Result<()> {
    /// let hybrid_index = HybridIndex::new(lexical, vector);
    /// let stats = hybrid_index.stats()?;
    /// println!("Lexical docs: {}", stats.lexical_doc_count);
    /// println!("Vector docs: {}", stats.vector_doc_count);
    /// # Ok(())
    /// # }
    /// ```
    pub fn stats(&self) -> Result<HybridIndexStats> {
        Ok(HybridIndexStats {
            lexical_doc_count: self.lexical_index.doc_count(),
            vector_doc_count: self.vector_index.vector_count() as u64,
        })
    }

    /// Execute a hybrid search with the given request.
    ///
    /// This method performs both lexical and vector searches according to the request,
    /// and then uses the `ResultMerger` to combine the results into a single list.
    ///
    /// # Arguments
    ///
    /// * `request` - The hybrid search request containing queries and parameters.
    ///
    /// # Returns
    ///
    /// A `Result` containing the `HybridSearchResults`.
    pub async fn search(&self, request: HybridSearchRequest) -> Result<HybridSearchResults> {
        use std::collections::HashMap;
        use std::time::Instant;

        use crate::hybrid::search::merger::ResultMerger;
        use crate::lexical::index::inverted::searcher::InvertedIndexSearcher;
        use crate::lexical::search::searcher::LexicalSearchQuery;
        use crate::vector::index::flat::searcher::FlatVectorSearcher;
        use crate::vector::index::hnsw::searcher::HnswSearcher;
        use crate::vector::search::searcher::{VectorIndexSearchRequest, VectorIndexSearcher};

        let start_time = Instant::now();

        // 1. Lexical Search
        let (lexical_results, query_text) = if let Some(req) = request.lexical_request {
            let text = match &req.query {
                LexicalSearchQuery::Dsl(s) => s.clone(),
                LexicalSearchQuery::Obj(_) => "".to_string(),
            };
            let lexical_searcher = InvertedIndexSearcher::from_arc(self.lexical_index.clone());
            (lexical_searcher.search(req)?, text)
        } else {
            (
                crate::lexical::index::inverted::query::LexicalSearchResults {
                    hits: Vec::new(),
                    total_hits: 0,
                    max_score: 0.0,
                },
                "".to_string(),
            )
        };

        // 2. Vector Search
        let vector_results = if let Some(vector_req) = &request.vector_request {
            // Determine the correct searcher based on index type
            let metadata = self.vector_index.metadata()?;
            let index_type = metadata.index_type.to_lowercase();

            let vector_searcher: Box<dyn VectorIndexSearcher> = match index_type.as_str() {
                "hnsw" => Box::new(HnswSearcher::new(self.vector_index.clone())?),
                "flat" => Box::new(FlatVectorSearcher::new(self.vector_index.clone())?),
                _ => Box::new(FlatVectorSearcher::new(self.vector_index.clone())?),
            };

            let mut all_hits: HashMap<u64, crate::vector::engine::response::VectorHit> =
                HashMap::new();

            // Iterate over all query vectors in the request
            for query_vector in &vector_req.query_vectors {
                let data = query_vector.vector.data.to_vec();
                let vector_obj = crate::vector::core::vector::Vector::new(data);

                let mut search_req = VectorIndexSearchRequest::new(vector_obj)
                    .top_k(vector_req.limit)
                    .min_similarity(vector_req.min_score);

                let target_fields = if let Some(fields) = &query_vector.fields {
                    fields.clone()
                } else {
                    self.vector_index.field_names()?
                };

                for field in target_fields {
                    search_req.field_name = Some(field.clone());

                    let search_res = vector_searcher.search(&search_req)?;

                    for res in search_res.results {
                        let entry = all_hits.entry(res.doc_id).or_insert_with(|| {
                            crate::vector::engine::response::VectorHit {
                                doc_id: res.doc_id,
                                score: 0.0,
                                field_hits: Vec::new(),
                            }
                        });

                        if res.similarity > entry.score {
                            entry.score = res.similarity;
                        }

                        entry.field_hits.push(crate::vector::field::FieldHit {
                            doc_id: res.doc_id,
                            field: res.field_name,
                            score: res.similarity,
                            distance: res.distance,
                            metadata: res.metadata,
                        });
                    }
                }
            }

            let hits: Vec<_> = all_hits.into_values().collect();
            Some(crate::vector::engine::response::VectorSearchResults { hits })
        } else {
            None
        };

        // 3. Merge Results
        let merger = ResultMerger::new(request.params.clone());
        merger
            .merge_results(
                lexical_results,
                vector_results,
                query_text,
                start_time.elapsed().as_millis() as u64,
            )
            .await
    }
}

/// Statistics about a hybrid index.
///
/// Contains document counts from both the lexical and vector components
/// of a hybrid index.
#[derive(Debug, Clone)]
pub struct HybridIndexStats {
    /// Number of documents in the lexical index.
    pub lexical_doc_count: u64,
    /// Number of vectors in the vector index.
    pub vector_doc_count: u64,
}
