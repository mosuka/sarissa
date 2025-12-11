//! Hybrid search engine implementation.
//!
//! This module provides the `HybridEngine` that combines lexical and vector search
//! engines to provide unified hybrid search functionality.

use std::collections::HashMap;

use crate::error::Result;
use crate::hybrid::search::searcher::{
    HybridSearchParams, HybridSearchRequest, HybridSearchResults, HybridVectorOptions,
};
use crate::vector::core::document::{DocumentPayload, DocumentVector, FieldPayload};
use crate::vector::engine::{
    FieldSelector, VectorEngine, VectorEngineSearchRequest, VectorEngineSearchResults,
};
use crate::vector::search::searcher::VectorSearchParams;

/// High-level hybrid search engine combining lexical and vector search.
///
/// This engine wraps both `LexicalEngine` and `VectorEngine` to provide
/// unified hybrid search functionality. It follows the same pattern as the
/// individual engines but coordinates searches across both indexes.
///
/// # Examples
///
/// ```no_run
/// use platypus::hybrid::engine::HybridEngine;
/// use platypus::hybrid::search::searcher::HybridSearchRequest;
/// use platypus::lexical::engine::LexicalEngine;
/// use platypus::vector::engine::VectorEngine;
/// use platypus::vector::Vector;
///
/// # async fn example(lexical_engine: LexicalEngine, vector_engine: VectorEngine) -> platypus::error::Result<()> {
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
    /// use platypus::hybrid::engine::HybridEngine;
    /// use platypus::lexical::engine::LexicalEngine;
    /// use platypus::vector::engine::VectorEngine;
    ///
    /// # fn example(lexical_engine: LexicalEngine, vector_engine: VectorEngine) -> platypus::error::Result<()> {
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

    /// Add a document to the lexical index.
    /// Returns the assigned document ID.
    ///
    /// This method assigns the same document ID that would be used for hybrid operations,
    /// but only writes to the lexical index. Vector ingestion should be done separately.
    ///
    /// # Arguments
    ///
    /// * `doc` - The document to add to the lexical index
    ///
    /// # Returns
    ///
    /// The assigned document ID
    pub async fn add_document(
        &mut self,
        doc: crate::lexical::document::document::Document,
    ) -> Result<u64> {
        let doc_id = self.next_doc_id;
        self.upsert_document(doc_id, doc).await?;
        Ok(doc_id)
    }

    /// Add a document to the lexical index using a specific document ID.
    ///
    /// # Arguments
    ///
    /// * `doc_id` - The document ID to use
    /// * `doc` - The document to add to the lexical index
    pub async fn upsert_document(
        &mut self,
        doc_id: u64,
        doc: crate::lexical::document::document::Document,
    ) -> Result<()> {
        self.lexical_engine.upsert_document(doc_id, doc.clone())?;

        // Update next_doc_id if necessary
        if doc_id >= self.next_doc_id {
            self.next_doc_id = doc_id + 1;
        }

        Ok(())
    }

    /// Commit changes to both lexical and vector indexes.
    pub fn commit(&mut self) -> Result<()> {
        self.lexical_engine.commit()?;
        Ok(())
    }

    /// Upsert vectors only (pre-embedded).
    ///
    /// This does not touch the lexical index. It upserts the provided
    /// `DocumentVector` into the vector engine and advances `next_doc_id`
    /// if the given `doc_id` is new/highest.
    pub fn upsert_vector_document(&mut self, doc_id: u64, vectors: DocumentVector) -> Result<()> {
        self.vector_engine.upsert_document(doc_id, vectors)?;
        if doc_id >= self.next_doc_id {
            self.next_doc_id = doc_id + 1;
        }
        Ok(())
    }

    /// Upsert vectors only from raw payloads (embedding inside vector engine).
    ///
    /// This does not touch the lexical index. It embeds the payload and
    /// upserts into the vector engine, advancing `next_doc_id` if needed.
    pub fn upsert_vector_payload(&mut self, doc_id: u64, payload: DocumentPayload) -> Result<()> {
        self.vector_engine
            .upsert_document_payload(doc_id, payload)?;
        if doc_id >= self.next_doc_id {
            self.next_doc_id = doc_id + 1;
        }
        Ok(())
    }

    /// Optimize both indexes.
    pub fn optimize(&mut self) -> Result<()> {
        self.lexical_engine.optimize()?;
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
    /// # use platypus::hybrid::engine::HybridEngine;
    /// # use platypus::hybrid::search::searcher::HybridSearchRequest;
    /// # async fn example(engine: HybridEngine) -> platypus::error::Result<()> {
    /// let request = HybridSearchRequest::new("rust programming");
    /// let results = engine.search(request).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn search(&self, request: HybridSearchRequest) -> Result<HybridSearchResults> {
        use std::collections::HashMap;
        use std::time::Instant;

        let HybridSearchRequest {
            text_query,
            vector_query,
            vector_payloads,
            params,
            lexical_params,
            vector_params,
            vector_overrides,
        } = request;

        let start = Instant::now();

        // Prepare lexical search request
        let lexical_request =
            crate::lexical::search::searcher::LexicalSearchRequest::new(text_query.clone())
                .max_docs(lexical_params.max_docs)
                .min_score(lexical_params.min_score)
                .load_documents(lexical_params.load_documents);

        // Execute both searches sequentially (engines are not Send, so can't use spawn_blocking)
        // However, we can still execute them efficiently using tokio's runtime
        let keyword_results = self.lexical_engine.search(lexical_request)?;

        let vector_query = Self::build_vector_engine_search_request(
            vector_overrides,
            vector_query,
            vector_payloads,
            &vector_params,
            &params,
            &self.vector_engine,
        )?;

        let vector_results = if let Some(query) = vector_query {
            let mut results = self.vector_engine.search(query)?;
            Self::apply_vector_constraints(&mut results, &vector_params);
            Some(results)
        } else {
            None
        };

        // Merge results
        let merger = crate::hybrid::search::merger::ResultMerger::new(params.clone());
        let query_time_ms = start.elapsed().as_millis() as u64;

        // TODO: Implement proper document store
        let document_store = HashMap::new();

        // merge_results is async, use .await directly
        merger
            .merge_results(
                keyword_results,
                vector_results,
                text_query,
                query_time_ms,
                &document_store,
            )
            .await
    }

    fn build_vector_engine_search_request(
        overrides: HybridVectorOptions,
        query: Option<VectorEngineSearchRequest>,
        vector_payloads: HashMap<String, FieldPayload>,
        vector_params: &VectorSearchParams,
        params: &HybridSearchParams,
        vector_engine: &VectorEngine,
    ) -> Result<Option<VectorEngineSearchRequest>> {
        let mut query = query.unwrap_or_default();
        let mut payload_fields: Vec<String> = Vec::new();

        for (field_name, payload) in vector_payloads {
            if payload.is_empty() {
                continue;
            }
            let embedded = vector_engine.embed_query_field_payload(&field_name, payload)?;
            if embedded.is_empty() {
                continue;
            }
            payload_fields.push(field_name);
            query.query_vectors.extend(embedded);
        }

        if query.query_vectors.is_empty() {
            return Ok(None);
        }

        HybridSearchRequest::apply_overrides_to_query(&overrides, &mut query);
        if query.limit == 0 {
            query.limit = Self::default_vector_limit(vector_params, params);
        }
        if query.fields.is_none() && overrides.fields.is_none() && !payload_fields.is_empty() {
            let mut dedup = Vec::new();
            for field in payload_fields {
                if !dedup.iter().any(|existing: &String| existing == &field) {
                    dedup.push(field);
                }
            }
            query.fields = Some(dedup.into_iter().map(FieldSelector::Exact).collect());
        }
        Ok(Some(query))
    }

    fn default_vector_limit(
        vector_params: &VectorSearchParams,
        params: &HybridSearchParams,
    ) -> usize {
        let mut candidates = Vec::new();
        if vector_params.top_k > 0 {
            candidates.push(vector_params.top_k);
        }
        if params.max_results > 0 {
            candidates.push(params.max_results);
        }
        candidates.into_iter().max().unwrap_or(1).max(1)
    }

    fn apply_vector_constraints(
        results: &mut VectorEngineSearchResults,
        vector_params: &VectorSearchParams,
    ) {
        if vector_params.min_similarity > 0.0 {
            results
                .hits
                .retain(|hit| hit.score >= vector_params.min_similarity);
        }

        if vector_params.top_k > 0 && results.hits.len() > vector_params.top_k {
            results.hits.truncate(vector_params.top_k);
        }
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
    use super::*;
    use crate::embedding::text_embedder::TextEmbedder;
    use crate::storage::Storage;
    use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};
    use crate::vector::DistanceMetric;
    use crate::vector::core::document::{FieldPayload, StoredVector, VectorType};
    use crate::vector::core::vector::Vector;
    use crate::vector::engine::{
        FieldSelector, MetadataFilter, QueryVector, VectorEmbedderConfig, VectorEmbedderProvider,
        VectorEngineFilter, VectorEngineHit, VectorFieldConfig, VectorIndexConfig, VectorIndexKind,
        VectorScoreMode,
    };
    use crate::vector::field::FieldHit;
    use async_trait::async_trait;
    use std::collections::HashMap;
    use std::sync::Arc;

    #[test]
    fn build_query_apply_overrides_and_limits() {
        let engine = mock_vector_engine();
        let mut overrides = HybridVectorOptions::default();
        overrides.fields = Some(vec![FieldSelector::Exact("body_embedding".into())]);
        overrides.score_mode = Some(VectorScoreMode::MaxSim);
        overrides.overfetch = Some(1.25);
        let mut field_filter = MetadataFilter::default();
        field_filter
            .equals
            .insert("section".to_string(), "body".to_string());
        overrides.filter = Some(VectorEngineFilter {
            document: MetadataFilter::default(),
            field: field_filter.clone(),
        });

        let mut query = VectorEngineSearchRequest::default();
        query.limit = 0;
        query.query_vectors.push(QueryVector {
            vector: StoredVector::new(
                Arc::<[f32]>::from([1.0_f32, 0.0, 0.0]),
                "mock".into(),
                VectorType::Text,
            ),
            weight: 1.0,
        });

        let vector_params = VectorSearchParams {
            top_k: 4,
            ..Default::default()
        };
        let mut hybrid_params = HybridSearchParams::default();
        hybrid_params.max_results = 2;

        let resolved = HybridEngine::build_vector_engine_search_request(
            overrides,
            Some(query),
            HashMap::new(),
            &vector_params,
            &hybrid_params,
            &engine,
        )
        .expect("vector query")
        .expect("query present");

        assert_eq!(resolved.limit, 4);
        assert!(matches!(resolved.score_mode, VectorScoreMode::MaxSim));
        let fields = resolved.fields.expect("fields");
        assert_eq!(fields.len(), 1);
        let filter = resolved.filter.expect("filter");
        assert_eq!(
            filter.field.equals.get("section"),
            field_filter.equals.get("section")
        );
    }

    #[test]
    fn apply_vector_constraints_respects_similarity_and_topk() {
        let mut results = VectorEngineSearchResults {
            hits: vec![
                VectorEngineHit {
                    doc_id: 1,
                    score: 0.9,
                    field_hits: vec![FieldHit {
                        doc_id: 1,
                        field: "title".into(),
                        score: 0.9,
                        distance: 0.1,
                        metadata: Default::default(),
                    }],
                },
                VectorEngineHit {
                    doc_id: 2,
                    score: 0.4,
                    field_hits: vec![FieldHit {
                        doc_id: 2,
                        field: "title".into(),
                        score: 0.4,
                        distance: 0.6,
                        metadata: Default::default(),
                    }],
                },
            ],
        };

        let params = VectorSearchParams {
            top_k: 1,
            min_similarity: 0.5,
            ..Default::default()
        };

        HybridEngine::apply_vector_constraints(&mut results, &params);

        assert_eq!(results.hits.len(), 1);
        assert_eq!(results.hits[0].doc_id, 1);
    }

    #[test]
    fn build_query_from_payload_embeds_vectors() {
        let engine = mock_vector_engine();
        let mut payloads = HashMap::new();
        let mut payload = FieldPayload::default();
        payload.add_text_segment("rust embeddings");
        payloads.insert("body".into(), payload);

        let resolved = HybridEngine::build_vector_engine_search_request(
            HybridVectorOptions::default(),
            None,
            payloads,
            &VectorSearchParams::default(),
            &HybridSearchParams::default(),
            &engine,
        )
        .expect("payload query")
        .expect("query present");

        assert!(!resolved.query_vectors.is_empty());
        let fields = resolved.fields.expect("fields inferred");
        assert_eq!(fields.len(), 1);
    }

    fn mock_vector_engine() -> VectorEngine {
        let mut fields = HashMap::new();
        fields.insert(
            "body".into(),
            VectorFieldConfig {
                dimension: 3,
                distance: DistanceMetric::Cosine,
                index: VectorIndexKind::Flat,
                embedder_id: "mock".into(),
                vector_type: VectorType::Text,
                embedder: Some("mock_embedder".into()),
                base_weight: 1.0,
            },
        );
        let embedders = HashMap::from([(
            "mock_embedder".into(),
            VectorEmbedderConfig {
                provider: VectorEmbedderProvider::External,
                model: "mock".into(),
                options: HashMap::new(),
            },
        )]);
        #[allow(deprecated)]
        let config = VectorIndexConfig {
            fields,
            embedders,
            default_fields: vec!["body".into()],
            metadata: HashMap::new(),
            embedder: None,
        };
        let storage: Arc<dyn Storage> =
            Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let engine = VectorEngine::new(storage, config).expect("engine");
        engine
            .register_embedder_instance(
                "mock_embedder".to_string(),
                Arc::new(MockTextEmbedder::new(3)),
            )
            .expect("register embedder");
        engine
    }

    #[derive(Debug)]
    struct MockTextEmbedder {
        dimension: usize,
    }

    impl MockTextEmbedder {
        fn new(dimension: usize) -> Self {
            Self { dimension }
        }
    }

    #[async_trait]
    impl TextEmbedder for MockTextEmbedder {
        async fn embed(&self, text: &str) -> Result<Vector> {
            let value = text.len() as f32;
            Ok(Vector::new(vec![value; self.dimension]))
        }

        fn dimension(&self) -> usize {
            self.dimension
        }

        fn name(&self) -> &str {
            "mock-hybrid-text-embedder"
        }
    }
}
