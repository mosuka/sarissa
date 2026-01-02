#[cfg(test)]
mod tests {
    use sarissa::lexical::core::document::Document;
    use sarissa::lexical::core::field::TextOption;
    use sarissa::lexical::index::inverted::writer::{
        InvertedIndexWriter, InvertedIndexWriterConfig,
    };
    use sarissa::storage::memory::MemoryStorageConfig;
    use sarissa::storage::{StorageConfig, StorageFactory};
    use sarissa::vector::collection::VectorCollection;
    use sarissa::vector::core::distance::DistanceMetric;
    use sarissa::vector::core::document::{DocumentVector, StoredVector};
    use sarissa::vector::engine::config::{VectorFieldConfig, VectorIndexConfig, VectorIndexKind};
    use sarissa::vector::engine::{VectorScoreMode, VectorSearchRequest};
    use std::collections::HashMap;
    use std::sync::Arc;

    #[test]
    fn test_vector_collection_logical_deletion() {
        // 1. Create VectorCollection
        // Configure field
        let field_config = VectorFieldConfig {
            dimension: 128,
            distance: DistanceMetric::Cosine,
            index: VectorIndexKind::Flat,
            metadata: HashMap::new(),
            base_weight: 1.0,
        };

        let config = VectorIndexConfig {
            fields: HashMap::from([("embedding".to_string(), field_config)]),
            default_fields: vec!["embedding".to_string()],
            metadata: HashMap::new(),
            default_distance: DistanceMetric::Cosine,
            default_dimension: None,
            default_index_kind: VectorIndexKind::Flat,
            default_base_weight: 1.0,
            implicit_schema: false,
            embedder: Arc::new(sarissa::embedding::precomputed::PrecomputedEmbedder::new()),
        };

        let storage =
            StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default())).unwrap();

        let collection = VectorCollection::new(config, storage, None).unwrap();

        // 2. Add a document
        let doc_id = 1;
        let mut doc_vector = DocumentVector::new();
        // Use set_field with StoredVector
        doc_vector.set_field("embedding", StoredVector::new(Arc::from(vec![0.1f32; 128])));

        collection.upsert_document(doc_id, doc_vector).unwrap();

        // 3. Verify it exists
        // Search request
        let request = VectorSearchRequest {
            query_vectors: vec![sarissa::vector::engine::QueryVector {
                vector: StoredVector::new(Arc::from(vec![0.1f32; 128])),
                weight: 1.0,
                fields: None,
            }],
            limit: 10,
            overfetch: 1.5,
            min_score: 0.0,
            score_mode: VectorScoreMode::MaxSim,
            filter: None,
            fields: None, // All fields
            query_payloads: vec![],
        };

        let searcher = collection.searcher().unwrap();
        let results = searcher.search(&request).unwrap();
        assert_eq!(
            results.hits.len(),
            1,
            "Document should be found before deletion"
        );
        assert_eq!(results.hits[0].doc_id, doc_id);

        // 4. Delete the document
        collection.delete_document(doc_id).unwrap();

        // 5. Verify it is GONE (Logical Deletion check)
        let searcher_after = collection.searcher().unwrap();
        let results_after = searcher_after.search(&request).unwrap();
        assert_eq!(
            results_after.hits.len(),
            0,
            "Document should NOT be found after deletion (logical check)"
        );
    }

    #[test]
    fn test_inverted_index_persisted_deletion() {
        // Setup storage
        let storage =
            StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default())).unwrap();
        let config = InvertedIndexWriterConfig::default();

        // Create writer
        let mut writer = InvertedIndexWriter::new(storage.clone(), config.clone()).unwrap();

        // Add document
        let doc_id = 100;

        let doc = Document::builder()
            .add_text("title", "hello world", TextOption::default())
            .build();

        writer.upsert_document(doc_id, doc).unwrap();

        // Commit to persist
        writer.commit().unwrap();

        // Verify deletion call works (it calls mark_persisted_doc_deleted internally now)
        writer.delete_document(doc_id).unwrap();
    }
}
