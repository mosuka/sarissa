#[cfg(test)]
mod tests {
    use crate::error::Result;
    use crate::storage::StorageConfig;
    use crate::storage::StorageFactory;
    use crate::storage::memory::MemoryStorageConfig;
    use crate::vector::core::distance::DistanceMetric;
    use crate::vector::core::vector::Vector;
    use crate::vector::index::VectorIndex;
    use crate::vector::index::config::HnswIndexConfig;
    use crate::vector::index::hnsw::HnswIndex;

    #[test]
    fn test_hnsw_integration() -> Result<()> {
        let storage_config = StorageConfig::Memory(MemoryStorageConfig::default());
        let storage = StorageFactory::create(storage_config)?;

        // HNSW Config
        let config = HnswIndexConfig {
            dimension: 3,
            m: 16,
            ef_construction: 100,
            distance_metric: DistanceMetric::Cosine,
            ..Default::default()
        };

        let index = HnswIndex::create(storage.clone(), "default_index", config.clone())?;
        let mut writer = index.writer()?;

        // Add vectors
        let vectors = vec![
            (1, "test".to_string(), Vector::new(vec![1.0, 0.0, 0.0])), // A
            (2, "test".to_string(), Vector::new(vec![0.0, 1.0, 0.0])), // B
            (3, "test".to_string(), Vector::new(vec![0.0, 0.0, 1.0])), // C
            (4, "test".to_string(), Vector::new(vec![0.707, 0.707, 0.0])), // Between A and B
        ];

        writer.build(vectors.clone())?;
        writer.finalize()?;
        // Note: commit is handled by VectorIndexWriter trait default which calls write("default_index")
        // Since we are using HnswIndexWriter directly via trait object or concrete?
        // index.writer() returns Box<dyn VectorIndexWriter>.
        writer.commit("default_index")?;

        // Read back
        let reader = index.reader()?;

        // Check graph loading
        use crate::vector::index::hnsw::reader::HnswIndexReader;
        let hnsw_reader = reader
            .as_any()
            .downcast_ref::<HnswIndexReader>()
            .expect("Should be HnswIndexReader");
        assert!(hnsw_reader.graph.is_some());

        // Search using Graph
        use crate::vector::index::hnsw::searcher::HnswSearcher;
        use crate::vector::search::searcher::{VectorIndexSearchRequest, VectorIndexSearcher};

        let searcher = HnswSearcher::new(reader.clone())?;

        // Query close to A (1,0,0)
        let query = Vector::new(vec![0.9, 0.1, 0.0]);
        let request = VectorIndexSearchRequest::new(query)
            .top_k(1)
            .field_name("test".to_string());

        let results = searcher.search(&request)?;

        assert_eq!(results.results.len(), 1);
        assert_eq!(results.results[0].doc_id, 1);

        Ok(())
    }
}
