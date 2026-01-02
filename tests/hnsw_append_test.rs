use sarissa::storage::file::{FileStorage, FileStorageConfig};
use sarissa::vector::DistanceMetric;
use sarissa::vector::core::vector::Vector;
use sarissa::vector::index::config::HnswIndexConfig;
use sarissa::vector::index::hnsw::reader::HnswIndexReader;
use sarissa::vector::index::hnsw::searcher::HnswSearcher;
use sarissa::vector::index::hnsw::writer::HnswIndexWriter;
use sarissa::vector::search::searcher::{VectorIndexSearchRequest, VectorIndexSearcher};
use sarissa::vector::writer::{VectorIndexWriter, VectorIndexWriterConfig};
use std::sync::Arc;
use tempfile::tempdir;

#[test]
fn test_hnsw_append_persistence() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempdir()?;
    let path = dir.path();
    let index_name = "test_index";

    let index_config = HnswIndexConfig {
        dimension: 3,
        m: 16,
        ef_construction: 100,
        normalize_vectors: false,
        distance_metric: DistanceMetric::Euclidean,
        ..Default::default()
    };

    let writer_config = VectorIndexWriterConfig {
        parallel_build: true,
        ..Default::default()
    };

    // 1. Initial Build
    {
        let storage_config = FileStorageConfig::new(path);
        let storage = Arc::new(FileStorage::new(path, storage_config)?);

        // Use with_storage to ensure we can write
        let mut writer = HnswIndexWriter::with_storage(
            index_config.clone(),
            writer_config.clone(),
            index_name,
            storage,
        )?;

        let vectors = vec![
            (1, "doc1".to_string(), Vector::new(vec![1.0f32, 0.0, 0.0])),
            (2, "doc2".to_string(), Vector::new(vec![0.0f32, 1.0, 0.0])),
        ];

        writer.add_vectors(vectors)?;
        writer.finalize()?;
        writer.write()?;
    }

    // 2. Load and Append
    {
        let storage_config = FileStorageConfig::new(path);
        let storage = Arc::new(FileStorage::new(path, storage_config)?);

        let mut writer = HnswIndexWriter::load(
            index_config.clone(),
            writer_config.clone(),
            storage,
            index_name,
        )?;

        let new_vectors = vec![
            (3, "doc3".to_string(), Vector::new(vec![0.0f32, 0.0, 1.0])),
            (4, "doc4".to_string(), Vector::new(vec![1.0f32, 1.0, 0.0])),
        ];

        writer.add_vectors(new_vectors)?;
        writer.finalize()?;
        writer.write()?;
    }

    Ok(())
}

#[test]
fn test_hnsw_append_search_verification() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempdir()?;
    let path = dir.path();
    let index_name = "search_test";

    let index_config = HnswIndexConfig {
        dimension: 2,
        m: 16,
        ef_construction: 100,
        normalize_vectors: false,
        distance_metric: DistanceMetric::Euclidean,
        ..Default::default()
    };
    let writer_config = VectorIndexWriterConfig {
        parallel_build: true,
        ..Default::default()
    };

    // 1. Initial Build
    {
        let storage_config = FileStorageConfig::new(path);
        let storage = Arc::new(FileStorage::new(path, storage_config)?);

        let mut writer = HnswIndexWriter::with_storage(
            index_config.clone(),
            writer_config.clone(),
            index_name,
            storage,
        )?;

        writer.add_vectors(vec![(
            1,
            "doc1".to_string(),
            Vector::new(vec![1.0f32, 0.0]),
        )])?;
        writer.finalize()?;
        writer.write()?;
    }

    // 2. Load and Append
    {
        let storage_config = FileStorageConfig::new(path);
        let storage = Arc::new(FileStorage::new(path, storage_config)?);

        let mut writer = HnswIndexWriter::load(
            index_config.clone(),
            writer_config.clone(),
            storage,
            index_name,
        )?;

        writer.add_vectors(vec![(
            2,
            "doc2".to_string(),
            Vector::new(vec![0.0f32, 1.0]),
        )])?;
        writer.finalize()?;
        writer.write()?;
    }

    // 3. Search
    {
        let storage_config = FileStorageConfig::new(path);
        let storage = Arc::new(FileStorage::new(path, storage_config)?);

        let reader = Arc::new(HnswIndexReader::load(
            storage.as_ref(),
            index_name,
            DistanceMetric::Euclidean,
        )?);

        // Create Searcher
        let searcher = HnswSearcher::new(reader)?;

        // Search for closest to [1.0, 0.0] -> should be doc1
        let req1 = VectorIndexSearchRequest::new(Vector::new(vec![1.0f32, 0.0])).top_k(1);
        let results1 = searcher.search(&req1)?;
        assert_eq!(results1.len(), 1, "Expected 1 result for doc1");
        assert_eq!(results1.results[0].doc_id, 1, "Expected doc1");

        // Search for closest to [0.0, 1.0] -> should be doc2
        // If append works, doc2 is in the graph.
        let req2 = VectorIndexSearchRequest::new(Vector::new(vec![0.0f32, 1.0])).top_k(1);
        let results2 = searcher.search(&req2)?;
        assert_eq!(results2.len(), 1, "Expected 1 result for doc2");
        assert_eq!(results2.results[0].doc_id, 2, "Expected doc2");
    }

    Ok(())
}
