#[cfg(test)]
mod tests {
    use crate::storage::memory::MemoryStorage;
    use crate::vector::core::vector::Vector;
    use crate::vector::index::config::FlatIndexConfig;
    use crate::vector::index::flat::writer::FlatIndexWriter;
    use crate::vector::writer::{VectorIndexWriter, VectorIndexWriterConfig};
    use std::sync::Arc;

    #[test]
    fn test_vector_deletion() {
        // Test FlatIndexWriter deletion
        let storage = Arc::new(MemoryStorage::default());
        let config = FlatIndexConfig {
            dimension: 3,
            ..FlatIndexConfig::default()
        };
        let writer_config = VectorIndexWriterConfig::default();

        let mut writer =
            FlatIndexWriter::with_storage(config, writer_config, "test_deletion_vectors", storage)
                .unwrap();

        // Add 3 vectors
        let vectors = vec![
            (1, "f".to_string(), Vector::new(vec![1.0, 0.0, 0.0])),
            (2, "f".to_string(), Vector::new(vec![0.0, 1.0, 0.0])),
            (3, "f".to_string(), Vector::new(vec![0.0, 0.0, 1.0])),
        ];
        writer.add_vectors(vectors).unwrap();

        assert_eq!(writer.vectors().len(), 3);

        // Delete vector 2
        writer.delete_document(2).unwrap();

        // Verify it's gone from buffer
        assert_eq!(writer.vectors().len(), 2);

        let remaining_ids: Vec<u64> = writer.vectors().iter().map(|(id, _, _)| *id).collect();
        assert!(remaining_ids.contains(&1));
        assert!(remaining_ids.contains(&3));
        assert!(!remaining_ids.contains(&2));

        // Finalize
        writer.finalize().unwrap();

        // Verify finalized state (should still correspond to buffered state effectively for FlatIndex)
        assert_eq!(writer.vectors().len(), 2);
    }
}
