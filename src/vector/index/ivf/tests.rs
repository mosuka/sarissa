#[cfg(test)]
mod tests {
    use crate::storage::memory::MemoryStorage;
    use crate::vector::core::distance::DistanceMetric;
    use crate::vector::core::vector::Vector;
    use crate::vector::index::config::IvfIndexConfig;
    use crate::vector::index::ivf::writer::IvfIndexWriter;
    use crate::vector::writer::{VectorIndexWriter, VectorIndexWriterConfig};
    use std::sync::Arc;

    #[test]
    fn test_ivf_partition_rebalancing() {
        let storage = Arc::new(MemoryStorage::default());
        let config = IvfIndexConfig {
            dimension: 2,
            distance_metric: DistanceMetric::Euclidean,
            n_clusters: 4,
            n_probe: 2,
            normalize_vectors: false,
            ..IvfIndexConfig::default()
        };
        let writer_config = VectorIndexWriterConfig::default();

        let mut writer = IvfIndexWriter::with_storage(config, writer_config, storage).unwrap();

        // Create imbalanced clusters
        // Cluster 0: 1 vector (sparse)
        // Cluster 1: 10 vectors (dense)
        // Cluster 2: 4 vectors (normal)
        // Cluster 3: 4 vectors (normal)

        let mut vectors = Vec::new();

        // Cluster 0 area (around [0, 0])
        vectors.push((0, "f".to_string(), Vector::new(vec![0.0, 0.0])));

        // Cluster 1 area (around [10, 10]) - Dense
        for i in 0..10 {
            vectors.push((
                i + 1,
                "f".to_string(),
                Vector::new(vec![10.0 + i as f32 * 0.1, 10.0 + i as f32 * 0.1]),
            ));
        }

        // Cluster 2 area (around [0, 10])
        for i in 0..4 {
            vectors.push((
                i + 11,
                "f".to_string(),
                Vector::new(vec![0.0 + i as f32 * 0.1, 10.0 + i as f32 * 0.1]),
            ));
        }

        // Cluster 3 area (around [10, 0])
        for i in 0..4 {
            vectors.push((
                i + 15,
                "f".to_string(),
                Vector::new(vec![10.0 + i as f32 * 0.1, 0.0 + i as f32 * 0.1]),
            ));
        }

        writer.build(vectors).unwrap();
        writer.finalize().unwrap();

        let initial_stats = writer.get_cluster_stats();
        println!("Initial stats: {:?}", initial_stats);

        // Optimize should trigger merging and splitting
        // avg = 19 / 4 = 4.75
        // sparse_threshold = 4.75 / 4 = 1.18 -> Cluster 0 (1 vector) should be merged
        // dense_threshold = 4.75 * 4 = 19 -> No cluster is dense enough with factor 4?
        // Let's adjust thresholds for the test or use manual calls.

        // Manual merge test
        let merged = writer.merge_sparse_clusters(2).unwrap();
        assert!(merged > 0, "Should have merged at least one sparse cluster");

        let stats_after_merge = writer.get_cluster_stats();
        assert_eq!(stats_after_merge.len(), initial_stats.len() - merged);

        // Manual split test
        // Let's split anything > 5
        let split = writer.split_dense_clusters(5).unwrap();
        assert!(split > 0, "Should have split at least one dense cluster");

        let stats_after_split = writer.get_cluster_stats();
        assert_eq!(stats_after_split.len(), stats_after_merge.len() + split);

        // Verify all vectors are still present
        let total_vectors: usize = stats_after_split.iter().map(|s| s.count).sum();
        assert_eq!(total_vectors, 19);
    }
}
