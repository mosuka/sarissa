//! Vector Similarity Search example - demonstrates the new vector API structure.

use sage::prelude::*;
use sage::vector::{DistanceMetric, Vector, types::VectorSearchConfig};
use sage::vector_index::{VectorIndexBuildConfig, VectorIndexBuilderFactory, VectorIndexType};

fn main() -> Result<()> {
    println!("=== Vector Similarity Search Example - New API Structure ===\n");

    // Create vector index configuration
    let vector_config = VectorIndexBuildConfig {
        dimension: 4, // Small dimension for demo purposes
        distance_metric: DistanceMetric::Cosine,
        index_type: VectorIndexType::Flat,
        normalize_vectors: true,
        ..Default::default()
    };

    // Create vector index builder
    let mut vector_builder = VectorIndexBuilderFactory::create_builder(vector_config)?;
    println!("Created Flat index with 4 dimensions using Cosine distance\n");

    // Create sample vectors for demonstration
    let sample_vectors = vec![
        (1, Vector::new(vec![0.9, 0.1, 0.2, 0.1])), // Technology-focused
        (2, Vector::new(vec![0.8, 0.2, 0.3, 0.1])), // Similar to technology
        (3, Vector::new(vec![0.1, 0.9, 0.1, 0.2])), // Cooking-focused
        (4, Vector::new(vec![0.2, 0.8, 0.2, 0.1])), // Similar to cooking
        (5, Vector::new(vec![0.1, 0.1, 0.9, 0.2])), // Photography-focused
        (6, Vector::new(vec![0.2, 0.2, 0.1, 0.9])), // Sports-focused
    ];

    println!("=== Adding Vectors to Index ===");
    for (doc_id, vector) in &sample_vectors {
        println!("Added vector for document {}: {:?}", doc_id, vector.data);
    }

    // Add vectors to the builder
    vector_builder.add_vectors(sample_vectors)?;

    // Finalize the index
    vector_builder.finalize()?;
    println!("\nIndex finalized successfully!");

    // Display progress and stats
    println!("Build progress: {:.1}%", vector_builder.progress() * 100.0);
    println!(
        "Estimated memory usage: {} bytes",
        vector_builder.estimated_memory_usage()
    );

    // Optimize the index
    vector_builder.optimize()?;
    println!("Index optimized successfully!");

    // Demonstrate vector operations
    println!("\n=== Vector Operations ===");

    // Create test query vectors
    let tech_query = Vector::new(vec![0.85, 0.15, 0.25, 0.1]);
    let _cooking_query = Vector::new(vec![0.15, 0.85, 0.15, 0.2]);

    // Show distance calculations
    let sample_doc = Vector::new(vec![0.9, 0.1, 0.2, 0.1]);

    println!("Query vector: {:?}", tech_query.data);
    println!("Sample document vector: {:?}", sample_doc.data);

    let cosine_distance = DistanceMetric::Cosine.distance(&tech_query.data, &sample_doc.data)?;
    let cosine_similarity =
        DistanceMetric::Cosine.similarity(&tech_query.data, &sample_doc.data)?;

    println!("Cosine distance: {cosine_distance:.4}");
    println!("Cosine similarity: {cosine_similarity:.4}");

    // Demonstrate different distance metrics
    println!("\n=== Distance Metrics Comparison ===");
    let metrics = vec![
        DistanceMetric::Cosine,
        DistanceMetric::Euclidean,
        DistanceMetric::Manhattan,
        DistanceMetric::DotProduct,
    ];

    for metric in metrics {
        let distance = metric.distance(&tech_query.data, &sample_doc.data)?;
        let similarity = metric.similarity(&tech_query.data, &sample_doc.data)?;
        println!(
            "{}: distance={:.4}, similarity={:.4}",
            metric.name(),
            distance,
            similarity
        );
    }

    // Demonstrate vector normalization
    println!("\n=== Vector Normalization ===");
    let mut test_vector = Vector::new(vec![3.0, 4.0, 0.0, 0.0]);
    println!("Original vector: {:?}", test_vector.data);
    println!("Original norm: {:.4}", test_vector.norm());

    test_vector.normalize();
    println!("Normalized vector: {:?}", test_vector.data);
    println!("Normalized norm: {:.4}", test_vector.norm());

    // Demonstrate batch normalization
    let mut batch_vectors = vec![
        Vector::new(vec![1.0, 2.0, 3.0, 4.0]),
        Vector::new(vec![5.0, 6.0, 7.0, 8.0]),
        Vector::new(vec![0.5, 1.5, 2.5, 3.5]),
    ];

    println!("\n=== Batch Normalization ===");
    println!("Before normalization:");
    for (i, vector) in batch_vectors.iter().enumerate() {
        println!(
            "  Vector {}: {:?} (norm: {:.4})",
            i,
            vector.data,
            vector.norm()
        );
    }

    Vector::normalize_batch_parallel(&mut batch_vectors);

    println!("After parallel normalization:");
    for (i, vector) in batch_vectors.iter().enumerate() {
        println!(
            "  Vector {}: {:?} (norm: {:.4})",
            i,
            vector.data,
            vector.norm()
        );
    }

    // Demonstrate parallel distance calculations
    println!("\n=== Parallel Distance Calculations ===");
    let query = vec![1.0, 0.0, 0.0, 0.0];
    let vectors: Vec<&[f32]> = batch_vectors.iter().map(|v| v.data.as_slice()).collect();

    let distances = DistanceMetric::Cosine.batch_distance_parallel(&query, &vectors)?;
    let similarities = DistanceMetric::Cosine.batch_similarity_parallel(&query, &vectors)?;

    println!("Query: {query:?}");
    for (i, (dist, sim)) in distances.iter().zip(similarities.iter()).enumerate() {
        println!("  Vector {i}: distance={dist:.4}, similarity={sim:.4}");
    }

    println!("\n=== Vector Search Configuration ===");
    let search_config = VectorSearchConfig {
        top_k: 3,
        min_similarity: 0.5,
        include_scores: true,
        include_vectors: false,
        timeout_ms: Some(1000),
    };

    println!("Search configuration:");
    println!("  Top K: {}", search_config.top_k);
    println!("  Min similarity: {}", search_config.min_similarity);
    println!("  Include scores: {}", search_config.include_scores);
    println!("  Include vectors: {}", search_config.include_vectors);
    println!("  Timeout: {:?} ms", search_config.timeout_ms);

    Ok(())
}
