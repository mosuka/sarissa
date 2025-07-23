//! Vector Similarity Search example - demonstrates semantic similarity search using dense vectors.

use sarissa::index::index::IndexConfig;
use sarissa::prelude::*;
use sarissa::schema::{IdField, TextField};
use sarissa::search::SearchEngine;
use sarissa::vector::{
    DistanceMetric, Vector, VectorSearchConfig,
};
use sarissa::vector::index::{VectorIndexConfig, VectorIndexFactory, VectorIndexType};
use tempfile::TempDir;

fn main() -> Result<()> {
    println!("=== Vector Similarity Search Example - Semantic Search with Dense Vectors ===\n");

    // Create a temporary directory for the index
    let temp_dir = TempDir::new().unwrap();
    println!("Creating index in: {:?}", temp_dir.path());

    // Create a schema with text fields
    let mut schema = Schema::new();
    schema.add_field(
        "title",
        Box::new(TextField::new().stored(true).indexed(true)),
    )?;
    schema.add_field("content", Box::new(TextField::new().indexed(true)))?;
    schema.add_field("category", Box::new(TextField::new().indexed(true)))?;
    schema.add_field("id", Box::new(IdField::new()))?;

    // Create a search engine
    let mut engine = SearchEngine::create_in_dir(temp_dir.path(), schema, IndexConfig::default())?;

    println!("=== Vector Index Setup ===\n");

    // Create vector index configuration
    let vector_config = VectorIndexConfig {
        dimension: 4, // Small dimension for demo purposes
        distance_metric: DistanceMetric::Cosine,
        index_type: VectorIndexType::Flat,
        normalize_vectors: true,
        initial_capacity: 100,
    };

    // Create vector index
    let mut vector_index = VectorIndexFactory::create(vector_config)?;
    println!("Created {} index with {} dimensions", 
             vector_index.distance_metric().name(), 
             vector_index.dimension());

    // Add sample documents with corresponding vectors
    println!("\n=== Adding Documents and Vectors ===\n");

    let documents_with_vectors = vec![
        (
            Document::builder()
                .add_text("title", "Machine Learning Fundamentals")
                .add_text("content", "Introduction to algorithms, neural networks, and data science concepts")
                .add_text("category", "technology")
                .add_text("id", "doc001")
                .build(),
            Vector::new(vec![0.9, 0.1, 0.2, 0.1]), // Technology-focused vector
        ),
        (
            Document::builder()
                .add_text("title", "Deep Learning with Python")
                .add_text("content", "Advanced neural network architectures and machine learning frameworks")
                .add_text("category", "technology")
                .add_text("id", "doc002")
                .build(),
            Vector::new(vec![0.8, 0.2, 0.3, 0.1]), // Similar to ML fundamentals
        ),
        (
            Document::builder()
                .add_text("title", "Cooking Italian Pasta")
                .add_text("content", "Traditional recipes and techniques for making authentic Italian pasta dishes")
                .add_text("category", "cooking")
                .add_text("id", "doc003")
                .build(),
            Vector::new(vec![0.1, 0.9, 0.1, 0.2]), // Cooking-focused vector
        ),
        (
            Document::builder()
                .add_text("title", "French Cuisine Masterclass")
                .add_text("content", "Classic French cooking techniques and gourmet recipe collection")
                .add_text("category", "cooking")
                .add_text("id", "doc004")
                .build(),
            Vector::new(vec![0.2, 0.8, 0.2, 0.3]), // Similar to Italian cooking
        ),
        (
            Document::builder()
                .add_text("title", "Digital Photography Basics")
                .add_text("content", "Camera settings, composition techniques, and photo editing fundamentals")
                .add_text("category", "photography")
                .add_text("id", "doc005")
                .build(),
            Vector::new(vec![0.3, 0.1, 0.8, 0.2]), // Photography-focused vector
        ),
        (
            Document::builder()
                .add_text("title", "Portrait Photography Guide")
                .add_text("content", "Professional techniques for capturing stunning portrait photographs")
                .add_text("category", "photography")
                .add_text("id", "doc006")
                .build(),
            Vector::new(vec![0.2, 0.2, 0.9, 0.1]), // Similar to general photography
        ),
        (
            Document::builder()
                .add_text("title", "Travel Planning Essentials")
                .add_text("content", "Tips for booking flights, accommodations, and creating travel itineraries")
                .add_text("category", "travel")
                .add_text("id", "doc007")
                .build(),
            Vector::new(vec![0.1, 0.3, 0.2, 0.9]), // Travel-focused vector
        ),
        (
            Document::builder()
                .add_text("title", "Budget Travel Hacks")
                .add_text("content", "Money-saving strategies for affordable travel and backpacking")
                .add_text("category", "travel")
                .add_text("id", "doc008")
                .build(),
            Vector::new(vec![0.2, 0.2, 0.1, 0.8]), // Similar to travel planning
        ),
    ];

    // Add documents to text search engine
    let documents: Vec<Document> = documents_with_vectors.iter().map(|(doc, _)| doc.clone()).collect();
    engine.add_documents(documents)?;

    // Add vectors to vector index
    for (i, (_, vector)) in documents_with_vectors.iter().enumerate() {
        let doc_id = (i + 1) as u64; // Use 1-based indexing for doc IDs
        vector_index.add_vector(doc_id, vector.clone())?;
        println!("Added vector for document {}: {:?}", doc_id, vector.data);
    }

    println!("\nAdded {} documents and vectors to indices", documents_with_vectors.len());

    // Display vector index statistics
    let stats = vector_index.stats();
    println!("\n=== Vector Index Statistics ===");
    println!("Total vectors: {}", stats.total_vectors);
    println!("Vector dimension: {}", stats.dimension);
    println!("Average norm: {:.4}", stats.avg_norm);
    println!("Min norm: {:.4}", stats.min_norm);
    println!("Max norm: {:.4}", stats.max_norm);
    println!("Index size: {} bytes", stats.index_size_bytes);
    println!("Memory usage: {} bytes", stats.memory_usage_bytes);
    println!("Vectors per MB: {:.1}", stats.vectors_per_mb());

    println!("\n=== Vector Search Examples ===\n");

    // Example 1: Search for technology-related content
    println!("1. Searching for technology-related content:");
    let tech_query = Vector::new(vec![1.0, 0.0, 0.0, 0.0]); // Pure technology vector
    let search_config = VectorSearchConfig {
        distance_metric: DistanceMetric::Cosine,
        top_k: 3,
        min_similarity: 0.0,
        normalize_vectors: true,
        include_vectors: true,
        include_metadata: false,
        parallel: false,
    };
    
    let results = vector_index.search(&tech_query, &search_config)?;
    println!("   Query vector: {:?}", tech_query.data);
    println!("   Found {} results in {}ms", results.len(), results.query_time_ms);
    
    for (i, result) in results.results.iter().enumerate() {
        println!("   {}. Doc ID: {}, Similarity: {:.4}, Distance: {:.4}", 
                 i + 1, result.doc_id, result.similarity, result.distance);
        if let Some(vector) = &result.vector {
            println!("      Vector: {:?}", vector.data);
        }
    }

    // Example 2: Search for cooking-related content
    println!("\n2. Searching for cooking-related content:");
    let cooking_query = Vector::new(vec![0.0, 1.0, 0.0, 0.0]); // Pure cooking vector
    let results = vector_index.search(&cooking_query, &search_config)?;
    println!("   Query vector: {:?}", cooking_query.data);
    println!("   Found {} results in {}ms", results.len(), results.query_time_ms);
    
    for (i, result) in results.results.iter().enumerate() {
        println!("   {}. Doc ID: {}, Similarity: {:.4}, Distance: {:.4}", 
                 i + 1, result.doc_id, result.similarity, result.distance);
    }

    // Example 3: Search for photography-related content
    println!("\n3. Searching for photography-related content:");
    let photo_query = Vector::new(vec![0.0, 0.0, 1.0, 0.0]); // Pure photography vector
    let results = vector_index.search(&photo_query, &search_config)?;
    println!("   Query vector: {:?}", photo_query.data);
    println!("   Found {} results in {}ms", results.len(), results.query_time_ms);
    
    for (i, result) in results.results.iter().enumerate() {
        println!("   {}. Doc ID: {}, Similarity: {:.4}, Distance: {:.4}", 
                 i + 1, result.doc_id, result.similarity, result.distance);
    }

    // Example 4: Search with mixed interests
    println!("\n4. Searching with mixed interests (tech + cooking):");
    let mixed_query = Vector::new(vec![0.7, 0.3, 0.0, 0.0]); // 70% tech, 30% cooking
    let results = vector_index.search(&mixed_query, &search_config)?;
    println!("   Query vector: {:?}", mixed_query.data);
    println!("   Found {} results in {}ms", results.len(), results.query_time_ms);
    
    for (i, result) in results.results.iter().enumerate() {
        println!("   {}. Doc ID: {}, Similarity: {:.4}, Distance: {:.4}", 
                 i + 1, result.doc_id, result.similarity, result.distance);
    }

    println!("\n=== Distance Metrics Comparison ===\n");

    // Compare different distance metrics
    let query_vector = Vector::new(vec![0.5, 0.5, 0.0, 0.0]);
    let target_vector = Vector::new(vec![0.8, 0.2, 0.1, 0.1]);

    let metrics = vec![
        DistanceMetric::Cosine,
        DistanceMetric::Euclidean,
        DistanceMetric::Manhattan,
        DistanceMetric::DotProduct,
        DistanceMetric::Angular,
    ];

    println!("Comparing distance metrics between vectors:");
    println!("Query:  {:?}", query_vector.data);
    println!("Target: {:?}", target_vector.data);
    println!();

    for metric in metrics {
        let distance = metric.distance(&query_vector.data, &target_vector.data)?;
        let similarity = metric.similarity(&query_vector.data, &target_vector.data)?;
        println!("{:12}: Distance = {:.4}, Similarity = {:.4}", 
                 metric.name(), distance, similarity);
    }

    println!("\n=== Advanced Search Configuration ===\n");

    // Example 5: Search with minimum similarity threshold
    println!("5. Search with minimum similarity threshold (0.8):");
    let high_threshold_config = VectorSearchConfig {
        min_similarity: 0.8,
        top_k: 10,
        ..search_config
    };
    
    let tech_results = vector_index.search(&tech_query, &high_threshold_config)?;
    println!("   Found {} results with similarity >= 0.8", tech_results.len());
    
    for (i, result) in tech_results.results.iter().enumerate() {
        println!("   {}. Doc ID: {}, Similarity: {:.4}", 
                 i + 1, result.doc_id, result.similarity);
    }

    // Example 6: Search with different distance metrics
    println!("\n6. Comparing search results with different distance metrics:");
    
    let euclidean_config = VectorSearchConfig {
        distance_metric: DistanceMetric::Euclidean,
        top_k: 3,
        min_similarity: 0.0,
        normalize_vectors: true,
        include_vectors: false,
        include_metadata: false,
        parallel: false,
    };

    // Create euclidean index for comparison
    let euclidean_index_config = VectorIndexConfig {
        dimension: 4,
        distance_metric: DistanceMetric::Euclidean,
        index_type: VectorIndexType::Flat,
        normalize_vectors: true,
        initial_capacity: 100,
    };

    let mut euclidean_index = VectorIndexFactory::create(euclidean_index_config)?;
    
    // Add same vectors to euclidean index
    for (i, (_, vector)) in documents_with_vectors.iter().enumerate() {
        let doc_id = (i + 1) as u64;
        euclidean_index.add_vector(doc_id, vector.clone())?;
    }

    let cosine_results = vector_index.search(&tech_query, &search_config)?;
    let euclidean_results = euclidean_index.search(&tech_query, &euclidean_config)?;

    println!("   Cosine similarity results:");
    for (i, result) in cosine_results.results.iter().take(3).enumerate() {
        println!("     {}. Doc ID: {}, Similarity: {:.4}", 
                 i + 1, result.doc_id, result.similarity);
    }

    println!("   Euclidean distance results:");
    for (i, result) in euclidean_results.results.iter().take(3).enumerate() {
        println!("     {}. Doc ID: {}, Similarity: {:.4}", 
                 i + 1, result.doc_id, result.similarity);
    }

    println!("\n=== Vector Operations Demo ===\n");

    // Example 7: Vector operations and analysis
    println!("7. Vector operations and analysis:");
    
    let mut demo_vector = Vector::new(vec![3.0, 4.0, 0.0, 0.0]);
    println!("   Original vector: {:?}", demo_vector.data);
    println!("   Vector norm: {:.4}", demo_vector.norm());
    
    demo_vector.normalize();
    println!("   Normalized vector: {:?}", demo_vector.data);
    println!("   Normalized norm: {:.4}", demo_vector.norm());

    // Add metadata to vector
    demo_vector.add_metadata("type".to_string(), "demo".to_string());
    demo_vector.add_metadata("created".to_string(), "2024".to_string());
    println!("   Vector metadata: {:?}", demo_vector.metadata);

    // Vector validation
    let valid_vector = Vector::new(vec![1.0, 2.0, 3.0, 4.0]);
    let invalid_vector = Vector::new(vec![1.0, f32::NAN, 3.0, 4.0]);
    
    println!("   Valid vector check: {}", valid_vector.is_valid());
    println!("   Invalid vector check: {}", invalid_vector.is_valid());

    // Dimension validation
    println!("   Dimension validation (4D): {}", 
             valid_vector.validate_dimension(4).is_ok());
    println!("   Dimension validation (3D): {}", 
             valid_vector.validate_dimension(3).is_ok());

    println!("\n=== Batch Operations Demo ===\n");

    // Example 8: Batch vector operations
    println!("8. Batch vector operations:");
    
    let query_batch = vec![1.0, 0.0, 0.0, 0.0];
    let v1 = vec![0.9, 0.1, 0.0, 0.0];
    let v2 = vec![0.0, 1.0, 0.0, 0.0];
    let v3 = vec![0.0, 0.0, 1.0, 0.0];
    let v4 = vec![0.5, 0.5, 0.0, 0.0];
    let vector_batch = vec![v1.as_slice(), v2.as_slice(), v3.as_slice(), v4.as_slice()];

    let batch_distances = DistanceMetric::Cosine
        .batch_distance_parallel(&query_batch, &vector_batch)?;
    let batch_similarities = DistanceMetric::Cosine
        .batch_similarity_parallel(&query_batch, &vector_batch)?;

    println!("   Query vector: {:?}", query_batch);
    println!("   Batch distances: {:?}", batch_distances);
    println!("   Batch similarities: {:?}", batch_similarities);

    // Batch normalization
    let mut vector_batch_objects = vec![
        Vector::new(vec![3.0, 4.0, 0.0, 0.0]),
        Vector::new(vec![1.0, 1.0, 1.0, 1.0]),
        Vector::new(vec![2.0, 0.0, 0.0, 1.0]),
    ];

    println!("   Before normalization:");
    for (i, v) in vector_batch_objects.iter().enumerate() {
        println!("     Vector {}: {:?} (norm: {:.4})", i, v.data, v.norm());
    }

    Vector::normalize_batch_parallel(&mut vector_batch_objects);

    println!("   After normalization:");
    for (i, v) in vector_batch_objects.iter().enumerate() {
        println!("     Vector {}: {:?} (norm: {:.4})", i, v.data, v.norm());
    }

    println!("\n=== Performance Analysis ===\n");

    // Example 9: Performance measurement
    println!("9. Performance measurement:");
    
    let start_time = std::time::Instant::now();
    let performance_query = Vector::new(vec![0.25, 0.25, 0.25, 0.25]);
    let results = vector_index.search(&performance_query, &search_config)?;
    let search_duration = start_time.elapsed();

    println!("   Search completed in: {:?}", search_duration);
    println!("   Results returned: {}", results.len());
    println!("   Total vectors searched: {}", results.total_searched);
    println!("   Query time reported: {}ms", results.query_time_ms);

    // Index operations performance
    let start_time = std::time::Instant::now();
    let new_vector = Vector::new(vec![0.1, 0.2, 0.3, 0.4]);
    vector_index.add_vector(999, new_vector)?;
    let add_duration = start_time.elapsed();

    println!("   Vector addition time: {:?}", add_duration);

    let start_time = std::time::Instant::now();
    let retrieved = vector_index.get_vector(999)?;
    let retrieval_duration = start_time.elapsed();

    println!("   Vector retrieval time: {:?}", retrieval_duration);
    println!("   Retrieved vector: {:?}", retrieved.unwrap().data);

    println!("\n=== Vector Search Key Features ===");
    println!("• Dense vector similarity search");
    println!("• Multiple distance metrics (Cosine, Euclidean, Manhattan, etc.)");
    println!("• Configurable search parameters (top-k, similarity threshold)");
    println!("• Vector normalization and validation");
    println!("• Batch operations for performance");
    println!("• Parallel processing support");
    println!("• Memory-efficient flat index for exact search");
    println!("• Integration with text search engines");

    println!("\n=== Use Cases ===");
    println!("• Semantic search: 'Find documents similar in meaning'");
    println!("• Content recommendation: 'Users who liked this also liked'");
    println!("• Document clustering: 'Group similar documents together'");
    println!("• Duplicate detection: 'Find near-duplicate content'");
    println!("• Multi-modal search: 'Search images by text description'");
    println!("• Knowledge graph embedding: 'Entity relationship modeling'");
    println!("• Personalization: 'Content matching user preferences'");

    println!("\n=== Distance Metrics Guide ===");
    println!("Cosine Similarity:");
    println!("  • Best for: Text similarity, normalized vectors");
    println!("  • Range: 0-1 (1 = identical direction)");
    println!("  • Ignores magnitude, focuses on direction");

    println!("\nEuclidean Distance:");
    println!("  • Best for: Spatial data, when magnitude matters");
    println!("  • Range: 0-∞ (0 = identical points)");
    println!("  • Considers both direction and magnitude");

    println!("\nManhattan Distance:");
    println!("  • Best for: Grid-like data, feature differences");
    println!("  • Range: 0-∞ (0 = identical)");
    println!("  • Sum of absolute differences");

    println!("\nDot Product:");
    println!("  • Best for: When both direction and magnitude matter");
    println!("  • Range: -∞ to +∞ (higher = more similar)");
    println!("  • Raw similarity without normalization");

    engine.close()?;
    println!("\nVector Similarity Search example completed successfully!");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_similarity_search_example() {
        let result = main();
        assert!(result.is_ok());
    }
}