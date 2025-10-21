//! Vector Search using CandleTextEmbedder
//!
//! This example demonstrates:
//! - Using CandleTextEmbedder to generate text embeddings
//! - Building a vector index with real semantic embeddings
//! - Performing semantic similarity search on text documents
//! - Comparing search results across different query types
//!
//! To run this example:
//! ```bash
//! cargo run --example vector_search --features embeddings-candle
//! ```

#[cfg(feature = "embeddings-candle")]
use sage::embedding::{CandleTextEmbedder, TextEmbedder};
#[cfg(feature = "embeddings-candle")]
use sage::error::Result;
#[cfg(feature = "embeddings-candle")]
use sage::vector::DistanceMetric;
#[cfg(feature = "embeddings-candle")]
use sage::vector::index::{VectorIndexBuildConfig, VectorIndexBuilderFactory, VectorIndexType};
#[cfg(feature = "embeddings-candle")]
use sage::vector::types::VectorSearchConfig;

#[cfg(feature = "embeddings-candle")]
#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Vector Search with CandleTextEmbedder ===\n");

    // Step 1: Initialize the embedder
    println!("Loading embedding model: sentence-transformers/all-MiniLM-L6-v2");
    let embedder = CandleTextEmbedder::new("sentence-transformers/all-MiniLM-L6-v2")?;
    let dimension = embedder.dimension();

    println!("Model loaded successfully!");
    println!("Model name: {}", embedder.name());
    println!("Embedding dimension: {}\n", dimension);

    // Step 2: Prepare sample documents
    let documents = vec![
        (
            1,
            "Rust is a systems programming language focused on safety and performance",
        ),
        (
            2,
            "Python is a high-level programming language known for its simplicity",
        ),
        (
            3,
            "Machine learning algorithms can recognize patterns in data",
        ),
        (
            4,
            "Deep neural networks are the foundation of modern AI systems",
        ),
        (5, "The Eiffel Tower is a famous landmark in Paris, France"),
        (
            6,
            "Sushi is a traditional Japanese cuisine made with rice and fish",
        ),
        (
            7,
            "Quantum computing leverages quantum mechanics for computation",
        ),
        (
            8,
            "Climate change affects global weather patterns and ecosystems",
        ),
    ];

    println!("=== Sample Documents ===");
    for (id, text) in &documents {
        println!("Doc {}: {}", id, text);
    }
    println!();

    // Step 3: Generate embeddings for all documents
    println!("Generating embeddings for {} documents...", documents.len());
    let texts: Vec<&str> = documents.iter().map(|(_, text)| *text).collect();
    let vectors = embedder.embed_batch(&texts).await?;
    println!("Embeddings generated successfully!\n");

    // Step 4: Create vector index configuration
    let vector_config = VectorIndexBuildConfig {
        dimension,
        distance_metric: DistanceMetric::Cosine,
        index_type: VectorIndexType::Flat,
        normalize_vectors: true,
        ..Default::default()
    };

    // Step 5: Build the vector index
    println!("Building vector index...");
    let mut vector_builder = VectorIndexBuilderFactory::create_builder(vector_config)?;

    // Add document vectors to the index
    let doc_vectors: Vec<(u64, sage::vector::Vector)> = documents
        .iter()
        .zip(vectors.iter())
        .map(|((id, _), vector)| (*id, vector.clone()))
        .collect();

    vector_builder.add_vectors(doc_vectors)?;
    vector_builder.finalize()?;
    vector_builder.optimize()?;

    println!("Vector index built successfully!");
    println!("Build progress: {:.1}%", vector_builder.progress() * 100.0);
    println!(
        "Estimated memory usage: {} bytes\n",
        vector_builder.estimated_memory_usage()
    );

    // Step 6: Perform semantic searches
    println!("=== Semantic Search Examples ===\n");

    // Search 1: Programming language query
    println!("--- Search 1: Programming Languages ---");
    let query1 = "programming language features";
    println!("Query: \"{}\"", query1);

    let query_vector1 = embedder.embed(query1).await?;
    let results1 = perform_search(&documents, &vectors, &query_vector1, 3);

    println!("Top 3 results:");
    for (rank, (doc_id, text, similarity)) in results1.iter().enumerate() {
        println!(
            "  {}. Doc {} (similarity: {:.4}): {}",
            rank + 1,
            doc_id,
            similarity,
            text
        );
    }
    println!();

    // Search 2: AI and machine learning query
    println!("--- Search 2: Artificial Intelligence ---");
    let query2 = "artificial intelligence and neural networks";
    println!("Query: \"{}\"", query2);

    let query_vector2 = embedder.embed(query2).await?;
    let results2 = perform_search(&documents, &vectors, &query_vector2, 3);

    println!("Top 3 results:");
    for (rank, (doc_id, text, similarity)) in results2.iter().enumerate() {
        println!(
            "  {}. Doc {} (similarity: {:.4}): {}",
            rank + 1,
            doc_id,
            similarity,
            text
        );
    }
    println!();

    // Search 3: Food and culture query
    println!("--- Search 3: Food and Culture ---");
    let query3 = "traditional food and cuisine";
    println!("Query: \"{}\"", query3);

    let query_vector3 = embedder.embed(query3).await?;
    let results3 = perform_search(&documents, &vectors, &query_vector3, 3);

    println!("Top 3 results:");
    for (rank, (doc_id, text, similarity)) in results3.iter().enumerate() {
        println!(
            "  {}. Doc {} (similarity: {:.4}): {}",
            rank + 1,
            doc_id,
            similarity,
            text
        );
    }
    println!();

    // Step 7: Demonstrate search configuration
    println!("=== Vector Search Configuration ===");
    let search_config = VectorSearchConfig {
        top_k: 3,
        min_similarity: 0.3,
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

    println!("\n=== Example completed successfully! ===");
    Ok(())
}

#[cfg(feature = "embeddings-candle")]
fn perform_search(
    documents: &[(u64, &str)],
    vectors: &[sage::vector::Vector],
    query_vector: &sage::vector::Vector,
    top_k: usize,
) -> Vec<(u64, String, f32)> {
    let mut results: Vec<(u64, String, f32)> = documents
        .iter()
        .zip(vectors.iter())
        .map(|((doc_id, text), vector)| {
            // Calculate cosine similarity (vectors are normalized)
            let similarity: f32 = query_vector
                .data
                .iter()
                .zip(vector.data.iter())
                .map(|(a, b)| a * b)
                .sum();

            (*doc_id, text.to_string(), similarity)
        })
        .collect();

    // Sort by similarity (descending)
    results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

    // Return top K results
    results.into_iter().take(top_k).collect()
}

#[cfg(not(feature = "embeddings-candle"))]
fn main() {
    eprintln!("This example requires the 'embeddings-candle' feature.");
    eprintln!("Please run with: cargo run --example vector_search --features embeddings-candle");
    std::process::exit(1);
}
