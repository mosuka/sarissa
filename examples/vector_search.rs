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
use sage::vector::engine::VectorEngine;
#[cfg(feature = "embeddings-candle")]
use sage::vector::index::{VectorIndexWriterConfig, VectorIndexType};
#[cfg(feature = "embeddings-candle")]
use sage::vector::{Vector, VectorSearchRequest};

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
    let vector_config = VectorIndexWriterConfig {
        dimension,
        distance_metric: DistanceMetric::Cosine,
        index_type: VectorIndexType::Flat,
        normalize_vectors: true,
        ..Default::default()
    };

    // Step 5: Build the vector index using VectorEngine
    println!("Building vector index...");
    let mut engine = VectorEngine::create(vector_config)?;

    // Add document vectors to the index
    let doc_vectors: Vec<(u64, sage::vector::Vector)> = documents
        .iter()
        .zip(vectors.iter())
        .map(|((id, _), vector)| (*id, vector.clone()))
        .collect();

    engine.add_vectors(doc_vectors)?;
    engine.finalize()?;
    engine.optimize()?;

    println!("Vector index built successfully!");
    println!("Build progress: {:.1}%", engine.progress() * 100.0);
    println!(
        "Estimated memory usage: {} bytes\n",
        engine.estimated_memory_usage()
    );

    // Step 6: Perform semantic searches
    println!("=== Semantic Search Examples ===\n");

    // Search 1: Programming language query
    println!("--- Search 1: Programming Languages ---");
    let query1 = "programming language features";
    println!("Query: \"{}\"", query1);

    let query_vector1 = embedder.embed(query1).await?;
    let request1 = VectorSearchRequest::new(query_vector1).top_k(3);
    let results1 = engine.search(request1)?;

    println!("Top 3 results:");
    for (rank, result) in results1.results.iter().enumerate() {
        if let Some((text, _)) = documents.iter().find(|(id, _)| *id == result.doc_id) {
            println!(
                "  {}. Doc {} (score: {:.4}): {}",
                rank + 1,
                result.doc_id,
                result.similarity,
                text
            );
        }
    }
    println!();

    // Search 2: AI and machine learning query
    println!("--- Search 2: Artificial Intelligence ---");
    let query2 = "artificial intelligence and neural networks";
    println!("Query: \"{}\"", query2);

    let query_vector2 = embedder.embed(query2).await?;
    let request2 = VectorSearchRequest::new(query_vector2).top_k(3);
    let results2 = engine.search(request2)?;

    println!("Top 3 results:");
    for (rank, result) in results2.results.iter().enumerate() {
        if let Some((text, _)) = documents.iter().find(|(id, _)| *id == result.doc_id) {
            println!(
                "  {}. Doc {} (score: {:.4}): {}",
                rank + 1,
                result.doc_id,
                result.similarity,
                text
            );
        }
    }
    println!();

    // Search 3: Food and culture query
    println!("--- Search 3: Food and Culture ---");
    let query3 = "traditional food and cuisine";
    println!("Query: \"{}\"", query3);

    let query_vector3 = embedder.embed(query3).await?;
    let request3 = VectorSearchRequest::new(query_vector3).top_k(3);
    let results3 = engine.search(request3)?;

    println!("Top 3 results:");
    for (rank, result) in results3.results.iter().enumerate() {
        if let Some((text, _)) = documents.iter().find(|(id, _)| *id == result.doc_id) {
            println!(
                "  {}. Doc {} (score: {:.4}): {}",
                rank + 1,
                result.doc_id,
                result.similarity,
                text
            );
        }
    }
    println!();

    // Step 7: Demonstrate search configuration with builder pattern
    println!("=== Vector Search Configuration ===");
    let demo_query = Vector::new(vec![0.0; dimension]);
    let search_request = VectorSearchRequest::new(demo_query)
        .top_k(3)
        .min_similarity(0.3)
        .include_scores(true)
        .include_vectors(false)
        .timeout_ms(1000);

    println!("Search configuration:");
    println!("  Top K: {}", search_request.config.top_k);
    println!("  Min similarity: {}", search_request.config.min_similarity);
    println!("  Include scores: {}", search_request.config.include_scores);
    println!(
        "  Include vectors: {}",
        search_request.config.include_vectors
    );
    println!("  Timeout: {:?} ms", search_request.config.timeout_ms);

    println!("\n=== Example completed successfully! ===");
    Ok(())
}

#[cfg(not(feature = "embeddings-candle"))]
fn main() {
    eprintln!("This example requires the 'embeddings-candle' feature.");
    eprintln!("Please run with: cargo run --example vector_search --features embeddings-candle");
    std::process::exit(1);
}
