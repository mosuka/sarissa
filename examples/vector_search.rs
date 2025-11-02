//! Vector Search using CandleTextEmbedder
//!
//! This example demonstrates:
//! - Creating a VectorEngine from VectorIndexFactory
//! - Using CandleTextEmbedder to generate text embeddings
//! - Adding document vectors to the engine
//! - Performing semantic similarity search on text documents
//! - Comparing search results across different query types
//!
//! To run this example:
//! ```bash
//! cargo run --example vector_search --features embeddings-candle
//! ```

#[cfg(feature = "embeddings-candle")]
use tempfile::TempDir;

#[cfg(feature = "embeddings-candle")]
use yatagarasu::embedding::candle_text_embedder::CandleTextEmbedder;
#[cfg(feature = "embeddings-candle")]
use yatagarasu::embedding::text_embedder::TextEmbedder;
#[cfg(feature = "embeddings-candle")]
use yatagarasu::error::Result;
#[cfg(feature = "embeddings-candle")]
use yatagarasu::storage::file::FileStorageConfig;
#[cfg(feature = "embeddings-candle")]
use yatagarasu::storage::{StorageConfig, StorageFactory};
#[cfg(feature = "embeddings-candle")]
use yatagarasu::vector::DistanceMetric;
#[cfg(feature = "embeddings-candle")]
use yatagarasu::vector::Vector;
#[cfg(feature = "embeddings-candle")]
use yatagarasu::vector::engine::VectorEngine;
#[cfg(feature = "embeddings-candle")]
use yatagarasu::vector::index::factory::VectorIndexFactory;
#[cfg(feature = "embeddings-candle")]
use yatagarasu::vector::index::{FlatIndexConfig, VectorIndexConfig};
#[cfg(feature = "embeddings-candle")]
use yatagarasu::vector::search::searcher::VectorSearchRequest;

#[cfg(feature = "embeddings-candle")]
#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Vector Search with CandleTextEmbedder ===\n");

    // Step 1: Initialize the embedder
    println!("Step 1: Loading embedding model: sentence-transformers/all-MiniLM-L6-v2...");
    let embedder = CandleTextEmbedder::new("sentence-transformers/all-MiniLM-L6-v2")?;
    println!();

    // Step 2: Create a vector search engine
    println!("Step 2: Create a vector search engine...");

    // Create a storage backend
    let temp_dir = TempDir::new().unwrap();
    let storage =
        StorageFactory::create(StorageConfig::File(FileStorageConfig::new(temp_dir.path())))?;

    // Create vector index
    let vector_index_config = VectorIndexConfig::Flat(FlatIndexConfig {
        dimension: embedder.dimension(),
        distance_metric: DistanceMetric::Cosine,
        normalize_vectors: true,
        ..Default::default()
    });
    let vector_index = VectorIndexFactory::create(storage, vector_index_config)?;

    // Create a vector engine
    let mut vector_engine = VectorEngine::new(vector_index)?;
    println!();

    // Step 3: Prepare documents
    println!("Step 3: Prepare documents...");
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
    for (id, text) in &documents {
        println!("  Doc {}: {}", id, text);
    }
    println!();

    // Step 4: Generate embeddings for all documents
    println!(
        "Step 4: Generating embeddings for {} documents...",
        documents.len()
    );
    let texts: Vec<&str> = documents.iter().map(|(_, text)| *text).collect();
    let vectors = embedder.embed_batch(&texts).await?;
    println!();

    println!("Step 5: Add document vectors to the engine...");
    // Add document vectors to the engine
    let doc_vectors: Vec<(u64, Vector)> = documents
        .iter()
        .zip(vectors.iter())
        .map(|((id, _), vector)| (*id, vector.clone()))
        .collect();
    vector_engine.add_vectors(doc_vectors)?;
    vector_engine.commit()?;
    vector_engine.optimize()?;

    println!("  Vector index built successfully!");
    println!("  Build progress: {:.1}%", vector_engine.progress() * 100.0);
    println!(
        "  Estimated memory usage: {} bytes\n",
        vector_engine.estimated_memory_usage()
    );

    // Step 6: Perform semantic searches
    println!("Step 6: Demonstrating semantic search...\n");
    println!("{}", "=".repeat(80));

    // Search 1: Programming language query
    println!("\n[1] Programming Languages Search");
    println!("{}", "-".repeat(80));
    let query1 = "programming language features";
    println!("Query: \"{}\"", query1);

    let query_vector1 = embedder.embed(query1).await?;
    let request1 = VectorSearchRequest::new(query_vector1).top_k(3);
    let results1 = vector_engine.search(request1)?;

    println!("\nTop 3 results:");
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

    // Search 2: AI and machine learning query
    println!("\n[2] Artificial Intelligence Search");
    println!("{}", "-".repeat(80));
    let query2 = "artificial intelligence and neural networks";
    println!("Query: \"{}\"", query2);

    let query_vector2 = embedder.embed(query2).await?;
    let request2 = VectorSearchRequest::new(query_vector2).top_k(3);
    let results2 = vector_engine.search(request2)?;

    println!("\nTop 3 results:");
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

    // Search 3: Food and culture query
    println!("\n[3] Food and Culture Search");
    println!("{}", "-".repeat(80));
    let query3 = "traditional food and cuisine";
    println!("Query: \"{}\"", query3);

    let query_vector3 = embedder.embed(query3).await?;
    let request3 = VectorSearchRequest::new(query_vector3).top_k(3);
    let results3 = vector_engine.search(request3)?;

    println!("\nTop 3 results:");
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

    // Step 7: Demonstrate search configuration with builder pattern
    println!("\n{}", "=".repeat(80));
    println!("\n[Advanced] Vector Search Configuration");
    println!("{}", "-".repeat(80));
    let demo_query = Vector::new(vec![0.0; embedder.dimension()]);
    let search_request = VectorSearchRequest::new(demo_query)
        .top_k(3)
        .min_similarity(0.3)
        .include_scores(true)
        .include_vectors(false)
        .timeout_ms(1000);

    println!("Search configuration:");
    println!("  Top K: {}", search_request.params.top_k);
    println!("  Min similarity: {}", search_request.params.min_similarity);
    println!("  Include scores: {}", search_request.params.include_scores);
    println!(
        "  Include vectors: {}",
        search_request.params.include_vectors
    );
    println!("  Timeout: {:?} ms", search_request.params.timeout_ms);

    println!("\n{}", "=".repeat(80));
    println!("\n=== Example completed successfully! ===");
    Ok(())
}

#[cfg(not(feature = "embeddings-candle"))]
fn main() {
    eprintln!("This example requires the 'embeddings-candle' feature.");
    eprintln!("Please run with: cargo run --example vector_search --features embeddings-candle");
    std::process::exit(1);
}
