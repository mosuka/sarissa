//! Example of using CandleTextEmbedder with local BERT models
//!
//! This example demonstrates:
//! - Loading a sentence-transformers model from HuggingFace Hub
//! - Generating embeddings for text
//! - Batch processing multiple texts
//!
//! To run this example:
//! ```bash
//! cargo run --example embedding_with_candle --features embeddings-candle
//! ```

#[cfg(feature = "embeddings-candle")]
use yatagarasu::embedding::candle_text_embedder::CandleTextEmbedder;
#[cfg(feature = "embeddings-candle")]
use yatagarasu::embedding::text_embedder::TextEmbedder;

#[cfg(feature = "embeddings-candle")]
#[tokio::main]
async fn main() -> yatagarasu::error::Result<()> {
    println!("=== Candle Text Embedder Example ===\n");

    // Create embedder with a sentence-transformers model
    // This will download the model from HuggingFace Hub on first run
    println!("Loading model: sentence-transformers/all-MiniLM-L6-v2");
    let embedder = CandleTextEmbedder::new("sentence-transformers/all-MiniLM-L6-v2")?;

    println!("Model loaded successfully!");
    println!("Model name: {}", embedder.name());
    println!("Embedding dimension: {}\n", embedder.dimension());

    // Example 1: Single text embedding
    println!("--- Example 1: Single Text Embedding ---");
    let text = "Rust is a systems programming language";
    println!("Text: \"{}\"", text);

    let vector = embedder.embed(text).await?;
    println!("Generated embedding with {} dimensions", vector.dimension());
    println!(
        "First 5 values: {:?}\n",
        &vector.data[..5.min(vector.data.len())]
    );

    // Example 2: Batch processing
    println!("--- Example 2: Batch Processing ---");
    let texts = vec![
        "Machine learning is transforming technology",
        "Deep learning uses neural networks",
        "Natural language processing enables AI to understand text",
    ];

    println!("Processing {} texts...", texts.len());
    let vectors = embedder.embed_batch(&texts).await?;

    for (i, (text, vector)) in texts.iter().zip(vectors.iter()).enumerate() {
        println!("Text {}: \"{}\"", i + 1, text);
        println!(
            "  Embedding: {} dimensions, first 3 values: {:?}",
            vector.dimension(),
            &vector.data[..3.min(vector.data.len())]
        );
    }

    // Example 3: Semantic similarity
    println!("\n--- Example 3: Semantic Similarity ---");
    let query = "programming languages";
    let candidates = vec![
        "Rust and Python are popular programming languages",
        "I like to eat pizza for dinner",
        "Software development requires good tools",
    ];

    let query_vector = embedder.embed(query).await?;
    let candidate_vectors = embedder.embed_batch(&candidates).await?;

    println!("Query: \"{}\"", query);
    println!("\nSimilarity scores:");
    for (candidate, vector) in candidates.iter().zip(candidate_vectors.iter()) {
        // Calculate cosine similarity using dot product (vectors are normalized)
        let similarity: f32 = query_vector
            .data
            .iter()
            .zip(vector.data.iter())
            .map(|(a, b)| a * b)
            .sum();
        println!("  \"{}\": {:.4}", candidate, similarity);
    }

    println!("\n=== Example completed successfully! ===");
    Ok(())
}

#[cfg(not(feature = "embeddings-candle"))]
fn main() {
    eprintln!("This example requires the 'embeddings-candle' feature.");
    eprintln!(
        "Please run with: cargo run --example embedding_with_candle --features embeddings-candle"
    );
    std::process::exit(1);
}
