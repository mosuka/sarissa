//! Example of using OpenAITextEmbedder with OpenAI's Embeddings API
//!
//! This example demonstrates:
//! - Using OpenAI's text-embedding-3-small model
//! - Generating embeddings for text via API
//! - Batch processing for efficiency
//!
//! Prerequisites:
//! - Set OPENAI_API_KEY environment variable
//! - Ensure you have API credits
//!
//! To run this example:
//! ```bash
//! export OPENAI_API_KEY=your-api-key-here
//! cargo run --example embedding_with_openai --features embeddings-openai
//! ```

#[cfg(feature = "embeddings-openai")]
use sage::embedding::{OpenAITextEmbedder, TextEmbedder};

#[cfg(feature = "embeddings-openai")]
#[tokio::main]
async fn main() -> sage::error::Result<()> {
    println!("=== OpenAI Text Embedder Example ===\n");

    // Get API key from environment
    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| {
        eprintln!("Error: OPENAI_API_KEY environment variable not set");
        eprintln!("Please set it with: export OPENAI_API_KEY=your-api-key-here");
        std::process::exit(1);
    });

    // Create embedder with text-embedding-3-small model
    println!("Creating OpenAI embedder with model: text-embedding-3-small");
    let embedder = OpenAITextEmbedder::new(api_key, "text-embedding-3-small".to_string())?;

    println!("Embedder created successfully!");
    println!("Model name: {}", embedder.name());
    println!("Embedding dimension: {}\n", embedder.dimension());

    // Example 1: Single text embedding
    println!("--- Example 1: Single Text Embedding ---");
    let text = "Rust is a systems programming language";
    println!("Text: \"{}\"", text);
    println!("Calling OpenAI API...");

    let vector = embedder.embed(text).await?;
    println!("Generated embedding with {} dimensions", vector.dimension());
    println!(
        "First 5 values: {:?}\n",
        &vector.data[..5.min(vector.data.len())]
    );

    // Example 2: Batch processing (more efficient for multiple texts)
    println!("--- Example 2: Batch Processing ---");
    let texts = vec![
        "Machine learning is transforming technology",
        "Deep learning uses neural networks",
        "Natural language processing enables AI to understand text",
    ];

    println!("Processing {} texts in a single API call...", texts.len());
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

    println!("Query: \"{}\"", query);
    println!("Generating embeddings for query and candidates...");

    let query_vector = embedder.embed(query).await?;
    let candidate_vectors = embedder.embed_batch(&candidates).await?;

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

    // Example 4: Using custom dimension (cost optimization)
    println!("\n--- Example 4: Custom Dimension ---");
    println!("Creating embedder with reduced dimension (512 instead of 1536)");
    let api_key2 = std::env::var("OPENAI_API_KEY").unwrap();
    let custom_embedder =
        OpenAITextEmbedder::with_dimension(api_key2, "text-embedding-3-small".to_string(), 512)?;

    let text = "Smaller embeddings save storage and costs";
    println!("Text: \"{}\"", text);

    let custom_vector = custom_embedder.embed(text).await?;
    println!(
        "Generated embedding with {} dimensions",
        custom_vector.dimension()
    );
    println!(
        "First 5 values: {:?}",
        &custom_vector.data[..5.min(custom_vector.data.len())]
    );

    println!("\n=== Example completed successfully! ===");
    println!("Note: This example made several API calls to OpenAI.");
    Ok(())
}

#[cfg(not(feature = "embeddings-openai"))]
fn main() {
    eprintln!("This example requires the 'embeddings-openai' feature.");
    eprintln!(
        "Please run with: cargo run --example embedding_with_openai --features embeddings-openai"
    );
    std::process::exit(1);
}
