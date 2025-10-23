//! Example of dynamically switching between different TextEmbedder implementations
//!
//! This example demonstrates:
//! - Using trait objects (Arc<dyn TextEmbedder>) for runtime polymorphism
//! - Switching between Candle (local) and OpenAI (cloud) embedders
//! - Comparing embeddings from different sources
//!
//! Prerequisites:
//! - Set OPENAI_API_KEY environment variable for OpenAI embedder
//!
//! To run this example:
//! ```bash
//! export OPENAI_API_KEY=your-api-key-here
//! cargo run --example dynamic_embedder_switching --features embeddings-all
//! ```

#[cfg(all(feature = "embeddings-candle", feature = "embeddings-openai"))]
use std::sync::Arc;

#[cfg(all(feature = "embeddings-candle", feature = "embeddings-openai"))]
use sage::embedding::{CandleTextEmbedder, OpenAITextEmbedder, TextEmbedder};

#[cfg(all(feature = "embeddings-candle", feature = "embeddings-openai"))]
#[tokio::main]
async fn main() -> sage::error::Result<()> {
    println!("=== Dynamic Embedder Switching Example ===\n");

    // Create a vector to hold different embedders
    let mut embedders: Vec<Arc<dyn TextEmbedder>> = Vec::new();

    // Add Candle embedder (local inference)
    println!("Loading Candle embedder (local)...");
    let candle_embedder = CandleTextEmbedder::new("sentence-transformers/all-MiniLM-L6-v2")?;
    embedders.push(Arc::new(candle_embedder));
    println!("✓ Candle embedder loaded\n");

    // Add OpenAI embedder (cloud API)
    if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
        println!("Loading OpenAI embedder (cloud)...");
        let openai_embedder =
            OpenAITextEmbedder::new(api_key, "text-embedding-3-small".to_string())?;
        embedders.push(Arc::new(openai_embedder));
        println!("✓ OpenAI embedder loaded\n");
    } else {
        println!("⚠ OPENAI_API_KEY not set, skipping OpenAI embedder\n");
    }

    // Test text
    let test_texts = vec![
        "Rust is a systems programming language",
        "Machine learning enables computers to learn from data",
    ];

    // Use each embedder
    println!("--- Processing with Different Embedders ---\n");
    for embedder in &embedders {
        println!("Using embedder: {}", embedder.name());
        println!("Dimension: {}", embedder.dimension());

        for text in &test_texts {
            let vector = embedder.embed(text).await?;
            println!(
                "  \"{}\" → {} dimensions, first 3: {:?}",
                text,
                vector.dimension(),
                &vector.data[..3.min(vector.data.len())]
            );
        }
        println!();
    }

    // Example: Runtime selection based on configuration
    println!("--- Runtime Embedder Selection ---\n");

    let use_local = true; // Could be from config file or environment variable

    let selected_embedder: Arc<dyn TextEmbedder> = if use_local {
        println!("Selected: Local embedder (Candle)");
        embedders[0].clone()
    } else if embedders.len() > 1 {
        println!("Selected: Cloud embedder (OpenAI)");
        embedders[1].clone()
    } else {
        println!("Selected: Default embedder (Candle)");
        embedders[0].clone()
    };

    let query = "artificial intelligence and machine learning";
    println!("\nGenerating embedding for: \"{}\"", query);
    let result = selected_embedder.embed(query).await?;
    println!("Result: {} dimensions", result.dimension());

    // Example: Comparing embeddings from different sources
    if embedders.len() > 1 {
        println!("\n--- Comparing Embedders ---\n");

        let comparison_text = "natural language processing";
        println!("Text: \"{}\"", comparison_text);

        let candle_vec = embedders[0].embed(comparison_text).await?;
        let openai_vec = embedders[1].embed(comparison_text).await?;

        println!("\nCandle embedder:");
        println!("  Dimension: {}", candle_vec.dimension());
        println!("  First 5 values: {:?}", &candle_vec.data[..5]);

        println!("\nOpenAI embedder:");
        println!("  Dimension: {}", openai_vec.dimension());
        println!("  First 5 values: {:?}", &openai_vec.data[..5]);

        // Note: Direct comparison isn't meaningful as they use different models/dimensions
        println!("\nNote: Embeddings from different models are not directly comparable.");
        println!("Each model has its own semantic space and dimensionality.");
    }

    // Example: Batch processing with dynamic embedder
    println!("\n--- Batch Processing with Selected Embedder ---\n");

    let batch_texts = vec![
        "The quick brown fox",
        "jumps over the lazy dog",
        "in the forest at night",
    ];

    println!("Processing batch of {} texts...", batch_texts.len());
    let batch_results = selected_embedder.embed_batch(&batch_texts).await?;

    for (text, vector) in batch_texts.iter().zip(batch_results.iter()) {
        println!("  \"{}\" → {} dims", text, vector.dimension());
    }

    println!("\n=== Example completed successfully! ===");
    Ok(())
}

#[cfg(not(all(feature = "embeddings-candle", feature = "embeddings-openai")))]
fn main() {
    eprintln!("This example requires both 'embeddings-candle' and 'embeddings-openai' features.");
    eprintln!(
        "Please run with: cargo run --example dynamic_embedder_switching --features embeddings-all"
    );
    std::process::exit(1);
}
