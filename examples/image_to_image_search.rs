//! Image-to-Image search example using CLIP embeddings.
//!
//! This example demonstrates how to:
//! 1. Create a multimodal embedder using CLIP
//! 2. Index a collection of images
//! 3. Find similar images using an image query
//!
//! # Usage
//!
//! ```bash
//! cargo run --example image_to_image_search --features embeddings-multimodal -- query.jpg
//! ```
//!
//! # Prerequisites
//!
//! You'll need:
//! - A collection of images in an `images/` directory to search through
//! - A query image to search with (passed as command line argument)

use std::collections::HashMap;
use std::path::Path;

use sage::embedding::candle_multimodal_embedder::CandleMultimodalEmbedder;
use sage::embedding::image_embedder::ImageEmbedder;
use sage::error::Result;
use sage::vector::DistanceMetric;
use sage::vector::engine::VectorEngine;
use sage::vector::index::{VectorIndexType, VectorIndexWriterConfig};
use sage::vector::types::VectorSearchRequest;

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Image-to-Image Search Example ===\n");

    // Get query image from command line arguments
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        println!("Usage: {} <query_image_path>", args[0]);
        println!("\nExample:");
        println!(
            "  cargo run --example image_to_image_search --features embeddings-multimodal -- query.jpg"
        );
        return Ok(());
    }

    let query_image_path = &args[1];
    if !Path::new(query_image_path).exists() {
        println!("Error: Query image '{}' not found", query_image_path);
        return Ok(());
    }

    // Initialize multimodal embedder (CLIP)
    println!("Loading CLIP model...");
    let embedder = CandleMultimodalEmbedder::new("openai/clip-vit-base-patch32")?;
    println!("Model loaded: {}", ImageEmbedder::name(&embedder));
    println!(
        "Embedding dimension: {}\n",
        ImageEmbedder::dimension(&embedder)
    );

    // Create vector index configuration
    println!("Creating vector index configuration...");
    let config = VectorIndexWriterConfig {
        dimension: ImageEmbedder::dimension(&embedder),
        distance_metric: DistanceMetric::Cosine,
        index_type: VectorIndexType::HNSW,
        normalize_vectors: true,
        ..Default::default()
    };

    // Index sample images
    println!("Indexing image collection...");
    let image_dir = Path::new("resources/images");

    if !image_dir.exists() {
        println!("Error: 'resources/images/' directory not found.");
        println!("Please create an 'resources/images/' directory and add some image files.");
        println!("\nExample structure:");
        println!("  resources/images/");
        println!("    cat1.jpg");
        println!("    cat2.jpg");
        println!("    dog1.jpg");
        println!("    dog2.jpg");
        return Ok(());
    }

    let mut doc_id = 0u64;
    let mut doc_vectors = Vec::new();
    let mut image_metadata: HashMap<u64, String> = HashMap::new();

    for entry in std::fs::read_dir(image_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file()
            && let Some(ext) = path.extension()
        {
            let ext = ext.to_string_lossy().to_lowercase();
            if matches!(ext.as_str(), "jpg" | "jpeg" | "png" | "gif" | "bmp") {
                let path_str = path.to_string_lossy();
                println!("  Indexing: {}", path_str);

                // Generate image embedding
                let vector = embedder.embed_image(&path_str).await?;

                // Store metadata
                image_metadata.insert(doc_id, path_str.to_string());
                doc_vectors.push((doc_id, vector));

                doc_id += 1;
            }
        }
    }

    println!("\nIndexed {} images\n", doc_id);

    if doc_id == 0 {
        println!(
            "No images found to index. Please add image files to the 'resources/images/' directory."
        );
        return Ok(());
    }

    // Build the index using VectorEngine
    println!("Building HNSW index...");
    let mut engine = VectorEngine::create(config)?;
    engine.add_vectors(doc_vectors)?;
    engine.finalize()?;

    println!("Index built successfully!\n");

    // Generate embedding for query image
    println!("\n=== Searching for Similar Images ===");
    println!("Query image: {}\n", query_image_path);
    println!("Generating embedding for query image...");
    let query_vector = embedder.embed_image(query_image_path).await?;

    // Search for similar images
    println!("Searching for similar images...\n");

    // Score threshold for filtering results
    // CLIP similarities typically range from 0.2 to 0.35
    // A threshold of 0.25 filters out weak matches
    let score_threshold = 0.25;
    let max_results = 10;

    // Perform search using VectorEngine
    let request = VectorSearchRequest::new(query_vector).top_k(max_results);
    let search_results = engine.search(request)?;

    // Filter by threshold
    // Note: The query image itself will have similarity ~1.0 and should be first
    let filtered_results: Vec<_> = search_results
        .results
        .iter()
        .filter(|result| result.similarity >= score_threshold)
        .collect();

    if filtered_results.is_empty() {
        println!("No results above threshold ({:.2})", score_threshold);
    } else {
        println!(
            "Found {} similar images above threshold ({:.2}):",
            filtered_results.len(),
            score_threshold
        );
        for (i, result) in filtered_results.iter().enumerate() {
            if let Some(path) = image_metadata.get(&result.doc_id) {
                let filename = Path::new(path).file_name().unwrap().to_string_lossy();
                println!(
                    "  {}. {} (similarity: {:.4})",
                    i + 1,
                    filename,
                    result.similarity
                );
                println!("     Path: {}", path);
            }
        }
    }

    println!("\n=== Image Similarity Search Complete ===");
    println!("Found visually similar images using CLIP embeddings!");

    Ok(())
}
