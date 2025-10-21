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

use sage::embedding::{CandleMultimodalEmbedder, ImageEmbedder};
use sage::error::Result;
use sage::vector::index::{VectorIndexBuildConfig, VectorIndexBuilderFactory, VectorIndexType};
use sage::vector::{DistanceMetric, Vector};
use std::collections::HashMap;
use std::path::Path;

/// Calculate similarity between two L2-normalized vectors using dot product
/// Since CLIP embeddings are already L2-normalized, dot product equals cosine similarity
fn similarity(a: &Vector, b: &Vector) -> f32 {
    a.data.iter().zip(b.data.iter()).map(|(x, y)| x * y).sum()
}

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

    // Create vector index for images
    println!("Creating vector index...");
    let config = VectorIndexBuildConfig {
        dimension: ImageEmbedder::dimension(&embedder),
        distance_metric: DistanceMetric::Cosine,
        index_type: VectorIndexType::HNSW,
        normalize_vectors: true,
        ..Default::default()
    };
    let mut builder = VectorIndexBuilderFactory::create_builder(config)?;

    // Index sample images
    println!("Indexing image collection...");
    let image_dir = Path::new("images");

    if !image_dir.exists() {
        println!("Error: 'images/' directory not found.");
        println!("Please create an 'images/' directory and add some image files.");
        println!("\nExample structure:");
        println!("  images/");
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

        if path.is_file() {
            if let Some(ext) = path.extension() {
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
    }

    println!("\nIndexed {} images\n", doc_id);

    if doc_id == 0 {
        println!("No images found to index. Please add image files to the 'images/' directory.");
        return Ok(());
    }

    // Build the index
    println!("Building HNSW index...");
    builder.add_vectors(doc_vectors.clone())?;
    builder.finalize()?;

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

    // Perform manual similarity search
    let mut similarities: Vec<(u64, f32)> = doc_vectors
        .iter()
        .map(|(id, vec)| {
            let sim = similarity(&query_vector, vec);
            (*id, sim)
        })
        .collect();

    // Sort by similarity (descending)
    similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Filter by threshold and limit to max_results
    // Note: The query image itself will have similarity ~1.0 and should be first
    let filtered_results: Vec<_> = similarities
        .iter()
        .filter(|(_, score)| *score >= score_threshold)
        .take(max_results)
        .collect();

    if filtered_results.is_empty() {
        println!("No results above threshold ({:.2})", score_threshold);
    } else {
        println!(
            "Found {} similar images above threshold ({:.2}):",
            filtered_results.len(),
            score_threshold
        );
        for (i, (doc_id, sim_score)) in filtered_results.iter().enumerate() {
            if let Some(path) = image_metadata.get(doc_id) {
                let filename = Path::new(path).file_name().unwrap().to_string_lossy();
                println!("  {}. {} (similarity: {:.4})", i + 1, filename, sim_score);
                println!("     Path: {}", path);
            }
        }
    }

    println!("\n=== Image Similarity Search Complete ===");
    println!("Found visually similar images using CLIP embeddings!");

    Ok(())
}
