//! Text-to-Image search example using CLIP embeddings.
//!
//! This example demonstrates how to:
//! 1. Create a multimodal embedder using CLIP
//! 2. Index images using their visual embeddings
//! 3. Search for images using text queries
//!
//! # Usage
//!
//! ```bash
//! cargo run --example text_to_image_search --features embeddings-multimodal
//! ```
//!
//! # Prerequisites
//!
//! You'll need some image files to index. This example expects images in an `images/` directory.

use sage::embedding::{CandleMultimodalEmbedder, ImageEmbedder, TextEmbedder};
use sage::error::Result;
use sage::vector::index::{VectorIndexBuildConfig, VectorIndexBuilderFactory, VectorIndexType};
use sage::vector::{DistanceMetric, Vector};
use std::collections::HashMap;
use std::path::Path;

/// Calculate cosine similarity between two vectors
fn cosine_similarity(a: &Vector, b: &Vector) -> f32 {
    let dot_product: f32 = a.data.iter().zip(b.data.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.data.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.data.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Text-to-Image Search Example ===\n");

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
    println!("Indexing images...");
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

    // Example text queries
    let queries = vec![
        "a photo of a cat",
        "a photo of a dog",
        "an animal sleeping",
        "outdoor scene",
        "close-up portrait",
    ];

    println!("\n=== Search Results ===\n");

    for query_text in queries {
        println!("Query: \"{}\"", query_text);

        // Generate text embedding
        let query_vector = embedder.embed(query_text).await?;

        // Perform manual similarity search
        let mut similarities: Vec<(u64, f32)> = doc_vectors
            .iter()
            .map(|(id, vec)| {
                let similarity = cosine_similarity(&query_vector, vec);
                (*id, similarity)
            })
            .collect();

        // Sort by similarity (descending)
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Take top 3 results
        let top_results: Vec<_> = similarities.iter().take(3).collect();

        println!("  Top {} results:", top_results.len());
        for (i, (doc_id, similarity)) in top_results.iter().enumerate() {
            if let Some(path) = image_metadata.get(doc_id) {
                let filename = Path::new(path).file_name().unwrap().to_string_lossy();
                println!(
                    "    {}. {} (similarity: {:.4})",
                    i + 1,
                    filename,
                    similarity
                );
                println!("       Path: {}", path);
            }
        }
        println!();
    }

    println!("\n=== Cross-Modal Search Complete ===");
    println!("Text queries successfully matched with images!");

    Ok(())
}
