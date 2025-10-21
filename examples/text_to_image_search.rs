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

/// Calculate similarity between two L2-normalized vectors using dot product
/// Since CLIP embeddings are already L2-normalized, dot product equals cosine similarity
fn similarity(a: &Vector, b: &Vector) -> f32 {
    a.data.iter().zip(b.data.iter()).map(|(x, y)| x * y).sum()
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
    let image_dir = Path::new("resources/images");

    if !image_dir.exists() {
        println!("Error: 'resources/images/' directory not found.");
        println!("Please create a 'resources/images/' directory and add some image files.");
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
        "an animal running",
        "close-up portrait",
    ];

    println!("\n=== Search Results ===\n");

    // Score threshold for filtering results
    // CLIP similarities typically range from 0.2 to 0.35
    // A threshold of 0.25 filters out weak matches
    let score_threshold = 0.25;
    let max_results = 10;

    for query_text in queries {
        println!("Query: \"{}\"", query_text);

        // Generate text embedding
        let query_vector = embedder.embed(query_text).await?;

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

        // Show all scores for debugging
        println!("  All similarity scores:");
        for (doc_id, score) in similarities.iter() {
            if let Some(path) = image_metadata.get(doc_id) {
                let filename = Path::new(path).file_name().unwrap().to_string_lossy();
                println!("    {} = {:.4}", filename, score);
            }
        }
        println!();

        // Filter by threshold and limit to max_results
        let filtered_results: Vec<_> = similarities
            .iter()
            .filter(|(_, score)| *score >= score_threshold)
            .take(max_results)
            .collect();

        if filtered_results.is_empty() {
            println!("  No results above threshold ({:.2})", score_threshold);
        } else {
            println!(
                "  Found {} results above threshold ({:.2}):",
                filtered_results.len(),
                score_threshold
            );
            for (i, (doc_id, sim_score)) in filtered_results.iter().enumerate() {
                if let Some(path) = image_metadata.get(doc_id) {
                    let filename = Path::new(path).file_name().unwrap().to_string_lossy();
                    println!("    {}. {} (similarity: {:.4})", i + 1, filename, sim_score);
                    println!("       Path: {}", path);
                }
            }
        }
        println!();
    }

    println!("\n=== Cross-Modal Search Complete ===");
    println!("Text queries successfully matched with images!");

    Ok(())
}
