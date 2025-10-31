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

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use yatagarasu::embedding::candle_multimodal_embedder::CandleMultimodalEmbedder;
use yatagarasu::embedding::image_embedder::ImageEmbedder;
use yatagarasu::embedding::text_embedder::TextEmbedder;
use yatagarasu::error::Result;
use yatagarasu::storage::memory::MemoryStorage;
use yatagarasu::storage::memory::MemoryStorageConfig;
use yatagarasu::vector::DistanceMetric;
use yatagarasu::vector::engine::VectorEngine;
use yatagarasu::vector::index::{HnswIndexConfig, VectorIndexConfig};
use yatagarasu::vector::search::searcher::VectorSearchRequest;

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

    // Create vector index configuration
    println!("Creating vector index configuration...");
    let config = VectorIndexConfig::HNSW(HnswIndexConfig {
        dimension: ImageEmbedder::dimension(&embedder),
        distance_metric: DistanceMetric::Cosine,
        normalize_vectors: true,
        ..Default::default()
    });

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
        println!("No images found to index. Please add image files to the 'images/' directory.");
        return Ok(());
    }

    // Build the index using VectorEngine
    println!("Building HNSW index...");
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let index = yatagarasu::vector::index::VectorIndexFactory::create(storage, config)?;
    let mut engine = VectorEngine::new(index)?;
    engine.add_vectors(doc_vectors)?;
    engine.commit()?;

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

        // Perform search using VectorEngine
        let request = VectorSearchRequest::new(query_vector).top_k(max_results);
        let search_results = engine.search(request)?;

        // Show all scores for debugging
        println!("  All similarity scores:");
        for result in &search_results.results {
            if let Some(path) = image_metadata.get(&result.doc_id) {
                let filename = Path::new(path).file_name().unwrap().to_string_lossy();
                println!("    {} = {:.4}", filename, result.similarity);
            }
        }
        println!();

        // Filter by threshold
        let filtered_results: Vec<_> = search_results
            .results
            .iter()
            .filter(|result| result.similarity >= score_threshold)
            .collect();

        if filtered_results.is_empty() {
            println!("  No results above threshold ({:.2})", score_threshold);
        } else {
            println!(
                "  Found {} results above threshold ({:.2}):",
                filtered_results.len(),
                score_threshold
            );
            for (i, result) in filtered_results.iter().enumerate() {
                if let Some(path) = image_metadata.get(&result.doc_id) {
                    let filename = Path::new(path).file_name().unwrap().to_string_lossy();
                    println!(
                        "    {}. {} (similarity: {:.4})",
                        i + 1,
                        filename,
                        result.similarity
                    );
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
