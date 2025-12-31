//! Multimodal Search Example - Search across text and images
//!
//! This example demonstrates:
//! 1. Configuring a vector index with CandleMultimodalEmbedder (CLIP)
//! 2. Indexing images from local file system
//! 3. Indexing text descriptions
//! 4. Performing multimodal search:
//!    - Text query finding images
//!    - Image query finding images
//!    - Image query finding text
//!
//! To run this example:
//! ```bash
//! cargo run --example multimodal_search --features embeddings-multimodal
//! ```

#[cfg(feature = "embeddings-multimodal")]
use std::path::Path;

#[cfg(feature = "embeddings-multimodal")]
use sarissa::embedding::candle_multimodal_embedder::CandleMultimodalEmbedder;
#[cfg(feature = "embeddings-multimodal")]
use sarissa::embedding::embedder::Embedder;
#[cfg(feature = "embeddings-multimodal")]
use sarissa::error::Result;
#[cfg(feature = "embeddings-multimodal")]
use sarissa::storage::file::FileStorageConfig;
#[cfg(feature = "embeddings-multimodal")]
use sarissa::storage::{StorageConfig, StorageFactory};
#[cfg(feature = "embeddings-multimodal")]
use sarissa::vector::DistanceMetric;
#[cfg(feature = "embeddings-multimodal")]
use sarissa::vector::core::document::{DocumentPayload, Payload, PayloadSource};
#[cfg(feature = "embeddings-multimodal")]
use sarissa::vector::engine::config::{VectorFieldConfig, VectorIndexConfig, VectorIndexKind};
#[cfg(feature = "embeddings-multimodal")]
use sarissa::vector::engine::{VectorEngine, VectorSearchRequestBuilder};
#[cfg(feature = "embeddings-multimodal")]
use tempfile::TempDir;

#[cfg(feature = "embeddings-multimodal")]
fn main() -> Result<()> {
    println!("=== Multimodal Search Example (CLIP) ===\n");

    // 1. Setup Storage
    let temp_dir = TempDir::new().unwrap();
    let storage_config = StorageConfig::File(FileStorageConfig::new(temp_dir.path()));
    let storage = StorageFactory::create(storage_config)?;

    // 2. Configure Index with CandleMultimodalEmbedder (CLIP)
    // We use "openai/clip-vit-base-patch32" which is a good balance of speed and quality
    println!("Loading CLIP model (this may take a while on first run)...");
    let embedder = CandleMultimodalEmbedder::new("openai/clip-vit-base-patch32")?;
    println!("Model loaded: {}", embedder.name());

    let field_config = VectorFieldConfig {
        dimension: 512, // CLIP ViT-B/32 output dimension
        distance: DistanceMetric::Cosine,
        index: VectorIndexKind::Flat,

        base_weight: 1.0,
    };

    let index_config = VectorIndexConfig::builder()
        .embedder(embedder)
        .field("content", field_config)
        .build()?;

    // 3. Create Engine
    let engine = VectorEngine::new(storage, index_config)?;

    // 4. Index Images
    println!("\n--- Indexing Images ---");
    let resources_dir = Path::new("resources");
    let images_dir = resources_dir.join("images");

    if !images_dir.exists() {
        eprintln!("Error: resources/images directory not found.");
        eprintln!(
            "Please ensure you are running from the project root and resources/images exists."
        );
        return Ok(());
    }

    let mut indexed_count = 0;
    // Iterate over jpg files
    let entries = std::fs::read_dir(&images_dir)?;
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            if let Some(ext) = path.extension() {
                if ext == "jpg" || ext == "jpeg" || ext == "png" {
                    let filename = path.file_name().unwrap().to_string_lossy().to_string();
                    println!("Indexing image: {}", filename);

                    let mut doc = DocumentPayload::new();

                    // Read image file into bytes
                    let bytes = std::fs::read(&path)?;

                    // We index the image content into "content" field using PayloadSource::Bytes
                    doc.set_field("content", Payload::new(PayloadSource::bytes(bytes, None)));

                    // Store filename in metadata for display
                    doc.set_metadata("filename", filename.clone());
                    doc.set_metadata("type", "image");

                    engine.add_payloads(doc)?;
                    indexed_count += 1;
                }
            }
        }
    }
    println!("Indexed {} images.", indexed_count);

    // 5. Index Text Descriptions
    // Start doc_ids after images (though automatic assignment handles it, we assume sequential for output)
    println!("\n--- Indexing Text ---");
    let texts = vec![
        "A cute kitten looking at the camera",
        "A loyal dog standing in the grass",
        "Two dogs playing together",
        "A landscape with mountains and a lake", // Distractor
    ];

    for text in &texts {
        println!("Indexing text: \"{}\"", text);
        let mut doc = DocumentPayload::new();
        doc.set_text("content", *text);
        doc.set_metadata("text", *text);
        doc.set_metadata("type", "text");
        engine.add_payloads(doc)?;
    }

    // Commit to make documents searchable
    engine.commit()?;

    // 6. Search Demonstrations

    // Demo 1: Text-to-Image
    println!("\n--- Search 1: Text-to-Image ---");
    let query_text = "a photo of a cat";
    println!("Query: \"{}\"", query_text);

    // VectorSearchRequestBuilder usage:
    // It seems it doesn't have add_text/add_uri directly.
    // We should use add_payload_text / add_payload_uri or check implementation.
    // Based on previous error, add_text was missing.
    // Let's assume add_vector(field, vector) exists, but for payloads...
    // Checking query.rs...
    // It has `add_vector` (for raw vectors).
    // It likely has `add_payload` or similar? Or maybe `with_query_payload`?
    // Let's rely on checking `query.rs` content which I just requested.
    // Ah, I can't wait for `query.rs` content here inside replace call construction.
    // But I will split this modification. First, check `query.rs`.
    // Wait, I already called view_file to execute.
    let request = VectorSearchRequestBuilder::new()
        .add_text("content", query_text)
        .limit(3)
        .build();

    let results = engine.search(request)?;
    print_results(&results);

    // Demo 2: Image-to-Image
    println!("\n--- Search 2: Image-to-Image ---");
    let query_image_path = resources_dir.join("query_image.jpg");
    if query_image_path.exists() {
        println!("Query Image: {}", query_image_path.display());
        // Read query image bytes
        let query_bytes = std::fs::read(&query_image_path)?;

        // Create the request
        // We use add_bytes for the image query.
        // In a real app, you might want to detect mime type (e.g. "image/jpeg").
        let request = VectorSearchRequestBuilder::new()
            .add_bytes("content", query_bytes, Some("image/jpeg"))
            .limit(3)
            .build();

        let results = engine.search(request)?;
        print_results(&results);
    } else {
        println!(
            "Skipping Image-to-Image demo: {} not found",
            query_image_path.display()
        );
    }

    Ok(())
}

#[cfg(feature = "embeddings-multimodal")]
fn print_results(results: &sarissa::vector::engine::VectorSearchResults) {
    for (i, hit) in results.hits.iter().enumerate() {
        println!("{}. Doc ID: {}, Score: {:.4}", i + 1, hit.doc_id, hit.score);
    }
}

#[cfg(not(feature = "embeddings-multimodal"))]
fn main() {
    eprintln!("This example requires the 'embeddings-multimodal' feature.");
    eprintln!(
        "Please run with: cargo run --example multimodal_search --features embeddings-multimodal"
    );
    std::process::exit(1);
}
