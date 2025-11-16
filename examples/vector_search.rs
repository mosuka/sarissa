//! Vector Search with Multiple Fields Example
//!
//! This example demonstrates:
//! - Creating a VectorEngine with file storage
//! - Using CandleTextEmbedder to generate text embeddings
//! - Adding documents with MULTIPLE vector fields (title and content)
//! - Performing field-specific searches (searching only titles or only content)
//! - Searching across all fields
//!
//! To run this example:
//! ```bash
//! cargo run --example vector_search --features embeddings-candle
//! ```

#[cfg(feature = "embeddings-candle")]
use std::sync::Arc;

#[cfg(feature = "embeddings-candle")]
use tempfile::TempDir;

#[cfg(feature = "embeddings-candle")]
use platypus::document::document::Document;
#[cfg(feature = "embeddings-candle")]
use platypus::document::field::VectorOption;
#[cfg(feature = "embeddings-candle")]
use platypus::embedding::candle_text_embedder::CandleTextEmbedder;
#[cfg(feature = "embeddings-candle")]
use platypus::embedding::text_embedder::TextEmbedder;
#[cfg(feature = "embeddings-candle")]
use platypus::error::Result;
#[cfg(feature = "embeddings-candle")]
use platypus::storage::file::FileStorageConfig;
#[cfg(feature = "embeddings-candle")]
use platypus::storage::{StorageConfig, StorageFactory};
#[cfg(feature = "embeddings-candle")]
use platypus::vector::engine::VectorEngine;
#[cfg(feature = "embeddings-candle")]
use platypus::vector::index::config::{FlatIndexConfig, VectorIndexConfig};
#[cfg(feature = "embeddings-candle")]
use platypus::vector::index::factory::VectorIndexFactory;
#[cfg(feature = "embeddings-candle")]
use platypus::vector::search::searcher::VectorSearchRequest;

#[cfg(feature = "embeddings-candle")]
#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Vector Search with Multiple Fields Example ===\n");

    // Step 1: Initialize the embedder
    println!("Step 1: Loading embedding model: sentence-transformers/all-MiniLM-L6-v2...");
    let embedder = CandleTextEmbedder::new("sentence-transformers/all-MiniLM-L6-v2")?;
    let embedding_dim = embedder.dimension();
    println!("  Embedding dimension: {}\n", embedding_dim);

    // Wrap embedder in Arc for sharing
    let embedder_arc: Arc<dyn TextEmbedder> = Arc::new(embedder);

    // Step 2: Create VectorEngine with file storage
    println!("Step 2: Creating VectorEngine...");
    let temp_dir = TempDir::new().unwrap();
    let storage =
        StorageFactory::create(StorageConfig::File(FileStorageConfig::new(temp_dir.path())))?;

    let flat_config = FlatIndexConfig {
        dimension: embedding_dim,
        embedder: Arc::clone(&embedder_arc),
        ..Default::default()
    };
    let config = VectorIndexConfig::Flat(flat_config);
    let index = VectorIndexFactory::create(storage, config)?;
    let mut vector_engine = VectorEngine::new(index)?;
    println!("  VectorEngine created\n");

    // Step 3: Prepare sample documents with title and content fields
    println!("Step 3: Indexing sample documents with MULTIPLE vector fields...");

    let sample_docs = vec![
        (
            "Rust Programming",
            "Rust is a systems programming language focused on safety, concurrency, and performance.",
        ),
        (
            "Python for AI",
            "Python is widely used in artificial intelligence, machine learning, and data science applications.",
        ),
        (
            "JavaScript Web Development",
            "JavaScript powers interactive web applications and is essential for modern frontend development.",
        ),
        (
            "Database Systems",
            "Database management systems efficiently store, organize, and retrieve data for applications.",
        ),
        (
            "Machine Learning",
            "Machine learning algorithms enable computers to learn patterns from data without explicit programming.",
        ),
    ];

    for (doc_id, (title, content)) in sample_docs.iter().enumerate() {
        // Create document with TWO separate vector fields: title_embedding and content_embedding
        let doc = Document::builder()
            .add_vector("title_embedding", *title, VectorOption::default())
            .add_vector("content_embedding", *content, VectorOption::default())
            .build();

        // Add document to engine (async - converts text to vectors internally)
        vector_engine
            .add_document_with_id(doc_id as u64, doc)
            .await?;
        println!("  Added document {}: {}", doc_id, title);
        println!("    - title_embedding: \"{}\"", title);
        println!(
            "    - content_embedding: \"{}...\"",
            &content[..50.min(content.len())]
        );
    }

    // Commit and optimize
    vector_engine.commit()?;
    vector_engine.optimize()?;
    println!("\n  Indexing completed for {} documents", sample_docs.len());
    println!("  Each document has 2 vector fields: title_embedding, content_embedding\n");

    // Step 4: Perform searches
    println!("{}", "=".repeat(80));
    println!("Step 4: Demonstrating Field-Specific Vector Searches\n");

    // Search 1: Search across ALL fields (no field filter)
    println!("[1] Search ALL fields: \"programming language\"");
    println!("{}", "-".repeat(80));
    let query1 = "programming language";
    let query_vector1 = embedder_arc.embed(query1).await?;
    let request1 = VectorSearchRequest::new(query_vector1).top_k(3);
    let results1 = vector_engine.search(request1)?;
    display_results(&results1, &sample_docs, "All fields");

    // Search 2: Search ONLY in title_embedding field
    println!("\n[2] Search ONLY title_embedding field: \"programming language\"");
    println!("{}", "-".repeat(80));
    let query2 = "programming language";
    let query_vector2 = embedder_arc.embed(query2).await?;
    let request2 = VectorSearchRequest::new(query_vector2)
        .top_k(3)
        .field_name("title_embedding".to_string());
    let results2 = vector_engine.search(request2)?;
    display_results(&results2, &sample_docs, "title_embedding only");

    // Search 3: Search ONLY in content_embedding field
    println!("\n[3] Search ONLY content_embedding field: \"artificial intelligence\"");
    println!("{}", "-".repeat(80));
    let query3 = "artificial intelligence";
    let query_vector3 = embedder_arc.embed(query3).await?;
    let request3 = VectorSearchRequest::new(query_vector3)
        .top_k(3)
        .field_name("content_embedding".to_string());
    let results3 = vector_engine.search(request3)?;
    display_results(&results3, &sample_docs, "content_embedding only");

    // Search 4: Compare title vs content search
    println!("\n[4] Comparison: Same query on different fields");
    println!("{}", "-".repeat(80));
    let query4 = "web development";
    let query_vector4 = embedder_arc.embed(query4).await?;

    println!("Query: \"web development\" on title_embedding:");
    let request4a = VectorSearchRequest::new(query_vector4.clone())
        .top_k(2)
        .field_name("title_embedding".to_string());
    let results4a = vector_engine.search(request4a)?;
    display_results(&results4a, &sample_docs, "title_embedding");

    println!("\nQuery: \"web development\" on content_embedding:");
    let request4b = VectorSearchRequest::new(query_vector4)
        .top_k(2)
        .field_name("content_embedding".to_string());
    let results4b = vector_engine.search(request4b)?;
    display_results(&results4b, &sample_docs, "content_embedding");

    println!("\n{}", "=".repeat(80));
    println!("Vector search demonstration completed!");
    println!("\nFeatures demonstrated:");
    println!("  ✓ Multiple vector fields per document (title_embedding, content_embedding)");
    println!("  ✓ Field-specific search (filtering by field name)");
    println!("  ✓ Search across all fields (no filter)");
    println!("  ✓ Comparison of results from different fields");

    Ok(())
}

#[cfg(feature = "embeddings-candle")]
fn display_results(
    results: &platypus::vector::search::searcher::VectorSearchResults,
    docs: &[(&str, &str)],
    field_info: &str,
) {
    println!(
        "Results ({}): {} matches in {} ms",
        field_info,
        results.results.len(),
        results.search_time_ms
    );

    for (rank, result) in results.results.iter().enumerate() {
        if let Some((title, content)) = docs.get(result.doc_id as usize) {
            println!(
                "  {}. [Doc {}] {} - Similarity: {:.4} (field: {})",
                rank + 1,
                result.doc_id,
                title,
                result.similarity,
                result.field_name
            );
            let content_preview = if content.len() > 60 {
                format!("{}...", &content[..60])
            } else {
                content.to_string()
            };
            println!("     Content: {}", content_preview);
        }
    }
}

#[cfg(not(feature = "embeddings-candle"))]
fn main() {
    eprintln!("This example requires the 'embeddings-candle' feature.");
    eprintln!("Please run with: cargo run --example vector_search --features embeddings-candle");
    std::process::exit(1);
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "embeddings-candle")]
    #[tokio::test]
    async fn test_vector_search_example() {
        let result = super::main().await;
        assert!(result.is_ok());
    }
}
