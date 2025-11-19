//! Vector search example that mirrors the `lexical_search` flow:
//!
//! 1. ストレージを作成
//! 2. PerFieldEmbedder を構築
//! 3. VectorEngine を生成
//! 4. DocumentBuilder でドキュメントを作成し、フィールドごとにベクターを登録
//!
//! その後、複数の検索シナリオで field 指定のベクター検索を確認します。
//!
//! 実行手順:
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
use platypus::embedding::per_field::PerFieldEmbedder;
#[cfg(feature = "embeddings-candle")]
use platypus::embedding::text_embedder::TextEmbedder;
#[cfg(feature = "embeddings-candle")]
use platypus::error::Result;
#[cfg(feature = "embeddings-candle")]
use platypus::storage::file::FileStorageConfig;
#[cfg(feature = "embeddings-candle")]
use platypus::storage::{StorageConfig, StorageFactory};
#[cfg(feature = "embeddings-candle")]
use platypus::vector::core::vector::ORIGINAL_TEXT_METADATA_KEY;
#[cfg(feature = "embeddings-candle")]
use platypus::vector::engine::VectorEngine;
#[cfg(feature = "embeddings-candle")]
use platypus::vector::index::config::{FlatIndexConfig, VectorIndexConfig};
#[cfg(feature = "embeddings-candle")]
use platypus::vector::index::factory::VectorIndexFactory;
#[cfg(feature = "embeddings-candle")]
use platypus::vector::search::searcher::{VectorSearchRequest, VectorSearchResults};

#[cfg(feature = "embeddings-candle")]
#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Vector Search Pipeline (Storage → Embedders → Engine → Documents) ===\n");

    // Step 1. ストレージを作る
    println!("Step 1: Creating storage backend (TempDir + file storage)...");
    let temp_dir = TempDir::new().unwrap();
    let storage =
        StorageFactory::create(StorageConfig::File(FileStorageConfig::new(temp_dir.path())))?;
    println!("  Storage path: {}\n", temp_dir.path().display());

    // Step 2. PerFieldEmbedder を作る
    println!("Step 2: Configuring PerFieldEmbedder...");
    let base_model = "sentence-transformers/all-MiniLM-L6-v2";
    let default_embedder: Arc<dyn TextEmbedder> = Arc::new(CandleTextEmbedder::new(base_model)?);
    let mut per_field_builder = PerFieldEmbedder::new(Arc::clone(&default_embedder));
    per_field_builder.add_embedder("title", Arc::clone(&default_embedder));
    per_field_builder.add_embedder("content", Arc::clone(&default_embedder));
    println!("  Default model: {}", default_embedder.name());
    println!(
        "  Configured fields: {:?}\n",
        per_field_builder.configured_fields()
    );
    let per_field_embedder = Arc::new(per_field_builder);
    let embedder_for_index: Arc<dyn TextEmbedder> = per_field_embedder.clone();

    // Step 3. VectorEngine を作る
    println!("Step 3: Building VectorEngine with Flat index...");
    let mut flat_config = FlatIndexConfig::default();
    flat_config.dimension = embedder_for_index.dimension();
    flat_config.embedder = Arc::clone(&embedder_for_index);
    let config = VectorIndexConfig::Flat(flat_config);
    let index = VectorIndexFactory::create(storage, config)?;
    let mut vector_engine = VectorEngine::new(index)?;
    println!(
        "  VectorEngine ready (dimension = {})\n",
        embedder_for_index.dimension()
    );

    // Step 4. Add documents for testing boolean queries
    let documents = vec![
        Document::builder()
            .add_vector("title", "Rust Programming", VectorOption::default())
            .add_vector("content", "Rust is a systems programming language focused on safety, concurrency, and performance.", VectorOption::default())
            .build(),
        Document::builder()
            .add_vector("title", "Python for AI", VectorOption::default())
            .add_vector("content", "Python is widely used in artificial intelligence, machine learning, and data science applications.", VectorOption::default())
            .build(),
        Document::builder()
            .add_vector("title", "JavaScript Web Development", VectorOption::default())
            .add_vector("content", "JavaScript powers interactive web applications and is essential for modern frontend development.", VectorOption::default())
            .build(),
        Document::builder()
            .add_vector("title", "Database Systems", VectorOption::default())
            .add_vector("content", "Database management systems efficiently store, organize, and retrieve data for applications.", VectorOption::default())
            .build(),
        Document::builder()
            .add_vector("title", "Machine Learning", VectorOption::default())
            .add_vector("content", "Machine learning algorithms enable computers to learn patterns from data without explicit programming.", VectorOption::default())
            .build(),
    ];

    println!("Adding {} documents to the index...", documents.len());
    for document in documents {
        let doc_id = vector_engine.add_document(document.clone()).await?;
        println!(
            "  Indexed doc_id {}: {:?}",
            doc_id,
            document.get_field("title")
        );
    }

    vector_engine.commit()?;
    vector_engine.optimize()?;

    // Demonstrate vector search scenarios (mirroring lexical example style)
    println!("{}", "=".repeat(80));
    println!("Step 5: Running vector search scenarios\n");

    // 1. Search across all vector fields
    println!("[1] Search ALL fields: \"programming language\"");
    println!("{}", "-".repeat(80));
    let query_all = "programming language";
    let query_vec = embedder_for_index.embed(query_all).await?;
    let request = VectorSearchRequest::new(query_vec).top_k(3);
    let results = vector_engine.search(request)?;
    display_results(&results, "all fields");

    // 2. Search only title_embedding field
    println!("\n[2] Search ONLY field 'title': \"programming language\"");
    println!("{}", "-".repeat(80));
    let query_vec = embedder_for_index
        .embed_with_field("programming language", "title")
        .await?;
    let request = VectorSearchRequest::new(query_vec)
        .top_k(3)
        .field_name("title".to_string());
    let results = vector_engine.search(request)?;
    display_results(&results, "title");

    // 3. Search only content field
    println!("\n[3] Search ONLY field 'content': \"artificial intelligence\"");
    println!("{}", "-".repeat(80));
    let query_vec = embedder_for_index
        .embed_with_field("artificial intelligence", "content")
        .await?;
    let request = VectorSearchRequest::new(query_vec)
        .top_k(3)
        .field_name("content".to_string());
    let results = vector_engine.search(request)?;
    display_results(&results, "content");

    // 4. Compare title vs content results for the same query
    println!("\n[4] Field comparison for query: \"web development\"");
    println!("{}", "-".repeat(80));
    let query_vec = embedder_for_index.embed("web development").await?;
    let title_results = vector_engine.search(
        VectorSearchRequest::new(query_vec.clone())
            .top_k(2)
            .field_name("title".to_string()),
    )?;
    display_results(&title_results, "title");

    let content_results = vector_engine.search(
        VectorSearchRequest::new(query_vec)
            .top_k(2)
            .field_name("content".to_string()),
    )?;
    display_results(&content_results, "content");

    println!("\n{}", "=".repeat(80));
    println!("Vector search demonstration completed!\n");

    Ok(())
}

#[cfg(feature = "embeddings-candle")]
fn display_results(results: &VectorSearchResults, context: &str) {
    println!(
        "Results ({}): {} hits in {} ms",
        context,
        results.results.len(),
        results.search_time_ms
    );

    for (rank, hit) in results.results.iter().enumerate() {
        println!(
            "  {}. Doc#{:02} • similarity {:.4} • field {}",
            rank + 1,
            hit.doc_id,
            hit.similarity,
            hit.field_name
        );

        match hit.metadata.get(ORIGINAL_TEXT_METADATA_KEY) {
            Some(original_text) => {
                let label = hit.field_name.as_str();
                println!("     {} : {}", label, preview_text(original_text));
            }
            None => println!("     Stored Text: <not stored>"),
        }
    }
}

#[cfg(feature = "embeddings-candle")]
fn preview_text(text: &str) -> String {
    const LIMIT: usize = 80;
    if text.len() > LIMIT {
        format!("{}...", &text[..LIMIT])
    } else {
        text.to_string()
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
    #[test]
    fn test_vector_search_example() {
        let result = super::main();
        assert!(result.is_ok());
    }
}
