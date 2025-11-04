//! Hybrid Search Example - combining lexical and vector search
//!
//! This example demonstrates:
//! - Creating a hybrid search engine with both LexicalEngine and VectorEngine
//! - Adding documents to both indexes
//! - Performing actual hybrid searches that combine keyword and semantic matching
//! - Configuring different hybrid search strategies (balanced, keyword-focused, semantic-focused)
//! - Understanding score normalization and result fusion
//!
//! To run this example:
//! ```bash
//! cargo run --example hybrid_search --features embeddings-candle
//! ```

#[cfg(feature = "embeddings-candle")]
use std::sync::Arc;

#[cfg(feature = "embeddings-candle")]
use tempfile::TempDir;

#[cfg(feature = "embeddings-candle")]
use yatagarasu::analysis::analyzer::analyzer::Analyzer;
#[cfg(feature = "embeddings-candle")]
use yatagarasu::analysis::analyzer::keyword::KeywordAnalyzer;
#[cfg(feature = "embeddings-candle")]
use yatagarasu::analysis::analyzer::per_field::PerFieldAnalyzer;
#[cfg(feature = "embeddings-candle")]
use yatagarasu::analysis::analyzer::standard::StandardAnalyzer;
#[cfg(feature = "embeddings-candle")]
use yatagarasu::document::converter::DocumentConverter;
#[cfg(feature = "embeddings-candle")]
use yatagarasu::document::converter::jsonl::JsonlDocumentConverter;
#[cfg(feature = "embeddings-candle")]
use yatagarasu::document::field_value::FieldValue;
#[cfg(feature = "embeddings-candle")]
use yatagarasu::embedding::candle_text_embedder::CandleTextEmbedder;
#[cfg(feature = "embeddings-candle")]
use yatagarasu::embedding::text_embedder::TextEmbedder;
#[cfg(feature = "embeddings-candle")]
use yatagarasu::error::Result;
#[cfg(feature = "embeddings-candle")]
use yatagarasu::hybrid::engine::HybridEngine;
#[cfg(feature = "embeddings-candle")]
use yatagarasu::hybrid::search::searcher::{HybridSearchRequest, ScoreNormalization};
#[cfg(feature = "embeddings-candle")]
use yatagarasu::lexical::engine::LexicalEngine;
#[cfg(feature = "embeddings-candle")]
use yatagarasu::lexical::index::config::{InvertedIndexConfig, LexicalIndexConfig};
#[cfg(feature = "embeddings-candle")]
use yatagarasu::lexical::index::factory::LexicalIndexFactory;
#[cfg(feature = "embeddings-candle")]
use yatagarasu::storage::file::FileStorageConfig;
#[cfg(feature = "embeddings-candle")]
use yatagarasu::storage::{StorageConfig, StorageFactory};
#[cfg(feature = "embeddings-candle")]
use yatagarasu::vector::engine::VectorEngine;
#[cfg(feature = "embeddings-candle")]
use yatagarasu::vector::index::factory::VectorIndexFactory;
#[cfg(feature = "embeddings-candle")]
use yatagarasu::vector::index::{FlatIndexConfig, VectorIndexConfig};
#[cfg(feature = "embeddings-candle")]
use yatagarasu::vector::DistanceMetric;

#[cfg(feature = "embeddings-candle")]
#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Hybrid Search Example - Combining Lexical and Vector Search ===\n");

    // Step 1: Initialize the embedder for vector search
    println!("Step 1: Loading embedding model: sentence-transformers/all-MiniLM-L6-v2...");
    let embedder = CandleTextEmbedder::new("sentence-transformers/all-MiniLM-L6-v2")?;
    println!("  Embedding dimension: {}\n", embedder.dimension());

    // Step 2: Create LexicalEngine
    println!("Step 2: Creating LexicalEngine...");
    let lexical_temp_dir = TempDir::new().unwrap();
    let lexical_storage = StorageFactory::create(StorageConfig::File(FileStorageConfig::new(
        lexical_temp_dir.path(),
    )))?;

    let standard_analyzer: Arc<dyn Analyzer> = Arc::new(StandardAnalyzer::new()?);
    let keyword_analyzer: Arc<dyn Analyzer> = Arc::new(KeywordAnalyzer::new());
    let mut per_field_analyzer = PerFieldAnalyzer::new(Arc::clone(&standard_analyzer));
    per_field_analyzer.add_analyzer("id", Arc::clone(&keyword_analyzer));

    let lexical_index_config = LexicalIndexConfig::Inverted(InvertedIndexConfig {
        analyzer: Arc::new(per_field_analyzer.clone()),
        ..InvertedIndexConfig::default()
    });
    let lexical_index = LexicalIndexFactory::create(lexical_storage, lexical_index_config)?;
    let mut lexical_engine = LexicalEngine::new(lexical_index)?;
    println!("  LexicalEngine created\n");

    // Step 3: Create VectorEngine
    println!("Step 3: Creating VectorEngine...");
    let vector_temp_dir = TempDir::new().unwrap();
    let vector_storage = StorageFactory::create(StorageConfig::File(FileStorageConfig::new(
        vector_temp_dir.path(),
    )))?;

    let vector_index_config = VectorIndexConfig::Flat(FlatIndexConfig {
        dimension: embedder.dimension(),
        distance_metric: DistanceMetric::Cosine,
        normalize_vectors: true,
        ..Default::default()
    });
    let vector_index = VectorIndexFactory::create(vector_storage, vector_index_config)?;
    let mut vector_engine = VectorEngine::new(vector_index)?;
    println!("  VectorEngine created\n");

    // Step 4: Load documents from JSONL file
    println!("Step 4: Loading documents from resources/documents.jsonl...");
    let converter = JsonlDocumentConverter::new();
    let doc_iter = converter.convert("resources/documents.jsonl")?;

    // Collect documents for both indexing and embedding
    let mut documents = Vec::new();
    for doc_result in doc_iter {
        let doc = doc_result?;
        documents.push(doc);
    }
    println!("  Loaded {} documents\n", documents.len());

    // Step 5 & 6: Index documents in both LexicalEngine and VectorEngine
    // Note: We use sequential indexing to ensure document IDs match between engines.
    // LexicalEngine assigns IDs 0, 1, 2, ... internally, so we use the same IDs for VectorEngine.
    println!("Step 5 & 6: Indexing documents in both engines...");

    let mut doc_vectors = Vec::new();
    for (idx, doc) in documents.iter().enumerate() {
        let doc_id = idx as u64;

        // Add to LexicalEngine (internally assigns doc_id = 0, 1, 2, ...)
        lexical_engine.add_document(doc.clone())?;

        // Prepare for VectorEngine with the same doc_id
        let mut text_parts = Vec::new();
        if let Some(FieldValue::Text(title)) = doc.get_field("title") {
            text_parts.push(title.as_str());
        }
        if let Some(FieldValue::Text(body)) = doc.get_field("body") {
            text_parts.push(body.as_str());
        }
        let combined_text = text_parts.join(" ");

        // Generate embedding
        let vector = embedder.embed(&combined_text).await?;
        doc_vectors.push((doc_id, vector));
    }

    // Commit lexical index
    lexical_engine.commit()?;

    // Add vectors and commit
    vector_engine.add_vectors(doc_vectors)?;
    vector_engine.commit()?;
    vector_engine.optimize()?;

    println!("  Indexing completed for {} documents\n", documents.len());

    // Step 7: Create HybridEngine
    println!("Step 7: Creating HybridEngine...");
    let hybrid_engine = HybridEngine::new(lexical_engine, vector_engine)?;
    println!("  HybridEngine created successfully!\n");

    // Step 8: Perform various hybrid searches
    println!("{}", "=".repeat(80));
    println!("Step 8: Demonstrating Hybrid Search Strategies\n");

    // Search 1: Balanced hybrid search
    println!("\n[1] Balanced Hybrid Search");
    println!("{}", "-".repeat(80));
    let query1_text = "rust programming";
    let query1 = "body:rust body:programming";  // DSL query with field specification
    println!("Query: \"{}\"", query1_text);
    println!("Strategy: Balanced (50% keyword, 50% semantic)");

    let query_vector1 = embedder.embed(query1_text).await?;
    let request1 = HybridSearchRequest::new(query1)
        .with_vector(query_vector1)
        .keyword_weight(0.5)
        .vector_weight(0.5)
        .normalization(ScoreNormalization::MinMax)
        .max_results(5);

    let results1 = hybrid_engine.search(request1).await?;
    println!("\nResults:");
    println!("  Total matches: {}", results1.len());
    println!("  Keyword matches: {}", results1.keyword_matches);
    println!("  Vector matches: {}", results1.vector_matches);
    println!("  Query time: {} ms\n", results1.query_time_ms);

    display_hybrid_results(&results1);

    // Search 2: Keyword-focused search
    println!("\n[2] Keyword-Focused Hybrid Search");
    println!("{}", "-".repeat(80));
    let query2_text = "web development";
    let query2 = "body:web body:development";  // DSL query with field specification
    println!("Query: \"{}\"", query2_text);
    println!("Strategy: Keyword-focused (80% keyword, 20% semantic)");

    let query_vector2 = embedder.embed(query2_text).await?;
    let request2 = HybridSearchRequest::new(query2)
        .with_vector(query_vector2)
        .keyword_weight(0.8)
        .vector_weight(0.2)
        .normalization(ScoreNormalization::MinMax)
        .max_results(5);

    let results2 = hybrid_engine.search(request2).await?;
    println!("\nResults:");
    println!("  Total matches: {}", results2.len());
    println!("  Keyword matches: {}", results2.keyword_matches);
    println!("  Vector matches: {}", results2.vector_matches);
    println!("  Query time: {} ms\n", results2.query_time_ms);

    display_hybrid_results(&results2);

    // Search 3: Semantic-focused search
    println!("\n[3] Semantic-Focused Hybrid Search");
    println!("{}", "-".repeat(80));
    let query3_text = "machine learning algorithms";
    let query3 = "body:machine body:learning body:algorithms";  // DSL query with field specification
    println!("Query: \"{}\"", query3_text);
    println!("Strategy: Semantic-focused (30% keyword, 70% semantic)");

    let query_vector3 = embedder.embed(query3_text).await?;
    let request3 = HybridSearchRequest::new(query3)
        .with_vector(query_vector3)
        .keyword_weight(0.3)
        .vector_weight(0.7)
        .normalization(ScoreNormalization::MinMax)
        .max_results(5);

    let results3 = hybrid_engine.search(request3).await?;
    println!("\nResults:");
    println!("  Total matches: {}", results3.len());
    println!("  Keyword matches: {}", results3.keyword_matches);
    println!("  Vector matches: {}", results3.vector_matches);
    println!("  Query time: {} ms\n", results3.query_time_ms);

    display_hybrid_results(&results3);

    // Search 4: Strict matching (require both)
    println!("\n[4] Strict Hybrid Search (Require Both Matches)");
    println!("{}", "-".repeat(80));
    let query4_text = "python programming";
    let query4 = "body:python body:programming";  // DSL query with field specification
    println!("Query: \"{}\"", query4_text);
    println!("Strategy: Strict (50/50, requires both keyword and vector matches)");

    let query_vector4 = embedder.embed(query4_text).await?;
    let request4 = HybridSearchRequest::new(query4)
        .with_vector(query_vector4)
        .keyword_weight(0.5)
        .vector_weight(0.5)
        .min_keyword_score(0.1)
        .min_vector_similarity(0.3)
        .require_both(true)
        .normalization(ScoreNormalization::MinMax)
        .max_results(5);

    let results4 = hybrid_engine.search(request4).await?;
    println!("\nResults:");
    println!("  Total matches: {}", results4.len());
    println!("  Keyword matches: {}", results4.keyword_matches);
    println!("  Vector matches: {}", results4.vector_matches);
    println!("  Query time: {} ms\n", results4.query_time_ms);

    display_hybrid_results(&results4);

    // Step 9: Compare different normalization strategies
    println!("\n{}", "=".repeat(80));
    println!("\nStep 9: Comparing Normalization Strategies\n");

    let comparison_query_text = "database systems";
    let comparison_query = "body:database body:systems";  // DSL query with field specification
    println!("Query: \"{}\"", comparison_query_text);
    println!("Strategy: Testing different score normalization methods\n");

    let comparison_vector = embedder.embed(comparison_query_text).await?;

    for (name, norm_strategy) in [
        ("None", ScoreNormalization::None),
        ("MinMax", ScoreNormalization::MinMax),
        ("ZScore", ScoreNormalization::ZScore),
        ("Rank", ScoreNormalization::Rank),
    ] {
        println!("--- {} Normalization ---", name);
        let request = HybridSearchRequest::new(comparison_query)
            .with_vector(comparison_vector.clone())
            .keyword_weight(0.5)
            .vector_weight(0.5)
            .normalization(norm_strategy)
            .max_results(3);

        let results = hybrid_engine.search(request).await?;
        println!("Top 3 results:");
        for (i, result) in results.results.iter().take(3).enumerate() {
            println!(
                "  {}. Doc {} - Hybrid: {:.4}, Keyword: {:.4}, Vector: {:.4}",
                i + 1,
                result.doc_id,
                result.hybrid_score,
                result.keyword_score.unwrap_or(0.0),
                result.vector_similarity.unwrap_or(0.0)
            );
        }
        println!();
    }

    // Summary
    println!("{}", "=".repeat(80));
    println!("\n=== Summary ===\n");
    println!("This example demonstrated:");
    println!("  ✓ Creating a HybridEngine with LexicalEngine and VectorEngine");
    println!("  ✓ Indexing documents in both lexical and vector indexes");
    println!("  ✓ Performing actual hybrid searches with different strategies:");
    println!("    - Balanced: Equal weight for keyword and semantic");
    println!("    - Keyword-focused: Emphasizes exact matching");
    println!("    - Semantic-focused: Emphasizes conceptual similarity");
    println!("    - Strict: Requires matches from both search types");
    println!("  ✓ Comparing different score normalization strategies:");
    println!("    - None: Raw scores");
    println!("    - MinMax: Normalized to [0, 1]");
    println!("    - ZScore: Statistical normalization");
    println!("    - Rank: Reciprocal rank fusion");
    println!("\nHybrid search successfully combines the best of both worlds!");
    println!("Use keyword search for precise matching and vector search for semantic understanding.\n");

    Ok(())
}

#[cfg(feature = "embeddings-candle")]
fn display_hybrid_results(results: &yatagarasu::hybrid::search::searcher::HybridSearchResults) {
    for (i, result) in results.results.iter().enumerate() {
        println!("  {}. Doc {} - Hybrid Score: {:.4}", i + 1, result.doc_id, result.hybrid_score);

        if let Some(kw_score) = result.keyword_score {
            println!("     Keyword Score: {:.4}", kw_score);
        }
        if let Some(vec_sim) = result.vector_similarity {
            println!("     Vector Similarity: {:.4}", vec_sim);
        }

        if let Some(doc) = &result.document {
            if let Some(title) = doc.get("title") {
                println!("     Title: {}", title);
            }
            if let Some(category) = doc.get("category") {
                println!("     Category: {}", category);
            }
        }
        println!();
    }
}

#[cfg(not(feature = "embeddings-candle"))]
fn main() {
    eprintln!("This example requires the 'embeddings-candle' feature.");
    eprintln!("Please run with: cargo run --example hybrid_search --features embeddings-candle");
    std::process::exit(1);
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "embeddings-candle")]
    #[tokio::test]
    async fn test_hybrid_search_example() {
        let result = super::main().await;
        assert!(result.is_ok());
    }
}
