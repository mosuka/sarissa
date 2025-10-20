//! Example of parallel hybrid search combining keyword and vector search.

use sage::document::document::Document;
use sage::document::field_value::FieldValue;
use sage::error::Result;
use sage::parallel_hybrid_search::config::MergeStrategy;
use sage::parallel_hybrid_search::config::ParallelHybridSearchConfig;
use sage::parallel_hybrid_search::engine::ParallelHybridSearchEngine;
use sage::parallel_hybrid_search::mock_index::MockIndexReader;
use sage::query::term::TermQuery;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Parallel Hybrid Search Example ===\n");

    // Create configuration
    let config = ParallelHybridSearchConfig::new()
        .with_threads(4)
        .with_weights(0.6, 0.4) // 60% keyword, 40% vector
        .with_merge_strategy(MergeStrategy::LinearCombination)
        .with_max_results(10);

    // Create the parallel hybrid search engine
    let engine = ParallelHybridSearchEngine::new(config)?;

    // Add multiple indices with mock data
    println!("Adding test indices...");
    let mock_readers = vec![
        Arc::new(MockIndexReader::new()),
        Arc::new(MockIndexReader::new()),
        Arc::new(MockIndexReader::new()),
    ];

    // Add documents to mock indices
    println!("Adding documents to indices...");

    // Index 0: Rust documents
    let mut doc1 = Document::new();
    doc1.add_field("title", FieldValue::Text("Rust Programming".to_string()));
    doc1.add_field(
        "content",
        FieldValue::Text("Rust is a systems programming language".to_string()),
    );
    mock_readers[0].add_document(1, doc1);

    let mut doc2 = Document::new();
    doc2.add_field("title", FieldValue::Text("Rust Tutorial".to_string()));
    doc2.add_field(
        "content",
        FieldValue::Text("Learn Rust programming step by step".to_string()),
    );
    mock_readers[0].add_document(4, doc2);

    // Index 1: Python documents
    let mut doc3 = Document::new();
    doc3.add_field("title", FieldValue::Text("Python Tutorial".to_string()));
    doc3.add_field(
        "content",
        FieldValue::Text("Python is great for machine learning and programming".to_string()),
    );
    mock_readers[1].add_document(2, doc3);

    let mut doc4 = Document::new();
    doc4.add_field("title", FieldValue::Text("Python Data Science".to_string()));
    doc4.add_field(
        "content",
        FieldValue::Text("Data science with Python programming".to_string()),
    );
    mock_readers[1].add_document(5, doc4);

    // Index 2: JavaScript documents
    let mut doc5 = Document::new();
    doc5.add_field("title", FieldValue::Text("JavaScript Guide".to_string()));
    doc5.add_field(
        "content",
        FieldValue::Text("JavaScript powers the modern web programming".to_string()),
    );
    mock_readers[2].add_document(3, doc5);

    let mut doc6 = Document::new();
    doc6.add_field(
        "title",
        FieldValue::Text("JavaScript Frameworks".to_string()),
    );
    doc6.add_field(
        "content",
        FieldValue::Text("Modern JavaScript frameworks for web development".to_string()),
    );
    mock_readers[2].add_document(6, doc6);

    // Add indices to engine
    for (i, reader) in mock_readers.into_iter().enumerate() {
        engine
            .add_index(
                format!("index_{i}"),
                reader as Arc<dyn sage::lexical::reader::IndexReader>,
                None, // No vector reader for this example
                1.0,  // Equal weight for all indices
            )
            .await?;
    }

    // Add all documents to the document store for display purposes
    println!("Adding sample documents to document store...");

    // Documents from Index 0 (Rust)
    let mut doc_fields = HashMap::new();
    doc_fields.insert("title".to_string(), "Rust Programming".to_string());
    doc_fields.insert(
        "content".to_string(),
        "Rust is a systems programming language".to_string(),
    );
    engine.add_document(1, doc_fields).await?;

    let mut doc_fields = HashMap::new();
    doc_fields.insert("title".to_string(), "Rust Tutorial".to_string());
    doc_fields.insert(
        "content".to_string(),
        "Learn Rust programming step by step".to_string(),
    );
    engine.add_document(4, doc_fields).await?;

    // Documents from Index 1 (Python)
    let mut doc_fields = HashMap::new();
    doc_fields.insert("title".to_string(), "Python Tutorial".to_string());
    doc_fields.insert(
        "content".to_string(),
        "Python is great for machine learning and programming".to_string(),
    );
    engine.add_document(2, doc_fields).await?;

    let mut doc_fields = HashMap::new();
    doc_fields.insert("title".to_string(), "Python Data Science".to_string());
    doc_fields.insert(
        "content".to_string(),
        "Data science with Python programming".to_string(),
    );
    engine.add_document(5, doc_fields).await?;

    // Documents from Index 2 (JavaScript)
    let mut doc_fields = HashMap::new();
    doc_fields.insert("title".to_string(), "JavaScript Guide".to_string());
    doc_fields.insert(
        "content".to_string(),
        "JavaScript powers the modern web programming".to_string(),
    );
    engine.add_document(3, doc_fields).await?;

    let mut doc_fields = HashMap::new();
    doc_fields.insert("title".to_string(), "JavaScript Frameworks".to_string());
    doc_fields.insert(
        "content".to_string(),
        "Modern JavaScript frameworks for web development".to_string(),
    );
    engine.add_document(6, doc_fields).await?;

    // Train embedder with sample documents (optional)
    println!("\nTraining embedder...");
    let training_docs = vec![
        "Rust is a systems programming language",
        "Python is great for machine learning",
        "JavaScript powers the modern web",
        "Programming languages comparison",
        "Software development best practices",
    ];
    engine.train_embedder(&training_docs).await?;

    // Perform searches
    println!("\n=== Single Query Search ===");
    let query = Box::new(TermQuery::new("content", "programming"));
    let start = Instant::now();
    let results = engine.search("programming languages", query).await?;
    let elapsed = start.elapsed();

    println!(
        "Search completed in {:.2}ms",
        elapsed.as_secs_f64() * 1000.0
    );
    println!("Found {} results", results.len());
    println!("Indices searched: {}", results.indices_searched);
    println!("Total keyword matches: {}", results.total_keyword_matches);
    println!("Total vector matches: {}", results.total_vector_matches);

    // Display results
    for (i, result) in results.results.iter().enumerate() {
        println!("\nResult #{}", i + 1);
        println!("  Doc ID: {}", result.doc_id);
        println!("  Combined Score: {:.4}", result.combined_score);
        if let Some(kw_score) = result.keyword_score {
            println!("  Keyword Score: {kw_score:.4}");
        }
        if let Some(vec_sim) = result.vector_similarity {
            println!("  Vector Similarity: {vec_sim:.4}");
        }
        println!("  From Index: {}", result.index_id);

        // Display document fields
        if !result.fields.is_empty() {
            println!("  Document:");
            for (field, value) in &result.fields {
                println!("    {field}: {value}");
            }
        }

        // Display score explanation
        if let Some(explanation) = &result.explanation {
            println!("  Score Explanation:");
            println!("    Method: {}", explanation.method);
            println!(
                "    Keyword Contribution: {:.4}",
                explanation.keyword_contribution
            );
            println!(
                "    Vector Contribution: {:.4}",
                explanation.vector_contribution
            );
        }
    }

    // Display timing breakdown
    println!("\n=== Search Time Breakdown ===");
    println!(
        "Keyword Search: {:.2}ms",
        results.time_breakdown.keyword_search_ms
    );
    println!(
        "Vector Search: {:.2}ms",
        results.time_breakdown.vector_search_ms
    );
    println!("Result Merging: {:.2}ms", results.time_breakdown.merge_ms);
    println!(
        "Query Expansion: {:.2}ms",
        results.time_breakdown.expansion_ms
    );
    println!("Result Ranking: {:.2}ms", results.time_breakdown.ranking_ms);

    // Display cache statistics
    println!("\n=== Cache Statistics ===");
    println!("Cache Hit Rate: {:.2}%", results.cache_hit_rate() * 100.0);
    println!(
        "Keyword Cache - Hits: {}, Misses: {}",
        results.cache_stats.keyword_hits, results.cache_stats.keyword_misses
    );
    println!(
        "Vector Cache - Hits: {}, Misses: {}",
        results.cache_stats.vector_hits, results.cache_stats.vector_misses
    );

    // Batch search example
    println!("\n=== Batch Search Example ===");
    let queries = vec![
        (
            "rust programming",
            Box::new(TermQuery::new("content", "rust")) as Box<dyn sage::query::query::Query>,
        ),
        (
            "python machine learning",
            Box::new(TermQuery::new("content", "python")) as Box<dyn sage::query::query::Query>,
        ),
        (
            "javascript web",
            Box::new(TermQuery::new("content", "javascript")) as Box<dyn sage::query::query::Query>,
        ),
    ];

    let start = Instant::now();
    let batch_results = engine.batch_search(queries).await?;
    let elapsed = start.elapsed();

    println!(
        "Batch search completed in {:.2}ms",
        elapsed.as_secs_f64() * 1000.0
    );
    println!("Processed {} queries", batch_results.len());

    for (i, results) in batch_results.iter().enumerate() {
        println!("\nQuery #{}: Found {} results", i + 1, results.len());
        if let Some(top_result) = results.top_result() {
            println!(
                "  Top result: Doc ID {} with score {:.4}",
                top_result.doc_id, top_result.combined_score
            );
        }
    }

    // Test different merge strategies
    println!("\n=== Testing Different Merge Strategies ===");
    let strategies = vec![
        MergeStrategy::LinearCombination,
        MergeStrategy::ReciprocalRankFusion,
        MergeStrategy::MaxScore,
        MergeStrategy::ScoreProduct,
        MergeStrategy::Adaptive,
    ];

    for strategy in strategies {
        let config = ParallelHybridSearchConfig::new()
            .with_merge_strategy(strategy)
            .with_weights(0.5, 0.5);

        let test_engine = ParallelHybridSearchEngine::new(config)?;

        // Add test indices with programming-related content
        let mock_reader = Arc::new(MockIndexReader::new());
        let mut doc1 = Document::new();
        doc1.add_field("title", FieldValue::Text("Rust Programming".to_string()));
        doc1.add_field(
            "content",
            FieldValue::Text("Rust is a systems programming language".to_string()),
        );
        mock_reader.add_document(1, doc1);

        let mut doc2 = Document::new();
        doc2.add_field("title", FieldValue::Text("Python Guide".to_string()));
        doc2.add_field(
            "content",
            FieldValue::Text("Python programming tutorial".to_string()),
        );
        mock_reader.add_document(2, doc2);

        test_engine
            .add_index(
                "test_index".to_string(),
                mock_reader as Arc<dyn sage::lexical::reader::IndexReader>,
                None,
                1.0,
            )
            .await?;

        // Add documents to document store
        let mut doc_fields = HashMap::new();
        doc_fields.insert("title".to_string(), "Rust Programming".to_string());
        doc_fields.insert(
            "content".to_string(),
            "Rust is a systems programming language".to_string(),
        );
        test_engine.add_document(1, doc_fields).await?;

        let mut doc_fields = HashMap::new();
        doc_fields.insert("title".to_string(), "Python Guide".to_string());
        doc_fields.insert(
            "content".to_string(),
            "Python programming tutorial".to_string(),
        );
        test_engine.add_document(2, doc_fields).await?;

        let query = Box::new(TermQuery::new("content", "programming"));
        let results = test_engine.search("programming languages", query).await?;

        println!("\nStrategy {:?}: Found {} results", strategy, results.len());
        if let Some(top) = results.top_result()
            && let Some(explanation) = &top.explanation
        {
            println!("  Scoring method: {}", explanation.method);
        }
    }

    // Clear caches and show stats
    println!("\n=== Final Statistics ===");
    engine.clear_caches().await;
    let final_cache_stats = engine.get_cache_stats().await;
    println!("Cache cleared - all stats reset to zero");
    println!(
        "Keyword hits: {}, Vector hits: {}",
        final_cache_stats.keyword_hits, final_cache_stats.vector_hits
    );

    Ok(())
}
