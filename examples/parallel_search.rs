//! Parallel Search example - demonstrates parallel indexing followed by concurrent searches across multiple indices.
//!
//! This example combines parallel indexing and parallel search to show a complete workflow:
//! 1. Create multiple indices using parallel indexing with different partitioning strategies
//! 2. Perform concurrent searches across the indices with various merge strategies
//! 3. Demonstrate performance metrics and optimization techniques

use sarissa::full_text_index::AdvancedIndexWriter;
use sarissa::full_text_index::AdvancedWriterConfig;
use sarissa::full_text_search::AdvancedIndexReader;
use sarissa::full_text_search::advanced_reader::AdvancedReaderConfig;
use sarissa::parallel_full_text_index::{
    HashPartitioner, ParallelIndexConfig, ParallelIndexEngine, PartitionConfig,
    config::IndexingOptions,
};
use sarissa::parallel_full_text_search::{
    MergeStrategyType, ParallelSearchConfig, ParallelSearchEngine, config::SearchOptions,
};
use sarissa::prelude::*;
use sarissa::query::{PhraseQuery, TermQuery};
use sarissa::storage::{MemoryStorage, StorageConfig};
use std::path::Path;
use std::sync::Arc;
use tempfile::TempDir;

fn main() -> Result<()> {
    println!("=== Parallel Search Example - Complete Indexing and Search Workflow ===\n");

    // ========================================================================
    // PART 1: PARALLEL INDEXING
    // ========================================================================

    println!("PART 1: PARALLEL INDEXING\n");

    // Create temporary directories for multiple indices
    let temp_dirs: Vec<TempDir> = (0..4).map(|_| TempDir::new().unwrap()).collect();
    let index_paths: Vec<&Path> = temp_dirs.iter().map(|dir| dir.path()).collect();

    println!("Creating {} parallel indices:", index_paths.len());
    for (i, path) in index_paths.iter().enumerate() {
        println!("  Index {i}: {path:?}");
    }

    // Create a comprehensive schema

    println!("\n=== Hash Partitioning for Document Indexing ===\n");

    // Configure parallel indexing engine
    let parallel_config = ParallelIndexConfig {
        max_concurrent_partitions: 4,
        default_batch_size: 10,
        max_buffer_memory: 512 * 1024 * 1024,
        commit_interval: std::time::Duration::from_secs(30),
        retry_attempts: 3,
        operation_timeout: std::time::Duration::from_secs(60),
        enable_metrics: true,
        thread_pool_size: None,
        allow_partial_failures: true,
        auto_commit_threshold: 10000,
    };

    let mut parallel_engine = ParallelIndexEngine::new(parallel_config)?;

    // Set up hash partitioner by user_id
    let hash_partitioner = HashPartitioner::new("user_id".to_string(), 4);
    parallel_engine.set_partitioner(Box::new(hash_partitioner))?;

    // Storage for later search operations
    let mut storages: Vec<Arc<dyn sarissa::storage::Storage>> = Vec::new();

    // Add writers for each partition
    for i in 0..4 {
        let storage: Arc<dyn sarissa::storage::Storage> =
            Arc::new(MemoryStorage::new(StorageConfig::default()));
        storages.push(Arc::clone(&storage));

        let writer = Box::new(AdvancedIndexWriter::new(
            Arc::clone(&storage),
            AdvancedWriterConfig::default(),
        )?);
        let partition_config = PartitionConfig::new(format!("partition_{i}"));
        parallel_engine.add_partition(format!("partition_{i}"), writer, partition_config)?;
    }

    // Create comprehensive test documents
    let documents = vec![
        // Technology products
        Document::builder()
            .add_text("id", "T001")
            .add_text("title", "Laptop Pro X Advanced")
            .add_text(
                "content",
                "High-performance laptop with latest processor technology",
            )
            .add_text(
                "description",
                "Professional laptop for developers and creators",
            )
            .add_text("author", "TechCorp")
            .add_text("category", "electronics")
            .add_integer("user_id", 1001)
            .add_text("region", "north-america")
            .add_integer("priority", 1)
            .add_float("price", 1299.99)
            .build(),
        Document::builder()
            .add_text("id", "T002")
            .add_text("title", "Smartphone Ultra Smart")
            .add_text(
                "content",
                "Latest smartphone with advanced camera system and AI",
            )
            .add_text("description", "Premium smartphone with professional camera")
            .add_text("author", "PhoneCorp")
            .add_text("category", "electronics")
            .add_integer("user_id", 2002)
            .add_text("region", "europe")
            .add_integer("priority", 2)
            .add_float("price", 999.99)
            .build(),
        Document::builder()
            .add_text("id", "T003")
            .add_text("title", "Smart Watch Pro")
            .add_text("content", "Advanced fitness and health tracking smartwatch")
            .add_text(
                "description",
                "Professional smartwatch for health monitoring",
            )
            .add_text("author", "WearTech")
            .add_text("category", "electronics")
            .add_integer("user_id", 3003)
            .add_text("region", "asia-pacific")
            .add_integer("priority", 1)
            .add_float("price", 399.99)
            .build(),
        // Books and education
        Document::builder()
            .add_text("id", "B001")
            .add_text("title", "The Rust Programming Language Guide")
            .add_text(
                "content",
                "Official comprehensive guide to learning Rust programming",
            )
            .add_text(
                "description",
                "Professional programming guide for developers",
            )
            .add_text("author", "Steve Klabnik")
            .add_text("category", "books")
            .add_integer("user_id", 4004)
            .add_text("region", "north-america")
            .add_integer("priority", 2)
            .add_float("price", 49.99)
            .build(),
        Document::builder()
            .add_text("id", "B002")
            .add_text("title", "Advanced Machine Learning")
            .add_text(
                "content",
                "Comprehensive guide to advanced machine learning algorithms",
            )
            .add_text(
                "description",
                "Professional guide to ML for data scientists",
            )
            .add_text("author", "Dr. Alice Wilson")
            .add_text("category", "books")
            .add_integer("user_id", 5005)
            .add_text("region", "europe")
            .add_integer("priority", 1)
            .add_float("price", 79.99)
            .build(),
        // Home products
        Document::builder()
            .add_text("id", "H001")
            .add_text("title", "Smart Home Hub Pro")
            .add_text(
                "content",
                "Central control system for all smart home devices",
            )
            .add_text("description", "Professional smart home automation center")
            .add_text("author", "HomeTech")
            .add_text("category", "home")
            .add_integer("user_id", 6006)
            .add_text("region", "asia-pacific")
            .add_integer("priority", 2)
            .add_float("price", 199.99)
            .build(),
        Document::builder()
            .add_text("id", "H002")
            .add_text("title", "Robot Vacuum Advanced")
            .add_text(
                "content",
                "Advanced robotic vacuum with smart mapping and AI",
            )
            .add_text(
                "description",
                "Professional cleaning robot with advanced features",
            )
            .add_text("author", "CleanBot")
            .add_text("category", "home")
            .add_integer("user_id", 7007)
            .add_text("region", "north-america")
            .add_integer("priority", 3)
            .add_float("price", 599.99)
            .build(),
        Document::builder()
            .add_text("id", "H003")
            .add_text("title", "Smart Security Camera Pro")
            .add_text(
                "content",
                "HD security camera with night vision and smart detection",
            )
            .add_text("description", "Professional security monitoring system")
            .add_text("author", "SecureTech")
            .add_text("category", "home")
            .add_integer("user_id", 8008)
            .add_text("region", "europe")
            .add_integer("priority", 1)
            .add_float("price", 249.99)
            .build(),
    ];

    println!(
        "Indexing {} documents with hash partitioning...",
        documents.len()
    );

    // Index documents using the parallel engine
    let indexing_result =
        parallel_engine.index_documents(documents.clone(), IndexingOptions::default())?;

    println!("Indexing Results:");
    println!("  Total documents: {}", indexing_result.total_documents);
    println!("  Documents indexed: {}", indexing_result.documents_indexed);
    println!("  Documents failed: {}", indexing_result.documents_failed);
    println!(
        "  Execution time: {:.2}ms",
        indexing_result.execution_time.as_millis()
    );

    // Commit all indices
    println!("\nCommitting all indices...");
    let failed_commits = parallel_engine.commit_all()?;
    if failed_commits.is_empty() {
        println!("All commits successful!");
    } else {
        println!("Some commits failed: {failed_commits:?}");
    }

    // ========================================================================
    // PART 2: PARALLEL SEARCH
    // ========================================================================

    println!("\n{}", "=".repeat(80));
    println!("PART 2: PARALLEL SEARCH\n");

    // Create readers from the indexed storages
    let mut index_readers = Vec::new();

    for (i, storage) in storages.into_iter().enumerate() {
        // Load segments for this storage
        let files = storage.list_files()?;
        let mut segments = Vec::new();
        for file in files {
            if file.starts_with("segment_") && file.ends_with(".meta") {
                let mut input = storage.open_input(&file)?;
                let mut data = Vec::new();
                std::io::Read::read_to_end(&mut input, &mut data)?;
                let segment_info: sarissa::full_text::SegmentInfo = serde_json::from_slice(&data)
                    .map_err(|e| sarissa::error::SarissaError::index(format!("Failed to parse segment metadata: {e}")))?;
                segments.push(segment_info);
            }
        }
        segments.sort_by_key(|s| s.generation);

        let reader = Box::new(AdvancedIndexReader::new(segments, storage, AdvancedReaderConfig::default())?);
        index_readers.push((format!("partition_{i}"), reader));
    }

    // Configure parallel search engine
    let search_config = ParallelSearchConfig {
        max_concurrent_tasks: 4,
        max_results_per_index: 100,
        enable_metrics: true,
        default_merge_strategy: MergeStrategyType::ScoreBased,
        allow_partial_results: true,
        ..Default::default()
    };

    let search_engine = ParallelSearchEngine::new(search_config)?;

    // Add indices to the search engine with different weights
    for (i, (name, reader)) in index_readers.into_iter().enumerate() {
        let weight = match i {
            0 => 1.0, // Electronics partition
            1 => 1.5, // Books partition - higher weight
            2 => 0.8, // Home partition - lower weight
            3 => 1.0, // Mixed partition
            _ => 1.0,
        };
        search_engine.add_index(name, reader, weight)?;
    }

    println!(
        "Added {} indices to parallel search engine",
        search_engine.index_count()?
    );
    println!("Index weights: [1.0, 1.5, 0.8, 1.0] for different content priorities");

    println!("\n=== Example 1: Smart Device Search ===\n");

    // Search for "Smart" across all indices
    let query = Box::new(TermQuery::new("title", "Smart"));
    let options = SearchOptions::new(10)
        .with_metrics(true)
        .with_timeout(std::time::Duration::from_secs(5));

    let results = search_engine.search(query, options)?;

    println!("Searching for 'Smart' in title field:");
    println!("Total hits: {}", results.total_hits);
    println!("Results returned: {}", results.hits.len());
    println!("Max score: {:.4}", results.max_score);

    for (i, hit) in results.hits.iter().enumerate() {
        if let Some(doc) = &hit.document {
            if let Some(title) = doc.get_field("title").and_then(|f| f.as_text()) {
                if let Some(id) = doc.get_field("id").and_then(|f| f.as_text()) {
                    if let Some(category) = doc.get_field("category").and_then(|f| f.as_text()) {
                        println!(
                            "  {}. {} [{}:{}] (score: {:.4})",
                            i + 1,
                            title,
                            id,
                            category,
                            hit.score
                        );
                    }
                }
            }
        }
    }

    println!("\n=== Example 2: Professional/Advanced Products ===\n");

    // Search for "Professional" or "Advanced"
    let query = Box::new(TermQuery::new("description", "Professional"));
    let options = SearchOptions::new(10)
        .with_merge_strategy(MergeStrategyType::Weighted)
        .with_metrics(true);

    let results = search_engine.search(query, options)?;

    println!("Searching for 'Professional' in description field:");
    println!("Using weighted merge strategy - Books index boosted 1.5x");
    println!("Total hits: {}", results.total_hits);

    for (i, hit) in results.hits.iter().enumerate() {
        if let Some(doc) = &hit.document {
            if let Some(title) = doc.get_field("title").and_then(|f| f.as_text()) {
                if let Some(price) = doc.get_field("price").and_then(|f| match f {
                    sarissa::document::FieldValue::Float(v) => Some(*v),
                    _ => None,
                }) {
                    if let Some(category) = doc.get_field("category").and_then(|f| f.as_text()) {
                        println!(
                            "  {}. {} [{}] - ${:.2} (weighted score: {:.4})",
                            i + 1,
                            title,
                            category,
                            price,
                            hit.score
                        );
                    }
                }
            }
        }
    }

    println!("\n=== Example 3: Programming Language Search ===\n");

    // Search for phrase "Programming Language"
    let query = Box::new(PhraseQuery::new(
        "title",
        vec!["Programming".to_string(), "Language".to_string()],
    ));
    let options = SearchOptions::new(5)
        .with_merge_strategy(MergeStrategyType::Weighted)
        .with_metrics(true);

    let results = search_engine.search(query, options)?;

    println!("Searching for phrase 'Programming Language':");
    println!("Total hits: {}", results.total_hits);

    for (i, hit) in results.hits.iter().enumerate() {
        if let Some(doc) = &hit.document {
            if let Some(title) = doc.get_field("title").and_then(|f| f.as_text()) {
                if let Some(author) = doc.get_field("author").and_then(|f| f.as_text()) {
                    println!(
                        "  {}. {} by {} (score: {:.4})",
                        i + 1,
                        title,
                        author,
                        hit.score
                    );
                }
            }
        }
    }

    println!("\n=== Example 4: Advanced Features Search ===\n");

    // Search for "Advanced" with minimum score threshold
    let query = Box::new(TermQuery::new("content", "Advanced"));
    let options = SearchOptions::new(10)
        .with_min_score(0.1)
        .with_metrics(true);

    let results = search_engine.search(query, options)?;

    println!("Searching for 'Advanced' in content with min score 0.1:");
    println!("Total hits: {}", results.total_hits);

    for (i, hit) in results.hits.iter().enumerate() {
        if let Some(doc) = &hit.document {
            if let Some(title) = doc.get_field("title").and_then(|f| f.as_text()) {
                if let Some(category) = doc.get_field("category").and_then(|f| f.as_text()) {
                    println!(
                        "  {}. {} [{}] (score: {:.4})",
                        i + 1,
                        title,
                        category,
                        hit.score
                    );
                }
            }
        }
    }

    println!("\n=== Example 5: Concurrent Multi-Term Search ===\n");

    // Execute multiple searches to demonstrate concurrency
    let search_terms = vec![
        ("title", "Pro"),
        ("content", "system"),
        ("description", "guide"),
        ("category", "electronics"),
        ("author", "Tech"),
    ];

    println!(
        "Executing {} concurrent searches across {} indices...",
        search_terms.len(),
        search_engine.index_count()?
    );

    let mut total_time = std::time::Duration::new(0, 0);
    let mut total_hits = 0;

    for (field, term) in &search_terms {
        let start = std::time::Instant::now();
        let query = Box::new(TermQuery::new(*field, *term));
        let options = SearchOptions::new(5).with_metrics(true);

        let results = search_engine.search(query, options)?;
        let elapsed = start.elapsed();
        total_time += elapsed;
        total_hits += results.total_hits;

        println!(
            "  Search '{}' in '{}': {} hits in {:.2}μs",
            term,
            field,
            results.total_hits,
            elapsed.as_micros()
        );
    }

    println!("\nConcurrent Search Summary:");
    println!("  Total search time: {:.2}μs", total_time.as_micros());
    println!(
        "  Average time per search: {:.2}μs",
        total_time.as_micros() / search_terms.len() as u128
    );
    println!("  Total hits across all searches: {total_hits}");

    println!("\n=== Performance Metrics ===\n");

    let search_metrics = search_engine.metrics();
    println!("Search Engine Performance:");
    println!("  Total searches: {}", search_metrics.total_searches);
    println!(
        "  Successful searches: {}",
        search_metrics.successful_searches
    );
    println!(
        "  Success rate: {:.1}%",
        (search_metrics.successful_searches as f64 / search_metrics.total_searches as f64) * 100.0
    );
    println!(
        "  Average execution time: {:.2}μs",
        search_metrics.avg_execution_time.as_micros()
    );
    println!(
        "  Total hits returned: {}",
        search_metrics.total_hits_returned
    );

    let index_metrics = parallel_engine.metrics();
    println!("\nIndexing Engine Performance:");
    println!("  Total operations: {}", index_metrics.total_operations);
    println!(
        "  Successful operations: {}",
        index_metrics.successful_operations
    );
    println!(
        "  Success rate: {:.1}%",
        (index_metrics.successful_operations as f64 / index_metrics.total_operations as f64)
            * 100.0
    );

    println!("\n=== Summary ===\n");

    println!("Parallel Search Workflow Complete!");
    println!(
        "✓ Indexed {} documents across {} partitions using hash partitioning",
        indexing_result.documents_indexed, 4
    );
    println!(
        "✓ Executed {} searches across {} indices with different merge strategies",
        search_metrics.total_searches,
        search_engine.index_count()?
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_search_example() {
        let result = main();
        assert!(result.is_ok());
    }
}
