//! Parallel Search example - demonstrates concurrent searches across multiple indices with result merging.

use sarissa::index::reader::BasicIndexReader;
use sarissa::index::writer::{BasicIndexWriter, WriterConfig};
use sarissa::parallel_search::{
    MergeStrategyType, ParallelSearchConfig, ParallelSearchEngine, config::SearchOptions,
};
use sarissa::prelude::*;
use sarissa::query::{PhraseQuery, TermQuery};
use sarissa::schema::{IdField, NumericField, TextField};
use sarissa::storage::{MemoryStorage, StorageConfig};
use std::sync::Arc;
use tempfile::TempDir;

fn main() -> Result<()> {
    println!("=== Parallel Search Example - Concurrent Multi-Index Search ===\n");

    // Create temporary directories for multiple indices
    let temp_dirs: Vec<TempDir> = (0..3).map(|_| TempDir::new().unwrap()).collect();

    println!("Creating {} indices for parallel search:", temp_dirs.len());
    for (i, dir) in temp_dirs.iter().enumerate() {
        println!("  Index {}: {:?}", i, dir.path());
    }

    // Create schema for product catalog
    let mut schema = Schema::new();
    schema.add_field("id", Box::new(IdField::new()))?;
    schema.add_field(
        "title",
        Box::new(TextField::new().stored(true).indexed(true)),
    )?;
    schema.add_field("description", Box::new(TextField::new().indexed(true)))?;
    schema.add_field("category", Box::new(TextField::new().indexed(true)))?;
    schema.add_field("price", Box::new(NumericField::f64().indexed(true)))?;

    println!("\n=== Setting Up Indices ===\n");

    // Create and populate indices with different product categories
    let indices_data = vec![
        // Index 0: Electronics
        vec![
            (
                "E001",
                "Laptop Pro X",
                "High-performance laptop with latest processor",
                "electronics",
                1299.99,
            ),
            (
                "E002",
                "Smartphone Ultra",
                "Latest smartphone with advanced camera system",
                "electronics",
                999.99,
            ),
            (
                "E003",
                "Wireless Headphones",
                "Premium noise-canceling wireless headphones",
                "electronics",
                349.99,
            ),
            (
                "E004",
                "Tablet Pro",
                "Professional tablet for creative work",
                "electronics",
                799.99,
            ),
            (
                "E005",
                "Smart Watch",
                "Advanced fitness and health tracking smartwatch",
                "electronics",
                399.99,
            ),
        ],
        // Index 1: Books
        vec![
            (
                "B001",
                "The Rust Programming Language",
                "Official guide to learning Rust programming",
                "books",
                49.99,
            ),
            (
                "B002",
                "Computer Science Fundamentals",
                "Comprehensive guide to CS concepts",
                "books",
                59.99,
            ),
            (
                "B003",
                "Machine Learning Basics",
                "Introduction to machine learning algorithms",
                "books",
                44.99,
            ),
            (
                "B004",
                "Data Structures and Algorithms",
                "Essential algorithms for programmers",
                "books",
                54.99,
            ),
            (
                "B005",
                "Cloud Architecture Patterns",
                "Best practices for cloud systems",
                "books",
                64.99,
            ),
        ],
        // Index 2: Home & Garden
        vec![
            (
                "H001",
                "Smart Home Hub",
                "Central control for all smart home devices",
                "home",
                199.99,
            ),
            (
                "H002",
                "Robot Vacuum Pro",
                "Advanced robotic vacuum with mapping",
                "home",
                599.99,
            ),
            (
                "H003",
                "Smart Thermostat",
                "Energy-efficient programmable thermostat",
                "home",
                249.99,
            ),
            (
                "H004",
                "Garden Tool Set",
                "Professional quality gardening tools",
                "garden",
                149.99,
            ),
            (
                "H005",
                "Smart Security Camera",
                "HD security camera with night vision",
                "home",
                179.99,
            ),
        ],
    ];

    // Create indices and add documents
    let mut index_readers = Vec::new();

    for (index_num, (_temp_dir, products)) in temp_dirs.iter().zip(indices_data.iter()).enumerate()
    {
        let storage: Arc<dyn sarissa::storage::Storage> =
            Arc::new(MemoryStorage::new(StorageConfig::default()));
        let mut writer = BasicIndexWriter::new(
            schema.clone(),
            Arc::clone(&storage),
            WriterConfig::default(),
        )?;

        println!(
            "Populating Index {} with {} products",
            index_num,
            products.len()
        );

        for (id, title, description, category, price) in products {
            let doc = Document::builder()
                .add_text("id", *id)
                .add_text("title", *title)
                .add_text("description", *description)
                .add_text("category", *category)
                .add_float("price", *price)
                .build();

            writer.add_document(doc)?;
        }

        writer.commit()?;

        // Create reader for this index
        let reader = Box::new(BasicIndexReader::new(schema.clone(), Arc::clone(&storage))?);
        index_readers.push((format!("index_{}", index_num), reader));
    }

    println!("\n=== Configuring Parallel Search Engine ===\n");

    // Configure parallel search engine
    let config = ParallelSearchConfig {
        max_concurrent_tasks: 3,
        max_results_per_index: 100,
        enable_metrics: true,
        default_merge_strategy: MergeStrategyType::ScoreBased,
        allow_partial_results: true,
        ..Default::default()
    };

    let engine = ParallelSearchEngine::new(config)?;

    // Add indices to the engine with different weights
    for (i, (name, reader)) in index_readers.into_iter().enumerate() {
        let weight = match i {
            0 => 1.0, // Electronics - normal weight
            1 => 1.5, // Books - higher weight
            2 => 0.8, // Home - lower weight
            _ => 1.0,
        };
        engine.add_index(name, reader, weight)?;
    }

    println!(
        "Added {} indices to parallel search engine",
        engine.index_count()?
    );
    println!("  - index_0 (Electronics): weight = 1.0");
    println!("  - index_1 (Books): weight = 1.5");
    println!("  - index_2 (Home & Garden): weight = 0.8");

    println!("\n=== Example 1: Simple Term Search ===\n");

    // Search for "Smart" across all indices
    let query = Box::new(TermQuery::new("title", "Smart"));
    let options = SearchOptions::new(10)
        .with_metrics(true)
        .with_timeout(std::time::Duration::from_secs(5));

    let results = engine.search(query, options)?;

    println!("Searching for 'Smart' in title field:");
    println!("Total hits: {}", results.total_hits);
    println!("Results returned: {}", results.hits.len());
    println!("Max score: {:.4}", results.max_score);

    for (i, hit) in results.hits.iter().enumerate() {
        if let Some(doc) = &hit.document {
            if let Some(title) = doc.get_field("title").and_then(|f| f.as_text()) {
                if let Some(id) = doc.get_field("id").and_then(|f| f.as_text()) {
                    println!("  {}. {} [{}] (score: {:.4})", i + 1, title, id, hit.score);
                } else {
                    println!("  {}. {} (score: {:.4})", i + 1, title, hit.score);
                }
            }
        }
    }

    // Debug: Show which Smart products should be found
    println!("  Expected products with 'Smart' in title:");
    println!("    - E005: Smart Watch");
    println!("    - H001: Smart Home Hub");
    println!("    - H003: Smart Thermostat");
    println!("    - H005: Smart Security Camera");

    println!("\n=== Example 2: Phrase Search with Different Merge Strategy ===\n");

    // Search for phrase "Programming Language"
    let query = Box::new(PhraseQuery::new(
        "title",
        vec!["Programming".to_string(), "Language".to_string()],
    ));
    let options = SearchOptions::new(5)
        .with_merge_strategy(MergeStrategyType::Weighted)
        .with_metrics(true);

    let results = engine.search(query, options)?;

    println!("Searching for phrase 'Programming Language':");
    println!("Using weighted merge strategy (Books index has 1.5x weight)");
    println!("Total hits: {}", results.total_hits);

    for (i, hit) in results.hits.iter().enumerate() {
        if let Some(doc) = &hit.document {
            if let Some(title) = doc.get_field("title").and_then(|f| f.as_text()) {
                if let Some(price) = doc.get_field("price").and_then(|f| match f {
                    sarissa::schema::FieldValue::Float(v) => Some(*v),
                    _ => None,
                }) {
                    println!(
                        "  {}. {} - ${:.2} (score: {:.4})",
                        i + 1,
                        title,
                        price,
                        hit.score
                    );
                }
            }
        }
    }

    println!("\n=== Example 3: Search with Minimum Score Filter ===\n");

    // Search for "Advanced" with minimum score threshold
    let query = Box::new(TermQuery::new("description", "Advanced"));
    let options = SearchOptions::new(10)
        .with_min_score(0.5)
        .with_metrics(true);

    let results = engine.search(query, options)?;

    println!("Searching for 'Advanced' in descriptions with min score 0.5:");
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

    println!("\n=== Example 4: Performance Metrics ===\n");

    let metrics = engine.metrics();
    println!("Search Performance Metrics:");
    println!("  Total searches: {}", metrics.total_searches);
    println!("  Successful searches: {}", metrics.successful_searches);
    println!("  Failed searches: {}", metrics.failed_searches);
    println!(
        "  Average execution time: {:.2}μs",
        metrics.avg_execution_time.as_micros()
    );
    println!("  Total hits returned: {}", metrics.total_hits_returned);

    println!("\n=== Example 5: Concurrent Search Demonstration ===\n");

    // Reset metrics for clean measurement
    engine.reset_metrics();

    // Execute multiple searches to demonstrate concurrency
    let search_terms = vec!["Pro", "Smart", "Advanced", "quality", "Premium"];
    let mut total_time = std::time::Duration::new(0, 0);

    println!(
        "Executing {} parallel searches across {} indices...",
        search_terms.len(),
        engine.index_count()?
    );

    let search_fields = vec![
        ("title", "Pro"),
        ("title", "Smart"),
        ("description", "Advanced"),
        ("description", "quality"),
        ("description", "Premium"),
    ];

    for (field, term) in &search_fields {
        let start = std::time::Instant::now();
        let query = Box::new(TermQuery::new(*field, *term));
        let options = SearchOptions::new(10).with_metrics(true);

        let results = engine.search(query, options)?;
        let elapsed = start.elapsed();
        total_time += elapsed;

        println!(
            "  Search '{}' in '{}': {} hits ({} returned) in {:.2}μs",
            term,
            field,
            results.total_hits,
            results.hits.len(),
            elapsed.as_micros()
        );
    }

    println!("\nTotal search time: {:.2}μs", total_time.as_micros());
    println!(
        "Average time per search: {:.2}μs",
        total_time.as_micros() / search_terms.len() as u128
    );
    println!(
        "Note: Searches are executed concurrently across {} indices",
        engine.index_count()?
    );

    println!("\n=== Example 6: Cross-Index Weighted Search ===\n");

    // Demonstrate how weighting affects ranking when items appear in multiple indices
    let query = Box::new(TermQuery::new("description", "Professional"));
    let options = SearchOptions::new(10)
        .with_merge_strategy(MergeStrategyType::Weighted)
        .with_metrics(true);

    let results = engine.search(query, options)?;

    println!("Searching for 'Professional' in description field:");
    println!(
        "Results are weighted by index importance (Books: 1.5x, Electronics: 1.0x, Home: 0.8x)"
    );
    println!("Total hits: {}", results.total_hits);

    for (i, hit) in results.hits.iter().enumerate() {
        if let Some(doc) = &hit.document {
            if let Some(title) = doc.get_field("title").and_then(|f| f.as_text()) {
                if let Some(category) = doc.get_field("category").and_then(|f| f.as_text()) {
                    if let Some(id) = doc.get_field("id").and_then(|f| f.as_text()) {
                        println!(
                            "  {}. {} [{}:{}] (weighted score: {:.4})",
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

    // Debug: Show which products should contain 'Professional'
    println!("  Expected products with 'Professional' in description:");
    println!("    - E004: Tablet Pro - 'Professional tablet for creative work'");
    println!("    - H004: Garden Tool Set - 'Professional quality gardening tools'");

    println!("\n=== Summary ===\n");

    let final_metrics = engine.metrics();
    println!("Parallel Search Engine Summary:");
    println!("  • Indices managed: {}", engine.index_count()?);
    println!(
        "  • Total searches executed: {}",
        final_metrics.total_searches
    );
    println!(
        "  • Success rate: {:.1}%",
        (final_metrics.successful_searches as f64 / final_metrics.total_searches as f64) * 100.0
    );
    println!(
        "  • Total hits returned: {}",
        final_metrics.total_hits_returned
    );
    println!(
        "  • Average latency: {:.2}μs",
        final_metrics.avg_execution_time.as_micros()
    );

    println!("\n=== Key Takeaways ===\n");
    println!("1. Parallel search enables concurrent queries across multiple indices");
    println!(
        "2. Different merge strategies (score-based, weighted, round-robin) optimize for different use cases"
    );
    println!(
        "3. Index weights allow prioritizing certain data sources (e.g., Books index has 1.5x weight)"
    );
    println!("4. Performance metrics help monitor search quality and latency");
    println!("5. The engine gracefully handles timeouts and partial failures");

    println!("\nParallel Search example completed successfully!");

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
