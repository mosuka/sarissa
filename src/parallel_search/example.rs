//! Example usage of the parallel search module.

use crate::error::Result;
use crate::index::reader::BasicIndexReader;
use crate::parallel_search::{
    config::{ParallelSearchConfig, SearchOptions},
    engine::ParallelSearchEngine,
};
use crate::query::TermQuery;
use crate::schema::{Schema, TextField};
use crate::storage::{MemoryStorage, StorageConfig};
use std::sync::Arc;

/// Example function demonstrating parallel search usage.
pub fn example_parallel_search() -> Result<()> {
    // Create a configuration for parallel search
    let config = ParallelSearchConfig {
        max_concurrent_tasks: 4,
        ..Default::default()
    };

    // Create the parallel search engine
    let engine = ParallelSearchEngine::new(config)?;

    // Create some test indices
    let mut schema = Schema::new();
    schema.add_field("title", Box::new(TextField::new()))?;
    schema.add_field("content", Box::new(TextField::new()))?;

    // Add multiple indices to the engine
    for i in 0..3 {
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let reader = Box::new(BasicIndexReader::new(schema.clone(), storage)?);
        engine.add_index(format!("index_{i}"), reader, 1.0)?;
    }

    // Create a query
    let query = Box::new(TermQuery::new("title", "rust"));

    // Configure search options
    let options = SearchOptions::new(10)
        .with_timeout(std::time::Duration::from_secs(5))
        .with_min_score(0.1);

    // Execute the parallel search
    let results = engine.search(query, options)?;

    println!(
        "Found {} hits across {} indices",
        results.hits.len(),
        engine.index_count()?
    );

    // Display metrics
    let metrics = engine.metrics();
    println!("Total searches: {}", metrics.total_searches);
    println!("Average execution time: {:?}", metrics.avg_execution_time);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_search_example() {
        // This test just ensures the example code compiles and runs
        let result = example_parallel_search();
        // We expect it to work even with empty indices
        assert!(result.is_ok());
    }
}
