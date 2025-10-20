//! Example usage of the parallel search module.

use std::sync::Arc;

use crate::error::Result;
use crate::full_text::search::advanced_reader::{AdvancedIndexReader, AdvancedReaderConfig};
use crate::parallel_full_text_search::config::{ParallelSearchConfig, SearchOptions};
use crate::parallel_full_text_search::engine::ParallelSearchEngine;
use crate::query::term::TermQuery;
use crate::storage::memory::MemoryStorage;
use crate::storage::traits::StorageConfig;

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

    // Add multiple indices to the engine
    for i in 0..3 {
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let reader = Box::new(AdvancedIndexReader::new(
            vec![],
            storage,
            AdvancedReaderConfig::default(),
        )?);
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
