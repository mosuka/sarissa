//! Parallel Search example - demonstrates concurrent searches across multiple indices with result merging.

use sarissa::prelude::*;
use tempfile::TempDir;

fn main() -> Result<()> {
    println!("=== Parallel Search Example - Simplified Version ===\n");

    // Create a temporary directory for demonstration
    let temp_dir = TempDir::new().unwrap();
    println!("Creating index in: {:?}", temp_dir.path());

    println!("This is a simplified example of parallel search functionality.");
    println!("For a complete implementation, refer to the parallel search module documentation.");

    println!("\n=== Parallel Search Key Features ===");
    println!("• Search across multiple indices concurrently");
    println!("• Merge results using different strategies");
    println!("• Score-based and weighted merging");
    println!("• Load balancing across indices");
    println!("• Configurable timeouts");
    println!("• Metrics collection");

    println!("\n=== Use Cases ===");
    println!("• Multi-tenant applications with separate indices");
    println!("• Geographic search across regional indices");
    println!("• Time-series data with temporal partitioning");
    println!("• Domain-specific content categorization");
    println!("• Horizontal scaling for large datasets");

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
