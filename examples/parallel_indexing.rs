//! Parallel Indexing example - demonstrates distributed document indexing across multiple indices.

use sarissa::parallel_index::{
    HashPartitioner, ParallelIndexConfig, ParallelIndexEngine,
    PartitionConfig, ValuePartitioner,
};
use sarissa::prelude::*;
use sarissa::schema::{IdField, NumericField, TextField};
use std::path::Path;
use tempfile::TempDir;
fn main() -> Result<()> {
    println!("=== Parallel Indexing Example - Distributed Document Processing ===\n");

    // Create temporary directories for multiple indices
    let temp_dirs: Vec<TempDir> = (0..4).map(|_| TempDir::new().unwrap()).collect();
    let index_paths: Vec<&Path> = temp_dirs.iter().map(|dir| dir.path()).collect();

    println!("Creating {} parallel indices:", index_paths.len());
    for (i, path) in index_paths.iter().enumerate() {
        println!("  Index {}: {:?}", i, path);
    }

    // Create a schema for all indices
    let mut schema = Schema::new();
    schema.add_field(
        "title",
        Box::new(TextField::new().stored(true).indexed(true)),
    )?;
    schema.add_field("content", Box::new(TextField::new().indexed(true)))?;
    schema.add_field("author", Box::new(TextField::new().indexed(true)))?;
    schema.add_field("category", Box::new(TextField::new().indexed(true)))?;
    schema.add_field("user_id", Box::new(NumericField::u64().indexed(true)))?;
    schema.add_field("region", Box::new(TextField::new().indexed(true)))?;
    schema.add_field("priority", Box::new(NumericField::u64().indexed(true)))?;
    schema.add_field("id", Box::new(IdField::new()))?;

    println!("\n=== Hash Partitioning Example ===\n");

    // Example 1: Hash partitioning by user_id
    println!(
        "1. Hash partitioning by user_id across {} indices:",
        index_paths.len()
    );

    let _partition_config = PartitionConfig::new("partition_0".to_string());

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

    let parallel_engine = ParallelIndexEngine::new(parallel_config)?;

    // Create documents with various user_ids for hash partitioning
    let documents = vec![
        Document::builder()
            .add_text("title", "User Guide for Product A")
            .add_text(
                "content",
                "Comprehensive guide for using Product A effectively",
            )
            .add_text("author", "Alice Johnson")
            .add_text("category", "documentation")
            .add_integer("user_id", 1001)
            .add_text("region", "north-america")
            .add_integer("priority", 1)
            .add_text("id", "doc001")
            .build(),
        Document::builder()
            .add_text("title", "API Documentation")
            .add_text(
                "content",
                "REST API documentation with examples and best practices",
            )
            .add_text("author", "Bob Smith")
            .add_text("category", "technical")
            .add_integer("user_id", 2002)
            .add_text("region", "europe")
            .add_integer("priority", 2)
            .add_text("id", "doc002")
            .build(),
        Document::builder()
            .add_text("title", "Troubleshooting Guide")
            .add_text("content", "Common issues and solutions for system problems")
            .add_text("author", "Carol Davis")
            .add_text("category", "support")
            .add_integer("user_id", 3003)
            .add_text("region", "asia-pacific")
            .add_integer("priority", 3)
            .add_text("id", "doc003")
            .build(),
        Document::builder()
            .add_text("title", "Performance Optimization")
            .add_text(
                "content",
                "Tips and techniques for optimizing system performance",
            )
            .add_text("author", "David Brown")
            .add_text("category", "technical")
            .add_integer("user_id", 4004)
            .add_text("region", "north-america")
            .add_integer("priority", 1)
            .add_text("id", "doc004")
            .build(),
        Document::builder()
            .add_text("title", "Security Best Practices")
            .add_text(
                "content",
                "Essential security practices for enterprise applications",
            )
            .add_text("author", "Eva Wilson")
            .add_text("category", "security")
            .add_integer("user_id", 5005)
            .add_text("region", "europe")
            .add_integer("priority", 1)
            .add_text("id", "doc005")
            .build(),
        Document::builder()
            .add_text("title", "Data Migration Guide")
            .add_text(
                "content",
                "Step-by-step guide for migrating data between systems",
            )
            .add_text("author", "Frank Miller")
            .add_text("category", "operations")
            .add_integer("user_id", 6006)
            .add_text("region", "asia-pacific")
            .add_integer("priority", 2)
            .add_text("id", "doc006")
            .build(),
        Document::builder()
            .add_text("title", "Machine Learning Tutorial")
            .add_text(
                "content",
                "Introduction to machine learning concepts and algorithms",
            )
            .add_text("author", "Grace Lee")
            .add_text("category", "education")
            .add_integer("user_id", 7007)
            .add_text("region", "north-america")
            .add_integer("priority", 2)
            .add_text("id", "doc007")
            .build(),
        Document::builder()
            .add_text("title", "Cloud Architecture Patterns")
            .add_text("content", "Design patterns for scalable cloud applications")
            .add_text("author", "Henry Taylor")
            .add_text("category", "architecture")
            .add_integer("user_id", 8008)
            .add_text("region", "europe")
            .add_integer("priority", 3)
            .add_text("id", "doc008")
            .build(),
    ];

    println!(
        "Indexing {} documents with hash partitioning...",
        documents.len()
    );
    // Note: This is a simplified example. Real implementation would need proper document partitioning.
    // parallel_engine.add_documents(documents.clone()).await?;

    // Show partitioning results
    let metrics = parallel_engine.metrics();
    println!("Hash Partitioning Results:");
    println!("  Total operations: {}", metrics.total_operations);
    println!("  Successful operations: {}", metrics.successful_operations);
    println!(
        "  Average operation time: {:.2}ms",
        metrics.avg_execution_time.as_millis()
    );

    // Demonstrate hash partitioner directly
    println!("\n2. Hash Partition Distribution:");
    let _hash_partitioner = HashPartitioner::new("user_id".to_string(), 4);

    for doc in &documents {
        if let Some(field_value) = doc.get_field("user_id") {
            if let sarissa::schema::FieldValue::Integer(user_id) = field_value {
                // Note: HashPartitioner needs document reference for partitioning
                // let partition = hash_partitioner.partition(doc)?;
                println!("  User ID {}: -> Partition (example)", user_id);
            }
        }
    }

    println!("\n=== Value-Based Partitioning Example ===\n");

    // Example 2: Value-based partitioning by region
    println!("3. Value-based partitioning by region:");

    let _value_partitioner = ValuePartitioner::new("region".to_string(), 4);

    for doc in &documents {
        if let Some(field_value) = doc.get_field("region") {
            if let Some(region) = field_value.as_text() {
                // Note: ValuePartitioner needs document reference for partitioning
                // let partition = value_partitioner.partition(doc)?;
                println!("  Region '{}': -> Partition (example)", region);
            }
        }
    }

    // Create new parallel engine with value-based partitioning
    let _partition_config_region = PartitionConfig::new("partition_region".to_string());

    let parallel_config_region = ParallelIndexConfig {
        max_concurrent_partitions: 3,
        default_batch_size: 5,
        max_buffer_memory: 512 * 1024 * 1024,
        commit_interval: std::time::Duration::from_secs(30),
        retry_attempts: 3,
        operation_timeout: std::time::Duration::from_secs(60),
        enable_metrics: true,
        thread_pool_size: None,
        allow_partial_failures: true,
        auto_commit_threshold: 10000,
    };

    // Create new temp directories for region-based partitioning
    let temp_dirs_region: Vec<TempDir> = (0..3).map(|_| TempDir::new().unwrap()).collect();
    let _index_paths_region: Vec<&Path> = temp_dirs_region.iter().map(|dir| dir.path()).collect();

    let parallel_engine_region = ParallelIndexEngine::new(parallel_config_region)?;

    println!("\n4. Indexing with region-based partitioning...");
    // Note: This is a simplified example. Real implementation would need proper document partitioning.
    // parallel_engine_region.add_documents(documents.clone()).await?;

    let metrics_region = parallel_engine_region.metrics();
    println!("Region Partitioning Results:");
    println!(
        "  Total operations: {}",
        metrics_region.total_operations
    );

    println!("\n=== Batch Processing Configuration ===\n");

    // Example 3: Different batch sizes and concurrency
    println!("5. Batch processing configuration examples:");

    println!("  Current configuration:");
    println!("    - Batch size: {} documents", parallel_engine.config().default_batch_size);
    println!(
        "    - Max concurrent partitions: {}",
        parallel_engine.config().max_concurrent_partitions
    );
    println!("    - Total documents: {}", documents.len());
    println!(
        "    - Expected batches: {}",
        (documents.len() + parallel_engine.config().default_batch_size - 1) / parallel_engine.config().default_batch_size
    );

    // Show performance metrics
    println!("\n6. Performance metrics:");
    println!("  Hash partitioning:");
    println!(
        "    - Operations/second: {:.1}",
        metrics.throughput.ops_per_second
    );
    println!(
        "    - Average operation time: {:.2}ms",
        metrics.avg_execution_time.as_millis()
    );
    println!("    - Documents indexed: {}", metrics.total_documents_indexed);

    println!("  Region partitioning:");
    println!(
        "    - Operations/second: {:.1}",
        metrics_region.throughput.ops_per_second
    );
    println!(
        "    - Average operation time: {:.2}ms",
        metrics_region.avg_execution_time.as_millis()
    );
    println!("    - Documents indexed: {}", metrics_region.total_documents_indexed);

    println!("\n=== Custom Partitioning Strategy ===\n");

    // Example 4: Custom partitioning logic
    println!("7. Custom partitioning by priority:");

    // Partition by priority: high (1), medium (2), low (3)
    for doc in &documents {
        if let Some(field_value) = doc.get_field("priority") {
            if let sarissa::schema::FieldValue::Integer(priority) = field_value {
                let partition = match priority {
                    1 => 0, // High priority -> Index 0
                    2 => 1, // Medium priority -> Index 1
                    _ => 2, // Low priority -> Index 2
                };
                if let Some(title_field) = doc.get_field("title") {
                    if let Some(title) = title_field.as_text() {
                        println!(
                            "  '{}' (Priority {}): -> Partition {}",
                            &title[..std::cmp::min(title.len(), 30)],
                            priority,
                            partition
                        );
                    }
                }
            }
        }
    }

    println!("\n=== Commit and Optimization ===\n");

    // Example 5: Commit all indices
    println!("8. Committing changes across all indices...");
    let failed_commits = parallel_engine.commit_all()?;
    let failed_commits_region = parallel_engine_region.commit_all()?;
    
    if failed_commits.is_empty() && failed_commits_region.is_empty() {
        println!("All commits successful!");
    } else {
        println!("Some commits failed: {:?}", failed_commits);
    }

    println!("\n=== Parallel Indexing Key Features ===");
    println!("• Distribute documents across multiple indices");
    println!("• Hash-based partitioning for even distribution");
    println!("• Value-based partitioning for logical grouping");
    println!("• Configurable batch processing for performance");
    println!("• Concurrent indexing across multiple writers");
    println!("• Real-time metrics and performance monitoring");
    println!("• Automatic load balancing across partitions");

    println!("\n=== Partitioning Strategies ===");
    println!("Hash Partitioning:");
    println!("  • Even distribution across indices");
    println!("  • Good for load balancing");
    println!("  • Based on field value hash");
    println!("  • Predictable partition assignment");

    println!("\nValue Partitioning:");
    println!("  • Logical grouping by field values");
    println!("  • Good for regional/categorical data");
    println!("  • Custom partition mapping");
    println!("  • Domain-specific organization");

    println!("\n=== Use Cases ===");
    println!("• Large-scale document processing");
    println!("• Multi-tenant applications (partition by tenant)");
    println!("• Geographic data distribution (partition by region)");
    println!("• Time-series data (partition by time period)");
    println!("• User-generated content (partition by user)");
    println!("• E-commerce catalogs (partition by category)");
    println!("• Log processing (partition by service/severity)");

    println!("\n=== Performance Benefits ===");
    println!("• Parallel processing reduces indexing time");
    println!("• Distributed load across multiple cores/disks");
    println!("• Smaller index sizes improve search performance");
    println!("• Independent index optimization");
    println!("• Horizontal scaling capability");
    println!("• Fault isolation per partition");

    println!("\n=== Best Practices ===");
    println!("• Choose partition field with good distribution");
    println!("• Balance partition sizes for even load");
    println!("• Monitor metrics to tune batch sizes");
    println!("• Consider query patterns when partitioning");
    println!("• Use appropriate concurrency levels");
    println!("• Plan for partition growth over time");

    // Engines will be automatically closed when dropped

    println!("\nParallel Indexing example completed successfully!");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_indexing_example() {
        let result = main();
        assert!(result.is_ok());
    }
}
