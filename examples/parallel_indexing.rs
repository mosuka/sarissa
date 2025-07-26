//! Parallel Indexing example - demonstrates distributed document indexing across multiple indices.

use sarissa::index::writer::{BasicIndexWriter, WriterConfig};
use sarissa::parallel_index::{
    DocumentPartitioner, HashPartitioner, ParallelIndexConfig, ParallelIndexEngine,
    PartitionConfig, ValuePartitioner, config::IndexingOptions,
};
use sarissa::prelude::*;
use sarissa::schema::{IdField, NumericField, TextField};
use sarissa::storage::{MemoryStorage, StorageConfig};
use std::path::Path;
use std::sync::Arc;
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

    let mut parallel_engine = ParallelIndexEngine::new(parallel_config)?;

    // Set up hash partitioner
    let hash_partitioner = HashPartitioner::new("user_id".to_string(), 4);
    parallel_engine.set_partitioner(Box::new(hash_partitioner.clone()))?;

    // Add writers for each partition
    for i in 0..4 {
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let writer = Box::new(BasicIndexWriter::new(
            schema.clone(),
            storage,
            WriterConfig::default(),
        )?);
        let partition_config = PartitionConfig::new(format!("partition_{}", i));
        parallel_engine.add_partition(format!("partition_{}", i), writer, partition_config)?;
    }

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

    // Index documents using the parallel engine
    let indexing_result =
        parallel_engine.index_documents(documents.clone(), IndexingOptions::default())?;

    // Show partitioning results
    println!("Hash Partitioning Results:");
    println!("  Total documents: {}", indexing_result.total_documents);
    println!("  Documents indexed: {}", indexing_result.documents_indexed);
    println!("  Documents failed: {}", indexing_result.documents_failed);
    println!(
        "  Execution time: {:.2}ms",
        indexing_result.execution_time.as_millis()
    );

    let metrics = parallel_engine.metrics();
    println!("  Engine metrics:");
    println!("    - Total operations: {}", metrics.total_operations);
    println!(
        "    - Successful operations: {}",
        metrics.successful_operations
    );

    // Demonstrate hash partitioner directly
    println!("\n2. Hash Partition Distribution:");
    let hash_partitioner = HashPartitioner::new("user_id".to_string(), 4);

    for doc in &documents {
        if let Some(field_value) = doc.get_field("user_id") {
            if let sarissa::schema::FieldValue::Integer(user_id) = field_value {
                let partition = hash_partitioner.partition(doc)?;
                println!("  User ID {}: -> Partition {}", user_id, partition);
            }
        }
    }

    println!("\n=== Value-Based Partitioning Example ===\n");

    // Example 2: Value-based partitioning by region
    println!("3. Value-based partitioning by region:");

    // Set up value-based mapping for regions
    let value_partitioner = ValuePartitioner::new("region".to_string(), 3)
        .add_mapping("north-america".to_string(), 0)?
        .add_mapping("europe".to_string(), 1)?
        .add_mapping("asia-pacific".to_string(), 2)?
        .with_default_partition(2)?; // Default to partition 2

    for doc in &documents {
        if let Some(field_value) = doc.get_field("region") {
            if let Some(region) = field_value.as_text() {
                let partition = value_partitioner.partition(doc)?;
                println!("  Region '{}': -> Partition {}", region, partition);
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

    let mut parallel_engine_region = ParallelIndexEngine::new(parallel_config_region)?;

    // Set up value partitioner for region-based processing
    parallel_engine_region.set_partitioner(Box::new(value_partitioner.clone()))?;

    // Add writers for region partitions
    for i in 0..3 {
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let writer = Box::new(BasicIndexWriter::new(
            schema.clone(),
            storage,
            WriterConfig::default(),
        )?);
        let partition_config = PartitionConfig::new(format!("region_partition_{}", i));
        parallel_engine_region.add_partition(
            format!("region_partition_{}", i),
            writer,
            partition_config,
        )?;
    }

    println!("\n4. Indexing with region-based partitioning...");
    let region_indexing_result =
        parallel_engine_region.index_documents(documents.clone(), IndexingOptions::default())?;

    println!("Region Partitioning Results:");
    println!(
        "  Total documents: {}",
        region_indexing_result.total_documents
    );
    println!(
        "  Documents indexed: {}",
        region_indexing_result.documents_indexed
    );
    println!(
        "  Documents failed: {}",
        region_indexing_result.documents_failed
    );
    println!(
        "  Execution time: {:.2}ms",
        region_indexing_result.execution_time.as_millis()
    );

    println!("\n=== Batch Processing Configuration ===\n");

    // Example 3: Different batch sizes and concurrency
    println!("5. Batch processing configuration examples:");

    println!("  Current configuration:");
    println!(
        "    - Batch size: {} documents",
        parallel_engine.config().default_batch_size
    );
    println!(
        "    - Max concurrent partitions: {}",
        parallel_engine.config().max_concurrent_partitions
    );
    println!("    - Total documents: {}", documents.len());
    println!(
        "    - Expected batches: {}",
        (documents.len() + parallel_engine.config().default_batch_size - 1)
            / parallel_engine.config().default_batch_size
    );

    // Show performance metrics
    println!("\n6. Performance metrics:");
    println!("  Hash partitioning:");
    println!(
        "    - Documents indexed: {}",
        indexing_result.documents_indexed
    );
    println!(
        "    - Execution time: {:.2}ms",
        indexing_result.execution_time.as_millis()
    );
    println!(
        "    - Throughput: {:.1} docs/sec",
        indexing_result.documents_indexed as f64 / indexing_result.execution_time.as_secs_f64()
    );

    println!("  Region partitioning:");
    println!(
        "    - Documents indexed: {}",
        region_indexing_result.documents_indexed
    );
    println!(
        "    - Execution time: {:.2}ms",
        region_indexing_result.execution_time.as_millis()
    );
    println!(
        "    - Throughput: {:.1} docs/sec",
        region_indexing_result.documents_indexed as f64
            / region_indexing_result.execution_time.as_secs_f64()
    );

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

    println!("\n=== Index Verification ===\n");

    // Example 8: Verify partition states and document distribution
    println!("8. Verifying partition states and document distribution...");

    // Check hash partitioning results
    println!("\n  Hash Partitioning Verification:");
    let hash_partition_stats = parallel_engine.partition_statistics()?;
    for (partition_id, stats) in hash_partition_stats {
        println!(
            "    Partition {}: {} documents indexed, {} operations",
            partition_id, stats.documents_indexed, stats.successful_operations
        );
        if stats.failed_operations > 0 {
            println!("      ⚠️  {} failed operations", stats.failed_operations);
        }
    }

    // Check region partitioning results
    println!("\n  Region Partitioning Verification:");
    let region_partition_stats = parallel_engine_region.partition_statistics()?;
    for (partition_id, stats) in region_partition_stats {
        println!(
            "    Partition {}: {} documents indexed, {} operations",
            partition_id, stats.documents_indexed, stats.successful_operations
        );
        if stats.failed_operations > 0 {
            println!("      ⚠️  {} failed operations", stats.failed_operations);
        }
    }

    // Aggregated statistics
    println!("\n  Aggregated Statistics:");
    let hash_aggregated = parallel_engine.aggregated_statistics()?;
    let region_aggregated = parallel_engine_region.aggregated_statistics()?;

    println!(
        "    Hash partitioning total: {} documents across {} commits",
        hash_aggregated.documents_indexed, hash_aggregated.commit_count
    );
    println!(
        "    Region partitioning total: {} documents across {} commits",
        region_aggregated.documents_indexed, region_aggregated.commit_count
    );

    println!("\n=== Commit and Optimization ===\n");

    // Example 9: Commit all indices
    println!("9. Committing changes across all indices...");
    let failed_commits = parallel_engine.commit_all()?;
    let failed_commits_region = parallel_engine_region.commit_all()?;

    if failed_commits.is_empty() && failed_commits_region.is_empty() {
        println!("All commits successful!");

        // Final verification after commit
        println!("\n  Post-commit verification:");
        let final_hash_stats = parallel_engine.aggregated_statistics()?;
        let final_region_stats = parallel_engine_region.aggregated_statistics()?;

        println!(
            "    Hash partitioning: {} commits completed",
            final_hash_stats.commit_count
        );
        println!(
            "    Region partitioning: {} commits completed",
            final_region_stats.commit_count
        );
    } else {
        println!("Some commits failed: {:?}", failed_commits);
    }

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
