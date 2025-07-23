//! Example usage of the parallel indexing module.

use crate::error::Result;
use crate::index::writer::BasicIndexWriter;
use crate::parallel_index::{
    config::{IndexingOptions, ParallelIndexConfig, PartitionConfig},
    engine::ParallelIndexEngine,
    partitioner::HashPartitioner,
};
use crate::schema::{Document, FieldValue, Schema, TextField};
use crate::storage::{MemoryStorage, StorageConfig};
use std::sync::Arc;

/// Example function demonstrating parallel indexing usage.
pub fn example_parallel_indexing() -> Result<()> {
    // Create configuration for parallel indexing
    let mut config = ParallelIndexConfig::default();
    config.max_concurrent_partitions = 3;
    config.default_batch_size = 100;
    config.enable_metrics = true;
    
    // Create the parallel indexing engine
    let mut engine = ParallelIndexEngine::new(config)?;
    
    // Create schema for documents
    let mut schema = Schema::new();
    schema.add_field("id", Box::new(TextField::new()))?;
    schema.add_field("category", Box::new(TextField::new()))?;
    schema.add_field("title", Box::new(TextField::new()))?;
    schema.add_field("content", Box::new(TextField::new()))?;
    
    // Add partitions to the engine
    for i in 0..3 {
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let writer = Box::new(BasicIndexWriter::new(
            schema.clone(), 
            storage, 
            crate::index::writer::WriterConfig::default()
        )?);
        let partition_config = PartitionConfig::new(format!("partition_{}", i))
            .with_weight(1.0)
            .with_batch_size(50);
        
        engine.add_partition(format!("partition_{}", i), writer, partition_config)?;
    }
    
    // Set up hash-based partitioning by category
    let partitioner = Box::new(HashPartitioner::new("category".to_string(), 3));
    engine.set_partitioner(partitioner)?;
    
    // Create sample documents
    let documents = vec![
        create_sample_document("1", "electronics", "Laptop", "High-performance laptop"),
        create_sample_document("2", "books", "Rust Programming", "Learn Rust programming"),
        create_sample_document("3", "electronics", "Smartphone", "Latest smartphone model"),
        create_sample_document("4", "books", "Database Design", "Database design principles"),
        create_sample_document("5", "clothing", "T-Shirt", "Cotton t-shirt"),
        create_sample_document("6", "electronics", "Headphones", "Noise-canceling headphones"),
        create_sample_document("7", "books", "Web Development", "Modern web development"),
        create_sample_document("8", "clothing", "Jeans", "Blue denim jeans"),
    ];
    
    println!("Indexing {} documents across {} partitions...", 
             documents.len(), 
             engine.partition_count()?);
    
    // Configure indexing options
    let options = IndexingOptions::new(50)
        .with_force_commit(true)
        .with_timeout(std::time::Duration::from_secs(30))
        .with_metrics(true)
        .with_validation(true);
    
    // Execute parallel indexing
    let result = engine.index_documents(documents, options)?;
    
    // Display results
    println!("Indexing completed:");
    println!("  Total documents: {}", result.total_documents);
    println!("  Successfully indexed: {}", result.documents_indexed);
    println!("  Failed: {}", result.documents_failed);
    println!("  Execution time: {:?}", result.execution_time);
    println!("  Processed {} partitions", result.partition_results.len());
    
    // Show per-partition results
    for partition_result in result.partition_results {
        println!("  Partition {}: {} docs indexed in {:?}",
                 partition_result.partition_index,
                 partition_result.documents_indexed,
                 partition_result.processing_time);
    }
    
    // Display metrics
    let metrics = engine.metrics();
    println!("\nIndexing Metrics:");
    println!("  Total operations: {}", metrics.total_operations);
    println!("  Success rate: {:.1}%", 
             metrics.successful_operations as f64 / metrics.total_operations as f64 * 100.0);
    println!("  Average time per operation: {:?}", metrics.avg_execution_time);
    println!("  Throughput: {:.1} docs/sec", metrics.throughput.avg_docs_per_second);
    
    // Commit all changes
    let failed_commits = engine.commit_all()?;
    if failed_commits.is_empty() {
        println!("All partitions committed successfully");
    } else {
        println!("Failed to commit partitions: {:?}", failed_commits);
    }
    
    Ok(())
}

/// Create a sample document with the given fields.
fn create_sample_document(id: &str, category: &str, title: &str, content: &str) -> Document {
    let mut doc = Document::new();
    doc.add_field("id".to_string(), FieldValue::Text(id.to_string()));
    doc.add_field("category".to_string(), FieldValue::Text(category.to_string()));
    doc.add_field("title".to_string(), FieldValue::Text(title.to_string()));
    doc.add_field("content".to_string(), FieldValue::Text(content.to_string()));
    doc
}

/// Example of range-based partitioning by price.
pub fn example_range_partitioning() -> Result<()> {
    use crate::parallel_index::partitioner::RangePartitioner;
    
    let mut config = ParallelIndexConfig::default();
    config.max_concurrent_partitions = 4;
    
    let mut engine = ParallelIndexEngine::new(config)?;
    
    // Create schema with price field
    let mut schema = Schema::new();
    schema.add_field("id", Box::new(TextField::new()))?;
    schema.add_field("price", Box::new(TextField::new()))?; // Store as text for simplicity
    schema.add_field("product", Box::new(TextField::new()))?;
    
    // Add price-range partitions
    let partition_names = ["low_price", "mid_price", "high_price", "premium"];
    for (i, name) in partition_names.iter().enumerate() {
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let writer = Box::new(BasicIndexWriter::new(
            schema.clone(), 
            storage, 
            crate::index::writer::WriterConfig::default()
        )?);
        let partition_config = PartitionConfig::new(format!("partition_{}", i))
            .with_metadata("range".to_string(), name.to_string());
        
        engine.add_partition(format!("partition_{}", i), writer, partition_config)?;
    }
    
    // Set up range partitioning by price: <100, 100-500, 500-1000, >1000
    let partitioner = Box::new(RangePartitioner::new(
        "price".to_string(),
        vec![100, 500, 1000]
    )?);
    engine.set_partitioner(partitioner)?;
    
    // Create sample products with different prices
    let documents = vec![
        create_price_document("1", "50", "Book"),
        create_price_document("2", "250", "Headphones"),
        create_price_document("3", "750", "Laptop"),
        create_price_document("4", "1500", "Gaming PC"),
        create_price_document("5", "25", "Magazine"),
        create_price_document("6", "800", "Smartphone"),
    ];
    
    let options = IndexingOptions::new(10).with_force_commit(true);
    let result = engine.index_documents(documents, options)?;
    
    println!("Range partitioning results:");
    println!("  Documents indexed: {}", result.documents_indexed);
    println!("  Partitions used: {}", result.partition_results.len());
    
    Ok(())
}

/// Create a document with price field.
fn create_price_document(id: &str, price: &str, product: &str) -> Document {
    let mut doc = Document::new();
    doc.add_field("id".to_string(), FieldValue::Text(id.to_string()));
    doc.add_field("price".to_string(), FieldValue::Text(price.to_string()));
    doc.add_field("product".to_string(), FieldValue::Text(product.to_string()));
    doc
}

/// Example of value-based partitioning by region.
pub fn example_value_partitioning() -> Result<()> {
    use crate::parallel_index::partitioner::ValuePartitioner;
    use std::collections::HashMap;
    
    let config = ParallelIndexConfig::default();
    let mut engine = ParallelIndexEngine::new(config)?;
    
    // Create schema
    let mut schema = Schema::new();
    schema.add_field("id", Box::new(TextField::new()))?;
    schema.add_field("region", Box::new(TextField::new()))?;
    schema.add_field("data", Box::new(TextField::new()))?;
    
    // Add region-specific partitions
    let regions = ["US", "EU", "Asia"];
    for (i, region) in regions.iter().enumerate() {
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let writer = Box::new(BasicIndexWriter::new(
            schema.clone(), 
            storage, 
            crate::index::writer::WriterConfig::default()
        )?);
        let partition_config = PartitionConfig::new(format!("partition_{}", i))
            .with_metadata("region".to_string(), region.to_string());
        
        engine.add_partition(format!("partition_{}", i), writer, partition_config)?;
    }
    
    // Set up value-based partitioning by region
    let mut mapping = HashMap::new();
    mapping.insert("US".to_string(), 0);
    mapping.insert("EU".to_string(), 1);
    mapping.insert("Asia".to_string(), 2);
    
    let partitioner = ValuePartitioner::from_mapping(
        "region".to_string(),
        mapping,
        Some(0) // Default to US partition
    )?;
    engine.set_partitioner(Box::new(partitioner))?;
    
    // Create region-specific documents
    let documents = vec![
        create_region_document("1", "US", "New York data"),
        create_region_document("2", "EU", "London data"),
        create_region_document("3", "Asia", "Tokyo data"),
        create_region_document("4", "US", "California data"),
        create_region_document("5", "Unknown", "Will go to default partition"),
    ];
    
    let options = IndexingOptions::new(10).with_force_commit(true);
    let result = engine.index_documents(documents, options)?;
    
    println!("Value partitioning results:");
    println!("  Documents indexed: {}", result.documents_indexed);
    
    Ok(())
}

/// Create a document with region field.
fn create_region_document(id: &str, region: &str, data: &str) -> Document {
    let mut doc = Document::new();
    doc.add_field("id".to_string(), FieldValue::Text(id.to_string()));
    doc.add_field("region".to_string(), FieldValue::Text(region.to_string()));
    doc.add_field("data".to_string(), FieldValue::Text(data.to_string()));
    doc
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parallel_indexing_example() {
        // This test ensures the example code compiles and runs without errors
        let result = example_parallel_indexing();
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_range_partitioning_example() {
        let result = example_range_partitioning();
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_value_partitioning_example() {
        let result = example_value_partitioning();
        assert!(result.is_ok());
    }
}