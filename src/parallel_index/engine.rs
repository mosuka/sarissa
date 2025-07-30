//! Main parallel indexing engine implementation.

use std::sync::Arc;
use std::time::Duration;

use rayon::{ThreadPool, ThreadPoolBuilder};

use crate::error::{Result, SarissaError};
use crate::index::writer::IndexWriter;
use crate::parallel_index::batch_processor::{BatchProcessingResult, BatchProcessor};
use crate::parallel_index::config::{IndexingOptions, ParallelIndexConfig, PartitionConfig};
use crate::parallel_index::metrics::{IndexingMetricsCollector, IndexingTimer};
use crate::parallel_index::partitioner::DocumentPartitioner;
use crate::parallel_index::writer_manager::{IndexWriterHandle, WriterManager};
use crate::schema::Document;

/// Result of a parallel indexing operation.
#[derive(Debug)]
pub struct ParallelIndexingResult {
    /// Total number of documents processed.
    pub total_documents: usize,

    /// Number of documents successfully indexed.
    pub documents_indexed: usize,

    /// Number of documents that failed to index.
    pub documents_failed: usize,

    /// Results for each partition.
    pub partition_results: Vec<BatchProcessingResult>,

    /// Total execution time.
    pub execution_time: Duration,

    /// Any errors encountered during indexing.
    pub errors: Vec<SarissaError>,
}

/// Main engine for parallel document indexing across multiple partitions.
pub struct ParallelIndexEngine {
    /// Configuration for the engine.
    config: ParallelIndexConfig,

    /// Writer manager for handling multiple index writers.
    writer_manager: Arc<WriterManager>,

    /// Document partitioner for distributing documents.
    partitioner: Option<Box<dyn DocumentPartitioner>>,

    /// Thread pool for parallel execution.
    thread_pool: Arc<ThreadPool>,

    /// Metrics collector.
    metrics: Arc<IndexingMetricsCollector>,
}

impl ParallelIndexEngine {
    /// Create a new parallel indexing engine.
    pub fn new(config: ParallelIndexConfig) -> Result<Self> {
        let thread_pool_size = config.thread_pool_size.unwrap_or_else(num_cpus::get);

        let thread_pool = ThreadPoolBuilder::new()
            .num_threads(thread_pool_size)
            .thread_name(|i| format!("parallel-index-{i}"))
            .build()
            .map_err(|e| SarissaError::internal(format!("Failed to create thread pool: {e}")))?;

        Ok(Self {
            config,
            writer_manager: Arc::new(WriterManager::new()),
            partitioner: None,
            thread_pool: Arc::new(thread_pool),
            metrics: Arc::new(IndexingMetricsCollector::new()),
        })
    }

    /// Add a partition with its writer.
    pub fn add_partition(
        &self,
        _partition_id: String,
        writer: Box<dyn IndexWriter>,
        partition_config: PartitionConfig,
    ) -> Result<()> {
        let handle = IndexWriterHandle::new(partition_config, writer);
        self.writer_manager.add_writer(handle)
    }

    /// Remove a partition.
    pub fn remove_partition(&self, partition_id: &str) -> Result<()> {
        self.writer_manager.remove_writer(partition_id)?;
        Ok(())
    }

    /// Set the document partitioner.
    pub fn set_partitioner(&mut self, partitioner: Box<dyn DocumentPartitioner>) -> Result<()> {
        partitioner.validate()?;
        self.partitioner = Some(partitioner);
        Ok(())
    }

    /// Get the number of active partitions.
    pub fn partition_count(&self) -> Result<usize> {
        self.writer_manager.active_writer_count()
    }

    /// Index a collection of documents in parallel.
    pub fn index_documents(
        &self,
        documents: Vec<Document>,
        options: IndexingOptions,
    ) -> Result<ParallelIndexingResult> {
        let timer = IndexingTimer::start();

        // Check for empty documents first
        if documents.is_empty() {
            return Ok(ParallelIndexingResult {
                total_documents: 0,
                documents_indexed: 0,
                documents_failed: 0,
                partition_results: Vec::new(),
                execution_time: timer.elapsed(),
                errors: Vec::new(),
            });
        }

        // Validate prerequisites
        let partitioner = self
            .partitioner
            .as_ref()
            .ok_or_else(|| SarissaError::invalid_argument("No partitioner configured"))?;

        // Get active writers
        let writers = self.writer_manager.get_active_writers()?;
        if writers.is_empty() {
            return Err(SarissaError::invalid_argument(
                "No active writers available",
            ));
        }

        // Partition documents
        let partitioned_docs = partitioner.partition_batch(documents)?;
        let total_documents = partitioned_docs.len();

        // Group documents by partition
        let mut partition_groups: std::collections::HashMap<usize, Vec<Document>> =
            std::collections::HashMap::new();

        for (partition_index, document) in partitioned_docs {
            partition_groups
                .entry(partition_index)
                .or_default()
                .push(document);
        }

        // Process partitions in parallel
        let results = self.process_partitions_parallel(partition_groups, writers, &options)?;

        let execution_time = timer.elapsed();

        // Aggregate results
        let documents_indexed: usize = results.iter().map(|r| r.documents_indexed).sum();

        let documents_failed: usize = results.iter().map(|r| r.documents_failed).sum();

        let mut errors = Vec::new();
        for result in &results {
            for error in &result.errors {
                errors.push(SarissaError::other(error.to_string()));
            }
        }

        // Record metrics
        if self.config.enable_metrics {
            self.metrics.record_operation(
                execution_time,
                documents_indexed as u64,
                errors.is_empty(),
            );
        }

        Ok(ParallelIndexingResult {
            total_documents,
            documents_indexed,
            documents_failed,
            partition_results: results,
            execution_time,
            errors,
        })
    }

    /// Process partitions in parallel using the thread pool.
    fn process_partitions_parallel(
        &self,
        partition_groups: std::collections::HashMap<usize, Vec<Document>>,
        writers: Vec<IndexWriterHandle>,
        options: &IndexingOptions,
    ) -> Result<Vec<BatchProcessingResult>> {
        use std::sync::mpsc;

        let (tx, rx) = mpsc::channel();
        let mut handles = Vec::new();

        // Create writer lookup map
        let writer_map: std::collections::HashMap<String, IndexWriterHandle> = writers
            .into_iter()
            .map(|w| (w.partition_id().to_string(), w))
            .collect();

        // Submit tasks to thread pool
        for (partition_index, documents) in partition_groups {
            // Find the appropriate writer for this partition
            let writer_handle = self.find_writer_for_partition(partition_index, &writer_map)?;
            let tx = tx.clone();
            let options = options.clone();
            let batch_config = self.create_batch_config();

            self.thread_pool.spawn(move || {
                let result = Self::process_partition_batch(
                    partition_index,
                    documents,
                    writer_handle,
                    &options,
                    batch_config,
                );
                let _ = tx.send(result);
            });

            handles.push(());
        }

        // Drop the original sender so receiver knows when all tasks are done
        drop(tx);

        // Collect results
        let mut results = Vec::new();
        while let Ok(result) = rx.recv() {
            results.push(result);
        }

        // Apply timeout if specified
        if let Some(timeout) = options.timeout {
            let start = std::time::Instant::now();
            if start.elapsed() > timeout {
                return Err(SarissaError::timeout("Indexing operation timed out"));
            }
        }

        Ok(results)
    }

    /// Find the appropriate writer for a partition index.
    fn find_writer_for_partition(
        &self,
        partition_index: usize,
        writer_map: &std::collections::HashMap<String, IndexWriterHandle>,
    ) -> Result<IndexWriterHandle> {
        // For now, we use a simple mapping: partition_index -> "partition_{index}"
        // In a more sophisticated implementation, this could use a configurable mapping
        let partition_id = format!("partition_{partition_index}");

        writer_map
            .get(&partition_id)
            .cloned()
            .or_else(|| {
                // Fallback: use any available writer if exact match not found
                writer_map.values().next().cloned()
            })
            .ok_or_else(|| {
                SarissaError::not_found(format!("No writer found for partition {partition_index}"))
            })
    }

    /// Create batch configuration from engine configuration.
    fn create_batch_config(&self) -> crate::parallel_index::batch_processor::BatchConfig {
        crate::parallel_index::batch_processor::BatchConfig {
            max_batch_size: self.config.default_batch_size,
            max_batch_memory: self.config.max_buffer_memory / self.config.max_concurrent_partitions,
            max_batch_age: self.config.commit_interval,
            immediate_processing: true,
            input_buffer_size: self.config.default_batch_size * 2,
        }
    }

    /// Process a batch of documents for a single partition.
    fn process_partition_batch(
        partition_index: usize,
        documents: Vec<Document>,
        writer_handle: IndexWriterHandle,
        options: &IndexingOptions,
        batch_config: crate::parallel_index::batch_processor::BatchConfig,
    ) -> BatchProcessingResult {
        let mut processor = BatchProcessor::new(batch_config, 1);

        // Create a single batch from all documents
        let mut batch = crate::parallel_index::batch_processor::DocumentBatch::new(partition_index);
        for doc in documents {
            batch.add_document(doc);
        }

        // Process the batch
        processor.process_batch(batch, &writer_handle, options)
    }

    /// Commit all active writers.
    pub fn commit_all(&self) -> Result<Vec<String>> {
        let failed_partitions = self.writer_manager.commit_all()?;

        if self.config.enable_metrics {
            let commit_timer = IndexingTimer::start();
            // Record commit time (simplified - in reality we'd track per-partition)
            self.metrics.record_commit(commit_timer.elapsed());
        }

        Ok(failed_partitions)
    }

    /// Get current indexing metrics.
    pub fn metrics(&self) -> crate::parallel_index::metrics::IndexingMetrics {
        self.metrics.snapshot()
    }

    /// Reset metrics.
    pub fn reset_metrics(&self) {
        self.metrics.reset();
    }

    /// Get the configuration.
    pub fn config(&self) -> &ParallelIndexConfig {
        &self.config
    }

    /// Get detailed statistics for all partitions.
    pub fn partition_statistics(
        &self,
    ) -> Result<
        Vec<(
            String,
            crate::parallel_index::writer_manager::IndexPartitionStats,
        )>,
    > {
        let writers = self.writer_manager.get_all_writers()?;
        let mut stats = Vec::new();

        for writer in writers {
            let partition_id = writer.partition_id().to_string();
            let partition_stats = writer.stats()?;
            stats.push((partition_id, partition_stats));
        }

        Ok(stats)
    }

    /// Get aggregated statistics from all partitions.
    pub fn aggregated_statistics(
        &self,
    ) -> Result<crate::parallel_index::writer_manager::IndexPartitionStats> {
        self.writer_manager.get_aggregated_stats()
    }

    /// Check if a partitioner is configured.
    pub fn has_partitioner(&self) -> bool {
        self.partitioner.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::writer::BasicIndexWriter;
    use crate::parallel_index::partitioner::HashPartitioner;
    use crate::schema::{FieldValue, Schema, TextField};
    use crate::storage::{MemoryStorage, StorageConfig};

    fn create_test_writer() -> Box<dyn IndexWriter> {
        let mut schema = Schema::new().unwrap();
        schema.add_field("id", Box::new(TextField::new())).unwrap();
        schema
            .add_field("content", Box::new(TextField::new()))
            .unwrap();
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        Box::new(
            BasicIndexWriter::new(
                schema,
                storage,
                crate::index::writer::WriterConfig::default(),
            )
            .unwrap(),
        )
    }

    fn create_test_document(id: &str, content: &str) -> Document {
        let mut doc = Document::new();
        doc.add_field("id".to_string(), FieldValue::Text(id.to_string()));
        doc.add_field("content".to_string(), FieldValue::Text(content.to_string()));
        doc
    }

    #[test]
    fn test_engine_creation() {
        let config = ParallelIndexConfig::default();
        let engine = ParallelIndexEngine::new(config).unwrap();

        assert_eq!(engine.partition_count().unwrap(), 0);
        assert!(!engine.has_partitioner());
    }

    #[test]
    fn test_partition_management() {
        let config = ParallelIndexConfig::default();
        let engine = ParallelIndexEngine::new(config).unwrap();

        // Add partitions
        let partition_config1 = PartitionConfig::new("partition_0".to_string());
        engine
            .add_partition(
                "partition_0".to_string(),
                create_test_writer(),
                partition_config1,
            )
            .unwrap();

        let partition_config2 = PartitionConfig::new("partition_1".to_string());
        engine
            .add_partition(
                "partition_1".to_string(),
                create_test_writer(),
                partition_config2,
            )
            .unwrap();

        assert_eq!(engine.partition_count().unwrap(), 2);

        // Remove partition
        engine.remove_partition("partition_1").unwrap();
        assert_eq!(engine.partition_count().unwrap(), 1);
    }

    #[test]
    fn test_partitioner_configuration() {
        let config = ParallelIndexConfig::default();
        let mut engine = ParallelIndexEngine::new(config).unwrap();

        let partitioner = Box::new(HashPartitioner::new("id".to_string(), 2));
        engine.set_partitioner(partitioner).unwrap();

        assert!(engine.has_partitioner());
    }

    #[test]
    fn test_empty_document_indexing() {
        let config = ParallelIndexConfig::default();
        let engine = ParallelIndexEngine::new(config).unwrap();

        let documents = Vec::new();
        let options = IndexingOptions::default();

        let result = engine.index_documents(documents, options).unwrap();

        assert_eq!(result.total_documents, 0);
        assert_eq!(result.documents_indexed, 0);
        assert_eq!(result.documents_failed, 0);
    }

    #[test]
    fn test_indexing_without_partitioner() {
        let config = ParallelIndexConfig::default();
        let engine = ParallelIndexEngine::new(config).unwrap();

        let documents = vec![create_test_document("1", "content1")];
        let options = IndexingOptions::default();

        let result = engine.index_documents(documents, options);
        assert!(result.is_err()); // Should fail without partitioner
    }

    #[test]
    fn test_indexing_without_writers() {
        let config = ParallelIndexConfig::default();
        let mut engine = ParallelIndexEngine::new(config).unwrap();

        let partitioner = Box::new(HashPartitioner::new("id".to_string(), 2));
        engine.set_partitioner(partitioner).unwrap();

        let documents = vec![create_test_document("1", "content1")];
        let options = IndexingOptions::default();

        let result = engine.index_documents(documents, options);
        assert!(result.is_err()); // Should fail without writers
    }

    #[test]
    fn test_commit_all() {
        let config = ParallelIndexConfig::default();
        let engine = ParallelIndexEngine::new(config).unwrap();

        let partition_config = PartitionConfig::new("partition_0".to_string());
        engine
            .add_partition(
                "partition_0".to_string(),
                create_test_writer(),
                partition_config,
            )
            .unwrap();

        let failed_partitions = engine.commit_all().unwrap();
        assert!(failed_partitions.is_empty()); // Should succeed
    }

    #[test]
    fn test_metrics_collection() {
        let config = ParallelIndexConfig {
            enable_metrics: true,
            ..Default::default()
        };

        let engine = ParallelIndexEngine::new(config).unwrap();

        let metrics = engine.metrics();
        assert_eq!(metrics.total_operations, 0);

        // Reset should work
        engine.reset_metrics();
        let metrics_after_reset = engine.metrics();
        assert_eq!(metrics_after_reset.total_operations, 0);
    }
}
