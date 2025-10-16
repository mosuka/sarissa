//! Writer management for parallel indexing operations.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::error::{Result, SageError};
use crate::full_text_index::writer::IndexWriter;
use crate::parallel_full_text_index::config::PartitionConfig;

/// Statistics for a specific index partition.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IndexPartitionStats {
    /// Number of documents indexed.
    pub documents_indexed: u64,

    /// Number of successful operations.
    pub successful_operations: u64,

    /// Number of failed operations.
    pub failed_operations: u64,

    /// Total indexing time.
    pub total_indexing_time_ms: u64,

    /// Average indexing time per document.
    pub avg_indexing_time_ms: f64,

    /// Last commit timestamp.
    pub last_commit_time: Option<chrono::DateTime<chrono::Utc>>,

    /// Current index size estimate in bytes.
    pub estimated_size_bytes: u64,

    /// Number of commits performed.
    pub commit_count: u64,
}

impl IndexPartitionStats {
    /// Update statistics with new indexing operation.
    pub fn update_indexing(&mut self, doc_count: u64, elapsed_ms: u64, success: bool) {
        if success {
            self.documents_indexed += doc_count;
            self.successful_operations += 1;
        } else {
            self.failed_operations += 1;
        }

        self.total_indexing_time_ms += elapsed_ms;

        if self.documents_indexed > 0 {
            self.avg_indexing_time_ms =
                self.total_indexing_time_ms as f64 / self.documents_indexed as f64;
        }
    }

    /// Record a commit operation.
    pub fn record_commit(&mut self) {
        self.commit_count += 1;
        self.last_commit_time = Some(chrono::Utc::now());
    }
}

/// Handle for managing an individual index writer with its configuration and statistics.
#[derive(Debug)]
pub struct IndexWriterHandle {
    /// Partition configuration.
    pub config: PartitionConfig,

    /// The actual index writer.
    writer: Arc<Mutex<Box<dyn IndexWriter>>>,

    /// Statistics for this partition.
    stats: Arc<RwLock<IndexPartitionStats>>,

    /// Whether this writer is currently active.
    is_active: Arc<RwLock<bool>>,

    /// Last operation timestamp.
    last_operation: Arc<RwLock<Option<Instant>>>,
}

impl IndexWriterHandle {
    /// Create a new index writer handle.
    pub fn new(config: PartitionConfig, writer: Box<dyn IndexWriter>) -> Self {
        Self {
            config,
            writer: Arc::new(Mutex::new(writer)),
            stats: Arc::new(RwLock::new(IndexPartitionStats::default())),
            is_active: Arc::new(RwLock::new(true)),
            last_operation: Arc::new(RwLock::new(None)),
        }
    }

    /// Get the partition ID.
    pub fn partition_id(&self) -> &str {
        &self.config.partition_id
    }

    /// Get current statistics.
    pub fn stats(&self) -> Result<IndexPartitionStats> {
        self.stats
            .read()
            .map(|s| s.clone())
            .map_err(|_| SageError::internal("Failed to read partition stats"))
    }

    /// Check if this writer is active.
    pub fn is_active(&self) -> Result<bool> {
        self.is_active
            .read()
            .map(|a| *a)
            .map_err(|_| SageError::internal("Failed to read active status"))
    }

    /// Set the active status.
    pub fn set_active(&self, active: bool) -> Result<()> {
        self.is_active
            .write()
            .map(|mut a| *a = active)
            .map_err(|_| SageError::internal("Failed to set active status"))
    }

    /// Execute an indexing operation.
    pub fn index_documents(
        &self,
        documents: Vec<crate::document::document::Document>,
    ) -> Result<()> {
        let start = Instant::now();

        // Update last operation timestamp
        self.last_operation
            .write()
            .map_err(|_| SageError::internal("Failed to update last operation time"))?
            .replace(start);

        let doc_count = documents.len() as u64;
        let result = {
            let mut writer = self
                .writer
                .lock()
                .map_err(|_| SageError::internal("Failed to acquire writer lock"))?;

            // Index documents one by one
            let mut success = true;
            for doc in documents {
                if writer.add_document(doc).is_err() {
                    success = false;
                    break;
                }
            }

            if success {
                Ok(())
            } else {
                Err(SageError::index("Failed to index some documents"))
            }
        };

        let elapsed_ms = start.elapsed().as_millis() as u64;

        // Update statistics
        self.stats
            .write()
            .map_err(|_| SageError::internal("Failed to update stats"))?
            .update_indexing(doc_count, elapsed_ms, result.is_ok());

        result
    }

    /// Commit changes.
    pub fn commit(&self) -> Result<()> {
        let mut writer = self
            .writer
            .lock()
            .map_err(|_| SageError::internal("Failed to acquire writer lock"))?;

        writer.commit()?;

        // Update statistics
        self.stats
            .write()
            .map_err(|_| SageError::internal("Failed to update stats"))?
            .record_commit();

        Ok(())
    }

    /// Get the last operation time.
    pub fn last_operation_time(&self) -> Result<Option<Instant>> {
        self.last_operation
            .read()
            .map(|t| *t)
            .map_err(|_| SageError::internal("Failed to read last operation time"))
    }
}

/// Manager for multiple index writers.
pub struct WriterManager {
    /// Map of partition ID to writer handle.
    writers: Arc<RwLock<HashMap<String, IndexWriterHandle>>>,
}

impl WriterManager {
    /// Create a new writer manager.
    pub fn new() -> Self {
        Self {
            writers: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Add a writer to the manager.
    pub fn add_writer(&self, handle: IndexWriterHandle) -> Result<()> {
        let partition_id = handle.partition_id().to_string();

        let mut writers = self
            .writers
            .write()
            .map_err(|_| SageError::internal("Failed to acquire writers lock"))?;

        if writers.contains_key(&partition_id) {
            return Err(SageError::invalid_argument(format!(
                "Writer with partition ID '{partition_id}' already exists"
            )));
        }

        writers.insert(partition_id, handle);
        Ok(())
    }

    /// Remove a writer from the manager.
    pub fn remove_writer(&self, partition_id: &str) -> Result<IndexWriterHandle> {
        let mut writers = self
            .writers
            .write()
            .map_err(|_| SageError::internal("Failed to acquire writers lock"))?;

        writers.remove(partition_id).ok_or_else(|| {
            SageError::not_found(format!(
                "Writer with partition ID '{partition_id}' not found"
            ))
        })
    }

    /// Get a writer by partition ID.
    pub fn get_writer(&self, partition_id: &str) -> Result<Option<IndexWriterHandle>> {
        let writers = self
            .writers
            .read()
            .map_err(|_| SageError::internal("Failed to acquire readers lock"))?;

        Ok(writers.get(partition_id).cloned())
    }

    /// Get all active writers.
    pub fn get_active_writers(&self) -> Result<Vec<IndexWriterHandle>> {
        let writers = self
            .writers
            .read()
            .map_err(|_| SageError::internal("Failed to acquire readers lock"))?;

        let mut active_writers = Vec::new();
        for writer in writers.values() {
            if writer.is_active()? {
                active_writers.push(writer.clone());
            }
        }

        Ok(active_writers)
    }

    /// Get all writers (including inactive).
    pub fn get_all_writers(&self) -> Result<Vec<IndexWriterHandle>> {
        let writers = self
            .writers
            .read()
            .map_err(|_| SageError::internal("Failed to acquire readers lock"))?;

        Ok(writers.values().cloned().collect())
    }

    /// Get the number of writers.
    pub fn writer_count(&self) -> Result<usize> {
        let writers = self
            .writers
            .read()
            .map_err(|_| SageError::internal("Failed to acquire readers lock"))?;

        Ok(writers.len())
    }

    /// Get the number of active writers.
    pub fn active_writer_count(&self) -> Result<usize> {
        let writers = self
            .writers
            .read()
            .map_err(|_| SageError::internal("Failed to acquire readers lock"))?;

        let mut count = 0;
        for writer in writers.values() {
            if writer.is_active()? {
                count += 1;
            }
        }

        Ok(count)
    }

    /// Commit all active writers.
    pub fn commit_all(&self) -> Result<Vec<String>> {
        let writers = self.get_active_writers()?;
        let mut failed_partitions = Vec::new();

        for writer in writers {
            if writer.commit().is_err() {
                failed_partitions.push(writer.partition_id().to_string());
            }
        }

        Ok(failed_partitions)
    }

    /// Get aggregated statistics from all writers.
    pub fn get_aggregated_stats(&self) -> Result<IndexPartitionStats> {
        let writers = self.get_all_writers()?;
        let mut aggregated = IndexPartitionStats::default();

        for writer in writers {
            let stats = writer.stats()?;
            aggregated.documents_indexed += stats.documents_indexed;
            aggregated.successful_operations += stats.successful_operations;
            aggregated.failed_operations += stats.failed_operations;
            aggregated.total_indexing_time_ms += stats.total_indexing_time_ms;
            aggregated.estimated_size_bytes += stats.estimated_size_bytes;
            aggregated.commit_count += stats.commit_count;
        }

        if aggregated.documents_indexed > 0 {
            aggregated.avg_indexing_time_ms =
                aggregated.total_indexing_time_ms as f64 / aggregated.documents_indexed as f64;
        }

        Ok(aggregated)
    }
}

impl Default for WriterManager {
    fn default() -> Self {
        Self::new()
    }
}

// Implement Clone for IndexWriterHandle to allow sharing between threads
impl Clone for IndexWriterHandle {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            writer: Arc::clone(&self.writer),
            stats: Arc::clone(&self.stats),
            is_active: Arc::clone(&self.is_active),
            last_operation: Arc::clone(&self.last_operation),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::full_text_index::advanced_writer::AdvancedIndexWriter;

    use crate::storage::memory::MemoryStorage;
    use crate::storage::traits::StorageConfig;

    fn create_test_writer() -> Box<dyn IndexWriter> {
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        Box::new(
            AdvancedIndexWriter::new(
                storage,
                crate::full_text_index::advanced_writer::AdvancedWriterConfig::default(),
            )
            .unwrap(),
        )
    }

    #[test]
    fn test_index_writer_handle_creation() {
        let config = PartitionConfig::new("test_partition".to_string());
        let writer = create_test_writer();
        let handle = IndexWriterHandle::new(config, writer);

        assert_eq!(handle.partition_id(), "test_partition");
        assert!(handle.is_active().unwrap());
    }

    #[test]
    fn test_writer_manager_operations() {
        let manager = WriterManager::new();

        // Add writers
        let config1 = PartitionConfig::new("partition1".to_string());
        let writer1 = create_test_writer();
        let handle1 = IndexWriterHandle::new(config1, writer1);
        manager.add_writer(handle1).unwrap();

        let config2 = PartitionConfig::new("partition2".to_string());
        let writer2 = create_test_writer();
        let handle2 = IndexWriterHandle::new(config2, writer2);
        manager.add_writer(handle2).unwrap();

        // Test counts
        assert_eq!(manager.writer_count().unwrap(), 2);
        assert_eq!(manager.active_writer_count().unwrap(), 2);

        // Test retrieval
        let retrieved = manager.get_writer("partition1").unwrap().unwrap();
        assert_eq!(retrieved.partition_id(), "partition1");

        // Test deactivation
        retrieved.set_active(false).unwrap();
        assert_eq!(manager.active_writer_count().unwrap(), 1);

        // Test removal
        let removed = manager.remove_writer("partition1").unwrap();
        assert_eq!(removed.partition_id(), "partition1");
        assert_eq!(manager.writer_count().unwrap(), 1);
    }

    #[test]
    fn test_duplicate_writer_error() {
        let manager = WriterManager::new();
        let config = PartitionConfig::new("duplicate".to_string());

        let writer1 = create_test_writer();
        let handle1 = IndexWriterHandle::new(config.clone(), writer1);
        manager.add_writer(handle1).unwrap();

        let writer2 = create_test_writer();
        let handle2 = IndexWriterHandle::new(config, writer2);
        let result = manager.add_writer(handle2);
        assert!(result.is_err());
    }

    #[test]
    fn test_stats_update() {
        let mut stats = IndexPartitionStats::default();

        stats.update_indexing(100, 500, true);
        assert_eq!(stats.documents_indexed, 100);
        assert_eq!(stats.successful_operations, 1);
        assert_eq!(stats.failed_operations, 0);
        assert_eq!(stats.avg_indexing_time_ms, 5.0);

        stats.update_indexing(50, 300, false);
        assert_eq!(stats.documents_indexed, 100); // Unchanged on failure
        assert_eq!(stats.successful_operations, 1);
        assert_eq!(stats.failed_operations, 1);
    }
}
