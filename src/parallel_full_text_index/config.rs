//! Configuration for parallel indexing operations.

use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Configuration for parallel index engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelIndexConfig {
    /// Maximum number of concurrent indexing partitions.
    pub max_concurrent_partitions: usize,

    /// Default batch size for document processing.
    pub default_batch_size: usize,

    /// Maximum memory buffer size in bytes.
    pub max_buffer_memory: usize,

    /// Commit interval for flushing changes.
    pub commit_interval: Duration,

    /// Number of retry attempts for failed operations.
    pub retry_attempts: usize,

    /// Timeout for individual indexing operations.
    pub operation_timeout: Duration,

    /// Whether to enable metrics collection.
    pub enable_metrics: bool,

    /// Thread pool size for parallel processing.
    /// If None, uses the number of CPU cores.
    pub thread_pool_size: Option<usize>,

    /// Whether to continue on partial failures.
    pub allow_partial_failures: bool,

    /// Auto-commit threshold (number of documents).
    pub auto_commit_threshold: usize,
}

impl Default for ParallelIndexConfig {
    fn default() -> Self {
        Self {
            max_concurrent_partitions: num_cpus::get(),
            default_batch_size: 1000,
            max_buffer_memory: 512 * 1024 * 1024, // 512MB
            commit_interval: Duration::from_secs(30),
            retry_attempts: 3,
            operation_timeout: Duration::from_secs(60),
            enable_metrics: true,
            thread_pool_size: None,
            allow_partial_failures: true,
            auto_commit_threshold: 10000,
        }
    }
}

/// Configuration for a specific partition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionConfig {
    /// Partition identifier.
    pub partition_id: String,

    /// Weight factor for load balancing.
    pub weight: f32,

    /// Whether this partition is active.
    pub is_active: bool,

    /// Partition-specific batch size override.
    pub batch_size: Option<usize>,

    /// Partition-specific commit interval override.
    pub commit_interval: Option<Duration>,

    /// Maximum number of documents this partition can hold.
    pub max_documents: Option<u64>,

    /// Custom metadata for this partition.
    pub metadata: std::collections::HashMap<String, String>,
}

impl PartitionConfig {
    /// Create a new partition configuration.
    pub fn new(partition_id: String) -> Self {
        Self {
            partition_id,
            weight: 1.0,
            is_active: true,
            batch_size: None,
            commit_interval: None,
            max_documents: None,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Set the weight for load balancing.
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }

    /// Set the active status.
    pub fn with_active(mut self, active: bool) -> Self {
        self.is_active = active;
        self
    }

    /// Set partition-specific batch size.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = Some(batch_size);
        self
    }

    /// Set partition-specific commit interval.
    pub fn with_commit_interval(mut self, interval: Duration) -> Self {
        self.commit_interval = Some(interval);
        self
    }

    /// Set maximum document limit.
    pub fn with_max_documents(mut self, max_docs: u64) -> Self {
        self.max_documents = Some(max_docs);
        self
    }

    /// Add custom metadata.
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// Options for indexing operations.
#[derive(Debug, Clone)]
pub struct IndexingOptions {
    /// Batch size for this operation.
    pub batch_size: usize,

    /// Whether to force commit after indexing.
    pub force_commit: bool,

    /// Timeout for this operation.
    pub timeout: Option<Duration>,

    /// Whether to collect detailed metrics.
    pub collect_metrics: bool,

    /// Whether to validate documents before indexing.
    pub validate_documents: bool,

    /// Maximum number of errors to tolerate.
    pub max_errors: Option<usize>,
}

impl Default for IndexingOptions {
    fn default() -> Self {
        Self {
            batch_size: 1000,
            force_commit: false,
            timeout: None,
            collect_metrics: false,
            validate_documents: true,
            max_errors: None,
        }
    }
}

impl IndexingOptions {
    /// Create new indexing options with specified batch size.
    pub fn new(batch_size: usize) -> Self {
        Self {
            batch_size,
            ..Default::default()
        }
    }

    /// Enable forced commit.
    pub fn with_force_commit(mut self, force: bool) -> Self {
        self.force_commit = force;
        self
    }

    /// Set operation timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Enable metrics collection.
    pub fn with_metrics(mut self, collect: bool) -> Self {
        self.collect_metrics = collect;
        self
    }

    /// Set document validation.
    pub fn with_validation(mut self, validate: bool) -> Self {
        self.validate_documents = validate;
        self
    }

    /// Set maximum error tolerance.
    pub fn with_max_errors(mut self, max_errors: usize) -> Self {
        self.max_errors = Some(max_errors);
        self
    }
}

/// Partition strategy type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PartitionStrategy {
    /// Hash-based partitioning by field value.
    Hash,

    /// Range-based partitioning (numeric/date ranges).
    Range,

    /// Value-based direct mapping.
    Value,

    /// Round-robin distribution.
    RoundRobin,

    /// Custom partitioning logic.
    Custom,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ParallelIndexConfig::default();
        assert!(config.max_concurrent_partitions > 0);
        assert_eq!(config.default_batch_size, 1000);
        assert_eq!(config.commit_interval, Duration::from_secs(30));
        assert_eq!(config.retry_attempts, 3);
        assert!(config.enable_metrics);
    }

    #[test]
    fn test_partition_config_builder() {
        let config = PartitionConfig::new("partition1".to_string())
            .with_weight(2.0)
            .with_active(true)
            .with_batch_size(500)
            .with_max_documents(100000)
            .with_metadata("region".to_string(), "us-east".to_string());

        assert_eq!(config.partition_id, "partition1");
        assert_eq!(config.weight, 2.0);
        assert!(config.is_active);
        assert_eq!(config.batch_size, Some(500));
        assert_eq!(config.max_documents, Some(100000));
        assert_eq!(config.metadata.get("region"), Some(&"us-east".to_string()));
    }

    #[test]
    fn test_indexing_options_builder() {
        let options = IndexingOptions::new(2000)
            .with_force_commit(true)
            .with_timeout(Duration::from_secs(120))
            .with_metrics(true)
            .with_validation(false)
            .with_max_errors(10);

        assert_eq!(options.batch_size, 2000);
        assert!(options.force_commit);
        assert_eq!(options.timeout, Some(Duration::from_secs(120)));
        assert!(options.collect_metrics);
        assert!(!options.validate_documents);
        assert_eq!(options.max_errors, Some(10));
    }
}
