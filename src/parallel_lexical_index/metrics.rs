//! Metrics collection for parallel indexing operations.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Metrics collected during parallel indexing operations.
#[derive(Debug, Clone)]
pub struct IndexingMetrics {
    /// Total number of indexing operations executed.
    pub total_operations: u64,

    /// Number of successful indexing operations.
    pub successful_operations: u64,

    /// Number of failed indexing operations.
    pub failed_operations: u64,

    /// Total number of documents indexed.
    pub total_documents_indexed: u64,

    /// Total execution time across all operations.
    pub total_execution_time: Duration,

    /// Average execution time per operation.
    pub avg_execution_time: Duration,

    /// Average execution time per document.
    pub avg_time_per_document: Duration,

    /// Maximum execution time observed.
    pub max_execution_time: Duration,

    /// Minimum execution time observed.
    pub min_execution_time: Duration,

    /// Number of commits performed.
    pub total_commits: u64,

    /// Total time spent on commits.
    pub total_commit_time: Duration,

    /// Number of retries performed.
    pub total_retries: u64,

    /// Per-partition metrics.
    pub partition_metrics: Vec<PartitionMetrics>,

    /// Throughput metrics.
    pub throughput: ThroughputMetrics,
}

impl Default for IndexingMetrics {
    fn default() -> Self {
        Self {
            total_operations: 0,
            successful_operations: 0,
            failed_operations: 0,
            total_documents_indexed: 0,
            total_execution_time: Duration::ZERO,
            avg_execution_time: Duration::ZERO,
            avg_time_per_document: Duration::ZERO,
            max_execution_time: Duration::ZERO,
            min_execution_time: Duration::MAX,
            total_commits: 0,
            total_commit_time: Duration::ZERO,
            total_retries: 0,
            partition_metrics: Vec::new(),
            throughput: ThroughputMetrics::default(),
        }
    }
}

/// Metrics for a specific partition.
#[derive(Debug, Clone)]
pub struct PartitionMetrics {
    /// Partition identifier.
    pub partition_id: String,

    /// Number of operations on this partition.
    pub operation_count: u64,

    /// Number of documents indexed in this partition.
    pub document_count: u64,

    /// Average response time for this partition.
    pub avg_response_time: Duration,

    /// Total indexing time for this partition.
    pub total_indexing_time: Duration,

    /// Number of errors for this partition.
    pub error_count: u64,

    /// Current load factor (0.0 to 1.0).
    pub load_factor: f32,
}

/// Throughput metrics.
#[derive(Debug, Clone, Default)]
pub struct ThroughputMetrics {
    /// Documents indexed per second (current rate).
    pub docs_per_second: f64,

    /// Operations per second (current rate).
    pub ops_per_second: f64,

    /// Peak documents per second observed.
    pub peak_docs_per_second: f64,

    /// Peak operations per second observed.
    pub peak_ops_per_second: f64,

    /// Average documents per second over entire session.
    pub avg_docs_per_second: f64,

    /// Average operations per second over entire session.
    pub avg_ops_per_second: f64,
}

/// Collector for gathering indexing metrics during operations.
pub struct IndexingMetricsCollector {
    /// Atomic counters for thread-safe collection.
    total_operations: Arc<AtomicU64>,
    successful_operations: Arc<AtomicU64>,
    failed_operations: Arc<AtomicU64>,
    total_documents_indexed: Arc<AtomicU64>,
    total_execution_nanos: Arc<AtomicU64>,
    max_execution_nanos: Arc<AtomicU64>,
    min_execution_nanos: Arc<AtomicU64>,
    total_commits: Arc<AtomicU64>,
    total_commit_nanos: Arc<AtomicU64>,
    total_retries: Arc<AtomicU64>,

    /// Start time for the collector.
    start_time: Instant,

    /// Window for calculating current throughput.
    throughput_window: Arc<parking_lot::Mutex<ThroughputWindow>>,
}

/// Window for calculating throughput metrics.
#[derive(Debug)]
struct ThroughputWindow {
    /// Timestamps and document counts for recent operations.
    recent_operations: std::collections::VecDeque<(Instant, u64)>,

    /// Maximum window size.
    window_size: Duration,

    /// Peak values observed.
    peak_docs_per_second: f64,
    peak_ops_per_second: f64,
}

impl ThroughputWindow {
    fn new(window_size: Duration) -> Self {
        Self {
            recent_operations: std::collections::VecDeque::new(),
            window_size,
            peak_docs_per_second: 0.0,
            peak_ops_per_second: 0.0,
        }
    }

    fn add_operation(&mut self, doc_count: u64) {
        let now = Instant::now();
        self.recent_operations.push_back((now, doc_count));

        // Remove old entries outside the window
        while let Some(&(timestamp, _)) = self.recent_operations.front() {
            if now.duration_since(timestamp) > self.window_size {
                self.recent_operations.pop_front();
            } else {
                break;
            }
        }

        // Calculate current throughput
        let total_docs: u64 = self.recent_operations.iter().map(|(_, count)| count).sum();
        let ops_count = self.recent_operations.len() as u64;

        if let Some((oldest_time, _)) = self.recent_operations.front() {
            let window_duration = now.duration_since(*oldest_time).as_secs_f64();
            if window_duration > 0.0 {
                let current_docs_per_sec = total_docs as f64 / window_duration;
                let current_ops_per_sec = ops_count as f64 / window_duration;

                if current_docs_per_sec > self.peak_docs_per_second {
                    self.peak_docs_per_second = current_docs_per_sec;
                }
                if current_ops_per_sec > self.peak_ops_per_second {
                    self.peak_ops_per_second = current_ops_per_sec;
                }
            }
        }
    }

    fn get_current_throughput(&self) -> (f64, f64) {
        if self.recent_operations.is_empty() {
            return (0.0, 0.0);
        }

        let total_docs: u64 = self.recent_operations.iter().map(|(_, count)| count).sum();
        let ops_count = self.recent_operations.len() as u64;

        if let (Some((oldest_time, _)), Some((newest_time, _))) = (
            self.recent_operations.front(),
            self.recent_operations.back(),
        ) {
            let window_duration = newest_time.duration_since(*oldest_time).as_secs_f64();
            if window_duration > 0.0 {
                let docs_per_sec = total_docs as f64 / window_duration;
                let ops_per_sec = ops_count as f64 / window_duration;
                return (docs_per_sec, ops_per_sec);
            }
        }

        (0.0, 0.0)
    }
}

impl IndexingMetricsCollector {
    /// Create a new metrics collector.
    pub fn new() -> Self {
        Self {
            total_operations: Arc::new(AtomicU64::new(0)),
            successful_operations: Arc::new(AtomicU64::new(0)),
            failed_operations: Arc::new(AtomicU64::new(0)),
            total_documents_indexed: Arc::new(AtomicU64::new(0)),
            total_execution_nanos: Arc::new(AtomicU64::new(0)),
            max_execution_nanos: Arc::new(AtomicU64::new(0)),
            min_execution_nanos: Arc::new(AtomicU64::new(u64::MAX)),
            total_commits: Arc::new(AtomicU64::new(0)),
            total_commit_nanos: Arc::new(AtomicU64::new(0)),
            total_retries: Arc::new(AtomicU64::new(0)),
            start_time: Instant::now(),
            throughput_window: Arc::new(parking_lot::Mutex::new(ThroughputWindow::new(
                Duration::from_secs(60),
            ))),
        }
    }

    /// Record an indexing operation.
    pub fn record_operation(&self, execution_time: Duration, doc_count: u64, success: bool) {
        self.total_operations.fetch_add(1, Ordering::Relaxed);

        if success {
            self.successful_operations.fetch_add(1, Ordering::Relaxed);
            self.total_documents_indexed
                .fetch_add(doc_count, Ordering::Relaxed);
        } else {
            self.failed_operations.fetch_add(1, Ordering::Relaxed);
        }

        let nanos = execution_time.as_nanos() as u64;
        self.total_execution_nanos
            .fetch_add(nanos, Ordering::Relaxed);

        // Update max execution time
        loop {
            let current_max = self.max_execution_nanos.load(Ordering::Relaxed);
            if nanos <= current_max {
                break;
            }
            if self
                .max_execution_nanos
                .compare_exchange_weak(current_max, nanos, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }

        // Update min execution time
        loop {
            let current_min = self.min_execution_nanos.load(Ordering::Relaxed);
            if nanos >= current_min {
                break;
            }
            if self
                .min_execution_nanos
                .compare_exchange_weak(current_min, nanos, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }

        // Update throughput window
        self.throughput_window.lock().add_operation(doc_count);
    }

    /// Record a commit operation.
    pub fn record_commit(&self, commit_time: Duration) {
        self.total_commits.fetch_add(1, Ordering::Relaxed);
        self.total_commit_nanos
            .fetch_add(commit_time.as_nanos() as u64, Ordering::Relaxed);
    }

    /// Record a retry operation.
    pub fn record_retry(&self) {
        self.total_retries.fetch_add(1, Ordering::Relaxed);
    }

    /// Get the current metrics snapshot.
    pub fn snapshot(&self) -> IndexingMetrics {
        let total_operations = self.total_operations.load(Ordering::Relaxed);
        let total_docs = self.total_documents_indexed.load(Ordering::Relaxed);
        let total_nanos = self.total_execution_nanos.load(Ordering::Relaxed);

        let avg_nanos = if total_operations > 0 {
            total_nanos / total_operations
        } else {
            0
        };

        let avg_doc_nanos = if total_docs > 0 {
            total_nanos / total_docs
        } else {
            0
        };

        let min_nanos = self.min_execution_nanos.load(Ordering::Relaxed);
        let min_duration = if min_nanos == u64::MAX {
            Duration::ZERO
        } else {
            Duration::from_nanos(min_nanos)
        };

        // Calculate overall throughput
        let total_time = self.start_time.elapsed();
        let avg_docs_per_second = if total_time.as_secs() > 0 {
            total_docs as f64 / total_time.as_secs_f64()
        } else {
            0.0
        };

        let avg_ops_per_second = if total_time.as_secs() > 0 {
            total_operations as f64 / total_time.as_secs_f64()
        } else {
            0.0
        };

        // Get current throughput from window
        let throughput_window = self.throughput_window.lock();
        let (current_docs_per_sec, current_ops_per_sec) =
            throughput_window.get_current_throughput();

        IndexingMetrics {
            total_operations,
            successful_operations: self.successful_operations.load(Ordering::Relaxed),
            failed_operations: self.failed_operations.load(Ordering::Relaxed),
            total_documents_indexed: total_docs,
            total_execution_time: Duration::from_nanos(total_nanos),
            avg_execution_time: Duration::from_nanos(avg_nanos),
            avg_time_per_document: Duration::from_nanos(avg_doc_nanos),
            max_execution_time: Duration::from_nanos(
                self.max_execution_nanos.load(Ordering::Relaxed),
            ),
            min_execution_time: min_duration,
            total_commits: self.total_commits.load(Ordering::Relaxed),
            total_commit_time: Duration::from_nanos(
                self.total_commit_nanos.load(Ordering::Relaxed),
            ),
            total_retries: self.total_retries.load(Ordering::Relaxed),
            partition_metrics: Vec::new(), // Would be populated from per-partition collectors
            throughput: ThroughputMetrics {
                docs_per_second: current_docs_per_sec,
                ops_per_second: current_ops_per_sec,
                peak_docs_per_second: throughput_window.peak_docs_per_second,
                peak_ops_per_second: throughput_window.peak_ops_per_second,
                avg_docs_per_second,
                avg_ops_per_second,
            },
        }
    }

    /// Reset all metrics.
    pub fn reset(&self) {
        self.total_operations.store(0, Ordering::Relaxed);
        self.successful_operations.store(0, Ordering::Relaxed);
        self.failed_operations.store(0, Ordering::Relaxed);
        self.total_documents_indexed.store(0, Ordering::Relaxed);
        self.total_execution_nanos.store(0, Ordering::Relaxed);
        self.max_execution_nanos.store(0, Ordering::Relaxed);
        self.min_execution_nanos.store(u64::MAX, Ordering::Relaxed);
        self.total_commits.store(0, Ordering::Relaxed);
        self.total_commit_nanos.store(0, Ordering::Relaxed);
        self.total_retries.store(0, Ordering::Relaxed);

        let mut window = self.throughput_window.lock();
        window.recent_operations.clear();
        window.peak_docs_per_second = 0.0;
        window.peak_ops_per_second = 0.0;
    }

    /// Get the uptime of this collector.
    pub fn uptime(&self) -> Duration {
        self.start_time.elapsed()
    }
}

impl Default for IndexingMetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper for timing indexing operations.
pub struct IndexingTimer {
    start: Instant,
}

impl IndexingTimer {
    /// Start a new timer.
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    /// Get elapsed time.
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Stop the timer and return elapsed time.
    pub fn stop(self) -> Duration {
        self.start.elapsed()
    }
}

/// Metrics for monitoring memory usage during indexing.
#[derive(Debug, Default)]
pub struct IndexingMemoryMetrics {
    /// Current memory usage in bytes.
    pub current_usage: Arc<AtomicUsize>,

    /// Peak memory usage observed.
    pub peak_usage: Arc<AtomicUsize>,

    /// Number of memory allocations.
    pub allocation_count: Arc<AtomicU64>,

    /// Total bytes allocated.
    pub total_bytes_allocated: Arc<AtomicU64>,
}

impl IndexingMemoryMetrics {
    /// Create new memory metrics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a memory allocation.
    pub fn record_allocation(&self, size: usize) {
        self.current_usage.fetch_add(size, Ordering::Relaxed);
        self.allocation_count.fetch_add(1, Ordering::Relaxed);
        self.total_bytes_allocated
            .fetch_add(size as u64, Ordering::Relaxed);

        // Update peak if necessary
        loop {
            let current = self.current_usage.load(Ordering::Relaxed);
            let peak = self.peak_usage.load(Ordering::Relaxed);
            if current <= peak {
                break;
            }
            if self
                .peak_usage
                .compare_exchange_weak(peak, current, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }
    }

    /// Record a memory deallocation.
    pub fn record_deallocation(&self, size: usize) {
        self.current_usage.fetch_sub(size, Ordering::Relaxed);
    }

    /// Get current memory usage.
    pub fn current(&self) -> usize {
        self.current_usage.load(Ordering::Relaxed)
    }

    /// Get peak memory usage.
    pub fn peak(&self) -> usize {
        self.peak_usage.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_collection() {
        let collector = IndexingMetricsCollector::new();

        // Record some operations
        collector.record_operation(Duration::from_millis(100), 10, true);
        collector.record_operation(Duration::from_millis(50), 5, true);
        collector.record_operation(Duration::from_millis(200), 20, false);

        let metrics = collector.snapshot();

        assert_eq!(metrics.total_operations, 3);
        assert_eq!(metrics.successful_operations, 2);
        assert_eq!(metrics.failed_operations, 1);
        assert_eq!(metrics.total_documents_indexed, 15); // Only successful operations

        // Check timing metrics
        assert_eq!(metrics.min_execution_time, Duration::from_millis(50));
        assert_eq!(metrics.max_execution_time, Duration::from_millis(200));
        assert!(metrics.avg_execution_time >= Duration::from_millis(100));
        assert!(metrics.avg_execution_time <= Duration::from_millis(125));
    }

    #[test]
    fn test_commit_recording() {
        let collector = IndexingMetricsCollector::new();

        collector.record_commit(Duration::from_millis(10));
        collector.record_commit(Duration::from_millis(20));

        let metrics = collector.snapshot();

        assert_eq!(metrics.total_commits, 2);
        assert_eq!(metrics.total_commit_time, Duration::from_millis(30));
    }

    #[test]
    fn test_timer() {
        let timer = IndexingTimer::start();
        std::thread::sleep(Duration::from_millis(10));
        let elapsed = timer.stop();

        assert!(elapsed >= Duration::from_millis(10));
        assert!(elapsed < Duration::from_millis(100));
    }

    #[test]
    fn test_memory_metrics() {
        let metrics = IndexingMemoryMetrics::new();

        metrics.record_allocation(1024);
        assert_eq!(metrics.current(), 1024);
        assert_eq!(metrics.peak(), 1024);

        metrics.record_allocation(2048);
        assert_eq!(metrics.current(), 3072);
        assert_eq!(metrics.peak(), 3072);

        metrics.record_deallocation(1024);
        assert_eq!(metrics.current(), 2048);
        assert_eq!(metrics.peak(), 3072); // Peak unchanged
    }

    #[test]
    fn test_throughput_window() {
        let mut window = ThroughputWindow::new(Duration::from_secs(1));

        window.add_operation(100);
        std::thread::sleep(Duration::from_millis(100));
        window.add_operation(200);

        let (docs_per_sec, ops_per_sec) = window.get_current_throughput();

        assert!(docs_per_sec > 0.0);
        assert!(ops_per_sec > 0.0);
    }
}
