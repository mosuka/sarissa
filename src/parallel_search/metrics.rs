//! Metrics collection for parallel search operations.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Metrics collected during parallel search operations.
#[derive(Debug, Clone)]
pub struct SearchMetrics {
    /// Total number of searches executed.
    pub total_searches: u64,

    /// Number of successful searches.
    pub successful_searches: u64,

    /// Number of failed searches.
    pub failed_searches: u64,

    /// Total execution time across all searches.
    pub total_execution_time: Duration,

    /// Average execution time per search.
    pub avg_execution_time: Duration,

    /// Maximum execution time observed.
    pub max_execution_time: Duration,

    /// Minimum execution time observed.
    pub min_execution_time: Duration,

    /// Total documents evaluated.
    pub total_docs_evaluated: u64,

    /// Total hits returned.
    pub total_hits_returned: u64,

    /// Number of timeouts.
    pub timeout_count: u64,

    /// Per-index metrics.
    pub index_metrics: Vec<IndexMetrics>,
}

impl Default for SearchMetrics {
    fn default() -> Self {
        Self {
            total_searches: 0,
            successful_searches: 0,
            failed_searches: 0,
            total_execution_time: Duration::ZERO,
            avg_execution_time: Duration::ZERO,
            max_execution_time: Duration::ZERO,
            min_execution_time: Duration::MAX,
            total_docs_evaluated: 0,
            total_hits_returned: 0,
            timeout_count: 0,
            index_metrics: Vec::new(),
        }
    }
}

/// Metrics for a specific index.
#[derive(Debug, Clone)]
pub struct IndexMetrics {
    /// Index identifier.
    pub index_id: String,

    /// Number of searches on this index.
    pub search_count: u64,

    /// Number of successful searches.
    pub success_count: u64,

    /// Number of failed searches.
    pub failure_count: u64,

    /// Average response time.
    pub avg_response_time: Duration,

    /// Total documents evaluated from this index.
    pub docs_evaluated: u64,

    /// Total hits from this index.
    pub hits_returned: u64,
}

/// Collector for gathering metrics during parallel search.
pub struct SearchMetricsCollector {
    /// Atomic counters for thread-safe collection.
    total_searches: Arc<AtomicU64>,
    successful_searches: Arc<AtomicU64>,
    failed_searches: Arc<AtomicU64>,
    total_execution_nanos: Arc<AtomicU64>,
    max_execution_nanos: Arc<AtomicU64>,
    min_execution_nanos: Arc<AtomicU64>,
    total_docs_evaluated: Arc<AtomicU64>,
    total_hits_returned: Arc<AtomicU64>,
    timeout_count: Arc<AtomicU64>,

    /// Start time for the collector.
    start_time: Instant,
}

impl SearchMetricsCollector {
    /// Create a new metrics collector.
    pub fn new() -> Self {
        Self {
            total_searches: Arc::new(AtomicU64::new(0)),
            successful_searches: Arc::new(AtomicU64::new(0)),
            failed_searches: Arc::new(AtomicU64::new(0)),
            total_execution_nanos: Arc::new(AtomicU64::new(0)),
            max_execution_nanos: Arc::new(AtomicU64::new(0)),
            min_execution_nanos: Arc::new(AtomicU64::new(u64::MAX)),
            total_docs_evaluated: Arc::new(AtomicU64::new(0)),
            total_hits_returned: Arc::new(AtomicU64::new(0)),
            timeout_count: Arc::new(AtomicU64::new(0)),
            start_time: Instant::now(),
        }
    }

    /// Record a search execution.
    pub fn record_search(
        &self,
        execution_time: Duration,
        success: bool,
        docs_evaluated: u64,
        hits_returned: u64,
        timed_out: bool,
    ) {
        self.total_searches.fetch_add(1, Ordering::Relaxed);

        if success {
            self.successful_searches.fetch_add(1, Ordering::Relaxed);
        } else {
            self.failed_searches.fetch_add(1, Ordering::Relaxed);
        }

        if timed_out {
            self.timeout_count.fetch_add(1, Ordering::Relaxed);
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

        self.total_docs_evaluated
            .fetch_add(docs_evaluated, Ordering::Relaxed);
        self.total_hits_returned
            .fetch_add(hits_returned, Ordering::Relaxed);
    }

    /// Get the current metrics snapshot.
    pub fn snapshot(&self) -> SearchMetrics {
        let total_searches = self.total_searches.load(Ordering::Relaxed);
        let total_nanos = self.total_execution_nanos.load(Ordering::Relaxed);

        let avg_nanos = if total_searches > 0 {
            total_nanos / total_searches
        } else {
            0
        };

        let min_nanos = self.min_execution_nanos.load(Ordering::Relaxed);
        let min_duration = if min_nanos == u64::MAX {
            Duration::ZERO
        } else {
            Duration::from_nanos(min_nanos)
        };

        SearchMetrics {
            total_searches,
            successful_searches: self.successful_searches.load(Ordering::Relaxed),
            failed_searches: self.failed_searches.load(Ordering::Relaxed),
            total_execution_time: Duration::from_nanos(total_nanos),
            avg_execution_time: Duration::from_nanos(avg_nanos),
            max_execution_time: Duration::from_nanos(
                self.max_execution_nanos.load(Ordering::Relaxed),
            ),
            min_execution_time: min_duration,
            total_docs_evaluated: self.total_docs_evaluated.load(Ordering::Relaxed),
            total_hits_returned: self.total_hits_returned.load(Ordering::Relaxed),
            timeout_count: self.timeout_count.load(Ordering::Relaxed),
            index_metrics: Vec::new(), // Would be populated from per-index collectors
        }
    }

    /// Reset all metrics.
    pub fn reset(&self) {
        self.total_searches.store(0, Ordering::Relaxed);
        self.successful_searches.store(0, Ordering::Relaxed);
        self.failed_searches.store(0, Ordering::Relaxed);
        self.total_execution_nanos.store(0, Ordering::Relaxed);
        self.max_execution_nanos.store(0, Ordering::Relaxed);
        self.min_execution_nanos.store(u64::MAX, Ordering::Relaxed);
        self.total_docs_evaluated.store(0, Ordering::Relaxed);
        self.total_hits_returned.store(0, Ordering::Relaxed);
        self.timeout_count.store(0, Ordering::Relaxed);
    }

    /// Get the uptime of this collector.
    pub fn uptime(&self) -> Duration {
        self.start_time.elapsed()
    }
}

impl Default for SearchMetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper for timing operations.
pub struct Timer {
    start: Instant,
}

impl Timer {
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

/// Metrics for monitoring memory usage.
#[derive(Debug, Default)]
pub struct MemoryMetrics {
    /// Current memory usage in bytes.
    pub current_usage: Arc<AtomicUsize>,

    /// Peak memory usage observed.
    pub peak_usage: Arc<AtomicUsize>,

    /// Number of allocations.
    pub allocation_count: Arc<AtomicU64>,
}

impl MemoryMetrics {
    /// Create new memory metrics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a memory allocation.
    pub fn record_allocation(&self, size: usize) {
        self.current_usage.fetch_add(size, Ordering::Relaxed);
        self.allocation_count.fetch_add(1, Ordering::Relaxed);

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
        let collector = SearchMetricsCollector::new();

        // Record some searches
        collector.record_search(Duration::from_millis(100), true, 1000, 10, false);
        collector.record_search(Duration::from_millis(50), true, 500, 5, false);
        collector.record_search(Duration::from_millis(200), false, 2000, 0, true);

        let metrics = collector.snapshot();

        assert_eq!(metrics.total_searches, 3);
        assert_eq!(metrics.successful_searches, 2);
        assert_eq!(metrics.failed_searches, 1);
        assert_eq!(metrics.timeout_count, 1);
        assert_eq!(metrics.total_docs_evaluated, 3500);
        assert_eq!(metrics.total_hits_returned, 15);

        // Check timing metrics
        assert_eq!(metrics.min_execution_time, Duration::from_millis(50));
        assert_eq!(metrics.max_execution_time, Duration::from_millis(200));
        assert!(metrics.avg_execution_time >= Duration::from_millis(100));
        assert!(metrics.avg_execution_time <= Duration::from_millis(120));
    }

    #[test]
    fn test_timer() {
        let timer = Timer::start();
        std::thread::sleep(Duration::from_millis(10));
        let elapsed = timer.stop();

        assert!(elapsed >= Duration::from_millis(10));
        assert!(elapsed < Duration::from_millis(100));
    }

    #[test]
    fn test_memory_metrics() {
        let metrics = MemoryMetrics::new();

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
}
