//! Search task definitions for parallel execution.

use crate::error::{Result, SarissaError};
use crate::query::{Query, SearchResults};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// A search task to be executed on a specific index.
#[derive(Debug)]
pub struct SearchTask {
    /// Unique identifier for this task.
    pub task_id: String,

    /// Index ID this task is targeting.
    pub index_id: String,

    /// Query to execute.
    pub query: Box<dyn Query>,

    /// Maximum number of results to collect.
    pub max_docs: usize,

    /// Minimum score threshold.
    pub min_score: Option<f32>,

    /// Timeout for this specific task.
    pub timeout: Option<Duration>,

    /// Whether to load full documents.
    pub load_documents: bool,

    /// Priority level for task scheduling.
    pub priority: TaskPriority,
}

impl SearchTask {
    /// Create a new search task.
    pub fn new(index_id: String, query: Box<dyn Query>, max_docs: usize) -> Self {
        let task_id = format!("{}_{}", index_id, uuid::Uuid::new_v4());
        Self {
            task_id,
            index_id,
            query,
            max_docs,
            min_score: None,
            timeout: None,
            load_documents: true,
            priority: TaskPriority::Normal,
        }
    }

    /// Set the minimum score threshold.
    pub fn with_min_score(mut self, min_score: f32) -> Self {
        self.min_score = Some(min_score);
        self
    }

    /// Set the timeout for this task.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Set whether to load documents.
    pub fn with_load_documents(mut self, load: bool) -> Self {
        self.load_documents = load;
        self
    }

    /// Set the priority level.
    pub fn with_priority(mut self, priority: TaskPriority) -> Self {
        self.priority = priority;
        self
    }
}

/// Priority levels for task scheduling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    /// Low priority tasks.
    Low = 0,

    /// Normal priority tasks.
    Normal = 1,

    /// High priority tasks.
    High = 2,

    /// Critical priority tasks (executed first).
    Critical = 3,
}

/// Result of executing a search task.
#[derive(Debug)]
pub struct TaskResult {
    /// Task ID this result belongs to.
    pub task_id: String,

    /// Index ID this result came from.
    pub index_id: String,

    /// Search results if successful.
    pub results: Option<SearchResults>,

    /// Error if the task failed.
    pub error: Option<SarissaError>,

    /// Execution time for this task.
    pub execution_time: Duration,

    /// Whether the task timed out.
    pub timed_out: bool,

    /// Additional metrics.
    pub metrics: TaskMetrics,
}

impl TaskResult {
    /// Create a successful task result.
    pub fn success(
        task_id: String,
        index_id: String,
        results: SearchResults,
        execution_time: Duration,
    ) -> Self {
        Self {
            task_id,
            index_id,
            results: Some(results),
            error: None,
            execution_time,
            timed_out: false,
            metrics: TaskMetrics::default(),
        }
    }

    /// Create a failed task result.
    pub fn failure(
        task_id: String,
        index_id: String,
        error: SarissaError,
        execution_time: Duration,
    ) -> Self {
        Self {
            task_id,
            index_id,
            results: None,
            error: Some(error),
            execution_time,
            timed_out: false,
            metrics: TaskMetrics::default(),
        }
    }

    /// Create a timeout task result.
    pub fn timeout(task_id: String, index_id: String, execution_time: Duration) -> Self {
        Self {
            task_id,
            index_id,
            results: None,
            error: Some(SarissaError::timeout("Search task timed out")),
            execution_time,
            timed_out: true,
            metrics: TaskMetrics::default(),
        }
    }

    /// Check if the task was successful.
    pub fn is_success(&self) -> bool {
        self.results.is_some() && self.error.is_none()
    }

    /// Get the number of hits if successful.
    pub fn hit_count(&self) -> usize {
        self.results.as_ref().map(|r| r.hits.len()).unwrap_or(0)
    }
}

/// Metrics collected during task execution.
#[derive(Debug, Default, Clone)]
pub struct TaskMetrics {
    /// Number of documents evaluated.
    pub docs_evaluated: u64,

    /// Number of segments accessed.
    pub segments_accessed: u32,

    /// Memory used during execution (bytes).
    pub memory_used: usize,

    /// Time spent in query parsing.
    pub parse_time: Option<Duration>,

    /// Time spent in actual search.
    pub search_time: Option<Duration>,

    /// Time spent loading documents.
    pub load_time: Option<Duration>,
}

/// Status of a task during execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskStatus {
    /// Task is queued for execution.
    Queued,

    /// Task is currently executing.
    Running,

    /// Task completed successfully.
    Completed,

    /// Task failed with an error.
    Failed,

    /// Task was cancelled.
    Cancelled,

    /// Task timed out.
    TimedOut,
}

/// Handle for tracking a task's progress.
pub struct TaskHandle {
    /// Task ID.
    pub task_id: String,

    /// Current status.
    pub status: Arc<std::sync::RwLock<TaskStatus>>,

    /// Start time if running.
    pub start_time: Arc<std::sync::RwLock<Option<Instant>>>,

    /// Cancellation token.
    pub cancel_token: Arc<std::sync::atomic::AtomicBool>,
}

impl TaskHandle {
    /// Create a new task handle.
    pub fn new(task_id: String) -> Self {
        Self {
            task_id,
            status: Arc::new(std::sync::RwLock::new(TaskStatus::Queued)),
            start_time: Arc::new(std::sync::RwLock::new(None)),
            cancel_token: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// Get the current status.
    pub fn status(&self) -> Result<TaskStatus> {
        self.status
            .read()
            .map(|s| *s)
            .map_err(|_| SarissaError::internal("Failed to read task status"))
    }

    /// Set the status.
    pub fn set_status(&self, status: TaskStatus) -> Result<()> {
        self.status
            .write()
            .map(|mut s| *s = status)
            .map_err(|_| SarissaError::internal("Failed to write task status"))
    }

    /// Mark the task as started.
    pub fn start(&self) -> Result<()> {
        self.set_status(TaskStatus::Running)?;
        self.start_time
            .write()
            .map(|mut t| *t = Some(Instant::now()))
            .map_err(|_| SarissaError::internal("Failed to set start time"))
    }

    /// Cancel the task.
    pub fn cancel(&self) {
        self.cancel_token
            .store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// Check if the task is cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.cancel_token.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Get the elapsed time if the task is running.
    pub fn elapsed(&self) -> Result<Option<Duration>> {
        self.start_time
            .read()
            .map(|t| t.map(|start| start.elapsed()))
            .map_err(|_| SarissaError::internal("Failed to read start time"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::TermQuery;

    #[test]
    fn test_search_task_creation() {
        let query = Box::new(TermQuery::new("field", "value"));
        let task = SearchTask::new("index1".to_string(), query, 100)
            .with_min_score(0.5)
            .with_timeout(Duration::from_secs(10))
            .with_priority(TaskPriority::High);

        assert_eq!(task.index_id, "index1");
        assert_eq!(task.max_docs, 100);
        assert_eq!(task.min_score, Some(0.5));
        assert_eq!(task.timeout, Some(Duration::from_secs(10)));
        assert_eq!(task.priority, TaskPriority::High);
    }

    #[test]
    fn test_task_priority_ordering() {
        assert!(TaskPriority::Critical > TaskPriority::High);
        assert!(TaskPriority::High > TaskPriority::Normal);
        assert!(TaskPriority::Normal > TaskPriority::Low);
    }

    #[test]
    fn test_task_result_creation() {
        let task_id = "task1".to_string();
        let index_id = "index1".to_string();

        // Test success result
        let results = SearchResults {
            hits: vec![],
            total_hits: 0,
            max_score: 0.0,
        };
        let success = TaskResult::success(
            task_id.clone(),
            index_id.clone(),
            results,
            Duration::from_millis(100),
        );
        assert!(success.is_success());
        assert_eq!(success.hit_count(), 0);

        // Test failure result
        let error = SarissaError::internal("Test error");
        let failure = TaskResult::failure(
            task_id.clone(),
            index_id.clone(),
            error,
            Duration::from_millis(50),
        );
        assert!(!failure.is_success());
        assert!(failure.error.is_some());

        // Test timeout result
        let timeout = TaskResult::timeout(task_id, index_id, Duration::from_secs(5));
        assert!(!timeout.is_success());
        assert!(timeout.timed_out);
    }

    #[test]
    fn test_task_handle() {
        let handle = TaskHandle::new("test_task".to_string());

        // Initial status
        assert_eq!(handle.status().unwrap(), TaskStatus::Queued);
        assert!(!handle.is_cancelled());

        // Start the task
        handle.start().unwrap();
        assert_eq!(handle.status().unwrap(), TaskStatus::Running);
        assert!(handle.elapsed().unwrap().is_some());

        // Cancel the task
        handle.cancel();
        assert!(handle.is_cancelled());

        // Update status
        handle.set_status(TaskStatus::Completed).unwrap();
        assert_eq!(handle.status().unwrap(), TaskStatus::Completed);
    }
}
