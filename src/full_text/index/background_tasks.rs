//! Background task management for segment operations.
//!
//! This module provides scheduling and execution of background tasks such as
//! segment merging, compaction, and optimization with proper resource management.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crossbeam_channel::{Receiver, Sender, bounded, unbounded};

use crate::error::{Result, SageError};
use crate::full_text::index::deletion::DeletionManager;
use crate::full_text::index::merge_engine::MergeEngine;
use crate::full_text::index::merge_policy::MergePolicy;
use crate::full_text::index::segment_manager::SegmentManager;

/// Type of background task.
#[derive(Debug, Clone, PartialEq)]
pub enum TaskType {
    /// Merge segments task.
    Merge {
        segment_ids: Vec<String>,
        priority: f64,
    },

    /// Compaction task for removing deleted documents.
    Compaction {
        segment_id: String,
        deletion_ratio: f64,
    },

    /// Index optimization task.
    Optimization {
        target_segments: Vec<String>,
        optimization_level: u8,
    },

    /// Cleanup task for removing obsolete files.
    Cleanup { file_paths: Vec<String> },

    /// Statistics update task.
    StatsUpdate,
}

/// Background task with metadata.
#[derive(Debug, Clone)]
pub struct BackgroundTask {
    /// Unique task ID.
    pub task_id: String,

    /// Task type and parameters.
    pub task_type: TaskType,

    /// Task priority (higher = more urgent).
    pub priority: f64,

    /// Timestamp when task was created.
    pub created_at: u64,

    /// Timestamp when task should be executed.
    pub scheduled_at: u64,

    /// Number of retry attempts.
    pub retry_count: u8,

    /// Maximum number of retries.
    pub max_retries: u8,

    /// Estimated duration in milliseconds.
    pub estimated_duration_ms: u64,

    /// Task metadata.
    pub metadata: Vec<(String, String)>,
}

impl BackgroundTask {
    /// Create a new background task.
    pub fn new(task_type: TaskType, priority: f64) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let task_id = format!("{now:016x}_{}", rand::random::<u32>());

        BackgroundTask {
            task_id,
            task_type,
            priority,
            created_at: now,
            scheduled_at: now,
            retry_count: 0,
            max_retries: 3,
            estimated_duration_ms: 10000, // Default 10 seconds
            metadata: Vec::new(),
        }
    }

    /// Check if task is ready to execute.
    pub fn is_ready(&self) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        now >= self.scheduled_at
    }

    /// Check if task has exceeded retry limit.
    pub fn is_failed(&self) -> bool {
        self.retry_count >= self.max_retries
    }

    /// Get task age in seconds.
    pub fn age_seconds(&self) -> u64 {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        now.saturating_sub(self.created_at)
    }
}

/// Status of task execution.
#[derive(Debug, Clone, PartialEq)]
pub enum TaskStatus {
    /// Task is pending execution.
    Pending,

    /// Task is currently running.
    Running,

    /// Task completed successfully.
    Completed,

    /// Task failed with error.
    Failed(String),

    /// Task was cancelled.
    Cancelled,
}

/// Result of task execution.
#[derive(Debug, Clone)]
pub struct TaskResult {
    /// Task ID.
    pub task_id: String,

    /// Execution status.
    pub status: TaskStatus,

    /// Execution time in milliseconds.
    pub execution_time_ms: u64,

    /// Number of items processed (segments, documents, etc.).
    pub items_processed: u64,

    /// Size of data processed in bytes.
    pub bytes_processed: u64,

    /// Error message if failed.
    pub error_message: Option<String>,

    /// Result metadata.
    pub metadata: Vec<(String, String)>,
}

/// Configuration for background task scheduler.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Number of worker threads.
    pub worker_threads: usize,

    /// Maximum number of pending tasks.
    pub max_pending_tasks: usize,

    /// Task execution timeout in seconds.
    pub task_timeout_secs: u64,

    /// Health check interval in seconds.
    pub health_check_interval_secs: u64,

    /// Maximum memory usage for background tasks (MB).
    pub max_memory_mb: u64,

    /// Enable task prioritization.
    pub enable_prioritization: bool,

    /// Enable task batching.
    pub enable_batching: bool,

    /// Batch size for similar tasks.
    pub batch_size: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        SchedulerConfig {
            worker_threads: 2,
            max_pending_tasks: 100,
            task_timeout_secs: 300, // 5 minutes
            health_check_interval_secs: 30,
            max_memory_mb: 512,
            enable_prioritization: true,
            enable_batching: true,
            batch_size: 5,
        }
    }
}

/// Statistics about task execution.
#[derive(Debug, Clone, Default)]
pub struct SchedulerStats {
    /// Total tasks submitted.
    pub tasks_submitted: u64,

    /// Tasks currently pending.
    pub tasks_pending: u64,

    /// Tasks currently running.
    pub tasks_running: u64,

    /// Tasks completed successfully.
    pub tasks_completed: u64,

    /// Tasks that failed.
    pub tasks_failed: u64,

    /// Tasks that were cancelled.
    pub tasks_cancelled: u64,

    /// Average execution time (milliseconds).
    pub avg_execution_time_ms: f64,

    /// Total bytes processed.
    pub total_bytes_processed: u64,

    /// Current memory usage (bytes).
    pub current_memory_usage: u64,

    /// Number of active worker threads.
    pub active_workers: u64,
}

/// Background task scheduler and executor.
#[derive(Debug)]
pub struct BackgroundScheduler {
    /// Configuration.
    config: SchedulerConfig,

    /// Task queue (priority queue).
    task_sender: Sender<BackgroundTask>,
    task_receiver: Receiver<BackgroundTask>,

    /// Result channel for completed tasks.
    result_sender: Sender<TaskResult>,
    result_receiver: Receiver<TaskResult>,

    /// Segment manager reference.
    segment_manager: Arc<SegmentManager>,

    /// Merge engine.
    merge_engine: Arc<MergeEngine>,

    /// Deletion manager.
    deletion_manager: Arc<DeletionManager>,

    /// Merge policy.
    merge_policy: Arc<dyn MergePolicy>,

    /// Running state.
    running: Arc<AtomicBool>,

    /// Statistics.
    stats: Arc<RwLock<SchedulerStats>>,

    /// Worker thread handles.
    workers: RwLock<Vec<thread::JoinHandle<()>>>,

    /// Task ID counter.
    task_counter: Arc<AtomicU64>,
}

impl BackgroundScheduler {
    /// Create a new background scheduler.
    pub fn new(
        config: SchedulerConfig,
        segment_manager: Arc<SegmentManager>,
        merge_engine: Arc<MergeEngine>,
        deletion_manager: Arc<DeletionManager>,
        merge_policy: Arc<dyn MergePolicy>,
    ) -> Result<Self> {
        let (task_sender, task_receiver) = bounded(config.max_pending_tasks);
        let (result_sender, result_receiver) = unbounded();

        Ok(BackgroundScheduler {
            config,
            task_sender,
            task_receiver,
            result_sender,
            result_receiver,
            segment_manager,
            merge_engine,
            deletion_manager,
            merge_policy,
            running: Arc::new(AtomicBool::new(false)),
            stats: Arc::new(RwLock::new(SchedulerStats::default())),
            workers: RwLock::new(Vec::new()),
            task_counter: Arc::new(AtomicU64::new(0)),
        })
    }

    /// Start the background scheduler.
    pub fn start(&self) -> Result<()> {
        if self.running.load(Ordering::Acquire) {
            return Err(SageError::index("Background scheduler already running"));
        }

        self.running.store(true, Ordering::Release);

        // Start worker threads
        let mut workers = self.workers.write().unwrap();
        for worker_id in 0..self.config.worker_threads {
            let worker = self.spawn_worker(worker_id)?;
            workers.push(worker);
        }

        // Start health check thread
        let health_checker = self.spawn_health_checker()?;
        workers.push(health_checker);

        Ok(())
    }

    /// Stop the background scheduler.
    pub fn stop(&self) -> Result<()> {
        self.running.store(false, Ordering::Release);

        // Wait for workers to finish
        let mut workers = self.workers.write().unwrap();
        while let Some(worker) = workers.pop() {
            let _ = worker.join();
        }

        Ok(())
    }

    /// Submit a task for background execution.
    pub fn submit_task(&self, task: BackgroundTask) -> Result<()> {
        if !self.running.load(Ordering::Acquire) {
            return Err(SageError::index("Background scheduler not running"));
        }

        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.tasks_submitted += 1;
            stats.tasks_pending += 1;
        }

        // Send task to queue
        self.task_sender
            .send(task)
            .map_err(|_| SageError::index("Failed to submit task to queue"))?;

        Ok(())
    }

    /// Submit a merge task.
    pub fn submit_merge_task(&self, segment_ids: Vec<String>, priority: f64) -> Result<()> {
        let task_type = TaskType::Merge {
            segment_ids,
            priority,
        };
        let task = BackgroundTask::new(task_type, priority);
        self.submit_task(task)
    }

    /// Submit a compaction task.
    pub fn submit_compaction_task(&self, segment_id: String, deletion_ratio: f64) -> Result<()> {
        let task_type = TaskType::Compaction {
            segment_id,
            deletion_ratio,
        };
        let task = BackgroundTask::new(task_type, 5.0 + (deletion_ratio * 10.0));
        self.submit_task(task)
    }

    /// Check for automatic merge triggers.
    pub fn check_auto_merge(&self) -> Result<()> {
        if !self.running.load(Ordering::Acquire) {
            return Ok(());
        }

        let segments = self.segment_manager.get_segments();

        if self.merge_policy.should_merge(&segments) {
            let candidates = self.merge_policy.select_merges(&segments);

            for candidate in candidates {
                self.submit_merge_task(candidate.segments, candidate.priority)?;
            }
        }

        Ok(())
    }

    /// Check for automatic compaction triggers.
    pub fn check_auto_compaction(&self) -> Result<()> {
        if !self.running.load(Ordering::Acquire) {
            return Ok(());
        }

        let candidates = self.deletion_manager.get_compaction_candidates();

        for segment_id in candidates {
            let deletion_ratio = self.deletion_manager.get_deletion_ratio(&segment_id);
            self.submit_compaction_task(segment_id, deletion_ratio)?;
        }

        Ok(())
    }

    /// Get task execution results.
    pub fn get_results(&self) -> Vec<TaskResult> {
        let mut results = Vec::new();

        while let Ok(result) = self.result_receiver.try_recv() {
            results.push(result);
        }

        results
    }

    /// Get current statistics.
    pub fn get_stats(&self) -> SchedulerStats {
        self.stats.read().unwrap().clone()
    }

    /// Spawn a worker thread.
    fn spawn_worker(&self, worker_id: usize) -> Result<thread::JoinHandle<()>> {
        let task_receiver = self.task_receiver.clone();
        let result_sender = self.result_sender.clone();
        let segment_manager = self.segment_manager.clone();
        let merge_engine = self.merge_engine.clone();
        let deletion_manager = self.deletion_manager.clone();
        let running = Arc::clone(&self.running);
        let stats = Arc::clone(&self.stats);
        let timeout = Duration::from_secs(self.config.task_timeout_secs);

        let handle = thread::Builder::new()
            .name(format!("bg-worker-{worker_id}"))
            .spawn(move || {
                while running.load(Ordering::Acquire) {
                    match task_receiver.recv_timeout(Duration::from_secs(1)) {
                        Ok(task) => {
                            let result = Self::execute_task(
                                task,
                                &segment_manager,
                                &merge_engine,
                                &deletion_manager,
                                timeout,
                            );

                            // Update statistics
                            {
                                let mut stats = stats.write().unwrap();
                                stats.tasks_pending = stats.tasks_pending.saturating_sub(1);

                                match result.status {
                                    TaskStatus::Completed => stats.tasks_completed += 1,
                                    TaskStatus::Failed(_) => stats.tasks_failed += 1,
                                    TaskStatus::Cancelled => stats.tasks_cancelled += 1,
                                    _ => {}
                                }

                                stats.total_bytes_processed += result.bytes_processed;

                                // Update average execution time
                                let total_completed = stats.tasks_completed + stats.tasks_failed;
                                if total_completed > 0 {
                                    stats.avg_execution_time_ms = (stats.avg_execution_time_ms
                                        * (total_completed - 1) as f64
                                        + result.execution_time_ms as f64)
                                        / total_completed as f64;
                                }
                            }

                            let _ = result_sender.send(result);
                        }
                        Err(_) => {
                            // Timeout or channel closed, continue
                        }
                    }
                }
            })?;

        Ok(handle)
    }

    /// Spawn health checker thread.
    fn spawn_health_checker(&self) -> Result<thread::JoinHandle<()>> {
        let running = Arc::clone(&self.running);
        let _segment_manager = self.segment_manager.clone();
        let _deletion_manager = self.deletion_manager.clone();
        let _merge_policy = Arc::clone(&self.merge_policy);
        let scheduler = self.clone();
        let interval = Duration::from_secs(self.config.health_check_interval_secs);

        let handle = thread::Builder::new()
            .name("bg-health-checker".to_string())
            .spawn(move || {
                while running.load(Ordering::Acquire) {
                    // Check for auto-merge conditions
                    let _ = scheduler.check_auto_merge();

                    // Check for auto-compaction conditions
                    let _ = scheduler.check_auto_compaction();

                    thread::sleep(interval);
                }
            })?;

        Ok(handle)
    }

    /// Execute a background task.
    fn execute_task(
        task: BackgroundTask,
        segment_manager: &SegmentManager,
        merge_engine: &MergeEngine,
        deletion_manager: &DeletionManager,
        _timeout: Duration,
    ) -> TaskResult {
        let start_time = SystemTime::now();
        let task_id = task.task_id.clone();

        let (status, items_processed, bytes_processed, error_message) = match task.task_type {
            TaskType::Merge { segment_ids, .. } => {
                Self::execute_merge_task(&segment_ids, segment_manager, merge_engine)
            }

            TaskType::Compaction { segment_id, .. } => {
                Self::execute_compaction_task(&segment_id, deletion_manager)
            }

            TaskType::Optimization {
                target_segments,
                optimization_level,
            } => Self::execute_optimization_task(&target_segments, optimization_level),

            TaskType::Cleanup { file_paths } => {
                Self::execute_cleanup_task(&file_paths, segment_manager)
            }

            TaskType::StatsUpdate => {
                Self::execute_stats_update_task(segment_manager, deletion_manager)
            }
        };

        let execution_time_ms = start_time.elapsed().unwrap_or_default().as_millis() as u64;

        TaskResult {
            task_id,
            status,
            execution_time_ms,
            items_processed,
            bytes_processed,
            error_message,
            metadata: Vec::new(),
        }
    }

    /// Execute merge task.
    fn execute_merge_task(
        segment_ids: &[String],
        _segment_manager: &SegmentManager,
        _merge_engine: &MergeEngine,
    ) -> (TaskStatus, u64, u64, Option<String>) {
        // TODO: Implement actual merge execution
        // This is a placeholder implementation

        let segments = _segment_manager.get_segments();
        let segments_to_merge: Vec<_> = segments
            .iter()
            .filter(|seg| segment_ids.contains(&seg.segment_info.segment_id))
            .collect();

        if segments_to_merge.is_empty() {
            return (
                TaskStatus::Failed("No segments found to merge".to_string()),
                0,
                0,
                Some("No segments found to merge".to_string()),
            );
        }

        let items_processed = segments_to_merge.len() as u64;
        let bytes_processed = segments_to_merge.iter().map(|s| s.size_bytes).sum();

        (
            TaskStatus::Completed,
            items_processed,
            bytes_processed,
            None,
        )
    }

    /// Execute compaction task.
    fn execute_compaction_task(
        segment_id: &str,
        deletion_manager: &DeletionManager,
    ) -> (TaskStatus, u64, u64, Option<String>) {
        let deleted_docs = deletion_manager.get_deleted_docs(segment_id);
        let items_processed = deleted_docs.len() as u64;

        // TODO: Implement actual compaction

        (TaskStatus::Completed, items_processed, 0, None)
    }

    /// Execute optimization task.
    fn execute_optimization_task(
        target_segments: &[String],
        _optimization_level: u8,
    ) -> (TaskStatus, u64, u64, Option<String>) {
        // TODO: Implement optimization
        (TaskStatus::Completed, target_segments.len() as u64, 0, None)
    }

    /// Execute cleanup task.
    fn execute_cleanup_task(
        file_paths: &[String],
        _segment_manager: &SegmentManager,
    ) -> (TaskStatus, u64, u64, Option<String>) {
        // TODO: Implement cleanup
        (TaskStatus::Completed, file_paths.len() as u64, 0, None)
    }

    /// Execute stats update task.
    fn execute_stats_update_task(
        _segment_manager: &SegmentManager,
        _deletion_manager: &DeletionManager,
    ) -> (TaskStatus, u64, u64, Option<String>) {
        // Update statistics
        let _stats = _segment_manager.get_stats();
        let _del_stats = _deletion_manager.get_stats();

        (TaskStatus::Completed, 1, 0, None)
    }
}

// Clone implementation needed for health checker
impl Clone for BackgroundScheduler {
    fn clone(&self) -> Self {
        // This is a shallow clone for the health checker
        // Not all fields are properly cloned, but it's sufficient for our use case
        BackgroundScheduler {
            config: self.config.clone(),
            task_sender: self.task_sender.clone(),
            task_receiver: self.task_receiver.clone(),
            result_sender: self.result_sender.clone(),
            result_receiver: self.result_receiver.clone(),
            segment_manager: self.segment_manager.clone(),
            merge_engine: self.merge_engine.clone(),
            deletion_manager: self.deletion_manager.clone(),
            merge_policy: Arc::clone(&self.merge_policy),
            running: Arc::clone(&self.running),
            stats: Arc::clone(&self.stats),
            workers: RwLock::new(Vec::new()), // Empty for clone
            task_counter: Arc::clone(&self.task_counter),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(dead_code)]
    #[test]
    fn test_background_task_creation() {
        let task_type = TaskType::Merge {
            segment_ids: vec!["seg1".to_string(), "seg2".to_string()],
            priority: 5.0,
        };

        let task = BackgroundTask::new(task_type, 10.0);

        assert_eq!(task.priority, 10.0);
        assert_eq!(task.retry_count, 0);
        assert_eq!(task.max_retries, 3);
        assert!(task.is_ready());
        assert!(!task.is_failed());
    }

    #[test]
    fn test_task_status_checks() {
        let mut task = BackgroundTask::new(TaskType::StatsUpdate, 1.0);

        // Test ready status
        assert!(task.is_ready());

        // Test future scheduling
        task.scheduled_at += 3600; // 1 hour in future
        assert!(!task.is_ready());

        // Test failure status
        task.retry_count = 5;
        assert!(task.is_failed());
    }

    #[test]
    fn test_scheduler_config_default() {
        let config = SchedulerConfig::default();

        assert_eq!(config.worker_threads, 2);
        assert_eq!(config.max_pending_tasks, 100);
        assert_eq!(config.task_timeout_secs, 300);
        assert!(config.enable_prioritization);
        assert!(config.enable_batching);
    }
}
