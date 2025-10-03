//! Main parallel search engine implementation.

use std::sync::Arc;
use std::time::Instant;

use rayon::{ThreadPool, ThreadPoolBuilder};

use crate::error::{Result, SarissaError};
use crate::index::reader::IndexReader;
use crate::parallel_search::config::{ParallelSearchConfig, SearchOptions};
use crate::parallel_search::index_manager::{IndexHandle, IndexManager};
use crate::parallel_search::merger::MergerFactory;
use crate::parallel_search::metrics::{SearchMetricsCollector, Timer};
use crate::parallel_search::search_task::{SearchTask, TaskHandle, TaskResult, TaskStatus};
use crate::query::{Query, SearchResults};
use crate::search::{Search, SearchRequest};

/// Parallel search engine for executing queries across multiple indices.
pub struct ParallelSearchEngine {
    /// Configuration for the engine.
    config: ParallelSearchConfig,

    /// Index manager.
    index_manager: Arc<IndexManager>,

    /// Thread pool for parallel execution.
    thread_pool: Arc<ThreadPool>,

    /// Metrics collector.
    metrics: Arc<SearchMetricsCollector>,
}

impl ParallelSearchEngine {
    /// Create a new parallel search engine.
    pub fn new(config: ParallelSearchConfig) -> Result<Self> {
        let thread_pool_size = config.thread_pool_size.unwrap_or_else(num_cpus::get);

        let thread_pool = ThreadPoolBuilder::new()
            .num_threads(thread_pool_size)
            .thread_name(|i| format!("parallel-search-{i}"))
            .build()
            .map_err(|e| SarissaError::internal(format!("Failed to create thread pool: {e}")))?;

        Ok(Self {
            config,
            index_manager: Arc::new(IndexManager::new()),
            thread_pool: Arc::new(thread_pool),
            metrics: Arc::new(SearchMetricsCollector::new()),
        })
    }

    /// Add an index to the engine.
    pub fn add_index(&self, id: String, reader: Box<dyn IndexReader>, weight: f32) -> Result<()> {
        let handle = IndexHandle::new(id, Arc::from(reader)).with_weight(weight);
        self.index_manager.add_index(handle)
    }

    /// Remove an index from the engine.
    pub fn remove_index(&self, id: &str) -> Result<()> {
        self.index_manager.remove_index(id)?;
        Ok(())
    }

    /// Update the weight of an index.
    pub fn update_weight(&self, id: &str, weight: f32) -> Result<()> {
        self.index_manager.update_weight(id, weight)
    }

    /// Get the number of indices.
    pub fn index_count(&self) -> Result<usize> {
        self.index_manager.len()
    }

    /// Execute a search across all active indices.
    pub fn search(&self, query: Box<dyn Query>, options: SearchOptions) -> Result<SearchResults> {
        let timer = Timer::start();

        // Get active indices
        let indices = self.index_manager.get_active_indices()?;
        if indices.is_empty() {
            return Ok(SearchResults {
                hits: Vec::new(),
                total_hits: 0,
                max_score: 0.0,
            });
        }

        // Create search tasks
        let tasks = self.create_search_tasks(query.as_ref(), &indices, &options)?;

        // Execute tasks in parallel
        let results = self.execute_tasks_parallel(tasks, &options)?;

        // Check if we have any successful results
        let successful_results = results.iter().filter(|r| r.is_success()).count();

        if successful_results == 0 && !self.config.allow_partial_results {
            return Err(SarissaError::internal("All search tasks failed"));
        }

        // Create merger based on strategy
        let merge_strategy = options
            .merge_strategy
            .unwrap_or(self.config.default_merge_strategy);
        let merger = MergerFactory::create(merge_strategy);

        // Merge results
        let merged_results = merger.merge(results, options.max_docs, options.min_score)?;

        // Record metrics
        if self.config.enable_metrics && options.collect_metrics {
            let execution_time = timer.stop();
            self.metrics.record_search(
                execution_time,
                true,
                merged_results.total_hits,
                merged_results.hits.len() as u64,
                false,
            );
        }

        Ok(merged_results)
    }

    /// Create search tasks for each index.
    fn create_search_tasks(
        &self,
        query: &dyn Query,
        indices: &[IndexHandle],
        options: &SearchOptions,
    ) -> Result<Vec<SearchTask>> {
        let mut tasks = Vec::with_capacity(indices.len());

        for index in indices {
            // Clone the query for each task
            let query_clone = query.clone_box();
            let task = SearchTask::new(
                index.id.clone(),
                query_clone,
                self.config.max_results_per_index.min(options.max_docs),
            )
            .with_min_score(options.min_score.unwrap_or(0.0))
            .with_timeout(options.timeout.unwrap_or(self.config.default_timeout))
            .with_load_documents(options.load_documents);

            tasks.push(task);
        }

        Ok(tasks)
    }

    /// Execute tasks in parallel.
    fn execute_tasks_parallel(
        &self,
        tasks: Vec<SearchTask>,
        options: &SearchOptions,
    ) -> Result<Vec<TaskResult>> {
        let num_tasks = tasks.len();
        let (tx, rx) = std::sync::mpsc::channel();
        let timeout = options.timeout.unwrap_or(self.config.default_timeout);

        // Create task handles
        let handles: Vec<_> = tasks
            .iter()
            .map(|task| Arc::new(TaskHandle::new(task.task_id.clone())))
            .collect();

        // Submit tasks to thread pool
        for (task, handle) in tasks.into_iter().zip(handles.iter()) {
            let tx = tx.clone();
            let handle = Arc::clone(handle);
            let index_manager = Arc::clone(&self.index_manager);

            self.thread_pool.spawn(move || {
                let result = Self::execute_single_task(task, handle, index_manager);
                let _ = tx.send(result);
            });
        }

        // Drop the original sender so receiver knows when all tasks are done
        drop(tx);

        // Collect results with timeout
        let deadline = Instant::now() + timeout;
        let mut results = Vec::with_capacity(num_tasks);

        for _ in 0..num_tasks {
            let remaining = deadline.saturating_duration_since(Instant::now());
            match rx.recv_timeout(remaining) {
                Ok(result) => results.push(result),
                Err(_) => {
                    // Timeout or channel closed
                    // Cancel remaining tasks
                    for handle in &handles {
                        handle.cancel();
                    }

                    // Add timeout results for incomplete tasks
                    while results.len() < num_tasks {
                        results.push(TaskResult::timeout(String::new(), String::new(), timeout));
                    }
                    break;
                }
            }
        }

        Ok(results)
    }

    /// Execute a single search task.
    fn execute_single_task(
        task: SearchTask,
        handle: Arc<TaskHandle>,
        index_manager: Arc<IndexManager>,
    ) -> TaskResult {
        let timer = Timer::start();

        // Mark task as running
        if let Err(e) = handle.start() {
            return TaskResult::failure(task.task_id, task.index_id, e, timer.elapsed());
        }

        // Get the index
        let index_handle = match index_manager.get_index(&task.index_id) {
            Ok(h) => h,
            Err(e) => {
                return TaskResult::failure(task.task_id, task.index_id, e, timer.elapsed());
            }
        };

        // Create searcher for this index
        let searcher = crate::search::Searcher::from_arc(Arc::clone(&index_handle.reader));

        // Create search request
        let mut request = SearchRequest::new(task.query)
            .max_docs(task.max_docs)
            .min_score(task.min_score.unwrap_or(0.0))
            .load_documents(task.load_documents);

        if let Some(timeout) = task.timeout {
            request = request.timeout_ms(timeout.as_millis() as u64);
        }

        // Execute search with cancellation check
        let search_result = if handle.is_cancelled() {
            Err(SarissaError::cancelled("Task was cancelled"))
        } else {
            searcher.search(request)
        };

        // Create task result
        match search_result {
            Ok(results) => {
                let _ = handle.set_status(TaskStatus::Completed);
                TaskResult::success(task.task_id, task.index_id, results, timer.elapsed())
            }
            Err(e) => {
                let _ = handle.set_status(TaskStatus::Failed);
                TaskResult::failure(task.task_id, task.index_id, e, timer.elapsed())
            }
        }
    }

    /// Get current metrics snapshot.
    pub fn metrics(&self) -> crate::parallel_search::metrics::SearchMetrics {
        self.metrics.snapshot()
    }

    /// Reset metrics.
    pub fn reset_metrics(&self) {
        self.metrics.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::reader::BasicIndexReader;
    use crate::query::TermQuery;

    use crate::storage::{MemoryStorage, StorageConfig};

    fn create_test_reader() -> Box<dyn IndexReader> {
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        Box::new(BasicIndexReader::new(storage).unwrap())
    }

    #[test]
    fn test_engine_creation() {
        let config = ParallelSearchConfig::default();
        let engine = ParallelSearchEngine::new(config).unwrap();

        assert_eq!(engine.index_count().unwrap(), 0);
    }

    #[test]
    fn test_index_management() {
        let config = ParallelSearchConfig::default();
        let engine = ParallelSearchEngine::new(config).unwrap();

        // Add indices
        engine
            .add_index("index1".to_string(), create_test_reader(), 1.0)
            .unwrap();
        engine
            .add_index("index2".to_string(), create_test_reader(), 2.0)
            .unwrap();

        assert_eq!(engine.index_count().unwrap(), 2);

        // Update weight
        engine.update_weight("index1", 3.0).unwrap();

        // Remove index
        engine.remove_index("index2").unwrap();
        assert_eq!(engine.index_count().unwrap(), 1);
    }

    #[test]
    fn test_empty_search() {
        let config = ParallelSearchConfig::default();
        let engine = ParallelSearchEngine::new(config).unwrap();

        let query = Box::new(TermQuery::new("text", "test"));
        let options = SearchOptions::default();

        let results = engine.search(query, options).unwrap();

        assert_eq!(results.hits.len(), 0);
        assert_eq!(results.total_hits, 0);
    }

    #[test]
    fn test_search_with_indices() {
        let config = ParallelSearchConfig::default();
        let engine = ParallelSearchEngine::new(config).unwrap();

        // Add test indices
        engine
            .add_index("index1".to_string(), create_test_reader(), 1.0)
            .unwrap();
        engine
            .add_index("index2".to_string(), create_test_reader(), 1.0)
            .unwrap();

        let query = Box::new(TermQuery::new("text", "test"));
        let options = SearchOptions::new(10).with_timeout(std::time::Duration::from_secs(5));

        let results = engine.search(query, options).unwrap();

        // Should complete without errors (empty indices will return no results)
        assert_eq!(results.hits.len(), 0);
    }

    #[test]
    fn test_metrics_collection() {
        let config = ParallelSearchConfig {
            enable_metrics: true,
            ..Default::default()
        };

        let engine = ParallelSearchEngine::new(config).unwrap();
        engine
            .add_index("index1".to_string(), create_test_reader(), 1.0)
            .unwrap();

        let query = Box::new(TermQuery::new("text", "test"));
        let options = SearchOptions::default()
            .with_metrics(true)
            .with_timeout(std::time::Duration::from_secs(1));

        let _ = engine.search(query, options).unwrap();

        let metrics = engine.metrics();
        assert_eq!(metrics.total_searches, 1);
        assert_eq!(metrics.successful_searches, 1);
    }
}
