//! Parallel execution engine for vector search tasks.

#![allow(clippy::await_holding_lock)]

use super::{
    LoadBalancingStrategy, ParallelSearchStats, ParallelVectorSearchConfig, VectorResultMerger,
};
use crate::error::{Result, SarissaError};
use crate::vector::Vector;
use crate::vector::types::{VectorSearchConfig, VectorSearchResults};
use crate::vector_search::{AdvancedSearchConfig, VectorSearchEngine};
use rayon::ThreadPool;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

/// Task for parallel vector search execution.
#[derive(Debug, Clone)]
pub struct SearchTask {
    /// Unique task identifier.
    pub task_id: usize,
    /// Query vector.
    pub query: Vector,
    /// Search configuration.
    pub config: VectorSearchConfig,
    /// Optional advanced configuration.
    pub advanced_config: Option<AdvancedSearchConfig>,
}

/// Result of a search task execution.
#[derive(Debug)]
pub struct SearchTaskResult {
    /// Task identifier.
    pub task_id: usize,
    /// Search results.
    pub result: Result<VectorSearchResults>,
    /// Execution time in milliseconds.
    pub execution_time_ms: f64,
    /// Number of vectors examined.
    pub vectors_examined: usize,
    /// Cache hit indicator.
    pub cache_hit: bool,
}

/// Parallel executor for vector search operations.
pub struct ParallelVectorSearchExecutor {
    config: ParallelVectorSearchConfig,
    thread_pool: Arc<ThreadPool>,
    search_engines: Vec<Arc<Mutex<VectorSearchEngine>>>,
    _result_merger: VectorResultMerger,
    result_cache: Arc<RwLock<HashMap<String, VectorSearchResults>>>,
    stats: Arc<Mutex<ParallelSearchStats>>,
    next_engine_index: Arc<Mutex<usize>>,
}

impl ParallelVectorSearchExecutor {
    /// Create a new parallel vector search executor.
    pub fn new(config: ParallelVectorSearchConfig) -> Result<Self> {
        let thread_pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(config.num_threads)
                .build()
                .map_err(|e| {
                    SarissaError::InvalidOperation(format!("Failed to create thread pool: {e}"))
                })?,
        );

        Self::with_thread_pool(config, thread_pool)
    }

    /// Create an executor with an existing thread pool.
    pub fn with_thread_pool(
        config: ParallelVectorSearchConfig,
        thread_pool: Arc<ThreadPool>,
    ) -> Result<Self> {
        // Create multiple search engines for parallel execution
        let mut search_engines = Vec::new();
        for _ in 0..config.num_threads {
            let engine = VectorSearchEngine::new(config.base_config.clone())?;
            search_engines.push(Arc::new(Mutex::new(engine)));
        }

        let result_merger = VectorResultMerger::new(config.merge_strategy)?;
        let result_cache = Arc::new(RwLock::new(HashMap::new()));
        let stats = Arc::new(Mutex::new(ParallelSearchStats::default()));
        let next_engine_index = Arc::new(Mutex::new(0));

        Ok(Self {
            config,
            thread_pool,
            search_engines,
            _result_merger: result_merger,
            result_cache,
            stats,
            next_engine_index,
        })
    }

    /// Execute a single search query.
    pub async fn search(
        &self,
        query: &Vector,
        config: &VectorSearchConfig,
    ) -> Result<VectorSearchResults> {
        let start_time = Instant::now();

        // Check cache first
        if self.config.enable_result_caching {
            let cache_key = self.compute_cache_key(query, config);
            if let Some(cached_result) = self.get_cached_result(&cache_key) {
                self.update_cache_stats(true);
                return Ok(cached_result);
            }
        }

        // Execute search
        let engine = self.select_search_engine()?;
        let result = {
            let mut engine_guard = engine.lock().unwrap();
            engine_guard.search(query, config).await?
        };

        // Update cache
        if self.config.enable_result_caching {
            let cache_key = self.compute_cache_key(query, config);
            self.update_cache(cache_key, result.clone());
        }

        // Update statistics
        let search_time = start_time.elapsed().as_secs_f64() * 1000.0;
        self.update_search_stats(search_time, result.candidates_examined, false);

        Ok(result)
    }

    /// Execute multiple search queries in parallel.
    pub async fn batch_search(
        &self,
        queries: &[Vector],
        config: &VectorSearchConfig,
    ) -> Result<Vec<VectorSearchResults>> {
        let start_time = Instant::now();

        // Create search tasks
        let tasks: Vec<SearchTask> = queries
            .iter()
            .enumerate()
            .map(|(i, query)| SearchTask {
                task_id: i,
                query: query.clone(),
                config: config.clone(),
                advanced_config: None,
            })
            .collect();

        // Execute tasks in parallel
        let results = self.execute_search_tasks(tasks).await?;

        // Extract results in order
        let mut search_results = vec![None; queries.len()];
        for task_result in results {
            if let Ok(result) = task_result.result {
                search_results[task_result.task_id] = Some(result);
            }
        }

        // Convert to Vec, handling any missing results
        let final_results: Result<Vec<VectorSearchResults>> = search_results
            .into_iter()
            .enumerate()
            .map(|(i, opt_result)| {
                opt_result.ok_or_else(|| {
                    SarissaError::InvalidOperation(format!("Search failed for query {i}"))
                })
            })
            .collect();

        // Update batch statistics
        let batch_time = start_time.elapsed().as_secs_f64() * 1000.0;
        self.update_batch_stats(batch_time, queries.len());

        final_results
    }

    /// Execute advanced search with multiple strategies.
    pub async fn advanced_search(
        &self,
        query: &Vector,
        config: &AdvancedSearchConfig,
    ) -> Result<VectorSearchResults> {
        let engine = self.select_search_engine()?;
        let explained_results = {
            let mut engine_guard = engine.lock().unwrap();
            engine_guard.advanced_search(query, config).await?
        };
        Ok(explained_results.results)
    }

    /// Load index for all search engines.
    pub async fn load_index(&mut self, index_path: &str) -> Result<()> {
        for engine in &self.search_engines {
            {
                let mut engine_guard = engine.lock().unwrap();
                engine_guard.load_index(index_path).await?;
            }
        }
        Ok(())
    }

    /// Warm up all search engines.
    pub async fn warmup(&mut self) -> Result<()> {
        for engine in &self.search_engines {
            {
                let mut engine_guard = engine.lock().unwrap();
                engine_guard.warmup().await?;
            }
        }
        Ok(())
    }

    /// Get search statistics.
    pub fn stats(&self) -> ParallelSearchStats {
        self.stats.lock().unwrap().clone()
    }

    /// Clear result cache.
    pub fn clear_cache(&self) {
        let mut cache = self.result_cache.write().unwrap();
        cache.clear();
    }

    /// Execute search tasks in parallel.
    async fn execute_search_tasks(&self, tasks: Vec<SearchTask>) -> Result<Vec<SearchTaskResult>> {
        let task_count = tasks.len();
        let mut results = Vec::with_capacity(task_count);

        // Process tasks in batches to control parallelism
        let batch_size = self
            .config
            .batch_size
            .min(self.config.max_concurrent_searches);

        for task_batch in tasks.chunks(batch_size) {
            let batch_results = self.execute_task_batch(task_batch.to_vec()).await?;
            results.extend(batch_results);
        }

        Ok(results)
    }

    /// Execute a batch of tasks in parallel.
    async fn execute_task_batch(&self, tasks: Vec<SearchTask>) -> Result<Vec<SearchTaskResult>> {
        let task_count = tasks.len();
        let results_arc = Arc::new(std::sync::Mutex::new(Vec::with_capacity(task_count)));

        // Use thread pool scope for parallel execution
        self.thread_pool.scope(|scope| {
            for task in tasks {
                let engine = match self.select_search_engine() {
                    Ok(engine) => engine,
                    Err(_) => continue,
                };

                let results_clone = Arc::clone(&results_arc);
                scope.spawn(move |_| {
                    let result = self.execute_single_task(task, engine);
                    results_clone.lock().unwrap().push(result);
                });
            }
        });

        // Extract results from Arc<Mutex<Vec<_>>>
        let results = Arc::try_unwrap(results_arc)
            .map_err(|_| {
                SarissaError::InvalidOperation("Failed to unwrap results Arc".to_string())
            })?
            .into_inner()
            .map_err(|_| {
                SarissaError::InvalidOperation("Failed to unwrap results Mutex".to_string())
            })?;

        Ok(results)
    }

    /// Execute a single search task.
    fn execute_single_task(
        &self,
        task: SearchTask,
        engine: Arc<Mutex<VectorSearchEngine>>,
    ) -> SearchTaskResult {
        let start_time = Instant::now();

        // Check cache first
        let cache_hit = if self.config.enable_result_caching {
            let cache_key = self.compute_cache_key(&task.query, &task.config);
            if let Some(cached_result) = self.get_cached_result(&cache_key) {
                return SearchTaskResult {
                    task_id: task.task_id,
                    result: Ok(cached_result),
                    execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
                    vectors_examined: 0,
                    cache_hit: true,
                };
            }
            false
        } else {
            false
        };

        // Execute search
        // For this synchronous context, we'll use a blocking approach
        // In a real implementation, you might want to use async/await properly
        #[allow(clippy::await_holding_lock)]
        let search_result = futures::executor::block_on(async {
            let mut engine_guard = engine.lock().unwrap();
            engine_guard.search(&task.query, &task.config).await
        });

        let execution_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;
        let vectors_examined = match &search_result {
            Ok(results) => results.candidates_examined,
            Err(_) => 0,
        };

        // Update cache if successful
        if let Ok(ref results) = search_result {
            if self.config.enable_result_caching {
                let cache_key = self.compute_cache_key(&task.query, &task.config);
                self.update_cache(cache_key, results.clone());
            }
        }

        SearchTaskResult {
            task_id: task.task_id,
            result: search_result,
            execution_time_ms,
            vectors_examined,
            cache_hit,
        }
    }

    /// Select a search engine based on load balancing strategy.
    fn select_search_engine(&self) -> Result<Arc<Mutex<VectorSearchEngine>>> {
        match self.config.load_balancing {
            LoadBalancingStrategy::RoundRobin => {
                let mut next_index = self.next_engine_index.lock().unwrap();
                let index = *next_index;
                *next_index = (index + 1) % self.search_engines.len();
                Ok(self.search_engines[index].clone())
            }
            LoadBalancingStrategy::Random => {
                use rand::Rng;
                let mut rng = rand::rng();
                let index = rng.random_range(0..self.search_engines.len());
                Ok(self.search_engines[index].clone())
            }
            LoadBalancingStrategy::LeastLoaded | LoadBalancingStrategy::QueryAware => {
                // For now, fallback to round-robin
                // TODO: Implement actual load balancing
                self.select_search_engine()
            }
        }
    }

    /// Compute cache key for a search query.
    fn compute_cache_key(&self, query: &Vector, config: &VectorSearchConfig) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();

        // Hash query vector
        for &value in &query.data {
            value.to_bits().hash(&mut hasher);
        }

        // Hash config
        config.top_k.hash(&mut hasher);
        config.min_similarity.to_bits().hash(&mut hasher);
        config.include_scores.hash(&mut hasher);
        config.include_vectors.hash(&mut hasher);

        format!("{:x}", hasher.finish())
    }

    /// Get cached result.
    fn get_cached_result(&self, cache_key: &str) -> Option<VectorSearchResults> {
        let cache = self.result_cache.read().unwrap();
        cache.get(cache_key).cloned()
    }

    /// Update cache with new result.
    fn update_cache(&self, cache_key: String, result: VectorSearchResults) {
        let mut cache = self.result_cache.write().unwrap();
        if cache.len() < self.config.cache_size_limit {
            cache.insert(cache_key, result);
        }
    }

    /// Update search statistics.
    fn update_search_stats(&self, search_time_ms: f64, vectors_examined: usize, cache_hit: bool) {
        let mut stats = self.stats.lock().unwrap();
        stats.total_searches += 1;

        // Update average search time
        let total = stats.total_searches as f64;
        stats.avg_search_time_ms =
            (stats.avg_search_time_ms * (total - 1.0) + search_time_ms) / total;

        // Update cache hit rate
        if cache_hit {
            stats.cache_hit_rate =
                (stats.cache_hit_rate * (total as f32 - 1.0) + 1.0) / total as f32;
        } else {
            stats.cache_hit_rate = (stats.cache_hit_rate * (total as f32 - 1.0)) / total as f32;
        }

        stats.total_vectors_searched += vectors_examined as u64;
    }

    /// Update batch search statistics.
    fn update_batch_stats(&self, batch_time_ms: f64, _query_count: usize) {
        let mut stats = self.stats.lock().unwrap();
        stats.total_batch_searches += 1;

        let total = stats.total_batch_searches as f64;
        stats.avg_batch_time_ms = (stats.avg_batch_time_ms * (total - 1.0) + batch_time_ms) / total;
    }

    /// Update cache statistics.
    fn update_cache_stats(&self, cache_hit: bool) {
        let mut stats = self.stats.lock().unwrap();
        stats.total_searches += 1;

        let total = stats.total_searches as f32;
        if cache_hit {
            stats.cache_hit_rate = (stats.cache_hit_rate * (total - 1.0) + 1.0) / total;
        } else {
            stats.cache_hit_rate = (stats.cache_hit_rate * (total - 1.0)) / total;
        }
    }
}
