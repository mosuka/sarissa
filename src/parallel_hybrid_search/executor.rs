//! Parallel executor for hybrid search tasks.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use rayon::ThreadPool;

use super::config::ParallelHybridSearchConfig;
use super::engine::HybridIndexHandle;
use super::merger::ParallelHybridResultMerger;
use super::types::{
    HybridSearchTask, HybridSearchTaskResult, IndexSearchStats, ParallelHybridSearchResults,
    SearchTimeBreakdown,
};

use crate::error::{Result, SageError};
use crate::full_text::reader::IndexReader;
use crate::query::{Query, SearchResults};
use crate::vector::Vector;

/// Executor for parallel hybrid search operations.
pub struct ParallelHybridSearchExecutor {
    config: ParallelHybridSearchConfig,
    thread_pool: Arc<ThreadPool>,
    merger: ParallelHybridResultMerger,
    indices: Arc<Mutex<HashMap<String, HybridIndexHandle>>>,
}

impl ParallelHybridSearchExecutor {
    /// Create a new parallel hybrid search executor.
    pub fn new(config: ParallelHybridSearchConfig) -> Result<Self> {
        let thread_pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(config.num_threads)
                .build()
                .map_err(|e| {
                    SageError::InvalidOperation(format!("Failed to create thread pool: {e}"))
                })?,
        );

        let merger = ParallelHybridResultMerger::new(config.clone());

        Ok(Self {
            config,
            thread_pool,
            merger,
            indices: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Set indices for the executor.
    pub fn set_indices(&self, indices: Vec<HybridIndexHandle>) {
        let mut indices_map = self.indices.lock().unwrap();
        indices_map.clear();
        for handle in indices {
            indices_map.insert(handle.id.clone(), handle);
        }
    }

    /// Execute parallel hybrid search across multiple indices.
    pub async fn execute_parallel_search(
        &self,
        query_text: &str,
        keyword_query: Box<dyn Query>,
        query_vector: Option<Vector>,
        indices: Vec<HybridIndexHandle>,
        document_store: &HashMap<u64, HashMap<String, String>>,
    ) -> Result<ParallelHybridSearchResults> {
        let start_time = Instant::now();
        let indices_count = indices.len();

        // Create search tasks
        let tasks = self.create_search_tasks(query_text, keyword_query, query_vector, indices)?;

        // Execute tasks in parallel
        let (task_results, time_breakdown) = self.execute_tasks_parallel(tasks).await?;

        // Separate keyword and vector results
        let mut keyword_results = Vec::new();
        let mut vector_results = Vec::new();
        let mut total_keyword_matches = 0u64;
        let mut total_vector_matches = 0u64;
        let mut index_stats = Vec::new();

        for result in task_results {
            let mut stats = IndexSearchStats {
                index_id: result.index_id.clone(),
                keyword_matches: 0,
                vector_matches: 0,
                search_time_ms: result.execution_time_ms,
                timed_out: false,
                error: result.error.as_ref().map(|e| e.to_string()),
            };

            if let Some(kw_results) = result.keyword_results {
                stats.keyword_matches = kw_results.len() as u64;
                total_keyword_matches += stats.keyword_matches;
                keyword_results.push((result.index_id.clone(), kw_results));
            }

            if let Some(vec_results) = result.vector_results {
                stats.vector_matches = vec_results.len() as u64;
                total_vector_matches += stats.vector_matches;
                vector_results.push((result.index_id.clone(), vec_results));
            }

            index_stats.push(stats);
        }

        // Merge results
        let merge_start = Instant::now();
        let (merged_results, ranking_time) =
            self.merger
                .merge(keyword_results, vector_results, document_store);
        let merge_time = merge_start.elapsed().as_secs_f64() * 1000.0;

        // Update time breakdown
        let mut final_breakdown = time_breakdown;
        final_breakdown.merge_ms = merge_time;
        final_breakdown.ranking_ms = ranking_time;

        let total_time = start_time.elapsed().as_secs_f64() * 1000.0;

        Ok(ParallelHybridSearchResults {
            results: merged_results,
            total_keyword_matches,
            total_vector_matches,
            indices_searched: indices_count,
            search_time_ms: total_time,
            time_breakdown: final_breakdown,
            cache_stats: Default::default(), // Populated by engine
            index_stats,
        })
    }

    /// Create search tasks for each index.
    fn create_search_tasks(
        &self,
        query_text: &str,
        keyword_query: Box<dyn Query>,
        query_vector: Option<Vector>,
        indices: Vec<HybridIndexHandle>,
    ) -> Result<Vec<HybridSearchTask>> {
        let mut tasks = Vec::with_capacity(indices.len());

        for (i, index) in indices.into_iter().enumerate() {
            let task = HybridSearchTask {
                task_id: i,
                index_id: index.id,
                query_text: query_text.to_string(),
                keyword_query: keyword_query.clone_box(),
                query_vector: query_vector.clone(),
            };
            tasks.push(task);
        }

        Ok(tasks)
    }

    /// Execute tasks in parallel.
    async fn execute_tasks_parallel(
        &self,
        tasks: Vec<HybridSearchTask>,
    ) -> Result<(Vec<HybridSearchTaskResult>, SearchTimeBreakdown)> {
        let task_count = tasks.len();
        let results = Arc::new(Mutex::new(Vec::with_capacity(task_count)));

        let keyword_time = Arc::new(Mutex::new(0.0));
        let vector_time = Arc::new(Mutex::new(0.0));

        // Get indices map for task execution
        let indices_map = self.create_indices_map(&tasks)?;

        // Execute tasks in thread pool
        self.thread_pool.scope(|scope| {
            for task in tasks {
                let results_clone = Arc::clone(&results);
                let keyword_time_clone = Arc::clone(&keyword_time);
                let vector_time_clone = Arc::clone(&vector_time);
                let indices_map_ref = &indices_map;
                let config = &self.config;

                scope.spawn(move |_| {
                    let result = Self::execute_single_task(
                        task,
                        indices_map_ref,
                        config,
                        keyword_time_clone,
                        vector_time_clone,
                    );
                    results_clone.lock().unwrap().push(result);
                });
            }
        });

        // Extract results
        let final_results = Arc::try_unwrap(results)
            .map_err(|_| SageError::InvalidOperation("Failed to unwrap results".to_string()))?
            .into_inner()
            .map_err(|_| SageError::InvalidOperation("Failed to unlock results".to_string()))?;

        let time_breakdown = SearchTimeBreakdown {
            keyword_search_ms: *keyword_time.lock().unwrap(),
            vector_search_ms: *vector_time.lock().unwrap(),
            merge_ms: 0.0,     // Set later
            expansion_ms: 0.0, // TODO: Implement query expansion timing
            ranking_ms: 0.0,   // TODO: Implement ranking timing
        };

        Ok((final_results, time_breakdown))
    }

    /// Execute a single hybrid search task.
    fn execute_single_task(
        task: HybridSearchTask,
        indices_map: &HashMap<String, HybridIndexHandle>,
        config: &ParallelHybridSearchConfig,
        keyword_time: Arc<Mutex<f64>>,
        vector_time: Arc<Mutex<f64>>,
    ) -> HybridSearchTaskResult {
        let start_time = Instant::now();

        let index_handle = match indices_map.get(&task.index_id) {
            Some(handle) => handle,
            None => {
                return HybridSearchTaskResult {
                    task_id: task.task_id,
                    index_id: task.index_id,
                    keyword_results: None,
                    vector_results: None,
                    execution_time_ms: 0.0,
                    error: Some(SageError::internal("Index not found")),
                };
            }
        };

        // Execute keyword search
        let kw_start = Instant::now();
        let keyword_results = match Self::execute_keyword_search(
            task.keyword_query.as_ref(),
            &index_handle.keyword_reader,
            config,
            &task.index_id,
        ) {
            Ok(results) => Some(results.hits),
            Err(e) => {
                return HybridSearchTaskResult {
                    task_id: task.task_id,
                    index_id: task.index_id,
                    keyword_results: None,
                    vector_results: None,
                    execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
                    error: Some(e),
                };
            }
        };
        let kw_time = kw_start.elapsed().as_secs_f64() * 1000.0;
        *keyword_time.lock().unwrap() += kw_time;

        // Execute vector search if available
        let vec_start = Instant::now();
        let vector_results = if let (Some(vector_reader), Some(query_vector)) =
            (&index_handle.vector_reader, &task.query_vector)
        {
            match Self::execute_vector_search(query_vector, vector_reader, config) {
                Ok(results) => Some(results.results),
                Err(_) => None, // Continue without vector results
            }
        } else {
            None
        };
        let vec_time = vec_start.elapsed().as_secs_f64() * 1000.0;
        *vector_time.lock().unwrap() += vec_time;

        HybridSearchTaskResult {
            task_id: task.task_id,
            index_id: task.index_id,
            keyword_results,
            vector_results,
            execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            error: None,
        }
    }

    /// Execute keyword search on an index.
    fn execute_keyword_search(
        query: &dyn Query,
        reader: &Arc<dyn IndexReader>,
        config: &ParallelHybridSearchConfig,
        _index_id: &str,
    ) -> Result<SearchResults> {
        use crate::full_text_search::{SearchConfig, SearchRequest, Searcher};

        // Create a searcher with the index reader using from_arc
        let searcher = Searcher::from_arc(reader.clone());

        // Create search configuration
        let search_config = SearchConfig {
            max_docs: config.max_keyword_results_per_index,
            min_score: config.min_keyword_score,
            load_documents: false, // Don't load documents for performance
            timeout_ms: Some(config.index_timeout.as_millis() as u64),
            parallel: false, // Already running in parallel at the executor level
        };

        // Create search request
        let request = SearchRequest {
            query: query.clone_box(),
            config: search_config,
        };

        // Execute the search
        searcher.search(request)
    }

    /// Execute vector search on an index.
    fn execute_vector_search(
        query_vector: &Vector,
        reader: &Arc<dyn crate::vector::reader::VectorIndexReader>,
        config: &ParallelHybridSearchConfig,
    ) -> Result<crate::vector::types::VectorSearchResults> {
        use crate::vector::types::{VectorSearchResult, VectorSearchResults};
        use std::time::Instant;

        let start_time = Instant::now();

        // Get all vector IDs from the index
        let vector_ids = reader.vector_ids()?;
        if vector_ids.is_empty() {
            return Ok(VectorSearchResults {
                results: Vec::new(),
                candidates_examined: 0,
                search_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
                query_metadata: std::collections::HashMap::new(),
            });
        }

        // Get distance metric (no Result wrapping)
        let metric = reader.distance_metric();

        // Calculate similarities for all vectors
        let mut scored_results: Vec<(u64, f32, f32)> = Vec::new();

        for doc_id in vector_ids {
            if let Some(vector) = reader.get_vector(doc_id)? {
                // Calculate distance and similarity
                let distance = metric.distance(&query_vector.data, &vector.data)?;
                let similarity = metric.similarity(&query_vector.data, &vector.data)?;

                // Apply minimum similarity threshold
                if similarity >= config.min_vector_similarity {
                    scored_results.push((doc_id, similarity, distance));
                }
            }
        }

        // Sort by similarity (descending)
        scored_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Limit to max results per index
        scored_results.truncate(config.max_vector_results_per_index);

        // Convert to VectorSearchResult
        let results: Vec<VectorSearchResult> = scored_results
            .into_iter()
            .map(|(doc_id, similarity, distance)| VectorSearchResult {
                doc_id,
                similarity,
                distance,
                vector: None, // Don't include vectors in results for efficiency
                metadata: std::collections::HashMap::new(),
            })
            .collect();

        let candidates_examined = reader.vector_count();
        let search_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;

        Ok(VectorSearchResults {
            results,
            candidates_examined,
            search_time_ms,
            query_metadata: std::collections::HashMap::new(),
        })
    }

    /// Create a map of index handles for efficient lookup.
    fn create_indices_map(
        &self,
        tasks: &[HybridSearchTask],
    ) -> Result<HashMap<String, HybridIndexHandle>> {
        let indices = self.indices.lock().unwrap();
        let mut map = HashMap::new();

        for task in tasks {
            if let Some(handle) = indices.get(&task.index_id) {
                map.insert(task.index_id.clone(), handle.clone());
            }
        }

        Ok(map)
    }
}
