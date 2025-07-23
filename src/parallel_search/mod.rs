//! Parallel search module for executing queries across multiple indices concurrently.
//!
//! This module provides functionality to:
//! - Manage multiple index readers
//! - Execute searches in parallel across indices
//! - Merge results using configurable strategies
//! - Monitor performance metrics

pub mod config;
pub mod engine;
pub mod example;
pub mod index_manager;
pub mod merger;
pub mod metrics;
pub mod search_task;

pub use config::{MergeStrategyType, ParallelSearchConfig};
pub use engine::ParallelSearchEngine;
pub use index_manager::{IndexHandle, IndexManager};
pub use merger::{MergeStrategy, ScoreBasedMerger, WeightedMerger};
pub use metrics::{SearchMetrics, SearchMetricsCollector};
pub use search_task::{SearchTask, TaskResult};