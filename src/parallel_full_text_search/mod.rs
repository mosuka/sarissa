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
