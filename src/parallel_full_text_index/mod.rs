//! Parallel index distribution module for distributing documents across multiple indices.
//!
//! This module provides functionality to:
//! - Partition documents based on field values
//! - Distribute indexing operations across multiple writers
//! - Execute parallel indexing with configurable strategies
//! - Monitor indexing performance metrics

pub mod batch_processor;
pub mod config;
pub mod engine;
pub mod metrics;
pub mod partitioner;
pub mod writer_manager;
