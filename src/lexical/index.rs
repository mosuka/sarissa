//! Full-text index writing and management.
//!
//! This module provides functionality for creating and maintaining full-text indexes,
//! including document indexing, segment management, and index optimization.

pub mod advanced_writer;
pub mod background_tasks;
pub mod deletion;
pub mod merge_engine;
pub mod merge_policy;
pub mod optimization;
pub mod segment_manager;
pub mod transaction;

// Re-export common types from inverted_index
pub use crate::lexical::inverted_index::{
    FileIndex, Index, IndexConfig, IndexMetadata, IndexStats, SegmentInfo,
};
