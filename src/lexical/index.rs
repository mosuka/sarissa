//! Inverted index writing and management.
//!
//! This module provides functionality for creating and maintaining inverted indexes,
//! including document indexing, segment management, and index optimization.

pub mod background_tasks;
pub mod deletion;
pub mod merge_engine;
pub mod merge_policy;
pub mod optimization;
pub mod reader;
pub mod segment_manager;
pub mod transaction;
pub mod writer;
