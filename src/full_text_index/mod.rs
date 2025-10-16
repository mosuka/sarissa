//! Full-text index building and maintenance.
//!
//! This module handles all write operations for full-text indexes:
//! - Index creation and updates
//! - Inverted index construction
//! - Segment merging and optimization
//! - Deletion management
//! - Background maintenance tasks

pub mod advanced_writer;
pub mod background_tasks;
pub mod deletion;
pub mod merge_engine;
pub mod merge_policy;
pub mod optimization;
pub mod optimize;
pub mod segment_manager;
pub mod transaction;
pub mod writer;
