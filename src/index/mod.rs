//! Index module for Sarissa.
//!
//! This module provides the core indexing functionality including
//! index creation, reading, writing, and management.

pub mod advanced_reader;
pub mod advanced_writer;
pub mod background_tasks;
pub mod deletion;
pub mod dictionary;
#[allow(clippy::module_inception)]
pub mod index;
pub mod merge_engine;
pub mod merge_policy;
pub mod optimization;
pub mod optimize;
pub mod posting;
pub mod reader;
pub mod segment;
pub mod segment_manager;
pub mod transaction;
pub mod writer;

// Re-export commonly used types
pub use advanced_reader::AdvancedIndexReader;
pub use advanced_writer::AdvancedIndexWriter;
pub use background_tasks::BackgroundTask;
pub use deletion::DeletionManager;
pub use dictionary::{HashTermDictionary, TermInfo};
pub use index::{Index, IndexStats, SegmentInfo};
pub use merge_engine::MergeEngine;
pub use merge_policy::MergePolicy;
pub use posting::{InvertedIndex, Posting, PostingList};
pub use reader::IndexReader;
pub use segment::Segment;
pub use segment_manager::{ManagedSegmentInfo, SegmentManager};
pub use writer::IndexWriter;
