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

// Re-export commonly used types
pub use advanced_writer::{AdvancedIndexWriter, AdvancedWriterConfig, AnalyzedDocument, AnalyzedTerm};
pub use background_tasks::BackgroundTask;
pub use deletion::DeletionManager;
pub use merge_engine::MergeEngine;
pub use merge_policy::MergePolicy;
pub use optimization::IndexOptimizer;
pub use optimize::{OptimizationRecommendation, OptimizationResult};
pub use segment_manager::{ManagedSegmentInfo, SegmentManager};
pub use transaction::Transaction;
pub use writer::IndexWriter;
