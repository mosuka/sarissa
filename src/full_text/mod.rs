//! Core full-text types and interfaces.
//!
//! This module provides the fundamental data structures and interfaces
//! for full-text search. It does not contain implementation logic for
//! writing or searching - those are in `full_text_index` and `full_text_search`.

pub mod bkd_tree;
pub mod dictionary;
#[allow(clippy::module_inception)]
pub mod index;
pub mod posting;
pub mod reader;
pub mod segment;

// Re-export core types
pub use bkd_tree::{BKDTreeStats, SimpleBKDTree};
pub use dictionary::{HashTermDictionary, TermInfo};
pub use index::{Index, IndexStats, SegmentInfo};
pub use posting::{InvertedIndex, Posting, PostingList};
pub use reader::IndexReader;
pub use segment::Segment;
