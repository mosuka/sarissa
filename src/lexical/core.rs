//! Core data structures for lexical search.
//!
//! This module provides fundamental data structures used throughout the lexical search system:
//! - Automaton for pattern matching
//! - BKD tree for spatial indexing
//! - Dictionary for term storage
//! - Doc values for fast field access
//! - Posting lists for inverted indexes
//! - Segment management
//! - Term operations

pub mod automaton;
pub mod bkd_tree;
pub mod dictionary;
pub mod doc_values;
pub mod posting;
pub mod segment;
pub mod terms;
