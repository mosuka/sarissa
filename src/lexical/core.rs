//! Core data structures for lexical search.
//!
//! This module provides fundamental data structures used throughout the lexical search system:
//! - BKD tree for spatial indexing
//! - Dictionary for term storage
//! - Doc values for fast field access

pub mod bkd_tree;
pub mod dictionary;
pub mod doc_values;
