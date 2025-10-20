//! Core full-text types and interfaces.
//!
//! This module provides the fundamental data structures and interfaces
//! for full-text search, including index writing and searching capabilities.

// Core data structures
pub mod bkd_tree;
pub mod dictionary;
pub mod doc_values;
pub mod inverted_index;
pub mod posting;
pub mod reader;
pub mod segment;

// Sub-modules
pub mod index;
pub mod search;
