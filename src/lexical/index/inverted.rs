//! Inverted index implementation for full-text search.
//!
//! This module provides the core inverted index implementation:
//! - Index creation and management
//! - Writer for building the index
//! - Reader for querying the index
//! - Searcher for executing searches

pub mod index;
pub mod reader;
pub mod searcher;
pub mod types;
pub mod writer;
