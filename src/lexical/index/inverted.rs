//! Inverted index implementation for full-text search.
//!
//! This module provides the core inverted index implementation:
//! - Core data structures (posting lists, term enumeration)
//! - Index creation and management
//! - Writer for building the index
//! - Reader for querying the index
//! - Searcher for executing searches
//! - Segment management and merging
//! - Index maintenance operations
//! - Query types for searching

pub mod core;
pub mod index;
pub mod maintenance;
pub mod query;
pub mod reader;
pub mod searcher;
pub mod segment;
pub mod types;
pub mod writer;
