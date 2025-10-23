//! IVF (Inverted File) vector index implementation.
//!
//! This module provides an IVF index that partitions vectors into clusters
//! for efficient approximate nearest neighbor search with reduced memory usage.

pub mod builder;
pub mod reader;

// Re-export main types for convenience
pub use builder::IvfIndexWriter;
pub use reader::IvfIndexReader;
