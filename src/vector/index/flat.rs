//! Flat vector index implementation.
//!
//! This module provides a simple flat (brute-force) vector index that stores
//! all vectors in memory and performs exact nearest neighbor search through
//! exhaustive comparison.

pub mod builder;
pub mod reader;

// Re-export main types for convenience
pub use builder::FlatIndexWriter;
pub use reader::FlatVectorIndexReader;
