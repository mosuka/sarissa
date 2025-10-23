//! HNSW (Hierarchical Navigable Small World) vector index implementation.
//!
//! This module provides an approximate nearest neighbor search index based on
//! the HNSW algorithm, which offers excellent search performance with relatively
//! low memory overhead.

pub mod builder;
pub mod reader;

// Re-export main types for convenience
pub use builder::HnswIndexWriter;
pub use reader::HnswIndexReader;
