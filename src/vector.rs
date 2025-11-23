//! Vector search implementation using approximate nearest neighbor algorithms.
//!
//! This module provides vector (semantic) search functionality through
//! various index structures (Flat, HNSW, IVF), supporting cosine similarity,
//! Euclidean distance, and other distance metrics.
//!
//! # Module Structure
//!
//! - `core`: Core data structures (vector, distance, quantization)
//! - `index`: Index management (config, factory, traits, flat, hnsw, ivf)
//! - `search`: Search execution (similarity, ranking, result processing)
//! - `engine`: High-level engine interface
//! - `writer`: Index writer trait

pub mod core;
pub mod index;
pub mod search;

pub mod engine;
pub mod reader;
pub mod writer;

pub mod field;

pub use self::core::distance::DistanceMetric;
pub use self::core::vector::Vector;
