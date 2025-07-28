//! Text embedding generation module.
//!
//! This module provides functionality to convert text into vector representations
//! for use in vector search and machine learning applications.

pub mod engine;

// Re-export commonly used types
pub use engine::{EmbeddingConfig, EmbeddingEngine, EmbeddingMethod};
