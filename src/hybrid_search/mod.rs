//! Hybrid search module combining keyword and vector search.
//!
//! This module provides the ability to combine traditional keyword-based search
//! with vector-based semantic search, offering the best of both approaches:
//! - Precise keyword matching for exact terms
//! - Semantic understanding through vector embeddings
//! - Configurable weighting between the two approaches

pub mod config;
pub mod engine;
pub mod merger;
pub mod scorer;
pub mod stats;
pub mod types;

pub use config::{HybridSearchConfig, ScoreNormalization};
pub use engine::{HybridSearchEngine, Searchable};
pub use merger::ResultMerger;
pub use scorer::ScoreNormalizer;
pub use stats::HybridSearchStats;
pub use types::{HybridSearchResult, HybridSearchResults};
