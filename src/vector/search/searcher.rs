//! Vector searcher implementations and utilities.

pub mod flat;
pub mod hnsw;
pub mod ivf;

use crate::error::Result;
use crate::vector::types::{VectorSearchRequest, VectorSearchResults};

/// Trait for vector searchers.
pub trait VectorSearcher: Send + Sync {
    /// Execute a vector similarity search.
    fn search(&self, request: &VectorSearchRequest) -> Result<VectorSearchResults>;

    /// Warm up the searcher (pre-load data, etc.).
    fn warmup(&mut self) -> Result<()> {
        // デフォルト実装: 何もしない
        Ok(())
    }
}
