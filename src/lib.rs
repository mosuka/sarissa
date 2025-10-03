//! # Sarissa
//!
//! A fast, featureful full-text search library for Rust, inspired by Whoosh.
//!
//! ## Features
//!
//! - Pure Rust implementation
//! - Fast indexing and searching
//! - Flexible text analysis pipeline
//! - Pluggable storage backends
//! - Multiple query types
//! - BM25 scoring

pub mod analysis;
pub mod cli;
pub mod document;
pub mod embeding;
pub mod error;
pub mod hybrid_search;
pub mod index;
pub mod ml;
pub mod parallel_hybrid_search;
pub mod parallel_index;
pub mod parallel_search;
pub mod parallel_vector_index;
pub mod parallel_vector_search;
pub mod query;
pub mod search;
pub mod spelling;
pub mod storage;
pub mod util;
pub mod vector;
pub mod vector_index;
pub mod vector_search;

// Re-export commonly used types
pub mod prelude {
    // Core types
    pub use crate::analysis::{Analyzer, StandardAnalyzer};
    pub use crate::document::{Document, FieldValue};
    pub use crate::error::{Result, SarissaError};
    pub use crate::index::{Index, IndexReader, IndexWriter};
    pub use crate::query::{BM25Scorer, Hit, Query, RangeQuery, SearchResults, TermQuery};
    pub use crate::search::{Search, SearchConfig};
    pub use crate::storage::{MemoryStorage, Storage, StorageConfig};
}

// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
