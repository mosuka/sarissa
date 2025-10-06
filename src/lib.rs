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
pub mod full_text;
pub mod full_text_index;
pub mod full_text_search;
pub mod hybrid_search;
pub mod ml;
pub mod parallel_full_text_index;
pub mod parallel_full_text_search;
pub mod parallel_hybrid_search;
pub mod parallel_vector_index;
pub mod parallel_vector_search;
pub mod query;
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
    pub use crate::full_text::{Index, IndexReader};
    pub use crate::full_text_index::{AdvancedIndexWriter, IndexWriter};
    pub use crate::full_text_search::{SearchConfig, SearchEngine, SearchRequest};
    pub use crate::query::{BM25Scorer, Hit, Query, RangeQuery, SearchResults, TermQuery};
    pub use crate::storage::{MemoryStorage, Storage, StorageConfig};
}

// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
