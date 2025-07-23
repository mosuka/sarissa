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
//!
//! ## Quick Start
//!
//! ```rust
//! use sarissa::prelude::*;
//!
//! // Create a schema
//! let mut schema = Schema::new();
//! schema.add_field("title", Box::new(TextField::new())).unwrap();
//! schema.add_field("body", Box::new(TextField::new())).unwrap();
//!
//! // Create an index
//! // let index = Index::create_in_dir("index_dir", schema)?;
//! ```

pub mod analysis;
pub mod cli;
pub mod error;
pub mod index;
// Temporarily disable ML module to test core functionality
// pub mod ml;
pub mod parallel_index;
pub mod parallel_search;
pub mod query;
pub mod schema;
pub mod search;
pub mod spelling;
pub mod storage;
pub mod util;
pub mod vector;

// Re-export commonly used types
pub mod prelude {
    // Core types
    pub use crate::analysis::{Analyzer, StandardAnalyzer};
    pub use crate::error::{SarissaError, Result};
    pub use crate::index::{Index, IndexReader, IndexWriter};
    pub use crate::query::{BM25Scorer, Hit, Query, RangeQuery, SearchResults, TermQuery};
    pub use crate::schema::{
        BooleanField, Document, FieldType, IdField, KeywordField, NumericField, Schema, TextField,
    };
    pub use crate::search::{Search, SearchConfig};
    pub use crate::storage::{MemoryStorage, Storage, StorageConfig};
}

// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
