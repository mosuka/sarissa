//! # Sage
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
pub mod document;
pub mod embeding;
pub mod error;
pub mod hybrid_search;
pub mod lexical;
pub mod ml;
pub mod parallel_hybrid_search;
pub mod parallel_lexical_index;
pub mod parallel_lexical_search;
pub mod parallel_vector_index;
pub mod parallel_vector_search;
pub mod query;
pub mod spelling;
pub mod storage;
pub mod util;
pub mod vector;

pub mod prelude {}

// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
