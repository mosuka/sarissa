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
pub mod embedding;
pub mod error;
pub mod hybrid;
pub mod lexical;
pub mod ml;
pub mod query;
pub mod spelling;
pub mod storage;
pub mod util;
pub mod vector;

pub mod prelude {}

// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
