//! Hybrid search module combining lexical and vector search.
//!
//! This module provides unified search capabilities that combine:
//! - Lexical (keyword-based) search with BM25 scoring
//! - Vector (semantic) search with embeddings
//! - Configurable fusion algorithms (RRF, weighted sum, etc.)
//!
//! # Architecture
//!
//! The hybrid search module follows the same pattern as the lexical module:
//!
//! - **Core data structure**: `index` - Combines lexical and vector indexes
//! - **Configuration and types**: Configuration, statistics, and type definitions
//! - **Engine**: High-level interface for hybrid search operations
//! - **Writer**: Hybrid index writing functionality
//! - **Search submodule**: Search execution, scoring, and result merging
//!
//! # Example
//!
//! ```rust
//! use yatagarasu::hybrid::config::HybridSearchConfig;
//! use yatagarasu::hybrid::engine::HybridSearchEngine;
//! use yatagarasu::error::Result;
//!
//! fn example() -> Result<()> {
//!     // Create hybrid search configuration
//!     let config = HybridSearchConfig::default();
//!
//!     // Create hybrid search engine
//!     let engine = HybridSearchEngine::new(config)?;
//!
//!     // Access configuration
//!     println!("Keyword weight: {}", engine.config().keyword_weight);
//!     println!("Vector weight: {}", engine.config().vector_weight);
//!
//!     Ok(())
//! }
//! # example().unwrap();
//! ```
//!
//! For a complete working example, see `examples/hybrid_search.rs`.

// Core data structure
pub mod index; // Core hybrid index combining lexical and vector indexes

// Configuration and types
pub mod config;
pub mod stats;
pub mod types;

// High-level interface
pub mod engine;

// Writer and search modules
pub mod search;
pub mod writer; // Index writer // Search execution submodule
