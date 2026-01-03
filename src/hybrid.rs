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
//! ```no_run
//! use sarissa::hybrid::engine::HybridEngine;
//! use sarissa::hybrid::search::searcher::HybridSearchRequest;
//! use sarissa::lexical::engine::LexicalEngine;
//! use sarissa::vector::engine::VectorEngine;
//! use sarissa::error::Result;
//!
//! async fn example(lexical_engine: LexicalEngine, vector_engine: VectorEngine) -> Result<()> {
//!     // Create hybrid search engine
//!     let engine = HybridEngine::new(lexical_engine, vector_engine)?;
//!
//!     // Create search request
//!     let request = HybridSearchRequest::new("rust programming")
//!         .keyword_weight(0.6)
//!         .vector_weight(0.4);
//!
//!     // Execute search
//!     let results = engine.search(request).await?;
//!
//!     Ok(())
//! }
//! ```
//!
//! For a complete working example, see `examples/hybrid_search.rs`.

// Core data structure
pub mod core;
pub mod index; // Core hybrid index combining lexical and vector indexes

// Configuration and types
pub mod stats;

// High-level interface
pub mod engine;

// Writer and search modules
pub mod search; // Search execution submodule (contains request, params, results)
pub mod writer; // Index writer
