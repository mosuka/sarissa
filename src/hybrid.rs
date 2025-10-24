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
//! ```rust,ignore
//! use sage::hybrid::index::HybridIndex;
//! use sage::hybrid::config::HybridSearchConfig;
//! use sage::hybrid::engine::HybridSearchEngine;
//!
//! // Create hybrid index
//! let hybrid_index = HybridIndex::new(lexical_index, vector_index);
//!
//! // Create search configuration
//! let config = HybridSearchConfig::default();
//!
//! // Perform hybrid search
//! let results = hybrid_index.search("rust programming", Some(&query_vector), &config)?;
//! ```

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
