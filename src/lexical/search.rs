//! Full-text search execution and result processing.
//!
//! This module handles all search execution and result processing:
//! - Query execution
//! - Scoring and ranking
//! - Result collection and processing
//! - Faceting and aggregation
//! - Highlighting and similarity
//!
//! # Module Structure
//!
//! - `scoring`: Scoring algorithms (BM25, similarity)
//! - `features`: Search features (faceting, highlighting, spell correction)
//! - `result_processor`: Result processing utilities

pub mod features;
pub mod result_processor;
pub mod scoring;
pub mod searcher;
