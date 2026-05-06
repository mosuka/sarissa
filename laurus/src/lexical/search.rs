//! Full-text search execution and result processing.
//!
//! This module handles all search execution and result processing:
//! - Query execution
//! - Result collection and processing
//! - Faceting and aggregation
//! - Highlighting and spell correction
//!
//! # Module Structure
//!
//! - `features`: Search features (faceting, highlighting, spell correction)
//! - `result_processor`: Result processing utilities
//! - `searcher`: Query execution
//!
//! Note: BM25 / similarity scoring lives in [`crate::lexical::query::scorer`]
//! (the production path used by [`crate::lexical::index::inverted::searcher`])
//! — the `scoring` submodule that previously lived here was a parallel
//! plug-in scoring API that no production caller invoked.

pub mod features;
pub mod result_processor;
pub mod searcher;
