//! Full-text search execution and result processing.
//!
//! This module handles all read operations and search execution:
//! - Query execution
//! - Scoring and ranking
//! - Result collection and processing
//! - Faceting and aggregation
//! - Highlighting and similarity

pub mod advanced_reader;
pub mod facet;
pub mod highlight;
pub mod result_processor;
pub mod scoring;
pub mod searcher;
pub mod similarity;
pub mod spell_corrected;
