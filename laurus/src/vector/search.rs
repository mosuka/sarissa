//! Vector search module for executing similarity searches on vector indexes.
//!
//! This module handles all vector search operations:
//! - Approximate and exact nearest neighbor search
//! - Search result processing and filtering
//!
//! Note: Distance metrics live in [`crate::vector::core::distance`] — the
//! `scoring` submodule that previously lived here was a parallel
//! similarity-metric / ranking API that no production caller invoked.

pub mod filter_set;
pub(crate) mod rerank;
pub mod searcher;
