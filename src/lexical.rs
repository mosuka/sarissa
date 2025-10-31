//! Lexical search implementation using inverted indexes.
//!
//! This module provides lexical (keyword-based) search functionality through
//! inverted index structures, supporting BM25 scoring, phrase queries, and
//! various query types based on token matching.
//!
//! # Module Structure
//!
//! - `core`: Core data structures (posting, dictionary, segment, etc.)
//! - `index`: Index management (config, factory, traits, inverted, segment, maintenance)
//! - `search`: Search execution (scoring, features, result processing)
//! - `engine`: High-level engine interface
//! - `types`: Type definitions
//! - `reader`: Index reader trait
//! - `writer`: Index writer trait

pub mod core;
pub mod index;
pub mod search;

pub mod engine;
pub mod reader;
pub mod types;
pub mod writer;
