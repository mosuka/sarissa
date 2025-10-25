//! Lexical search implementation using inverted indexes.
//!
//! This module provides lexical (keyword-based) search functionality through
//! inverted index structures, supporting BM25 scoring, phrase queries, and
//! various query types based on token matching.

// Core data structures
pub mod bkd_tree;
pub mod dictionary;
pub mod doc_values;
pub mod posting;
pub mod reader;
pub mod segment;
pub mod writer;

// High-level interface
pub mod engine;

// Type definitions
pub mod types;

// Sub-modules
pub mod index;
pub mod search;
