//! Core data structures for inverted index.
//!
//! This module provides fundamental data structures specific to inverted indexes:
//! - Automaton for fuzzy matching
//! - Posting lists for term-to-document mapping
//! - Term enumeration and statistics

pub mod automaton;
pub mod posting;
pub mod terms;
