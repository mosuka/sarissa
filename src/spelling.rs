//! Spelling correction and suggestion system for Sage.
//!
//! This module provides functionality for correcting typos in user queries,
//! generating spelling suggestions, and implementing "Did you mean?" features.

pub mod corrector;
pub mod dictionary;
pub mod levenshtein;
pub mod suggest;
