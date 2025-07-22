//! Spelling correction and suggestion system for Sarissa.
//!
//! This module provides functionality for correcting typos in user queries,
//! generating spelling suggestions, and implementing "Did you mean?" features.

pub mod corrector;
pub mod dictionary;
pub mod levenshtein;
pub mod suggest;

// Re-export commonly used types
pub use corrector::*;
pub use dictionary::*;
pub use levenshtein::*;
pub use suggest::*;
