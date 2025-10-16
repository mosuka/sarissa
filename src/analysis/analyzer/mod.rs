//! Analyzer implementations that combine tokenizers and filters.

#[allow(clippy::module_inception)]
pub mod analyzer;
pub mod keyword;
pub mod language;
pub mod noop;
pub mod per_field;
pub mod pipeline;
pub mod simple;
pub mod standard;
