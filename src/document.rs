//! Document module for schema-less indexing.
//!
//! This module provides document structure and field value types
//! for schema-less, Apache Lucene-style indexing.

pub mod converter;
#[allow(clippy::module_inception)]
pub mod document;
pub mod field_value;
pub mod parser;
