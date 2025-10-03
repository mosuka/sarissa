//! Document module for schema-less indexing.
//!
//! This module provides document structure and field value types
//! for schema-less, Apache Lucene-style indexing.

#[allow(clippy::module_inception)]
pub mod document;
pub mod field_value;

// Re-export commonly used types
pub use document::{Document, DocumentBuilder};
pub use field_value::{FieldValue, NumericType};
