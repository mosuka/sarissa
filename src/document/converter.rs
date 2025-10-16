//! Document converter for creating documents from strings.
//!
//! This module provides a DocumentConverter trait and various implementations
//! that can convert different formats into Document objects.

use crate::document::document::Document;
use crate::error::Result;

pub mod csv;
pub mod field_value;
pub mod json;

/// A trait for converting various formats into Document objects.
///
/// This trait allows for extensible document conversion from different formats
/// like field:value, JSON, YAML, PDF, etc.
///
/// # Example
///
/// ```
/// use sage::document::converter::DocumentConverter;
/// use sage::document::converter::field_value::FieldValueDocumentConverter;
///
/// let converter = FieldValueDocumentConverter::new();
/// let doc = converter.convert("title:Rust Programming\nbody:Search engine tutorial").unwrap();
/// ```
pub trait DocumentConverter {
    /// Convert input string into a Document.
    fn convert(&self, input: &str) -> Result<Document>;
}
