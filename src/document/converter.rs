//! Document converter for creating documents from files.
//!
//! This module provides a DocumentConverter trait and various implementations
//! that can convert different file formats into Document objects.

use std::path::Path;

use crate::document::document::Document;
use crate::error::Result;

pub mod csv;
pub mod jsonl;

/// A trait for converting various file formats into Document iterators.
///
/// This trait allows for extensible document conversion from different file formats
/// like CSV, JSONL, etc.
///
/// # Example
///
/// ```no_run
/// use yatagarasu::document::converter::DocumentConverter;
/// use yatagarasu::document::converter::csv::CsvDocumentConverter;
/// use yatagarasu::document::converter::jsonl::JsonlDocumentConverter;
///
/// // CSV converter
/// let csv_converter = CsvDocumentConverter::new();
/// for doc in csv_converter.convert("documents.csv").unwrap() {
///     let doc = doc.unwrap();
///     println!("CSV Document: {:?}", doc);
/// }
///
/// // JSONL converter
/// let jsonl_converter = JsonlDocumentConverter::new();
/// for doc in jsonl_converter.convert("documents.jsonl").unwrap() {
///     let doc = doc.unwrap();
///     println!("JSONL Document: {:?}", doc);
/// }
/// ```
pub trait DocumentConverter {
    /// The iterator type that yields documents.
    type Iter: Iterator<Item = Result<Document>>;

    /// Convert a file into an iterator of Documents.
    fn convert<P: AsRef<Path>>(&self, path: P) -> Result<Self::Iter>;
}
