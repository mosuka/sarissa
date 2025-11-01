//! Document converter for creating documents from files.
//!
//! This module provides the [`DocumentConverter`] trait and various implementations
//! for converting different file formats (CSV, JSONL) into [`Document`] objects.
//!
//! # Overview
//!
//! Document converters enable batch indexing from structured data files. Each
//! converter reads a specific file format and yields an iterator of documents,
//! allowing for efficient streaming processing of large datasets.
//!
//! # Available Converters
//!
//! - [`csv::CsvDocumentConverter`] - Converts CSV files with header row
//! - [`jsonl::JsonlDocumentConverter`] - Converts JSONL (JSON Lines) files
//!
//! # Design Pattern
//!
//! All converters implement the `DocumentConverter` trait which returns an
//! iterator, enabling:
//!
//! - **Streaming**: Process large files without loading everything into memory
//! - **Error handling**: Each document can fail independently
//! - **Flexibility**: Easy to add new file format converters
//!
//! # Examples
//!
//! Converting a CSV file:
//!
//! ```no_run
//! use yatagarasu::document::converter::DocumentConverter;
//! use yatagarasu::document::converter::csv::CsvDocumentConverter;
//!
//! let converter = CsvDocumentConverter::new();
//!
//! for doc_result in converter.convert("books.csv").unwrap() {
//!     match doc_result {
//!         Ok(doc) => {
//!             println!("Document: {:?}", doc);
//!             // Index the document...
//!         }
//!         Err(e) => eprintln!("Error: {}", e),
//!     }
//! }
//! ```
//!
//! Converting a JSONL file:
//!
//! ```no_run
//! use yatagarasu::document::converter::DocumentConverter;
//! use yatagarasu::document::converter::jsonl::JsonlDocumentConverter;
//!
//! let converter = JsonlDocumentConverter::new();
//!
//! let documents: Vec<_> = converter
//!     .convert("products.jsonl")
//!     .unwrap()
//!     .filter_map(|r| r.ok())
//!     .collect();
//!
//! println!("Loaded {} documents", documents.len());
//! ```
//!
//! Using custom delimiters (CSV):
//!
//! ```no_run
//! use yatagarasu::document::converter::DocumentConverter;
//! use yatagarasu::document::converter::csv::CsvDocumentConverter;
//!
//! // Tab-separated values
//! let converter = CsvDocumentConverter::new()
//!     .with_delimiter('\t')
//!     .with_flexible(true);
//!
//! for doc in converter.convert("data.tsv").unwrap() {
//!     // Process TSV document...
//! }
//! ```

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
