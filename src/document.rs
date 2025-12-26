//! Document module for schema-less indexing.
//!
//! This module provides the core document structures and utilities for
//! schema-less, Apache Lucene-style indexing. Documents are flexible
//! collections of field-value pairs that can be dynamically created without
//! predefined schemas.
//!
//! # Core Components
//!
//! - [`document::Document`] - The main document structure with field-value pairs
//! - [`field::FieldValue`] - Enum representing different field value types
//! - [`parser::DocumentParser`] - Converts documents to analyzed tokens for indexing
//! - [`converter`] - Utilities for converting files (CSV, JSONL) to documents
//!
//! # Supported Field Types
//!
//! - **Text** - Analyzed text for full-text search
//! - **Integer** - i64 integer values
//! - **Float** - f64 floating-point values
//! - **Boolean** - true/false values
//! - **Binary** - Raw binary data
//! - **DateTime** - UTC timestamps
//! - **Geo** - Geographic coordinates (lat/lon)
//! - **Null** - Explicit null values
//!
//! # Examples
//!
//! Creating a document:
//!
//! ```
//! use sarissa::document::document::Document;
//! use sarissa::document::field::{TextOption, IntegerOption, FloatOption, BooleanOption};
//!
//! let doc = Document::builder()
//!     .add_text("title", "Rust Programming Guide", TextOption::default())
//!     .add_text("author", "Jane Doe", TextOption::default())
//!     .add_integer("year", 2024, IntegerOption::default())
//!     .add_float("price", 39.99, FloatOption::default())
//!     .add_boolean("in_stock", true, BooleanOption::default())
//!     .build();
//!
//! assert_eq!(doc.len(), 5);
//! assert!(doc.has_field("title"));
//! ```
//!
//! Parsing documents for indexing:
//!
//! ```
//! use sarissa::document::document::Document;
//! use sarissa::document::parser::DocumentParser;
//! use sarissa::document::field::TextOption;
//! use sarissa::analysis::analyzer::standard::StandardAnalyzer;
//! use std::sync::Arc;
//!
//! let parser = DocumentParser::new(Arc::new(StandardAnalyzer::new().unwrap()));
//!
//! let doc = Document::builder()
//!     .add_text("title", "Rust Programming", TextOption::default())
//!     .build();
//!
//! let analyzed = parser.parse(doc).unwrap();
//! ```
//!
pub use crate::lexical::document::{analyzed, document, field, parser};
