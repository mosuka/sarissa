//! Document parser for converting documents into analyzed documents.
//!
//! This module provides [`DocumentParser`] that works similarly to QueryParser,
//! analyzing document fields and producing tokenized, index-ready documents.
//!
//! # Overview
//!
//! The `DocumentParser` bridges the gap between raw documents and the inverted
//! index by:
//!
//! 1. Analyzing text fields with configured analyzers (per-field or default)
//! 2. Converting non-text fields (numbers, dates, geo) to indexable terms
//! 3. Calculating term frequencies and positions
//! 4. Preserving both indexed and stored field values
//!
//! # Architecture
//!
//! ```text
//! Document → DocumentParser → AnalyzedDocument → Index
//!              ↓
//!        PerFieldAnalyzer
//!              ↓
//!        Tokenizer + Filters
//! ```
//!
//! # Field Type Handling
//!
//! - **Text fields**: Analyzed with tokenizers and filters
//! - **Integer/Float**: Converted to string representation for indexing
//! - **Boolean**: Converted to "true"/"false" strings
//! - **DateTime**: Converted to RFC3339 format
//! - **Geo**: Converted to "lat,lon" format
//! - **Binary**: Stored only, not indexed
//! - **Null**: Stored only, not indexed
//!
//! # Examples
//!
//! Basic usage with default analyzer:
//!
//! ```
//! use yatagarasu::document::document::Document;
//! use yatagarasu::document::parser::DocumentParser;
//! use yatagarasu::analysis::analyzer::standard::StandardAnalyzer;
//! use std::sync::Arc;
//!
//! let parser = DocumentParser::new(Arc::new(StandardAnalyzer::new().unwrap()));
//!
//! let doc = Document::builder()
//!     .add_text("title", "Rust Programming Language")
//!     .add_integer("year", 2024)
//!     .build();
//!
//! let analyzed = parser.parse(doc).unwrap();
//! assert!(analyzed.field_terms.contains_key("title"));
//! assert!(analyzed.field_terms.contains_key("year"));
//! ```
//!
//! With per-field analyzers:
//!
//! ```
//! use yatagarasu::document::document::Document;
//! use yatagarasu::document::parser::DocumentParser;
//! use yatagarasu::analysis::analyzer::per_field::PerFieldAnalyzer;
//! use yatagarasu::analysis::analyzer::standard::StandardAnalyzer;
//! use yatagarasu::analysis::analyzer::keyword::KeywordAnalyzer;
//! use std::sync::Arc;
//!
//! // Configure per-field analyzers
//! let mut per_field = PerFieldAnalyzer::new(Arc::new(StandardAnalyzer::new().unwrap()));
//! per_field.add_analyzer("id", Arc::new(KeywordAnalyzer::new()));
//!
//! let parser = DocumentParser::new(Arc::new(per_field));
//!
//! let doc = Document::builder()
//!     .add_text("title", "Getting Started")  // Uses StandardAnalyzer
//!     .add_text("id", "DOC-001")             // Uses KeywordAnalyzer
//!     .build();
//!
//! let analyzed = parser.parse(doc).unwrap();
//! // "id" field is treated as a single keyword token
//! assert_eq!(analyzed.field_terms.get("id").unwrap()[0].term, "DOC-001");
//! ```

use std::sync::Arc;

use ahash::AHashMap;

use crate::analysis::analyzer::analyzer::Analyzer;
use crate::analysis::analyzer::per_field::PerFieldAnalyzer;
use crate::analysis::token::Token;
use crate::document::document::Document;
use crate::document::field_value::FieldValue;
use crate::error::Result;
use crate::document::analyzed::{AnalyzedDocument, AnalyzedTerm};

/// A document parser that converts Documents into AnalyzedDocuments.
///
/// Similar to how QueryParser analyzes query strings, DocumentParser
/// analyzes Document fields using a PerFieldAnalyzer to produce
/// tokenized, indexed-ready AnalyzedDocuments.
///
/// # Example
///
/// ```
/// use yatagarasu::document::document::Document;
/// use yatagarasu::document::parser::DocumentParser;
/// use yatagarasu::analysis::analyzer::per_field::PerFieldAnalyzer;
/// use yatagarasu::analysis::analyzer::standard::StandardAnalyzer;
/// use yatagarasu::analysis::analyzer::keyword::KeywordAnalyzer;
/// use std::sync::Arc;
///
/// let mut per_field = PerFieldAnalyzer::new(Arc::new(StandardAnalyzer::new().unwrap()));
/// per_field.add_analyzer("id", Arc::new(KeywordAnalyzer::new()));
///
/// let parser = DocumentParser::new(Arc::new(per_field));
///
/// let doc = Document::builder()
///     .add_text("title", "Rust Programming")
///     .add_text("id", "BOOK-001")
///     .build();
///
/// let analyzed = parser.parse(doc).unwrap();
/// ```
pub struct DocumentParser {
    /// Analyzer (typically PerFieldAnalyzerWrapper) for analyzing fields.
    analyzer: Arc<dyn Analyzer>,
}

impl std::fmt::Debug for DocumentParser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DocumentParser")
            .field("analyzer", &self.analyzer.name())
            .finish()
    }
}

impl DocumentParser {
    /// Create a new document parser with the given analyzer.
    ///
    /// Typically, you would pass a PerFieldAnalyzer here,
    /// similar to how it's used with QueryParser.
    pub fn new(analyzer: Arc<dyn Analyzer>) -> Self {
        DocumentParser { analyzer }
    }

    /// Parse a document into an AnalyzedDocument.
    ///
    /// This converts text fields into tokenized terms with position information,
    /// ready to be written to the inverted index. The document ID will be assigned
    /// automatically by the index writer when the document is added.
    ///
    /// # Arguments
    ///
    /// * `doc` - The document to parse
    pub fn parse(&self, doc: Document) -> Result<AnalyzedDocument> {
        let mut field_terms = AHashMap::new();
        let mut stored_fields = AHashMap::new();

        // Process each field in the document
        for (field_name, field_value) in doc.fields() {
            match field_value {
                FieldValue::Text(text) => {
                    // Analyze text field with per-field analyzer
                    let tokens = if let Some(per_field) =
                        self.analyzer.as_any().downcast_ref::<PerFieldAnalyzer>()
                    {
                        per_field.analyze_field(field_name, text)?
                    } else {
                        self.analyzer.analyze(text)?
                    };

                    let token_vec: Vec<Token> = tokens.collect();
                    let analyzed_terms = self.tokens_to_analyzed_terms(token_vec);

                    field_terms.insert(field_name.clone(), analyzed_terms);
                    stored_fields.insert(field_name.clone(), FieldValue::Text(text.to_string()));
                }
                FieldValue::Integer(num) => {
                    // Convert integer to text for indexing
                    let text = num.to_string();

                    let analyzed_term = AnalyzedTerm {
                        term: text.clone(),
                        position: 0,
                        frequency: 1,
                        offset: (0, text.len()),
                    };

                    field_terms.insert(field_name.clone(), vec![analyzed_term]);
                    stored_fields.insert(field_name.clone(), FieldValue::Integer(*num));
                }
                FieldValue::Float(num) => {
                    // Convert float to text for indexing
                    let text = num.to_string();

                    let analyzed_term = AnalyzedTerm {
                        term: text.clone(),
                        position: 0,
                        frequency: 1,
                        offset: (0, text.len()),
                    };

                    field_terms.insert(field_name.clone(), vec![analyzed_term]);
                    stored_fields.insert(field_name.clone(), FieldValue::Float(*num));
                }
                FieldValue::Boolean(b) => {
                    // Convert boolean to text
                    let text = b.to_string();

                    let analyzed_term = AnalyzedTerm {
                        term: text.clone(),
                        position: 0,
                        frequency: 1,
                        offset: (0, text.len()),
                    };

                    field_terms.insert(field_name.clone(), vec![analyzed_term]);
                    stored_fields.insert(field_name.clone(), FieldValue::Boolean(*b));
                }
                FieldValue::Binary(_) => {
                    // Binary fields are not indexed, only stored
                    stored_fields.insert(field_name.clone(), field_value.clone());
                }
                FieldValue::DateTime(dt) => {
                    // Convert datetime to RFC3339 string
                    let text = dt.to_rfc3339();

                    let analyzed_term = AnalyzedTerm {
                        term: text.clone(),
                        position: 0,
                        frequency: 1,
                        offset: (0, text.len()),
                    };

                    field_terms.insert(field_name.clone(), vec![analyzed_term]);
                    stored_fields.insert(field_name.clone(), FieldValue::DateTime(*dt));
                }
                FieldValue::Geo(point) => {
                    // Convert geo point to string representation
                    let text = format!("{},{}", point.lat, point.lon);

                    let analyzed_term = AnalyzedTerm {
                        term: text.clone(),
                        position: 0,
                        frequency: 1,
                        offset: (0, text.len()),
                    };

                    field_terms.insert(field_name.clone(), vec![analyzed_term]);
                    stored_fields.insert(field_name.clone(), FieldValue::Geo(*point));
                }
                FieldValue::Null => {
                    // Null fields are not indexed, only stored
                    stored_fields.insert(field_name.clone(), FieldValue::Null);
                }
            }
        }

        // Calculate field lengths (number of tokens per field)
        let mut field_lengths = AHashMap::new();
        for (field_name, terms) in &field_terms {
            field_lengths.insert(field_name.clone(), terms.len() as u32);
        }

        Ok(AnalyzedDocument {
            field_terms,
            stored_fields,
            field_lengths,
        })
    }

    /// Convert tokens to analyzed terms with position and frequency information.
    fn tokens_to_analyzed_terms(&self, tokens: Vec<Token>) -> Vec<AnalyzedTerm> {
        // Type alias for clarity: maps term text to list of (position, (start_offset, end_offset))
        type TermPositionMap = AHashMap<String, Vec<(u32, (usize, usize))>>;
        let mut term_positions: TermPositionMap = AHashMap::new();

        // Group positions by term
        for token in tokens {
            term_positions.entry(token.text.clone()).or_default().push((
                token.position as u32,
                (token.start_offset, token.end_offset),
            ));
        }

        // Create analyzed terms
        term_positions
            .into_iter()
            .map(|(term, positions)| {
                let frequency = positions.len() as u32;
                let position = positions[0].0; // Use first position
                let offset = positions[0].1; // Use first offset

                AnalyzedTerm {
                    term,
                    position,
                    frequency,
                    offset,
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::analyzer::keyword::KeywordAnalyzer;
    use crate::analysis::analyzer::standard::StandardAnalyzer;

    #[test]
    fn test_basic_parsing() {
        let parser = DocumentParser::new(Arc::new(StandardAnalyzer::new().unwrap()));

        let doc = Document::builder()
            .add_text("title", "Rust Programming")
            .add_text("body", "Learn Rust")
            .build();

        let analyzed = parser.parse(doc).unwrap();

        assert!(analyzed.field_terms.contains_key("title"));
        assert!(analyzed.field_terms.contains_key("body"));
    }

    #[test]
    fn test_per_field_analyzer() {
        let mut per_field = PerFieldAnalyzer::new(Arc::new(StandardAnalyzer::new().unwrap()));
        per_field.add_analyzer("id", Arc::new(KeywordAnalyzer::new()));

        let parser = DocumentParser::new(Arc::new(per_field));

        let doc = Document::builder()
            .add_text("title", "Rust Programming")
            .add_text("id", "BOOK-001")
            .build();

        let analyzed = parser.parse(doc).unwrap();

        // title should be tokenized
        assert!(!analyzed.field_terms.get("title").unwrap().is_empty());
        // id should be one token (KeywordAnalyzer)
        assert_eq!(analyzed.field_terms.get("id").unwrap().len(), 1);
        assert_eq!(analyzed.field_terms.get("id").unwrap()[0].term, "BOOK-001"); // KeywordAnalyzer preserves case
    }

    #[test]
    fn test_numeric_fields() {
        let parser = DocumentParser::new(Arc::new(StandardAnalyzer::new().unwrap()));

        let doc = Document::builder()
            .add_text("title", "Test")
            .add_integer("year", 2024)
            .add_float("price", 19.99)
            .add_boolean("active", true)
            .build();

        let analyzed = parser.parse(doc).unwrap();

        assert!(analyzed.field_terms.contains_key("year"));
        assert!(analyzed.field_terms.contains_key("price"));
        assert!(analyzed.field_terms.contains_key("active"));
    }
}
