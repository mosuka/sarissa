//! Analyzed document structures for indexing.
//!
//! This module defines the data structures that represent documents after
//! analysis (tokenization and filtering), ready to be written to an index.
//!
//! # Overview
//!
//! The analysis pipeline transforms raw documents into analyzed documents:
//!
//! ```text
//! Document → Analyzer → AnalyzedDocument → Index
//! ```
//!
//! An [`AnalyzedDocument`] contains:
//! - Analyzed terms with positions for each field
//! - Stored field values (for retrieval)
//! - Field length statistics (for ranking)
//!
//! # Examples
//!
//! Creating an analyzed document (typically done by DocumentParser):
//!
//! ```
//! use sarissa::document::analyzed::{AnalyzedDocument, AnalyzedTerm};
//! use sarissa::document::field::FieldValue;
//! use ahash::AHashMap;
//!
//! let mut field_terms = AHashMap::new();
//! field_terms.insert(
//!     "content".to_string(),
//!     vec![
//!         AnalyzedTerm {
//!             term: "rust".to_string(),
//!             position: 0,
//!             frequency: 1,
//!             offset: (0, 4),
//!         },
//!         AnalyzedTerm {
//!             term: "programming".to_string(),
//!             position: 1,
//!             frequency: 1,
//!             offset: (5, 16),
//!         },
//!     ],
//! );
//!
//! let mut stored_fields = AHashMap::new();
//! stored_fields.insert("content".to_string(), FieldValue::Text("rust programming".to_string()));
//!
//! let mut field_lengths = AHashMap::new();
//! field_lengths.insert("content".to_string(), 2);
//!
//! let analyzed_doc = AnalyzedDocument {
//!     field_terms,
//!     stored_fields,
//!     field_lengths,
//! };
//!
//! assert_eq!(analyzed_doc.field_lengths["content"], 2);
//! ```

use ahash::AHashMap;

use crate::lexical::core::field::FieldValue;

/// A document with analyzed terms ready for indexing.
///
/// This structure represents a document after analysis (tokenization),
/// ready to be written to the inverted index. The document ID is assigned
/// automatically by the index writer when the document is added.
///
/// # Fields
///
/// - `field_terms` - Map of field names to their analyzed terms
/// - `stored_fields` - Original field values to be stored (for retrieval)
/// - `field_lengths` - Number of terms per field (used for BM25 scoring)
///
/// # Usage
///
/// Typically created by [`DocumentParser`](crate::document::parser::DocumentParser)
/// during the indexing process. Can also be constructed manually for
/// pre-analyzed documents from external systems.
#[derive(Debug, Clone)]
pub struct AnalyzedDocument {
    /// Field name to analyzed terms mapping.
    pub field_terms: AHashMap<String, Vec<AnalyzedTerm>>,
    /// Stored field values with original types preserved.
    pub stored_fields: AHashMap<String, FieldValue>,
    /// Field name to field length (number of tokens) mapping.
    pub field_lengths: AHashMap<String, u32>,
}

/// An analyzed term with position and metadata.
///
/// This represents a single token after analysis, including
/// position information for phrase queries and proximity searches.
///
/// # Fields
///
/// - `term` - The normalized term text (after tokenization/filtering)
/// - `position` - Position in the field (0-based)
/// - `frequency` - How many times this term appears in the document
/// - `offset` - Character offsets in original text (start, end)
///
/// # Examples
///
/// ```
/// use sarissa::document::analyzed::AnalyzedTerm;
///
/// let term = AnalyzedTerm {
///     term: "search".to_string(),
///     position: 5,
///     frequency: 2,
///     offset: (25, 31),
/// };
///
/// assert_eq!(term.term, "search");
/// assert_eq!(term.position, 5);
/// ```
#[derive(Debug, Clone)]
pub struct AnalyzedTerm {
    /// The term text.
    pub term: String,
    /// Position in the field.
    pub position: u32,
    /// Term frequency in the document.
    pub frequency: u32,
    /// Offset in the original text.
    pub offset: (usize, usize),
}

impl AnalyzedDocument {
    /// Create a new empty analyzed document.
    pub fn new() -> Self {
        Self {
            field_terms: AHashMap::new(),
            stored_fields: AHashMap::new(),
            field_lengths: AHashMap::new(),
        }
    }

    /// Get the number of fields in this document.
    pub fn field_count(&self) -> usize {
        self.field_terms.len()
    }

    /// Get the total number of terms across all fields.
    pub fn total_terms(&self) -> usize {
        self.field_terms.values().map(|terms| terms.len()).sum()
    }

    /// Get the length (number of terms) for a specific field.
    pub fn field_length(&self, field: &str) -> Option<u32> {
        self.field_lengths.get(field).copied()
    }
}

impl Default for AnalyzedDocument {
    fn default() -> Self {
        Self::new()
    }
}

impl AnalyzedTerm {
    /// Create a new analyzed term.
    pub fn new(term: String, position: u32, frequency: u32, offset: (usize, usize)) -> Self {
        Self {
            term,
            position,
            frequency,
            offset,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyzed_document_new() {
        let doc = AnalyzedDocument::new();
        assert_eq!(doc.field_count(), 0);
        assert_eq!(doc.total_terms(), 0);
    }

    #[test]
    fn test_analyzed_document_field_count() {
        let mut doc = AnalyzedDocument::new();
        doc.field_terms.insert("title".to_string(), vec![]);
        doc.field_terms.insert("content".to_string(), vec![]);
        assert_eq!(doc.field_count(), 2);
    }

    #[test]
    fn test_analyzed_document_total_terms() {
        let mut doc = AnalyzedDocument::new();
        doc.field_terms.insert(
            "title".to_string(),
            vec![
                AnalyzedTerm::new("hello".to_string(), 0, 1, (0, 5)),
                AnalyzedTerm::new("world".to_string(), 1, 1, (6, 11)),
            ],
        );
        doc.field_terms.insert(
            "content".to_string(),
            vec![AnalyzedTerm::new("test".to_string(), 0, 1, (0, 4))],
        );
        assert_eq!(doc.total_terms(), 3);
    }

    #[test]
    fn test_analyzed_term_new() {
        let term = AnalyzedTerm::new("search".to_string(), 5, 2, (10, 16));
        assert_eq!(term.term, "search");
        assert_eq!(term.position, 5);
        assert_eq!(term.frequency, 2);
        assert_eq!(term.offset, (10, 16));
    }
}
