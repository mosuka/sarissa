//! Index reader for searching and retrieving documents.

use crate::document::document::Document;
use crate::error::Result;
use crate::full_text::bkd_tree::SimpleBKDTree;

/// Trait for index readers.
pub trait IndexReader: Send + Sync + std::fmt::Debug {
    /// Get the number of documents in the index.
    fn doc_count(&self) -> u64;

    /// Get the maximum document ID in the index.
    fn max_doc(&self) -> u64;

    /// Check if a document is deleted.
    fn is_deleted(&self, doc_id: u64) -> bool;

    /// Get a document by ID.
    fn document(&self, doc_id: u64) -> Result<Option<Document>>;

    /// Get term information for a field and term.
    fn term_info(&self, field: &str, term: &str) -> Result<Option<ReaderTermInfo>>;

    /// Get posting list for a field and term.
    fn postings(&self, field: &str, term: &str) -> Result<Option<Box<dyn PostingIterator>>>;

    /// Get field statistics.
    fn field_stats(&self, field: &str) -> Result<Option<FieldStats>>;

    /// Close the reader and release resources.
    fn close(&mut self) -> Result<()>;

    /// Check if the reader is closed.
    fn is_closed(&self) -> bool;

    /// Get BKD Tree for a numeric field, if available.
    fn get_bkd_tree(&self, field: &str) -> Result<Option<&SimpleBKDTree>> {
        // Default implementation returns None (no BKD Tree support)
        let _ = field;
        Ok(None)
    }

    /// Get document frequency for a specific term in a field.
    fn term_doc_freq(&self, field: &str, term: &str) -> Result<u64> {
        match self.term_info(field, term)? {
            Some(term_info) => Ok(term_info.doc_freq),
            None => Ok(0),
        }
    }

    /// Get field statistics including average field length.
    fn field_statistics(&self, field: &str) -> Result<FieldStatistics> {
        match self.field_stats(field)? {
            Some(field_stats) => Ok(FieldStatistics {
                avg_field_length: field_stats.avg_length,
                doc_count: field_stats.doc_count,
                total_terms: field_stats.total_terms,
            }),
            None => Ok(FieldStatistics {
                avg_field_length: 10.0, // Default fallback
                doc_count: 0,
                total_terms: 0,
            }),
        }
    }

    /// Get this reader as Any for downcasting.
    fn as_any(&self) -> &dyn std::any::Any;
}

/// Information about a term in the index.
#[derive(Debug, Clone)]
pub struct ReaderTermInfo {
    /// The field name.
    pub field: String,

    /// The term text.
    pub term: String,

    /// Number of documents containing this term.
    pub doc_freq: u64,

    /// Total number of occurrences of this term.
    pub total_freq: u64,

    /// Position of the term in the posting list.
    pub posting_offset: u64,

    /// Size of the posting list in bytes.
    pub posting_size: u64,
}

/// Statistics about a field in the index.
#[derive(Debug, Clone)]
pub struct FieldStats {
    /// The field name.
    pub field: String,

    /// Number of unique terms in this field.
    pub unique_terms: u64,

    /// Total number of term occurrences.
    pub total_terms: u64,

    /// Number of documents with this field.
    pub doc_count: u64,

    /// Average field length.
    pub avg_length: f64,

    /// Minimum field length.
    pub min_length: u64,

    /// Maximum field length.
    pub max_length: u64,
}

/// Simplified field statistics for query scoring.
#[derive(Debug, Clone)]
pub struct FieldStatistics {
    /// Average field length.
    pub avg_field_length: f64,

    /// Number of documents with this field.
    pub doc_count: u64,

    /// Total number of terms.
    pub total_terms: u64,
}

/// Iterator over posting lists.
pub trait PostingIterator: Send + std::fmt::Debug {
    /// Get the current document ID.
    fn doc_id(&self) -> u64;

    /// Get the term frequency in the current document.
    fn term_freq(&self) -> u64;

    /// Get the positions of the term in the current document.
    fn positions(&self) -> Result<Vec<u64>>;

    /// Move to the next document.
    fn next(&mut self) -> Result<bool>;

    /// Skip to the first document >= target.
    fn skip_to(&mut self, target: u64) -> Result<bool>;

    /// Get the cost of iterating through this posting list.
    fn cost(&self) -> u64;
}
