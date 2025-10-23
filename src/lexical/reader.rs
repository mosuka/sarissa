//! Index reader traits for searching and retrieving documents.

use crate::document::document::Document;
use crate::document::field_value::FieldValue;
use crate::error::Result;
use crate::lexical::bkd_tree::SimpleBKDTree;
use crate::lexical::types::{FieldStatistics, FieldStats, ReaderTermInfo};

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

    /// Get a DocValues field value for a document.
    /// Returns None if DocValues are not available for this field or document.
    fn get_doc_value(&self, field: &str, doc_id: u64) -> Result<Option<FieldValue>> {
        // Default implementation returns None (no DocValues support)
        let _ = (field, doc_id);
        Ok(None)
    }

    /// Check if DocValues are available for a field.
    fn has_doc_values(&self, field: &str) -> bool {
        // Default implementation returns false
        let _ = field;
        false
    }
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
