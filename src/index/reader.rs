//! Index reader for searching and retrieving documents.

use std::collections::HashMap;
use std::sync::Arc;

use crate::error::{Result, SarissaError};
use crate::index::bkd_tree::SimpleBKDTree;
use crate::schema::{Document, FieldValue, Schema};
use crate::storage::Storage;

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

    /// Get the schema for this reader.
    fn schema(&self) -> &Schema;

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

/// A basic index reader implementation.
#[derive(Debug)]
pub struct BasicIndexReader {
    /// The schema for this reader.
    schema: Schema,

    /// The storage backend.
    storage: Arc<dyn Storage>,

    /// Cached documents.
    document_cache: Vec<Document>,

    /// BKD Trees for numeric fields.
    bkd_trees: HashMap<String, SimpleBKDTree>,

    /// Whether the reader is closed.
    closed: bool,
}

impl BasicIndexReader {
    /// Create a new index reader.
    pub fn new(schema: Schema, storage: Arc<dyn Storage>) -> Result<Self> {
        let mut reader = BasicIndexReader {
            schema,
            storage,
            document_cache: Vec::new(),
            bkd_trees: HashMap::new(),
            closed: false,
        };

        // Load existing segments
        reader.load_segments()?;

        // Build BKD Trees for numeric fields
        reader.build_bkd_trees()?;

        Ok(reader)
    }

    /// Load segments from storage.
    #[allow(dead_code)]
    fn load_segments(&mut self) -> Result<()> {
        let files = self.storage.list_files()?;

        // Find and load all segment files
        for file in files {
            if file.starts_with("segment_") && file.ends_with(".json") {
                self.load_segment(&file)?;
            }
        }

        // Debug: Print document cache size

        Ok(())
    }

    /// Load a segment file.
    fn load_segment(&mut self, filename: &str) -> Result<()> {
        let mut input = self.storage.open_input(filename)?;
        let mut segment_data = String::new();
        std::io::Read::read_to_string(&mut input, &mut segment_data)?;

        let documents: Vec<Document> = serde_json::from_str(&segment_data)
            .map_err(|e| SarissaError::index(format!("Failed to deserialize segment: {e}")))?;

        // Add documents to cache
        self.document_cache.extend(documents);

        Ok(())
    }

    /// Build BKD Trees for all numeric fields.
    fn build_bkd_trees(&mut self) -> Result<()> {
        // Find all numeric fields in the schema
        for (field_name, field_def) in self.schema.fields() {
            // Check if this is a numeric field by type name
            if field_def.field_type().type_name() == "numeric" {
                let mut entries = Vec::new();

                // Extract numeric values from all documents
                for (doc_id, doc) in self.document_cache.iter().enumerate() {
                    if let Some(field_value) = doc.get_field(field_name) {
                        if let Some(numeric_value) = self.extract_numeric_value(field_value) {
                            entries.push((numeric_value, doc_id as u64));
                        }
                    }
                }

                // Build BKD Tree for this field
                if !entries.is_empty() {
                    let bkd_tree = SimpleBKDTree::new(field_name.to_string(), entries);
                    self.bkd_trees.insert(field_name.to_string(), bkd_tree);
                }
            }
        }

        Ok(())
    }

    /// Extract numeric value from a field value.
    fn extract_numeric_value(&self, field_value: &FieldValue) -> Option<f64> {
        match field_value {
            FieldValue::Float(f) => Some(*f),
            FieldValue::Integer(i) => Some(*i as f64),
            _ => None,
        }
    }

    /// Check if the reader is closed.
    fn check_closed(&self) -> Result<()> {
        if self.closed {
            Err(SarissaError::index("Reader is closed"))
        } else {
            Ok(())
        }
    }
}

impl IndexReader for BasicIndexReader {
    fn doc_count(&self) -> u64 {
        self.document_cache.len() as u64
    }

    fn max_doc(&self) -> u64 {
        self.document_cache.len() as u64
    }

    fn is_deleted(&self, _doc_id: u64) -> bool {
        // For now, we don't support deletions
        false
    }

    fn document(&self, doc_id: u64) -> Result<Option<Document>> {
        self.check_closed()?;

        if doc_id >= self.document_cache.len() as u64 {
            return Ok(None);
        }

        let doc = self.document_cache[doc_id as usize].clone();
        Ok(Some(doc))
    }

    fn schema(&self) -> &Schema {
        &self.schema
    }

    fn term_info(&self, field: &str, term: &str) -> Result<Option<ReaderTermInfo>> {
        self.check_closed()?;

        // Get field type to determine matching strategy
        let field_type = self
            .schema
            .get_field(field)
            .map(|field_def| field_def.field_type().type_name());

        let mut doc_freq = 0u64;
        let mut total_term_freq = 0u64;

        for doc in &self.document_cache {
            if let Some(field_value) = doc.get_field(field) {
                if let Some(text) = field_value.as_text() {
                    match field_type {
                        Some("id") => {
                            // IdField: exact string matching (case-sensitive)
                            if text == term {
                                doc_freq += 1;
                                total_term_freq += 1;
                            }
                        }
                        _ => {
                            // TextField: token-based matching (case-sensitive)
                            let tokens: Vec<&str> = text.split_whitespace().collect();
                            let mut found_in_doc = false;
                            let mut term_count_in_doc = 0u64;

                            for token in tokens {
                                if token == term {
                                    if !found_in_doc {
                                        doc_freq += 1;
                                        found_in_doc = true;
                                    }
                                    term_count_in_doc += 1;
                                }
                            }
                            total_term_freq += term_count_in_doc;
                        }
                    }
                }
            }
        }

        if doc_freq > 0 {
            Ok(Some(ReaderTermInfo {
                field: field.to_string(),
                term: term.to_string(),
                doc_freq,
                total_freq: total_term_freq,
                posting_offset: 0, // Simplified implementation
                posting_size: 0,   // Simplified implementation
            }))
        } else {
            Ok(None)
        }
    }

    fn postings(&self, field: &str, term: &str) -> Result<Option<Box<dyn PostingIterator>>> {
        self.check_closed()?;

        // Get field type to determine matching strategy
        let field_type = self
            .schema
            .get_field(field)
            .map(|field_def| field_def.field_type().type_name());

        let mut matching_docs = Vec::new();

        for (doc_id, doc) in self.document_cache.iter().enumerate() {
            if let Some(field_value) = doc.get_field(field) {
                if let Some(text) = field_value.as_text() {
                    match field_type {
                        Some("id") => {
                            // IdField: exact string matching
                            if text == term {
                                matching_docs.push((doc_id as u64, 1, vec![0]));
                            }
                        }
                        _ => {
                            // TextField: token-based matching with positions
                            let tokens: Vec<&str> = text.split_whitespace().collect();
                            let mut positions = Vec::new();
                            let mut term_freq = 0;

                            for (pos, token) in tokens.iter().enumerate() {
                                if *token == term {
                                    positions.push(pos as u64);
                                    term_freq += 1;
                                }
                            }

                            if term_freq > 0 {
                                matching_docs.push((doc_id as u64, term_freq, positions));
                            }
                        }
                    }
                }
            }
        }

        if matching_docs.is_empty() {
            Ok(None)
        } else {
            let doc_ids: Vec<u64> = matching_docs.iter().map(|(id, _, _)| *id).collect();
            let term_freqs: Vec<u64> = matching_docs.iter().map(|(_, freq, _)| *freq).collect();
            let positions_vec: Vec<Vec<u64>> = matching_docs
                .iter()
                .map(|(_, _, pos)| pos.clone())
                .collect();

            Ok(Some(Box::new(BasicPostingIterator::with_positions(
                doc_ids,
                term_freqs,
                positions_vec,
            )?)))
        }
    }

    fn field_stats(&self, field: &str) -> Result<Option<FieldStats>> {
        self.check_closed()?;

        let mut doc_count = 0u64;
        let mut total_length = 0u64;
        let mut term_count = 0u64;
        let mut min_length = u64::MAX;
        let mut max_length = 0u64;

        for doc in &self.document_cache {
            if let Some(field_value) = doc.get_field(field) {
                if let Some(text) = field_value.as_text() {
                    doc_count += 1;
                    // Simple tokenization by whitespace
                    let tokens: Vec<&str> = text.split_whitespace().collect();
                    let field_length = tokens.len() as u64;
                    total_length += field_length;
                    term_count += field_length;
                    min_length = min_length.min(field_length);
                    max_length = max_length.max(field_length);
                }
            }
        }

        if doc_count > 0 {
            let avg_length = total_length as f64 / doc_count as f64;
            Ok(Some(FieldStats {
                field: field.to_string(),
                doc_count,
                unique_terms: term_count / doc_count.max(1), // Rough estimate
                total_terms: term_count,
                avg_length,
                min_length,
                max_length,
            }))
        } else {
            Ok(None)
        }
    }

    fn close(&mut self) -> Result<()> {
        if !self.closed {
            self.closed = true;
            self.document_cache.clear();
            self.bkd_trees.clear();
        }
        Ok(())
    }

    fn is_closed(&self) -> bool {
        self.closed
    }

    fn get_bkd_tree(&self, field: &str) -> Result<Option<&SimpleBKDTree>> {
        self.check_closed()?;
        Ok(self.bkd_trees.get(field))
    }
}

/// A basic posting iterator implementation.
#[derive(Debug)]
pub struct BasicPostingIterator {
    /// Document IDs in the posting list.
    doc_ids: Vec<u64>,

    /// Term frequencies for each document.
    term_freqs: Vec<u64>,

    /// Positions of terms within each document.
    positions: Vec<Vec<u64>>,

    /// Current position in the posting list.
    position: usize,

    /// Whether we've reached the end.
    exhausted: bool,

    /// Whether next() has been called at least once.
    started: bool,
}

impl BasicPostingIterator {
    /// Create a new posting iterator.
    pub fn new(doc_ids: Vec<u64>, term_freqs: Vec<u64>) -> Result<Self> {
        if doc_ids.len() != term_freqs.len() {
            return Err(SarissaError::index(
                "Document IDs and term frequencies must have the same length",
            ));
        }

        // Create empty positions for backward compatibility
        let positions = vec![Vec::new(); doc_ids.len()];

        Ok(BasicPostingIterator {
            doc_ids,
            term_freqs,
            positions,
            position: 0,
            exhausted: false,
            started: false,
        })
    }

    /// Create a new posting iterator with position information.
    pub fn with_positions(
        doc_ids: Vec<u64>,
        term_freqs: Vec<u64>,
        positions: Vec<Vec<u64>>,
    ) -> Result<Self> {
        if doc_ids.len() != term_freqs.len() || doc_ids.len() != positions.len() {
            return Err(SarissaError::index(
                "Document IDs, term frequencies, and positions must have the same length",
            ));
        }

        Ok(BasicPostingIterator {
            doc_ids,
            term_freqs,
            positions,
            position: 0,
            exhausted: false,
            started: false,
        })
    }

    /// Create an empty posting iterator.
    pub fn empty() -> Self {
        BasicPostingIterator {
            doc_ids: Vec::new(),
            term_freqs: Vec::new(),
            positions: Vec::new(),
            position: 0,
            exhausted: true,
            started: false,
        }
    }
}

impl PostingIterator for BasicPostingIterator {
    fn doc_id(&self) -> u64 {
        if self.exhausted || self.position >= self.doc_ids.len() {
            u64::MAX
        } else {
            self.doc_ids[self.position]
        }
    }

    fn term_freq(&self) -> u64 {
        if self.exhausted || self.position >= self.term_freqs.len() {
            0
        } else {
            self.term_freqs[self.position]
        }
    }

    fn positions(&self) -> Result<Vec<u64>> {
        if self.exhausted || self.position >= self.positions.len() {
            Ok(Vec::new())
        } else {
            Ok(self.positions[self.position].clone())
        }
    }

    fn next(&mut self) -> Result<bool> {
        if self.exhausted || self.doc_ids.is_empty() {
            return Ok(false);
        }

        if !self.started {
            // First call - position at first document
            self.started = true;
            Ok(true)
        } else {
            // Move to next document
            self.position += 1;

            if self.position >= self.doc_ids.len() {
                self.exhausted = true;
                Ok(false)
            } else {
                Ok(true)
            }
        }
    }

    fn skip_to(&mut self, target: u64) -> Result<bool> {
        if self.exhausted || self.doc_ids.is_empty() {
            return Ok(false);
        }

        // Mark as started
        self.started = true;

        // Use binary search for efficient skip_to operation
        let search_range = &self.doc_ids[self.position..];
        match search_range.binary_search(&target) {
            Ok(index) => {
                // Exact match found
                self.position += index;
                Ok(true)
            }
            Err(index) => {
                // Target not found, index is insertion point
                self.position += index;
                if self.position >= self.doc_ids.len() {
                    self.exhausted = true;
                    Ok(false)
                } else {
                    Ok(true)
                }
            }
        }
    }

    fn cost(&self) -> u64 {
        self.doc_ids.len() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{Schema, TextField};
    use crate::storage::{MemoryStorage, StorageConfig};
    use std::sync::Arc;

    #[allow(dead_code)]
    fn create_test_schema() -> Schema {
        let mut schema = Schema::new().unwrap();
        schema
            .add_field("title", Box::new(TextField::new().stored(true)))
            .unwrap();
        schema
            .add_field("body", Box::new(TextField::new()))
            .unwrap();
        schema
    }

    #[test]
    fn test_reader_creation() {
        let schema = create_test_schema();
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));

        let reader = BasicIndexReader::new(schema, storage).unwrap();

        assert!(!reader.is_closed());
        assert_eq!(reader.doc_count(), 0);
        assert_eq!(reader.max_doc(), 0);
        assert_eq!(reader.schema().len(), 2);
    }

    #[test]
    fn test_reader_close() {
        let schema = create_test_schema();
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));

        let mut reader = BasicIndexReader::new(schema, storage).unwrap();

        assert!(!reader.is_closed());

        reader.close().unwrap();

        assert!(reader.is_closed());

        // Operations should fail after close
        let result = reader.document(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_posting_iterator() {
        let doc_ids = vec![1, 5, 10, 15];
        let term_freqs = vec![2, 1, 3, 1];

        let mut iter = BasicPostingIterator::new(doc_ids, term_freqs).unwrap();

        // Initially not positioned at any document
        assert_eq!(iter.doc_id(), 1);
        assert_eq!(iter.term_freq(), 2);

        assert!(iter.next().unwrap()); // Move to first document
        assert_eq!(iter.doc_id(), 1);
        assert_eq!(iter.term_freq(), 2);

        assert!(iter.next().unwrap());
        assert_eq!(iter.doc_id(), 5);
        assert_eq!(iter.term_freq(), 1);

        assert!(iter.next().unwrap());
        assert_eq!(iter.doc_id(), 10);
        assert_eq!(iter.term_freq(), 3);

        assert!(iter.next().unwrap());
        assert_eq!(iter.doc_id(), 15);
        assert_eq!(iter.term_freq(), 1);

        assert!(!iter.next().unwrap());
        assert_eq!(iter.doc_id(), u64::MAX);
    }

    #[test]
    fn test_posting_iterator_skip_to() {
        let doc_ids = vec![1, 5, 10, 15, 20];
        let term_freqs = vec![2, 1, 3, 1, 2];

        let mut iter = BasicPostingIterator::new(doc_ids, term_freqs).unwrap();

        // Skip to document 8
        assert!(iter.skip_to(8).unwrap());
        assert_eq!(iter.doc_id(), 10);

        // Skip to document 18
        assert!(iter.skip_to(18).unwrap());
        assert_eq!(iter.doc_id(), 20);

        // Skip beyond the end
        assert!(!iter.skip_to(25).unwrap());
        assert_eq!(iter.doc_id(), u64::MAX);
    }

    #[test]
    fn test_empty_posting_iterator() {
        let iter = BasicPostingIterator::empty();

        assert_eq!(iter.doc_id(), u64::MAX);
        assert_eq!(iter.term_freq(), 0);
        assert_eq!(iter.cost(), 0);
    }

    #[test]
    fn test_posting_iterator_cost() {
        let doc_ids = vec![1, 2, 3, 4, 5];
        let term_freqs = vec![1, 1, 1, 1, 1];

        let iter = BasicPostingIterator::new(doc_ids, term_freqs).unwrap();
        assert_eq!(iter.cost(), 5);
    }

    #[test]
    fn test_posting_iterator_mismatched_lengths() {
        let doc_ids = vec![1, 2, 3];
        let term_freqs = vec![1, 1]; // Different length

        let result = BasicPostingIterator::new(doc_ids, term_freqs);
        assert!(result.is_err());
    }
}
