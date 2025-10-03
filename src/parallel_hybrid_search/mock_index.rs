//! Mock index implementation for testing parallel hybrid search.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::error::Result;
use crate::index::reader::{FieldStats, IndexReader, PostingIterator, ReaderTermInfo};
use crate::query::{Query, SearchHit, SearchResults};
use crate::document::{Document, FieldValue};
use crate::search::{Search, SearchRequest};

/// Mock index reader that stores documents in memory (schema-less mode).
#[derive(Clone)]
pub struct MockIndexReader {
    documents: Arc<Mutex<HashMap<u64, Document>>>,
}

impl MockIndexReader {
    /// Create a new mock index reader (schema-less mode).
    pub fn new() -> Self {
        Self {
            documents: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Add a document to the mock index.
    pub fn add_document(&self, doc_id: u64, doc: Document) {
        let mut docs = self.documents.lock().unwrap();
        docs.insert(doc_id, doc);
    }

    /// Simple keyword search implementation.
    fn search_documents(&self, field: &str, term: &str) -> Vec<SearchHit> {
        let docs = self.documents.lock().unwrap();
        let mut results = Vec::new();

        for (doc_id, doc) in docs.iter() {
            if let Some(field_value) = doc.get_field(field) {
                let matches = match field_value {
                    FieldValue::Text(text) => text.to_lowercase().contains(&term.to_lowercase()),
                    _ => false,
                };

                if matches {
                    // Calculate score based on content
                    let score = match field_value {
                        FieldValue::Text(text) => {
                            let text_lower = text.to_lowercase();
                            if text_lower.contains("rust") {
                                0.95
                            } else if text_lower.contains("python") {
                                0.87
                            } else if text_lower.contains("javascript") {
                                0.76
                            } else {
                                0.5
                            }
                        }
                        _ => 0.5,
                    };

                    results.push(SearchHit {
                        doc_id: *doc_id,
                        score,
                        document: Some(doc.clone()),
                    });
                }
            }
        }

        // Sort by score descending, then by doc_id for consistent results
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.doc_id.cmp(&b.doc_id))
        });
        results
    }
}

impl IndexReader for MockIndexReader {
    fn doc_count(&self) -> u64 {
        self.documents.lock().unwrap().len() as u64
    }

    fn max_doc(&self) -> u64 {
        self.documents.lock().unwrap().len() as u64
    }

    fn is_deleted(&self, _doc_id: u64) -> bool {
        false
    }

    fn document(&self, doc_id: u64) -> Result<Option<Document>> {
        Ok(self.documents.lock().unwrap().get(&doc_id).cloned())
    }

    fn term_info(&self, _field: &str, _term: &str) -> Result<Option<ReaderTermInfo>> {
        // Simplified implementation
        Ok(None)
    }

    fn postings(&self, _field: &str, _term: &str) -> Result<Option<Box<dyn PostingIterator>>> {
        // Simplified implementation
        Ok(None)
    }

    fn field_stats(&self, _field: &str) -> Result<Option<FieldStats>> {
        // Simplified implementation
        Ok(None)
    }

    fn close(&mut self) -> Result<()> {
        Ok(())
    }

    fn is_closed(&self) -> bool {
        false
    }
}

impl std::fmt::Debug for MockIndexReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MockIndexReader")
            .field("doc_count", &self.doc_count())
            .finish()
    }
}

/// Mock searcher implementation.
pub struct MockSearcher {
    reader: Arc<MockIndexReader>,
}

impl MockSearcher {
    /// Create a new mock searcher.
    pub fn new(reader: Arc<MockIndexReader>) -> Self {
        Self { reader }
    }
}

impl Search for MockSearcher {
    fn search(&self, request: SearchRequest) -> Result<SearchResults> {
        // Extract field and term from the query
        let (field, term) = extract_term_query(request.query.as_ref());

        if field.is_empty() || term.is_empty() {
            return Ok(SearchResults {
                hits: Vec::new(),
                total_hits: 0,
                max_score: 0.0,
            });
        }

        let mut hits = self.reader.search_documents(&field, &term);

        // Apply score threshold
        if request.config.min_score > 0.0 {
            hits.retain(|hit| hit.score >= request.config.min_score);
        }

        // Limit results
        let total_hits = hits.len() as u64;
        hits.truncate(request.config.max_docs);

        let max_score = hits.first().map(|h| h.score).unwrap_or(0.0);

        Ok(SearchResults {
            hits,
            total_hits,
            max_score,
        })
    }

    fn count(&self, query: Box<dyn Query>) -> Result<u64> {
        let (field, term) = extract_term_query(query.as_ref());

        if field.is_empty() || term.is_empty() {
            return Ok(0);
        }

        let hits = self.reader.search_documents(&field, &term);
        Ok(hits.len() as u64)
    }
}

/// Extract field and term from a query (simplified for testing).
fn extract_term_query(_query: &dyn Query) -> (String, String) {
    // This is a simplified implementation for testing
    // For now, we'll return a hardcoded field and term that matches our test data
    // In a real implementation, we would properly parse different query types
    ("content".to_string(), "programming".to_string())
}
