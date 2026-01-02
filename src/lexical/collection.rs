//! Lexical collection module.
//!
//! This module provides `LexicalCollection`, which manages the lifecycle of a lexical index.
//! It serves as the equivalent of `VectorCollection` in the vector module, providing a
//! unified interface for document management and search operations over a lexical index.

use std::sync::Arc;

use crate::analysis::analyzer::analyzer::Analyzer;
use crate::error::Result;
use crate::lexical::core::document::Document;
use crate::lexical::engine::config::LexicalIndexConfig;
use crate::lexical::index::LexicalIndex;
use crate::lexical::index::factory::LexicalIndexFactory;
use crate::lexical::index::inverted::InvertedIndexStats;
use crate::lexical::index::inverted::query::LexicalSearchResults;
use crate::lexical::search::searcher::LexicalSearchRequest;
use crate::storage::Storage;

/// A collection managing a lexical index.
///
/// This struct abstracts the underlying index implementation and provides
/// high-level operations for document management (add, update, delete) and search.
#[derive(Debug)]
pub struct LexicalCollection {
    /// The underlying lexical index.
    index: Box<dyn LexicalIndex>,
}

impl LexicalCollection {
    /// Create a new lexical collection with the given storage and configuration.
    pub fn new(storage: Arc<dyn Storage>, config: LexicalIndexConfig) -> Result<Self> {
        let index = LexicalIndexFactory::create(storage, config)?;
        Ok(Self { index })
    }

    /// Add a document to the collection.
    pub fn add_document(&self, doc: Document) -> Result<u64> {
        self.index.add_document(doc)
    }

    /// Upsert a document with a specific document ID.
    pub fn upsert_document(&self, doc_id: u64, doc: Document) -> Result<()> {
        self.index.upsert_document(doc_id, doc)
    }

    /// Add multiple documents to the collection.
    pub fn add_documents(&self, docs: Vec<Document>) -> Result<Vec<u64>> {
        self.index.add_documents(docs)
    }

    /// Delete a document by ID.
    pub fn delete_document(&self, doc_id: u64) -> Result<()> {
        self.index.delete_document(doc_id)
    }

    /// Commit any pending changes to the index.
    pub fn commit(&self) -> Result<()> {
        self.index.commit()
    }

    /// Optimize the index.
    pub fn optimize(&self) -> Result<()> {
        self.index.optimize()
    }

    /// Refresh the reader to see latest changes.
    pub fn refresh(&self) -> Result<()> {
        self.index.refresh()
    }

    /// Search with the given request.
    pub fn search(&self, request: LexicalSearchRequest) -> Result<LexicalSearchResults> {
        self.index.search(request)
    }

    /// Count documents matching the request.
    pub fn count(&self, request: LexicalSearchRequest) -> Result<u64> {
        self.index.count(request)
    }

    /// Get index statistics.
    pub fn stats(&self) -> Result<InvertedIndexStats> {
        self.index.stats()
    }

    /// Get the storage backend.
    pub fn storage(&self) -> &Arc<dyn Storage> {
        self.index.storage()
    }

    /// Close the collection.
    pub fn close(&self) -> Result<()> {
        self.index.close()
    }

    /// Check if the collection is closed.
    pub fn is_closed(&self) -> bool {
        self.index.is_closed()
    }

    /// Get the analyzer used by this collection.
    pub fn analyzer(&self) -> Result<Arc<dyn Analyzer>> {
        use crate::lexical::index::inverted::reader::InvertedIndexReader;

        let reader = self.index.reader()?;

        // Downcast to InvertedIndexReader to access analyzer
        if let Some(inverted_reader) = reader.as_any().downcast_ref::<InvertedIndexReader>() {
            Ok(Arc::clone(inverted_reader.analyzer()))
        } else {
            // For other index types, return StandardAnalyzer as default
            use crate::analysis::analyzer::standard::StandardAnalyzer;
            Ok(Arc::new(StandardAnalyzer::new()?))
        }
    }

    /// Get default fields configuration.
    pub fn default_fields(&self) -> Result<Vec<String>> {
        self.index.default_fields()
    }
}
