//! High-level lexical search engine that combines indexing and searching.
//!
//! This module provides a unified interface for lexical indexing and search,
//! similar to the VectorEngine for vector search.

use std::cell::{Ref, RefCell, RefMut};
use std::path::Path;
use std::sync::Arc;

use crate::document::document::Document;
use crate::error::Result;
use crate::lexical::inverted_index::{FileIndex, Index, IndexConfig, IndexStats};
use crate::lexical::search::searcher::inverted_index::Searcher;
use crate::lexical::types::SearchRequest;
use crate::lexical::writer::IndexWriter;
use crate::query::SearchResults;
use crate::query::query::Query;
use crate::storage::traits::Storage;

/// A high-level lexical search engine that provides both indexing and searching capabilities.
/// This is similar to the VectorEngine but for lexical search.
pub struct LexicalEngine {
    /// The underlying index.
    index: FileIndex,
    /// The searcher for executing queries.
    searcher: RefCell<Option<Searcher>>,
    /// The writer for adding/updating documents (cached for efficiency).
    writer: RefCell<Option<Box<dyn IndexWriter>>>,
}

impl std::fmt::Debug for LexicalEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LexicalEngine")
            .field("index", &self.index)
            .field("searcher", &self.searcher)
            .field("writer", &"<cached writer>")
            .finish()
    }
}

impl LexicalEngine {
    /// Create a new lexical search engine with the given index.
    pub fn new(index: FileIndex) -> Self {
        LexicalEngine {
            index,
            searcher: RefCell::new(None),
            writer: RefCell::new(None),
        }
    }

    /// Create a new lexical search engine in the given directory (schema-less mode).
    pub fn create_in_dir<P: AsRef<Path>>(dir: P, index_config: IndexConfig) -> Result<Self> {
        let index = FileIndex::create_in_dir(dir, index_config)?;
        Ok(LexicalEngine::new(index))
    }

    /// Open an existing lexical search engine from the given directory.
    pub fn open_dir<P: AsRef<Path>>(dir: P, index_config: IndexConfig) -> Result<Self> {
        let index = FileIndex::open_dir(dir, index_config)?;
        Ok(LexicalEngine::new(index))
    }

    /// Get the storage backend.
    pub fn storage(&self) -> &Arc<dyn Storage> {
        self.index.storage()
    }

    /// Get or create a writer for this engine.
    fn get_or_create_writer(&self) -> Result<RefMut<'_, Box<dyn IndexWriter>>> {
        {
            let mut writer_ref = self.writer.borrow_mut();
            if writer_ref.is_none() {
                *writer_ref = Some(self.index.writer()?);
            }
        }

        // Return a mutable reference to the writer
        Ok(RefMut::map(self.writer.borrow_mut(), |opt| {
            opt.as_mut().unwrap()
        }))
    }

    /// Add a document to the index.
    /// Note: You must call `commit()` to persist the changes.
    pub fn add_document(&mut self, doc: Document) -> Result<()> {
        let mut writer = self.get_or_create_writer()?;
        writer.add_document(doc)?;

        Ok(())
    }

    /// Add multiple documents to the index.
    /// Note: You must call `commit()` to persist the changes.
    pub fn add_documents(&mut self, docs: Vec<Document>) -> Result<()> {
        let mut writer = self.get_or_create_writer()?;

        for doc in docs {
            writer.add_document(doc)?;
        }

        Ok(())
    }

    /// Delete documents matching the given term.
    /// Note: You must call `commit()` to persist the changes.
    pub fn delete_documents(&mut self, field: &str, value: &str) -> Result<u64> {
        let mut writer = self.get_or_create_writer()?;
        let count = writer.delete_documents(field, value)?;

        Ok(count)
    }

    /// Update a document (delete old, add new).
    /// Note: You must call `commit()` to persist the changes.
    pub fn update_document(&mut self, field: &str, value: &str, doc: Document) -> Result<()> {
        let mut writer = self.get_or_create_writer()?;
        writer.update_document(field, value, doc)?;

        Ok(())
    }

    /// Commit any pending changes to the index.
    pub fn commit(&mut self) -> Result<()> {
        // Take the cached writer if it exists
        if let Some(mut writer) = self.writer.borrow_mut().take() {
            writer.commit()?;
        }

        // Invalidate searcher cache to reflect the new changes
        *self.searcher.borrow_mut() = None;

        Ok(())
    }

    /// Optimize the index.
    pub fn optimize(&mut self) -> Result<()> {
        self.index.optimize()?;

        // Invalidate searcher cache
        *self.searcher.borrow_mut() = None;

        Ok(())
    }

    /// Get or create a searcher for this engine.
    fn get_searcher(&'_ self) -> Result<Ref<'_, Searcher>> {
        {
            let mut searcher_ref = self.searcher.borrow_mut();
            if searcher_ref.is_none() {
                let reader = self.index.reader()?;
                *searcher_ref = Some(Searcher::new(reader));
            }
        }

        // Return a reference to the searcher
        Ok(Ref::map(self.searcher.borrow(), |opt| {
            opt.as_ref().unwrap()
        }))
    }

    /// Refresh the searcher to see latest changes.
    pub fn refresh(&mut self) -> Result<()> {
        *self.searcher.borrow_mut() = None;
        Ok(())
    }

    /// Get index statistics.
    pub fn stats(&self) -> Result<IndexStats> {
        self.index.stats()
    }

    /// Close the search engine.
    pub fn close(&mut self) -> Result<()> {
        // Drop the cached writer
        *self.writer.borrow_mut() = None;
        // Drop the cached searcher
        *self.searcher.borrow_mut() = None;
        self.index.close()
    }

    /// Check if the engine is closed.
    pub fn is_closed(&self) -> bool {
        self.index.is_closed()
    }
}

impl LexicalEngine {
    /// Search with the given request.
    pub fn search(&self, request: SearchRequest) -> Result<SearchResults> {
        let searcher = self.get_searcher()?;
        Searcher::search(&searcher, request)
    }

    /// Count documents matching the query.
    pub fn count(&self, query: Box<dyn Query>) -> Result<u64> {
        let searcher = self.get_searcher()?;
        Searcher::count(&searcher, query)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::term::TermQuery;

    use tempfile::TempDir;

    #[allow(dead_code)]
    fn create_test_document(title: &str, body: &str) -> Document {
        Document::builder()
            .add_text("title", title)
            .add_text("body", body)
            .build()
    }

    #[test]
    fn test_search_engine_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = IndexConfig::default();

        let engine = LexicalEngine::create_in_dir(temp_dir.path(), config).unwrap();

        // Schema-less mode: no schema() method available
        assert!(!engine.is_closed());
    }

    #[test]
    fn test_search_engine_open() {
        let temp_dir = TempDir::new().unwrap();
        let config = IndexConfig::default();

        // Create engine
        let mut engine = LexicalEngine::create_in_dir(temp_dir.path(), config.clone()).unwrap();
        engine.close().unwrap();

        // Open engine
        let engine = LexicalEngine::open_dir(temp_dir.path(), config).unwrap();

        // Schema-less mode: no schema() method available
        assert!(!engine.is_closed());
    }

    #[test]
    fn test_add_document() {
        let temp_dir = TempDir::new().unwrap();
        let config = IndexConfig::default();

        let mut engine = LexicalEngine::create_in_dir(temp_dir.path(), config).unwrap();

        let doc = create_test_document("Hello World", "This is a test document");
        engine.add_document(doc).unwrap();
        engine.commit().unwrap();

        // Check that document was added (through stats)
        let _stats = engine.stats().unwrap();
        // Note: stats might not reflect the added document immediately
        // depending on the index implementation
        // doc_count is usize, so >= 0 check is redundant
    }

    #[test]
    fn test_add_multiple_documents() {
        let temp_dir = TempDir::new().unwrap();
        let config = IndexConfig::default();

        let mut engine = LexicalEngine::create_in_dir(temp_dir.path(), config).unwrap();

        let docs = vec![
            create_test_document("First Document", "Content of first document"),
            create_test_document("Second Document", "Content of second document"),
            create_test_document("Third Document", "Content of third document"),
        ];

        engine.add_documents(docs).unwrap();
        engine.commit().unwrap();

        let _stats = engine.stats().unwrap();
        // doc_count is usize, so >= 0 check is redundant
    }

    #[test]
    fn test_search_empty_index() {
        let temp_dir = TempDir::new().unwrap();
        let config = IndexConfig::default();

        let engine = LexicalEngine::create_in_dir(temp_dir.path(), config).unwrap();

        let query = Box::new(TermQuery::new("title", "hello"));
        let request = SearchRequest::new(query);
        let results = engine.search(request).unwrap();

        assert_eq!(results.hits.len(), 0);
        assert_eq!(results.total_hits, 0);
        assert_eq!(results.max_score, 0.0);
    }

    #[test]
    fn test_search_with_documents() {
        let temp_dir = TempDir::new().unwrap();
        let config = IndexConfig::default();

        let mut engine = LexicalEngine::create_in_dir(temp_dir.path(), config).unwrap();

        // Add some documents
        let docs = vec![
            create_test_document("Hello World", "This is a test document"),
            create_test_document("Goodbye World", "This is another test document"),
        ];
        engine.add_documents(docs).unwrap();
        engine.commit().unwrap();

        // Search for documents
        let query = Box::new(TermQuery::new("title", "Hello"));
        let request = SearchRequest::new(query);
        let _results = engine.search(request).unwrap();

        // Results depend on the actual indexing implementation
        // For now, we just check that search doesn't fail
        // hits.len() is usize, so >= 0 check is redundant
        // total_hits is u64, so >= 0 check is redundant
    }

    #[test]
    fn test_count_query() {
        let temp_dir = TempDir::new().unwrap();
        let config = IndexConfig::default();

        let engine = LexicalEngine::create_in_dir(temp_dir.path(), config).unwrap();

        let query = Box::new(TermQuery::new("title", "hello"));
        let count = engine.count(query).unwrap();

        // Should return 0 for empty index
        assert_eq!(count, 0);
    }

    #[test]
    fn test_engine_refresh() {
        let temp_dir = TempDir::new().unwrap();
        let config = IndexConfig::default();

        let mut engine = LexicalEngine::create_in_dir(temp_dir.path(), config).unwrap();

        // Add a document
        let doc = create_test_document("Test Document", "Test content");
        engine.add_document(doc).unwrap();
        engine.commit().unwrap();

        // Refresh should not fail
        engine.refresh().unwrap();

        // Search should still work
        let query = Box::new(TermQuery::new("title", "Test"));
        let request = SearchRequest::new(query);
        let _results = engine.search(request).unwrap();
        // hits.len() is usize, so >= 0 check is redundant
    }

    #[test]
    fn test_engine_stats() {
        let temp_dir = TempDir::new().unwrap();
        let config = IndexConfig::default();

        let engine = LexicalEngine::create_in_dir(temp_dir.path(), config).unwrap();

        let stats = engine.stats().unwrap();
        // doc_count is usize, so >= 0 check is redundant
        // term_count is usize, so >= 0 check is redundant
        assert!(stats.last_modified > 0);
    }

    #[test]
    fn test_engine_close() {
        let temp_dir = TempDir::new().unwrap();
        let config = IndexConfig::default();

        let mut engine = LexicalEngine::create_in_dir(temp_dir.path(), config).unwrap();

        assert!(!engine.is_closed());

        engine.close().unwrap();

        assert!(engine.is_closed());
    }

    #[test]
    fn test_search_request_configuration() {
        let temp_dir = TempDir::new().unwrap();
        let config = IndexConfig::default();

        let engine = LexicalEngine::create_in_dir(temp_dir.path(), config).unwrap();

        let query = Box::new(TermQuery::new("title", "hello"));
        let request = SearchRequest::new(query)
            .max_docs(5)
            .min_score(0.5)
            .load_documents(false);

        let results = engine.search(request).unwrap();

        // Should respect the configuration
        assert_eq!(results.hits.len(), 0); // No matching documents
        assert_eq!(results.total_hits, 0);
    }

    #[test]
    fn test_search_with_query_parser() {
        let temp_dir = TempDir::new().unwrap();
        let config = IndexConfig::default();

        let mut engine = LexicalEngine::create_in_dir(temp_dir.path(), config).unwrap();

        // Add some documents with lowercase titles for testing
        let docs = vec![
            create_test_document("hello world", "This is a test document"),
            create_test_document("goodbye world", "This is another test document"),
        ];
        engine.add_documents(docs).unwrap();
        engine.commit().unwrap();

        // Search with QueryParser (Lucene style)
        use crate::query::parser::QueryParser;
        let parser = QueryParser::with_standard_analyzer()
            .unwrap()
            .with_default_field("title");

        // QueryParser analyzes "Hello" to "hello" before creating TermQuery
        let query = parser.parse("Hello").unwrap();
        let results = engine.search(SearchRequest::new(query)).unwrap();

        // Should find the document
        // QueryParser analyzes "Hello" -> "hello", which matches the indexed "hello"
        assert_eq!(results.hits.len(), 1);
        assert_eq!(results.total_hits, 1);
    }

    #[test]
    fn test_search_field_with_string() {
        let temp_dir = TempDir::new().unwrap();
        let config = IndexConfig::default();

        let engine = LexicalEngine::create_in_dir(temp_dir.path(), config).unwrap();

        // Search specific field
        use crate::query::parser::QueryParser;
        let parser = QueryParser::new();
        let query = parser.parse_field("title", "hello world").unwrap();
        let results = engine.search(SearchRequest::new(query)).unwrap();

        // Should parse and execute the query
        assert_eq!(results.hits.len(), 0);
        assert_eq!(results.total_hits, 0);
    }

    #[test]
    fn test_query_parser_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = IndexConfig::default();

        let _engine = LexicalEngine::create_in_dir(temp_dir.path(), config).unwrap();

        use crate::query::parser::QueryParser;
        let parser = QueryParser::new();
        assert!(parser.default_field().is_none());

        let parser_with_default = QueryParser::new().with_default_field("title");
        assert_eq!(parser_with_default.default_field(), Some("title"));
    }

    #[test]
    fn test_complex_string_query() {
        let temp_dir = TempDir::new().unwrap();
        let config = IndexConfig::default();

        let engine = LexicalEngine::create_in_dir(temp_dir.path(), config).unwrap();

        // Test complex query parsing
        use crate::query::parser::QueryParser;
        let parser = QueryParser::new().with_default_field("title");
        let query = parser.parse("title:hello AND body:world").unwrap();
        let results = engine.search(SearchRequest::new(query)).unwrap();

        // Should parse the complex query without error
        assert_eq!(results.hits.len(), 0);
        assert_eq!(results.total_hits, 0);
    }
}
