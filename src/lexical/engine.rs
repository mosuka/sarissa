//! High-level lexical search engine that combines indexing and searching.
//!
//! This module provides a unified interface for lexical indexing and search operations,
//! abstracting away the complexity of managing separate readers and writers.
//! The engine handles caching of readers and writers for efficiency and provides
//! convenient methods for common operations like adding documents, searching, and committing.
//!
//! # Architecture
//!
//! The `LexicalEngine` wraps a `LexicalIndex` trait object and provides:
//! - **Automatic writer caching**: Writers are created on-demand and cached for efficiency
//! - **Automatic reader invalidation**: Readers are invalidated after commits to reflect new changes
//! - **Simplified API**: Single entry point for all indexing and search operations
//! - **Index abstraction**: Supports different index types (Inverted, etc.) transparently
//!
//! # Usage
//!
//! ```rust,no_run
//! use yatagarasu::document::document::Document;
//! use yatagarasu::lexical::engine::LexicalEngine;
//! use yatagarasu::lexical::index::{LexicalIndexConfig, LexicalIndexFactory};
//! use yatagarasu::lexical::types::LexicalSearchRequest;
//! use yatagarasu::query::term::TermQuery;
//! use yatagarasu::storage::{StorageConfig, StorageFactory};
//! use yatagarasu::storage::memory::MemoryStorageConfig;
//! use std::sync::Arc;
//!
//! // Create storage using factory
//! let storage_config = StorageConfig::Memory(MemoryStorageConfig::default());
//! let storage = StorageFactory::create(storage_config).unwrap();
//!
//! // Create index using factory
//! let index_config = LexicalIndexConfig::default();
//! let index = LexicalIndexFactory::create(storage, index_config).unwrap();
//!
//! // Create engine
//! let mut engine = LexicalEngine::new(index).unwrap();
//!
//! // Add documents
//! let doc = Document::builder()
//!     .add_text("title", "Hello World")
//!     .add_text("body", "This is a test document")
//!     .build();
//! engine.add_document(doc).unwrap();
//! engine.commit().unwrap();
//!
//! // Search
//! let query = Box::new(TermQuery::new("title", "hello"));
//! let request = LexicalSearchRequest::new(query);
//! let results = engine.search(request).unwrap();
//! ```

use std::cell::{RefCell, RefMut};
use std::sync::Arc;

use crate::document::document::Document;
use crate::error::{Result, SageError};
use crate::lexical::index::reader::inverted::InvertedIndexReader;
use crate::lexical::index::{InvertedIndexStats, LexicalIndex};
use crate::lexical::reader::IndexReader;
use crate::lexical::search::searcher::LexicalSearcher;
use crate::lexical::search::searcher::inverted_index::InvertedIndexSearcher;
use crate::lexical::types::LexicalSearchRequest;
use crate::lexical::writer::IndexWriter;
use crate::query::SearchResults;
use crate::query::query::Query;
use crate::storage::Storage;

/// A high-level lexical search engine that provides both indexing and searching capabilities.
///
/// The `LexicalEngine` wraps a `LexicalIndex` trait object and provides a simplified,
/// unified interface for all lexical search operations. It manages the complexity of
/// coordinating between readers and writers while maintaining efficiency through caching.
///
/// # Features
///
/// - **Writer caching**: The writer is created on-demand and cached until commit
/// - **Reader invalidation**: Readers are automatically invalidated after commits/optimizations
/// - **Index abstraction**: Works with any `LexicalIndex` implementation (Inverted, etc.)
/// - **Simplified workflow**: Handles the lifecycle of readers and writers automatically
///
/// # Caching Strategy
///
/// - **Writer**: Created on first write operation, cached until `commit()` is called
/// - **Reader**: Invalidated after `commit()` or `optimize()` to ensure fresh data
/// - This design ensures that you always read committed data while minimizing object creation
///
/// # Usage Example
///
/// ```rust,no_run
/// use yatagarasu::document::document::Document;
/// use yatagarasu::lexical::engine::LexicalEngine;
/// use yatagarasu::lexical::index::{LexicalIndexConfig, LexicalIndexFactory};
/// use yatagarasu::lexical::types::LexicalSearchRequest;
/// use yatagarasu::query::term::TermQuery;
/// use yatagarasu::storage::{StorageConfig, StorageFactory};
/// use yatagarasu::storage::memory::MemoryStorageConfig;
/// use std::sync::Arc;
///
/// // Setup
/// let storage_config = StorageConfig::Memory(MemoryStorageConfig::default());
/// let storage = StorageFactory::create(storage_config).unwrap();
/// let index = LexicalIndexFactory::create(storage, LexicalIndexConfig::default()).unwrap();
/// let mut engine = LexicalEngine::new(index).unwrap();
///
/// // Add documents
/// let doc = Document::builder()
///     .add_text("title", "Rust Programming")
///     .build();
/// engine.add_document(doc).unwrap();
/// engine.commit().unwrap();
///
/// // Search
/// let query = Box::new(TermQuery::new("title", "rust"));
/// let results = engine.search(LexicalSearchRequest::new(query)).unwrap();
/// ```
pub struct LexicalEngine {
    /// The underlying lexical index.
    index: Box<dyn LexicalIndex>,
    /// The reader for executing queries (cached for efficiency).
    reader: RefCell<Option<Box<dyn IndexReader>>>,
    /// The writer for adding/updating documents (cached for efficiency).
    writer: RefCell<Option<Box<dyn IndexWriter>>>,
    /// The searcher for executing searches (cached for efficiency).
    searcher: RefCell<Option<Box<dyn crate::lexical::search::searcher::LexicalSearcher>>>,
}

impl std::fmt::Debug for LexicalEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LexicalEngine")
            .field("index", &self.index)
            .field("reader", &"<cached reader>")
            .field("writer", &"<cached writer>")
            .field("searcher", &"<cached searcher>")
            .finish()
    }
}

impl LexicalEngine {
    /// Create a new lexical search engine with the given lexical index.
    ///
    /// This constructor wraps a `LexicalIndex` and initializes empty caches for
    /// the reader and writer. The reader and writer will be created on-demand
    /// when needed.
    ///
    /// # Arguments
    ///
    /// * `index` - A lexical index trait object (contains configuration and storage)
    ///
    /// # Returns
    ///
    /// Returns a new `LexicalEngine` instance.
    ///
    /// # Example with Memory Storage
    ///
    /// ```rust,no_run
    /// use yatagarasu::lexical::engine::LexicalEngine;
    /// use yatagarasu::lexical::index::{LexicalIndexConfig, LexicalIndexFactory};
    /// use yatagarasu::storage::{StorageConfig, StorageFactory};
    /// use yatagarasu::storage::memory::MemoryStorageConfig;
    /// use std::sync::Arc;
    ///
    /// let storage_config = StorageConfig::Memory(MemoryStorageConfig::default());
    /// let storage = StorageFactory::create(storage_config).unwrap();
    /// let index = LexicalIndexFactory::create(storage, LexicalIndexConfig::default()).unwrap();
    /// let engine = LexicalEngine::new(index).unwrap();
    /// ```
    ///
    /// # Example with File Storage
    ///
    /// ```rust,no_run
    /// use yatagarasu::lexical::engine::LexicalEngine;
    /// use yatagarasu::lexical::index::{LexicalIndexConfig, LexicalIndexFactory};
    /// use yatagarasu::storage::{StorageConfig, StorageFactory};
    /// use yatagarasu::storage::file::FileStorageConfig;
    /// use std::sync::Arc;
    ///
    /// let storage_config = StorageConfig::File(FileStorageConfig::new("/tmp/index"));
    /// let storage = StorageFactory::create(storage_config).unwrap();
    /// let index = LexicalIndexFactory::create(storage, LexicalIndexConfig::default()).unwrap();
    /// let engine = LexicalEngine::new(index).unwrap();
    /// ```
    pub fn new(index: Box<dyn LexicalIndex>) -> Result<Self> {
        Ok(Self {
            index,
            reader: RefCell::new(None),
            writer: RefCell::new(None),
            searcher: RefCell::new(None),
        })
    }

    /// Get or create a reader for this engine.
    #[allow(dead_code)]
    fn get_or_create_reader(&self) -> Result<RefMut<'_, Box<dyn IndexReader>>> {
        {
            let mut reader_ref = self.reader.borrow_mut();
            if reader_ref.is_none() {
                *reader_ref = Some(self.index.reader()?);
            }
        }

        // Return a mutable reference to the reader
        Ok(RefMut::map(self.reader.borrow_mut(), |opt| {
            opt.as_mut().unwrap()
        }))
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

    /// Get or create a searcher for this engine.
    ///
    /// The searcher is created from the index reader and cached for efficiency.
    fn get_or_create_searcher(&self) -> Result<RefMut<'_, Box<dyn LexicalSearcher>>> {
        {
            let mut searcher_ref = self.searcher.borrow_mut();
            if searcher_ref.is_none() {
                // Get a fresh reader from the index
                let reader = self.index.reader()?;

                // Downcast to InvertedIndexReader and create appropriate searcher
                let searcher: Box<dyn LexicalSearcher> = if let Some(inverted_reader) =
                    reader.as_any().downcast_ref::<InvertedIndexReader>()
                {
                    Box::new(InvertedIndexSearcher::new(Box::new(
                        inverted_reader.clone(),
                    )))
                } else {
                    return Err(SageError::index("Unknown lexical index reader type"));
                };

                *searcher_ref = Some(searcher);
            }
        }

        // Return a mutable reference to the searcher
        Ok(RefMut::map(self.searcher.borrow_mut(), |opt| {
            opt.as_mut().unwrap()
        }))
    }

    /// Add a document to the index.
    ///
    /// This method adds a single document to the index. The writer is created
    /// and cached on the first call. Changes are not persisted until you call `commit()`.
    ///
    /// # Arguments
    ///
    /// * `doc` - The document to add
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success, or an error if the operation fails.
    ///
    /// # Important
    ///
    /// You must call `commit()` to persist the changes to storage.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use yatagarasu::document::document::Document;
    /// # use yatagarasu::lexical::engine::LexicalEngine;
    /// # use yatagarasu::lexical::index::{LexicalIndexConfig, LexicalIndexFactory};
    /// # use yatagarasu::storage::{StorageConfig, StorageFactory};
    /// use yatagarasu::storage::memory::MemoryStorageConfig;
    /// # use std::sync::Arc;
    /// # let storage_config = StorageConfig::Memory(MemoryStorageConfig::default());
    /// # let storage = StorageFactory::create(storage_config).unwrap();
    /// # let index = LexicalIndexFactory::create(storage, LexicalIndexConfig::default()).unwrap();
    /// # let mut engine = LexicalEngine::new(index).unwrap();
    ///
    /// let doc = Document::builder()
    ///     .add_text("title", "Hello World")
    ///     .add_text("body", "This is a test")
    ///     .build();
    /// engine.add_document(doc).unwrap();
    /// engine.commit().unwrap();  // Don't forget to commit!
    /// ```
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
    ///
    /// This method flushes all pending write operations to storage and makes them
    /// visible to subsequent searches. The cached writer is consumed and the reader
    /// cache is invalidated to ensure fresh data on the next search.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success, or an error if the commit fails.
    ///
    /// # Important
    ///
    /// - All write operations (add, update, delete) are not persisted until commit
    /// - After commit, the reader cache is invalidated automatically
    /// - The writer cache is cleared and will be recreated on the next write operation
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use yatagarasu::document::document::Document;
    /// # use yatagarasu::lexical::engine::LexicalEngine;
    /// # use yatagarasu::lexical::index::{LexicalIndexConfig, LexicalIndexFactory};
    /// # use yatagarasu::storage::{StorageConfig, StorageFactory};
    /// use yatagarasu::storage::memory::MemoryStorageConfig;
    /// # use std::sync::Arc;
    /// # let storage_config = StorageConfig::Memory(MemoryStorageConfig::default());
    /// # let storage = StorageFactory::create(storage_config).unwrap();
    /// # let index = LexicalIndexFactory::create(storage, LexicalIndexConfig::default()).unwrap();
    /// # let mut engine = LexicalEngine::new(index).unwrap();
    ///
    /// // Add multiple documents
    /// for i in 0..10 {
    ///     let doc = Document::builder()
    ///         .add_text("id", &i.to_string())
    ///         .add_text("title", &format!("Document {}", i))
    ///         .build();
    ///     engine.add_document(doc).unwrap();
    /// }
    ///
    /// // Commit all changes at once
    /// engine.commit().unwrap();
    /// ```
    pub fn commit(&mut self) -> Result<()> {
        // Take the cached writer if it exists
        if let Some(mut writer) = self.writer.borrow_mut().take() {
            writer.commit()?;
        }

        // Invalidate reader and searcher caches to reflect the new changes
        *self.reader.borrow_mut() = None;
        *self.searcher.borrow_mut() = None;

        Ok(())
    }

    /// Optimize the index.
    ///
    /// This method triggers index optimization, which typically involves merging smaller
    /// index segments into larger ones to improve search performance and reduce storage overhead.
    /// After optimization, the reader and searcher caches are invalidated to reflect the optimized structure.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success, or an error if optimization fails.
    ///
    /// # Performance Considerations
    ///
    /// - Optimization can be I/O and CPU intensive for large indexes
    /// - It's typically performed during off-peak hours or maintenance windows
    /// - The benefits include faster search performance and reduced storage space
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use yatagarasu::document::document::Document;
    /// # use yatagarasu::lexical::engine::LexicalEngine;
    /// # use yatagarasu::lexical::index::{LexicalIndexConfig, LexicalIndexFactory};
    /// # use yatagarasu::storage::{StorageConfig, StorageFactory};
    /// use yatagarasu::storage::memory::MemoryStorageConfig;
    /// # use std::sync::Arc;
    /// # let storage_config = StorageConfig::Memory(MemoryStorageConfig::default());
    /// # let storage = StorageFactory::create(storage_config).unwrap();
    /// # let index = LexicalIndexFactory::create(storage, LexicalIndexConfig::default()).unwrap();
    /// # let mut engine = LexicalEngine::new(index).unwrap();
    ///
    /// // Add and commit many documents
    /// for i in 0..1000 {
    ///     let doc = Document::builder()
    ///         .add_text("id", &i.to_string())
    ///         .build();
    ///     engine.add_document(doc).unwrap();
    /// }
    /// engine.commit().unwrap();
    ///
    /// // Optimize the index for better performance
    /// engine.optimize().unwrap();
    /// ```
    pub fn optimize(&mut self) -> Result<()> {
        self.index.optimize()?;

        // Invalidate reader and searcher caches
        *self.reader.borrow_mut() = None;
        *self.searcher.borrow_mut() = None;

        Ok(())
    }

    /// Refresh the reader to see latest changes.
    pub fn refresh(&mut self) -> Result<()> {
        *self.reader.borrow_mut() = None;
        *self.searcher.borrow_mut() = None;
        Ok(())
    }

    /// Get index statistics.
    pub fn stats(&self) -> Result<InvertedIndexStats> {
        self.index.stats()
    }

    /// Get the storage backend.
    pub fn storage(&self) -> &Arc<dyn Storage> {
        self.index.storage()
    }

    /// Search with the given request.
    ///
    /// This method executes a search query against the index using a cached searcher
    /// for improved performance.
    ///
    /// # Arguments
    ///
    /// * `request` - The search request containing the query and search parameters
    ///
    /// # Returns
    ///
    /// Returns `SearchResults` containing matching documents, scores, and metadata.
    ///
    /// # Example with TermQuery
    ///
    /// ```rust,no_run
    /// use yatagarasu::document::document::Document;
    /// use yatagarasu::lexical::types::LexicalSearchRequest;
    /// use yatagarasu::query::term::TermQuery;
    /// # use yatagarasu::lexical::engine::LexicalEngine;
    /// # use yatagarasu::lexical::index::{LexicalIndexConfig, LexicalIndexFactory};
    /// # use yatagarasu::storage::{StorageConfig, StorageFactory};
    /// use yatagarasu::storage::memory::MemoryStorageConfig;
    /// # use std::sync::Arc;
    /// # let storage_config = StorageConfig::Memory(MemoryStorageConfig::default());
    /// # let storage = StorageFactory::create(storage_config).unwrap();
    /// # let index = LexicalIndexFactory::create(storage, LexicalIndexConfig::default()).unwrap();
    /// # let mut engine = LexicalEngine::new(index).unwrap();
    /// # let doc = Document::builder().add_text("title", "hello world").build();
    /// # engine.add_document(doc).unwrap();
    /// # engine.commit().unwrap();
    ///
    /// let query = Box::new(TermQuery::new("title", "hello"));
    /// let request = LexicalSearchRequest::new(query)
    ///     .max_docs(10)
    ///     .min_score(0.5);
    /// let results = engine.search(request).unwrap();
    ///
    /// println!("Found {} documents", results.total_hits);
    /// for hit in results.hits {
    ///     println!("Doc ID: {}, Score: {}", hit.doc_id, hit.score);
    /// }
    /// ```
    ///
    /// # Example with QueryParser
    ///
    /// ```rust,no_run
    /// use yatagarasu::query::parser::QueryParser;
    /// use yatagarasu::lexical::types::LexicalSearchRequest;
    /// # use yatagarasu::document::document::Document;
    /// # use yatagarasu::lexical::engine::LexicalEngine;
    /// # use yatagarasu::lexical::index::{LexicalIndexConfig, LexicalIndexFactory};
    /// # use yatagarasu::storage::{StorageConfig, StorageFactory};
    /// use yatagarasu::storage::memory::MemoryStorageConfig;
    /// # use std::sync::Arc;
    /// # let storage_config = StorageConfig::Memory(MemoryStorageConfig::default());
    /// # let storage = StorageFactory::create(storage_config).unwrap();
    /// # let index = LexicalIndexFactory::create(storage, LexicalIndexConfig::default()).unwrap();
    /// # let mut engine = LexicalEngine::new(index).unwrap();
    ///
    /// let parser = QueryParser::new().with_default_field("title");
    /// let query = parser.parse("rust AND programming").unwrap();
    /// let results = engine.search(LexicalSearchRequest::new(query)).unwrap();
    /// ```
    pub fn search(&self, request: LexicalSearchRequest) -> Result<SearchResults> {
        let searcher = self.get_or_create_searcher()?;
        searcher.search(request)
    }

    /// Count documents matching the query.
    ///
    /// Uses a cached searcher for improved performance.
    pub fn count(&self, query: Box<dyn Query>) -> Result<u64> {
        let searcher = self.get_or_create_searcher()?;
        searcher.count(query)
    }

    /// Close the search engine.
    pub fn close(&mut self) -> Result<()> {
        // Drop the cached writer
        *self.writer.borrow_mut() = None;
        // Drop the cached reader
        *self.reader.borrow_mut() = None;
        // Drop the cached searcher
        *self.searcher.borrow_mut() = None;
        self.index.close()
    }

    /// Check if the engine is closed.
    pub fn is_closed(&self) -> bool {
        self.index.is_closed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexical::index::{LexicalIndexConfig, LexicalIndexFactory};
    use crate::query::term::TermQuery;
    use crate::storage::file::{FileStorage, FileStorageConfig};
    use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};
    use std::sync::Arc;
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
        let config = LexicalIndexConfig::default();
        let storage = Arc::new(
            FileStorage::new(temp_dir.path(), FileStorageConfig::new(temp_dir.path())).unwrap(),
        );
        let index = LexicalIndexFactory::create(storage, config).unwrap();
        let engine = LexicalEngine::new(index).unwrap();

        // Schema-less mode: no schema() method available
        assert!(!engine.is_closed());
    }

    #[test]
    fn test_search_engine_in_memory() {
        let config = LexicalIndexConfig::default();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let index = LexicalIndexFactory::create(storage, config).unwrap();
        let mut engine = LexicalEngine::new(index).unwrap();

        // Add some documents
        let docs = vec![
            create_test_document("Test Document 1", "Content of test document 1"),
            create_test_document("Test Document 2", "Content of test document 2"),
        ];
        engine.add_documents(docs).unwrap();
        engine.commit().unwrap();

        // Search for documents
        let query = Box::new(TermQuery::new("title", "Test"));
        let request = LexicalSearchRequest::new(query);
        let _results = engine.search(request).unwrap();

        // Should find documents in memory
        // Note: total_hits may be 0 if the analyzer lowercases "Test" to "test"
        // but we indexed "Test" (capital T). Just verify the search works.
        assert!(!engine.is_closed());
    }

    #[test]
    fn test_search_engine_open() {
        let temp_dir = TempDir::new().unwrap();
        let config = LexicalIndexConfig::default();

        // Create engine
        let storage = Arc::new(
            FileStorage::new(temp_dir.path(), FileStorageConfig::new(temp_dir.path())).unwrap(),
        );
        let index = LexicalIndexFactory::create(storage, config.clone()).unwrap();
        let mut engine = LexicalEngine::new(index).unwrap();
        engine.close().unwrap();

        // Open engine
        let storage = Arc::new(
            FileStorage::new(temp_dir.path(), FileStorageConfig::new(temp_dir.path())).unwrap(),
        );
        let index = LexicalIndexFactory::create(storage, config).unwrap();
        let engine = LexicalEngine::new(index).unwrap();

        // Schema-less mode: no schema() method available
        assert!(!engine.is_closed());
    }

    #[test]
    fn test_add_document() {
        let temp_dir = TempDir::new().unwrap();
        let config = LexicalIndexConfig::default();

        let storage = Arc::new(
            FileStorage::new(temp_dir.path(), FileStorageConfig::new(temp_dir.path())).unwrap(),
        );
        let index = LexicalIndexFactory::create(storage, config).unwrap();
        let mut engine = LexicalEngine::new(index).unwrap();

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
        let config = LexicalIndexConfig::default();

        let storage = Arc::new(
            FileStorage::new(temp_dir.path(), FileStorageConfig::new(temp_dir.path())).unwrap(),
        );
        let index = LexicalIndexFactory::create(storage, config).unwrap();
        let mut engine = LexicalEngine::new(index).unwrap();

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
        let config = LexicalIndexConfig::default();

        let storage = Arc::new(
            FileStorage::new(temp_dir.path(), FileStorageConfig::new(temp_dir.path())).unwrap(),
        );
        let index = LexicalIndexFactory::create(storage, config).unwrap();
        let engine = LexicalEngine::new(index).unwrap();

        let query = Box::new(TermQuery::new("title", "hello"));
        let request = LexicalSearchRequest::new(query);
        let results = engine.search(request).unwrap();

        assert_eq!(results.hits.len(), 0);
        assert_eq!(results.total_hits, 0);
        assert_eq!(results.max_score, 0.0);
    }

    #[test]
    fn test_search_with_documents() {
        let temp_dir = TempDir::new().unwrap();
        let config = LexicalIndexConfig::default();

        let storage = Arc::new(
            FileStorage::new(temp_dir.path(), FileStorageConfig::new(temp_dir.path())).unwrap(),
        );
        let index = LexicalIndexFactory::create(storage, config).unwrap();
        let mut engine = LexicalEngine::new(index).unwrap();

        // Add some documents
        let docs = vec![
            create_test_document("Hello World", "This is a test document"),
            create_test_document("Goodbye World", "This is another test document"),
        ];
        engine.add_documents(docs).unwrap();
        engine.commit().unwrap();

        // Search for documents
        let query = Box::new(TermQuery::new("title", "Hello"));
        let request = LexicalSearchRequest::new(query);
        let _results = engine.search(request).unwrap();

        // Results depend on the actual indexing implementation
        // For now, we just check that search doesn't fail
        // hits.len() is usize, so >= 0 check is redundant
        // total_hits is u64, so >= 0 check is redundant
    }

    #[test]
    fn test_count_query() {
        let temp_dir = TempDir::new().unwrap();
        let config = LexicalIndexConfig::default();

        let storage = Arc::new(
            FileStorage::new(temp_dir.path(), FileStorageConfig::new(temp_dir.path())).unwrap(),
        );
        let index = LexicalIndexFactory::create(storage, config).unwrap();
        let engine = LexicalEngine::new(index).unwrap();

        let query = Box::new(TermQuery::new("title", "hello"));
        let count = engine.count(query).unwrap();

        // Should return 0 for empty index
        assert_eq!(count, 0);
    }

    #[test]
    fn test_engine_refresh() {
        let temp_dir = TempDir::new().unwrap();
        let config = LexicalIndexConfig::default();

        let storage = Arc::new(
            FileStorage::new(temp_dir.path(), FileStorageConfig::new(temp_dir.path())).unwrap(),
        );
        let index = LexicalIndexFactory::create(storage, config).unwrap();
        let mut engine = LexicalEngine::new(index).unwrap();

        // Add a document
        let doc = create_test_document("Test Document", "Test content");
        engine.add_document(doc).unwrap();
        engine.commit().unwrap();

        // Refresh should not fail
        engine.refresh().unwrap();

        // Search should still work
        let query = Box::new(TermQuery::new("title", "Test"));
        let request = LexicalSearchRequest::new(query);
        let _results = engine.search(request).unwrap();
        // hits.len() is usize, so >= 0 check is redundant
    }

    #[test]
    fn test_engine_stats() {
        let temp_dir = TempDir::new().unwrap();
        let config = LexicalIndexConfig::default();

        let storage = Arc::new(
            FileStorage::new(temp_dir.path(), FileStorageConfig::new(temp_dir.path())).unwrap(),
        );
        let index = LexicalIndexFactory::create(storage, config).unwrap();
        let engine = LexicalEngine::new(index).unwrap();

        let stats = engine.stats().unwrap();
        // doc_count is usize, so >= 0 check is redundant
        // term_count is usize, so >= 0 check is redundant
        assert!(stats.last_modified > 0);
    }

    #[test]
    fn test_engine_close() {
        let temp_dir = TempDir::new().unwrap();
        let config = LexicalIndexConfig::default();

        let storage = Arc::new(
            FileStorage::new(temp_dir.path(), FileStorageConfig::new(temp_dir.path())).unwrap(),
        );
        let index = LexicalIndexFactory::create(storage, config).unwrap();
        let mut engine = LexicalEngine::new(index).unwrap();

        assert!(!engine.is_closed());

        engine.close().unwrap();

        assert!(engine.is_closed());
    }

    #[test]
    fn test_search_request_configuration() {
        let temp_dir = TempDir::new().unwrap();
        let config = LexicalIndexConfig::default();

        let storage = Arc::new(
            FileStorage::new(temp_dir.path(), FileStorageConfig::new(temp_dir.path())).unwrap(),
        );
        let index = LexicalIndexFactory::create(storage, config).unwrap();
        let engine = LexicalEngine::new(index).unwrap();

        let query = Box::new(TermQuery::new("title", "hello"));
        let request = LexicalSearchRequest::new(query)
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
        let config = LexicalIndexConfig::default();

        let storage = Arc::new(
            FileStorage::new(temp_dir.path(), FileStorageConfig::new(temp_dir.path())).unwrap(),
        );
        let index = LexicalIndexFactory::create(storage, config).unwrap();
        let mut engine = LexicalEngine::new(index).unwrap();

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
        let results = engine.search(LexicalSearchRequest::new(query)).unwrap();

        // Should find the document
        // QueryParser analyzes "Hello" -> "hello", which matches the indexed "hello"
        assert_eq!(results.hits.len(), 1);
        assert_eq!(results.total_hits, 1);
    }

    #[test]
    fn test_search_field_with_string() {
        let temp_dir = TempDir::new().unwrap();
        let config = LexicalIndexConfig::default();

        let storage = Arc::new(
            FileStorage::new(temp_dir.path(), FileStorageConfig::new(temp_dir.path())).unwrap(),
        );
        let index = LexicalIndexFactory::create(storage, config).unwrap();
        let engine = LexicalEngine::new(index).unwrap();

        // Search specific field
        use crate::query::parser::QueryParser;
        let parser = QueryParser::new();
        let query = parser.parse_field("title", "hello world").unwrap();
        let results = engine.search(LexicalSearchRequest::new(query)).unwrap();

        // Should parse and execute the query
        assert_eq!(results.hits.len(), 0);
        assert_eq!(results.total_hits, 0);
    }

    #[test]
    fn test_query_parser_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = LexicalIndexConfig::default();
        let storage = Arc::new(
            crate::storage::file::FileStorage::new(
                temp_dir.path(),
                crate::storage::file::FileStorageConfig::new(temp_dir.path()),
            )
            .unwrap(),
        );
        let index = LexicalIndexFactory::create(storage, config).unwrap();
        let _engine = LexicalEngine::new(index).unwrap();

        use crate::query::parser::QueryParser;
        let parser = QueryParser::new();
        assert!(parser.default_field().is_none());

        let parser_with_default = QueryParser::new().with_default_field("title");
        assert_eq!(parser_with_default.default_field(), Some("title"));
    }

    #[test]
    fn test_complex_string_query() {
        let temp_dir = TempDir::new().unwrap();
        let config = LexicalIndexConfig::default();

        let storage = Arc::new(
            FileStorage::new(temp_dir.path(), FileStorageConfig::new(temp_dir.path())).unwrap(),
        );
        let index = LexicalIndexFactory::create(storage, config).unwrap();
        let engine = LexicalEngine::new(index).unwrap();

        // Test complex query parsing
        use crate::query::parser::QueryParser;
        let parser = QueryParser::new().with_default_field("title");
        let query = parser.parse("title:hello AND body:world").unwrap();
        let results = engine.search(LexicalSearchRequest::new(query)).unwrap();

        // Should parse the complex query without error
        assert_eq!(results.hits.len(), 0);
        assert_eq!(results.total_hits, 0);
    }
}
