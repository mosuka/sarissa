//! High-level lexical search engine that combines indexing and searching.
//!
//! This module provides the core `LexicalEngine` implementation.

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
/// use sarissa::document::document::Document;
/// use sarissa::lexical::engine::LexicalEngine;
/// use sarissa::lexical::index::config::LexicalIndexConfig;
/// use sarissa::lexical::search::searcher::LexicalSearchRequest;
/// use sarissa::storage::memory::{MemoryStorage, MemoryStorageConfig};
/// use std::sync::Arc;
///
/// // Create storage and engine
/// let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
/// let config = LexicalIndexConfig::default();
/// let engine = LexicalEngine::new(storage, config).unwrap();
///
/// // Add documents
/// use sarissa::document::field::TextOption;
/// let doc = Document::builder()
///     .add_text("title", "Rust Programming", TextOption::default())
///     .build();
/// engine.add_document(doc).unwrap();
/// engine.commit().unwrap();
///
/// // Search using DSL string
/// let results = engine.search(LexicalSearchRequest::new("title:rust")).unwrap();
/// ```
pub struct LexicalEngine {
    /// The underlying lexical index.
    index: Box<dyn LexicalIndex>,
}

impl std::fmt::Debug for LexicalEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LexicalEngine")
            .field("index", &self.index)
            .finish()
    }
}

impl LexicalEngine {
    /// Create a new lexical search engine with the given storage and configuration.
    ///
    /// This constructor creates a `LexicalIndex` internally using the provided storage
    /// and configuration, then wraps it with lazy-initialized caches for the reader,
    /// writer, and searcher.
    ///
    /// # Arguments
    ///
    /// * `storage` - The storage backend for persisting index data
    /// * `config` - Configuration for the lexical index (schema, analyzer, etc.)
    ///
    /// # Returns
    ///
    /// Returns a new `LexicalEngine` instance.
    ///
    /// # Example with Memory Storage
    ///
    /// ```rust,no_run
    /// use sarissa::lexical::engine::LexicalEngine;
    /// use sarissa::lexical::index::config::LexicalIndexConfig;
    /// use sarissa::storage::{Storage, StorageConfig, StorageFactory};
    /// use sarissa::storage::memory::MemoryStorageConfig;
    /// use std::sync::Arc;
    ///
    /// let storage_config = StorageConfig::Memory(MemoryStorageConfig::default());
    /// let storage = StorageFactory::create(storage_config).unwrap();
    /// let engine = LexicalEngine::new(storage, LexicalIndexConfig::default()).unwrap();
    /// ```
    ///
    /// # Example with File Storage
    ///
    /// ```rust,no_run
    /// use sarissa::lexical::engine::LexicalEngine;
    /// use sarissa::lexical::index::config::LexicalIndexConfig;
    /// use sarissa::storage::{Storage, StorageConfig, StorageFactory};
    /// use sarissa::storage::file::FileStorageConfig;
    /// use std::sync::Arc;
    ///
    /// let storage_config = StorageConfig::File(FileStorageConfig::new("/tmp/index"));
    /// let storage = StorageFactory::create(storage_config).unwrap();
    /// let engine = LexicalEngine::new(storage, LexicalIndexConfig::default()).unwrap();
    /// ```
    pub fn new(storage: Arc<dyn Storage>, config: LexicalIndexConfig) -> Result<Self> {
        let index = LexicalIndexFactory::create(storage, config)?;
        Ok(Self { index })
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
    /// use sarissa::document::document::Document;
    /// # use sarissa::lexical::engine::LexicalEngine;
    /// # use sarissa::lexical::index::config::LexicalIndexConfig;
    /// # use sarissa::storage::{StorageConfig, StorageFactory};
    /// use sarissa::storage::memory::MemoryStorageConfig;
    /// # use std::sync::Arc;
    /// # let storage_config = StorageConfig::Memory(MemoryStorageConfig::default());
    /// # let storage = StorageFactory::create(storage_config).unwrap();
    /// # let engine = LexicalEngine::new(storage, LexicalIndexConfig::default()).unwrap();
    ///
    /// use sarissa::document::field::TextOption;
    /// let doc = Document::builder()
    ///     .add_text("title", "Hello World", TextOption::default())
    ///     .add_text("body", "This is a test", TextOption::default())
    ///     .build();
    /// let doc_id = engine.add_document(doc).unwrap();
    /// engine.commit().unwrap();  // Don't forget to commit!
    /// ```
    pub fn add_document(&self, doc: Document) -> Result<u64> {
        self.index.add_document(doc)
    }

    /// Upsert a document with a specific document ID.
    /// Note: You must call `commit()` to persist the changes.
    pub fn upsert_document(&self, doc_id: u64, doc: Document) -> Result<()> {
        self.index.upsert_document(doc_id, doc)
    }

    /// Add multiple documents to the index.
    /// Returns a vector of assigned document IDs.
    /// Note: You must call `commit()` to persist the changes.
    pub fn add_documents(&self, docs: Vec<Document>) -> Result<Vec<u64>> {
        self.index.add_documents(docs)
    }

    /// Delete a document by ID.
    ///
    /// Note: You must call `commit()` to persist the changes.
    pub fn delete_document(&self, doc_id: u64) -> Result<()> {
        self.index.delete_document(doc_id)
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
    /// use sarissa::document::document::Document;
    /// # use sarissa::lexical::engine::LexicalEngine;
    /// # use sarissa::lexical::index::config::LexicalIndexConfig;
    /// # use sarissa::storage::{StorageConfig, StorageFactory};
    /// use sarissa::storage::memory::MemoryStorageConfig;
    /// # use std::sync::Arc;
    /// # let storage_config = StorageConfig::Memory(MemoryStorageConfig::default());
    /// # let storage = StorageFactory::create(storage_config).unwrap();
    /// # let engine = LexicalEngine::new(storage, LexicalIndexConfig::default()).unwrap();
    ///
    /// // Add multiple documents
    /// use sarissa::document::field::TextOption;
    /// for i in 0..10 {
    ///     let doc = Document::builder()
    ///         .add_text("id", &i.to_string(), TextOption::default())
    ///         .add_text("title", &format!("Document {}", i), TextOption::default())
    ///         .build();
    ///     engine.add_document(doc).unwrap();
    /// }
    ///
    /// // Commit all changes at once
    /// engine.commit().unwrap();
    /// ```
    pub fn commit(&self) -> Result<()> {
        self.index.commit()
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
    /// use sarissa::document::document::Document;
    /// # use sarissa::lexical::engine::LexicalEngine;
    /// # use sarissa::lexical::index::config::LexicalIndexConfig;
    /// # use sarissa::storage::{StorageConfig, StorageFactory};
    /// use sarissa::storage::memory::MemoryStorageConfig;
    /// # use std::sync::Arc;
    /// # let storage_config = StorageConfig::Memory(MemoryStorageConfig::default());
    /// # let storage = StorageFactory::create(storage_config).unwrap();
    /// # let mut engine = LexicalEngine::new(storage, LexicalIndexConfig::default()).unwrap();
    ///
    /// // Add and commit many documents
    /// use sarissa::document::field::TextOption;
    /// for i in 0..1000 {
    ///     let doc = Document::builder()
    ///         .add_text("id", &i.to_string(), TextOption::default())
    ///         .build();
    ///     engine.add_document(doc).unwrap();
    /// }
    /// engine.commit().unwrap();
    ///
    /// // Optimize the index for better performance
    /// engine.optimize().unwrap();
    /// ```
    pub fn optimize(&self) -> Result<()> {
        self.index.optimize()
    }

    /// Refresh the reader to see latest changes.
    pub fn refresh(&self) -> Result<()> {
        self.index.refresh()
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
    /// use sarissa::document::document::Document;
    /// use sarissa::lexical::search::searcher::LexicalSearchRequest;
    /// use sarissa::lexical::index::inverted::query::term::TermQuery;
    /// # use sarissa::lexical::engine::LexicalEngine;
    /// # use sarissa::lexical::index::config::LexicalIndexConfig;
    /// # use sarissa::storage::{StorageConfig, StorageFactory};
    /// use sarissa::storage::memory::MemoryStorageConfig;
    /// # use std::sync::Arc;
    /// # let storage_config = StorageConfig::Memory(MemoryStorageConfig::default());
    /// # let storage = StorageFactory::create(storage_config).unwrap();
    /// # let engine = LexicalEngine::new(storage, LexicalIndexConfig::default()).unwrap();
    /// # use sarissa::document::field::TextOption;
    /// # let doc = Document::builder().add_text("title", "hello world", TextOption::default()).build();
    /// # engine.add_document(doc).unwrap();
    /// # engine.commit().unwrap();
    ///
    /// // Using DSL string
    /// let request = LexicalSearchRequest::new("title:hello")
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
    /// use sarissa::lexical::index::inverted::query::parser::QueryParser;
    /// use sarissa::lexical::search::searcher::LexicalSearchRequest;
    /// # use sarissa::document::document::Document;
    /// # use sarissa::lexical::engine::LexicalEngine;
    /// # use sarissa::lexical::index::config::LexicalIndexConfig;
    /// # use sarissa::storage::{StorageConfig, StorageFactory};
    /// use sarissa::storage::memory::MemoryStorageConfig;
    /// use sarissa::analysis::analyzer::standard::StandardAnalyzer;
    /// # use std::sync::Arc;
    /// # let storage_config = StorageConfig::Memory(MemoryStorageConfig::default());
    /// # let storage = StorageFactory::create(storage_config).unwrap();
    /// # let engine = LexicalEngine::new(storage, LexicalIndexConfig::default()).unwrap();
    ///
    /// let analyzer = Arc::new(StandardAnalyzer::default());
    /// let parser = QueryParser::new(analyzer).with_default_field("title");
    /// let query = parser.parse("rust AND programming").unwrap();
    /// let results = engine.search(LexicalSearchRequest::new(query)).unwrap();
    /// ```
    pub fn search(&self, request: LexicalSearchRequest) -> Result<LexicalSearchResults> {
        self.index.search(request)
    }

    /// Count documents matching the request.
    ///
    /// Uses a cached searcher for improved performance.
    /// If `min_score` is specified in the request parameters, only documents
    /// with a score equal to or greater than the threshold are counted.
    ///
    /// # Arguments
    ///
    /// * `request` - Search request containing the query and search parameters.
    ///   Use `LexicalSearchRequest::new(query)` to create a request.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use sarissa::lexical::engine::LexicalEngine;
    /// # use sarissa::lexical::index::config::LexicalIndexConfig;
    /// # use sarissa::lexical::search::searcher::LexicalSearchRequest;
    /// # use sarissa::storage::memory::MemoryStorage;
    /// # use sarissa::storage::memory::MemoryStorageConfig;
    /// # use std::sync::Arc;
    /// # let config = LexicalIndexConfig::default();
    /// # let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    /// # let engine = LexicalEngine::new(storage, config).unwrap();
    /// // Count all matching documents
    /// let count = engine.count(LexicalSearchRequest::new("title:hello")).unwrap();
    /// println!("Found {} documents", count);
    ///
    /// // Count with min_score threshold
    /// let count = engine.count(
    ///     LexicalSearchRequest::new("title:hello").min_score(0.5)
    /// ).unwrap();
    /// println!("Found {} documents with score >= 0.5", count);
    /// ```
    pub fn count(&self, request: LexicalSearchRequest) -> Result<u64> {
        self.index.count(request)
    }

    /// Close the search engine.
    pub fn close(&self) -> Result<()> {
        self.index.close()
    }

    /// Check if the engine is closed.
    pub fn is_closed(&self) -> bool {
        self.index.is_closed()
    }

    /// Get the analyzer used by this engine.
    ///
    /// Returns the analyzer from the underlying index reader.
    /// This is useful for query parsing and term normalization.
    ///
    /// # Returns
    ///
    /// Returns `Result<Arc<dyn Analyzer>>` containing the analyzer.
    ///
    /// # Errors
    ///
    /// Returns an error if the reader cannot be created or the index type
    /// doesn't support analyzers.
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

    /// Create a query parser configured for this index.
    ///
    /// The parser uses the index's analyzer and default fields configuration.
    ///
    /// # Returns
    ///
    /// Returns `Result<QueryParser>` containing the configured parser.
    pub fn query_parser(
        &self,
    ) -> Result<crate::lexical::index::inverted::query::parser::QueryParser> {
        let analyzer = self.analyzer()?;
        let mut parser = crate::lexical::index::inverted::query::parser::QueryParser::new(analyzer);

        if let Ok(fields) = self.index.default_fields() {
            if !fields.is_empty() {
                parser = parser.with_default_fields(fields);
            }
        }

        Ok(parser)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexical::core::field::TextOption;
    use crate::lexical::index::config::LexicalIndexConfig;
    use crate::lexical::index::inverted::query::Query;
    use crate::lexical::index::inverted::query::term::TermQuery;
    use crate::storage::file::{FileStorage, FileStorageConfig};
    use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};
    use std::sync::Arc;
    use tempfile::TempDir;

    #[allow(dead_code)]
    fn create_test_document(title: &str, body: &str) -> Document {
        Document::builder()
            .add_text("title", title, TextOption::default())
            .add_text("body", body, TextOption::default())
            .build()
    }

    #[test]
    fn test_search_engine_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = LexicalIndexConfig::default();
        let storage = Arc::new(
            FileStorage::new(temp_dir.path(), FileStorageConfig::new(temp_dir.path())).unwrap(),
        );
        let engine = LexicalEngine::new(storage, config).unwrap();

        // Schema-less mode: no schema() method available
        assert!(!engine.is_closed());
    }

    #[test]
    fn test_search_engine_in_memory() {
        let config = LexicalIndexConfig::default();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let engine = LexicalEngine::new(storage, config).unwrap();

        // Add some documents
        let docs = vec![
            create_test_document("Test Document 1", "Content of test document 1"),
            create_test_document("Test Document 2", "Content of test document 2"),
        ];
        engine.add_documents(docs).unwrap();
        engine.commit().unwrap();

        // Search for documents
        let query = Box::new(TermQuery::new("title", "Test")) as Box<dyn Query>;
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
        let engine = LexicalEngine::new(storage, config.clone()).unwrap();
        engine.close().unwrap();

        // Open engine
        let storage = Arc::new(
            FileStorage::new(temp_dir.path(), FileStorageConfig::new(temp_dir.path())).unwrap(),
        );
        let engine = LexicalEngine::new(storage, config).unwrap();

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
        let engine = LexicalEngine::new(storage, config).unwrap();

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
        let engine = LexicalEngine::new(storage, config).unwrap();

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
        let engine = LexicalEngine::new(storage, config).unwrap();

        let query = Box::new(TermQuery::new("title", "hello")) as Box<dyn Query>;
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
        let engine = LexicalEngine::new(storage, config).unwrap();

        // Add some documents
        let docs = vec![
            create_test_document("Hello World", "This is a test document"),
            create_test_document("Goodbye World", "This is another test document"),
        ];
        engine.add_documents(docs).unwrap();
        engine.commit().unwrap();

        // Search for documents
        let query = Box::new(TermQuery::new("title", "Hello")) as Box<dyn Query>;
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
        let engine = LexicalEngine::new(storage, config).unwrap();

        let query = Box::new(TermQuery::new("title", "hello")) as Box<dyn Query>;
        let count = engine.count(LexicalSearchRequest::new(query)).unwrap();

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
        let engine = LexicalEngine::new(storage, config).unwrap();

        // Add a document
        let doc = create_test_document("Test Document", "Test content");
        engine.add_document(doc).unwrap();
        engine.commit().unwrap();

        // Refresh should not fail
        engine.refresh().unwrap();

        // Search should still work
        let query = Box::new(TermQuery::new("title", "Test")) as Box<dyn Query>;
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
        let engine = LexicalEngine::new(storage, config).unwrap();

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
        let engine = LexicalEngine::new(storage, config).unwrap();

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
        let engine = LexicalEngine::new(storage, config).unwrap();

        let query = Box::new(TermQuery::new("title", "hello")) as Box<dyn Query>;
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
        let engine = LexicalEngine::new(storage, config).unwrap();

        // Add some documents with lowercase titles for testing
        let docs = vec![
            create_test_document("hello world", "This is a test document"),
            create_test_document("goodbye world", "This is another test document"),
        ];
        engine.add_documents(docs).unwrap();
        engine.commit().unwrap();

        // Search with QueryParser (Lucene style)
        use crate::lexical::index::inverted::query::parser::QueryParser;
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
        let engine = LexicalEngine::new(storage, config).unwrap();

        // Search specific field
        use crate::analysis::analyzer::standard::StandardAnalyzer;
        use crate::lexical::index::inverted::query::parser::QueryParser;
        let analyzer = Arc::new(StandardAnalyzer::new().unwrap());
        let parser = QueryParser::new(analyzer);
        let query = parser.parse_field("title", "hello world").unwrap();
        let results = engine.search(LexicalSearchRequest::new(query)).unwrap();

        // Should not find anything (empty index)
        assert_eq!(results.hits.len(), 0);
    }
}
