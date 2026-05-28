//! PHP-facing `Index` class — the primary entry point for the laurus binding.

use std::path::Path;
use std::sync::Arc;

use ext_php_rs::prelude::*;
use ext_php_rs::types::{ZendHashTable, Zval};
use laurus::{Engine, EngineStats, Storage, StorageConfig, StorageFactory};

use crate::convert::{document_to_hashtable, hashtable_to_document};
use crate::errors::laurus_err;
use crate::schema::PhpSchema;
use crate::search::{PhpSearchResult, build_request_from_php, to_php_search_result};

// ---------------------------------------------------------------------------
// Index
// ---------------------------------------------------------------------------

/// Laurus search index — the main entry point for the PHP binding
/// (`Laurus\Index`).
///
/// # Creating an index
///
/// ```php
/// use Laurus\Index;
/// use Laurus\Schema;
///
/// // In-memory (ephemeral)
/// $index = new Index();
///
/// // File-based (persistent)
/// $schema = new Schema();
/// $schema->addTextField("title");
/// $index = new Index("./myindex", $schema);
/// ```
///
/// # Searching
///
/// ```php
/// $results = $index->search("title:hello", 10);
/// $results = $index->search(new \Laurus\TermQuery("body", "rust"), 5);
/// ```
#[php_class]
#[php(name = "Laurus\\Index")]
pub struct PhpIndex {
    engine: Arc<Engine>,
    rt: Arc<tokio::runtime::Runtime>,
}

#[php_impl]
impl PhpIndex {
    /// Create a new index.
    ///
    /// # Arguments
    ///
    /// * `path` - Directory path for persistent storage. Pass null (default)
    ///   for an ephemeral in-memory index.
    /// * `schema` - Schema definition (optional).
    pub fn __construct(path: Option<String>, schema: Option<&PhpSchema>) -> PhpResult<Self> {
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| ext_php_rs::exception::PhpException::default(e.to_string()))?;

        let storage = create_storage(path.as_deref())?;

        let schema_val = match schema {
            Some(php_schema) => php_schema.inner.borrow().clone(),
            None => laurus::Schema::default(),
        };

        let engine = rt
            .block_on(Engine::new(storage, schema_val))
            .map_err(laurus_err)?;

        Ok(Self {
            engine: Arc::new(engine),
            rt: Arc::new(rt),
        })
    }

    // ── Document CRUD ─────────────────────────────────────────────────────

    /// Index a document, replacing any existing document with the same id.
    ///
    /// Call `commit()` to make the change visible to searches.
    ///
    /// # Arguments
    ///
    /// * `id` - External document identifier (string).
    /// * `doc` - An associative array mapping field names to values.
    pub fn put_document(&self, id: String, doc: &ZendHashTable) -> PhpResult<()> {
        let document = hashtable_to_document(doc)?;
        let engine = self.engine.clone();
        self.rt
            .block_on(engine.put_document(&id, document))
            .map_err(laurus_err)
    }

    /// Append a document version without removing existing versions.
    ///
    /// Laurus supports multiple versions of the same id (chunk-per-document
    /// RAG pattern). Use `putDocument()` to replace.
    ///
    /// # Arguments
    ///
    /// * `id` - External document identifier.
    /// * `doc` - An associative array mapping field names to values.
    pub fn add_document(&self, id: String, doc: &ZendHashTable) -> PhpResult<()> {
        let document = hashtable_to_document(doc)?;
        let engine = self.engine.clone();
        self.rt
            .block_on(engine.add_document(&id, document))
            .map_err(laurus_err)
    }

    /// Retrieve all document versions stored under `id`.
    ///
    /// # Arguments
    ///
    /// * `id` - External document identifier.
    ///
    /// # Returns
    ///
    /// An array of associative arrays, one per indexed version.
    pub fn get_documents(&self, id: String) -> PhpResult<Zval> {
        let engine = self.engine.clone();
        let docs = self
            .rt
            .block_on(engine.get_documents(&id))
            .map_err(laurus_err)?;
        let mut arr = ZendHashTable::new();
        for (i, doc) in docs.iter().enumerate() {
            let ht = document_to_hashtable(doc)?;
            let mut zv = Zval::new();
            zv.set_hashtable(ht);
            arr.insert_at_index(i as i64, zv)
                .map_err(|_| "failed to insert document")?;
        }
        let mut result = Zval::new();
        result.set_hashtable(arr);
        Ok(result)
    }

    /// Delete all document versions stored under `id`.
    ///
    /// Call `commit()` to make the deletion visible to searches.
    ///
    /// # Arguments
    ///
    /// * `id` - External document identifier.
    pub fn delete_documents(&self, id: String) -> PhpResult<()> {
        let engine = self.engine.clone();
        self.rt
            .block_on(engine.delete_documents(&id))
            .map_err(laurus_err)
    }

    /// Flush buffered writes and make all pending changes searchable.
    pub fn commit(&self) -> PhpResult<()> {
        let engine = self.engine.clone();
        self.rt.block_on(engine.commit()).map_err(laurus_err)
    }

    // ── Search ────────────────────────────────────────────────────────────

    /// Search the index and return an array of `SearchResult` objects.
    ///
    /// `$query` may be:
    ///   - A **DSL string** (e.g. `"title:hello"`)
    ///   - A **lexical query** object (`TermQuery`, `BooleanQuery`, etc.)
    ///   - A **vector query** object (`VectorQuery`, `VectorTextQuery`)
    ///   - A **`SearchRequest`** for full control
    ///
    /// # Arguments
    ///
    /// * `query` - The query to execute.
    /// * `limit` - Maximum number of results (default: 10).
    /// * `offset` - Pagination offset (default: 0).
    ///
    /// # Returns
    ///
    /// An array of `SearchResult` objects.
    #[php(defaults(limit = 10, offset = 0))]
    pub fn search(&self, query: &Zval, limit: i64, offset: i64) -> PhpResult<Vec<PhpSearchResult>> {
        let request = build_request_from_php(query, limit as usize, offset as usize)?;

        let engine = self.engine.clone();
        let results = self
            .rt
            .block_on(engine.search(request))
            .map_err(laurus_err)?;

        Ok(results.into_iter().map(to_php_search_result).collect())
    }

    /// Execute multiple independent searches in one call.
    ///
    /// Each entry of `queries` is dispatched in parallel on the
    /// underlying tokio runtime via `laurus::Engine::search_batch`.
    /// The same `limit` and `offset` are applied to every query in the
    /// batch. Each entry accepts the same kinds of values as `search`:
    /// a DSL string, a lexical / vector query object, or a
    /// `SearchRequest`.
    ///
    /// # Arguments
    ///
    /// * `queries` - An array of queries to execute.
    /// * `limit` - Maximum number of results per query (default: 10).
    /// * `offset` - Pagination offset per query (default: 0).
    ///
    /// # Returns
    ///
    /// An array of arrays: `results[i]` is the result array for
    /// `queries[i]`. Empty input returns an empty array without
    /// invoking the engine.
    ///
    /// Issue [#720](https://github.com/mosuka/laurus/issues/720)
    /// Phase 3e of [#648](https://github.com/mosuka/laurus/issues/648).
    #[php(defaults(limit = 10, offset = 0))]
    pub fn search_batch(
        &self,
        queries: &Zval,
        limit: i64,
        offset: i64,
    ) -> PhpResult<Vec<Vec<PhpSearchResult>>> {
        let arr = queries.array().ok_or_else(|| {
            PhpException::from(
                "search_batch: expected an array of queries (DSL string, Query object, or SearchRequest)".to_string(),
            )
        })?;

        if arr.is_empty() {
            return Ok(Vec::new());
        }

        let mut requests = Vec::with_capacity(arr.len());
        for (_, value) in arr.iter() {
            requests.push(build_request_from_php(
                value,
                limit as usize,
                offset as usize,
            )?);
        }

        let engine = self.engine.clone();
        let batch_results = self
            .rt
            .block_on(engine.search_batch(requests))
            .map_err(laurus_err)?;

        Ok(batch_results
            .into_iter()
            .map(|per_query_results| {
                per_query_results
                    .into_iter()
                    .map(to_php_search_result)
                    .collect()
            })
            .collect())
    }

    // ── Schema & stats ────────────────────────────────────────────────────

    /// Return index statistics as an associative array.
    ///
    /// # Returns
    ///
    /// An associative array with keys:
    ///   - `"documentCount"` (int): total indexed documents.
    ///   - `"vectorFields"` (array): per-field vector statistics.
    pub fn stats(&self) -> PhpResult<Zval> {
        let engine = self.engine.clone();
        let stats: EngineStats = self
            .rt
            .block_on(async { engine.stats() })
            .map_err(laurus_err)?;

        let mut ht = ZendHashTable::new();
        let mut count_zv = Zval::new();
        count_zv.set_long(stats.document_count as i64);
        ht.insert("documentCount", count_zv)
            .map_err(|_| "failed to insert documentCount")?;

        let mut vf_ht = ZendHashTable::new();
        for (field, field_stats) in &stats.vector_fields {
            let mut fd_ht = ZendHashTable::new();
            let mut count_zv = Zval::new();
            count_zv.set_long(field_stats.vector_count as i64);
            fd_ht
                .insert("count", count_zv)
                .map_err(|_| "failed to insert count")?;
            let mut dim_zv = Zval::new();
            dim_zv.set_long(field_stats.dimension as i64);
            fd_ht
                .insert("dimension", dim_zv)
                .map_err(|_| "failed to insert dimension")?;
            let mut fd_zv = Zval::new();
            fd_zv.set_hashtable(fd_ht);
            vf_ht
                .insert(field.as_str(), fd_zv)
                .map_err(|_| "failed to insert vector field")?;
        }
        let mut vf_zv = Zval::new();
        vf_zv.set_hashtable(vf_ht);
        ht.insert("vectorFields", vf_zv)
            .map_err(|_| "failed to insert vectorFields")?;

        let mut result = Zval::new();
        result.set_hashtable(ht);
        Ok(result)
    }

    /// Return a string representation.
    pub fn __to_string(&self) -> String {
        "Index()".to_string()
    }
}

// ---------------------------------------------------------------------------
// Storage factory helper
// ---------------------------------------------------------------------------

/// Create a storage backend from an optional path.
///
/// # Arguments
///
/// * `path` - Optional directory path. `None` means in-memory storage.
///
/// # Returns
///
/// An `Arc<dyn Storage>` for the engine.
fn create_storage(path: Option<&str>) -> PhpResult<Arc<dyn Storage>> {
    let config = match path {
        None => StorageConfig::Memory(Default::default()),
        Some(p) => {
            use laurus::storage::file::FileStorageConfig;
            StorageConfig::File(FileStorageConfig::new(Path::new(p)))
        }
    };
    StorageFactory::create(config).map_err(laurus_err)
}
