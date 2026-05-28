//! Python-facing [`Index`] class — the primary entry point for the laurus binding.

use std::path::Path;
use std::sync::Arc;

use crate::convert::{dict_to_document, document_to_dict};
use crate::errors::laurus_err;
use crate::schema::PySchema;
use crate::search::{PySearchResult, build_request_from_py, to_py_search_result};
use laurus::{Engine, EngineStats, Storage, StorageConfig, StorageFactory};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

// ---------------------------------------------------------------------------
// Index
// ---------------------------------------------------------------------------

/// Laurus search index — the main entry point for the Python binding.
///
/// ## Creating an index
///
/// ```python
/// import laurus
///
/// # In-memory (ephemeral, great for prototyping)
/// index = laurus.Index()
///
/// # File-based (persistent)
/// schema = laurus.Schema()
/// schema.add_text_field("title")
/// schema.add_text_field("body")
/// schema.add_hnsw_field("embedding", dimension=384)
/// index = laurus.Index(path="./myindex", schema=schema)
/// ```
///
/// ## Adding documents
///
/// ```python
/// index.put_document("doc1", {"title": "Hello", "body": "World"})
/// index.commit()
/// ```
///
/// ## Searching
///
/// ```python
/// # DSL string
/// results = index.search("title:hello", limit=10)
///
/// # Query object
/// results = index.search(laurus.TermQuery("body", "rust"), limit=5)
///
/// # Pre-computed vector
/// results = index.search(laurus.VectorQuery("embedding", vec), limit=5)
///
/// # Hybrid via SearchRequest
/// request = laurus.SearchRequest(
///     lexical_query=laurus.TermQuery("body", "async"),
///     vector_query=laurus.VectorTextQuery("embedding", "concurrent"),
///     fusion=laurus.RRF(k=60.0),
///     limit=3,
/// )
/// results = index.search(request)
/// ```
#[pyclass(name = "Index")]
pub struct PyIndex {
    engine: Arc<Engine>,
    rt: Arc<tokio::runtime::Runtime>,
}

#[pymethods]
impl PyIndex {
    /// Create a new index.
    ///
    /// Args:
    ///     path: Directory path for persistent storage.
    ///           Pass `None` (default) for an ephemeral in-memory index.
    ///     schema: Schema definition.  If omitted, an empty schema is used.
    #[new]
    #[pyo3(signature = (path=None, schema=None))]
    pub fn new(path: Option<String>, schema: Option<&PySchema>) -> PyResult<Self> {
        let rt =
            tokio::runtime::Runtime::new().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let storage = create_storage(path.as_deref())?;
        let schema = schema.map(|s| s.inner.clone()).unwrap_or_default();

        let engine = rt
            .block_on(Engine::new(storage, schema))
            .map_err(laurus_err)?;

        Ok(Self {
            engine: Arc::new(engine),
            rt: Arc::new(rt),
        })
    }

    // ── Document CRUD ─────────────────────────────────────────────────────

    /// Index a document, replacing any existing document with the same id.
    ///
    /// Args:
    ///     id: External document identifier (string).
    ///     doc: A `dict` mapping field names to values.
    ///
    /// Call [`commit`] to make the change visible to searches.
    pub fn put_document(&self, py: Python, id: &str, doc: &Bound<PyDict>) -> PyResult<()> {
        let document = dict_to_document(py, doc)?;
        let engine = self.engine.clone();
        let id = id.to_string();
        self.rt
            .block_on(engine.put_document(&id, document))
            .map_err(laurus_err)
    }

    /// Append a document version without removing existing versions.
    ///
    /// Laurus supports multiple versions of the same id (chunk-per-document
    /// RAG pattern).  Use [`put_document`] to replace.
    ///
    /// Args:
    ///     id: External document identifier.
    ///     doc: A `dict` mapping field names to values.
    pub fn add_document(&self, py: Python, id: &str, doc: &Bound<PyDict>) -> PyResult<()> {
        let document = dict_to_document(py, doc)?;
        let engine = self.engine.clone();
        let id = id.to_string();
        self.rt
            .block_on(engine.add_document(&id, document))
            .map_err(laurus_err)
    }

    /// Retrieve all document versions stored under `id`.
    ///
    /// Returns a list of dicts, one per indexed version.
    pub fn get_documents(&self, py: Python, id: &str) -> PyResult<Vec<Py<PyAny>>> {
        let engine = self.engine.clone();
        let id = id.to_string();
        let docs = self
            .rt
            .block_on(engine.get_documents(&id))
            .map_err(laurus_err)?;
        docs.iter().map(|doc| document_to_dict(py, doc)).collect()
    }

    /// Delete all document versions stored under `id`.
    ///
    /// Call [`commit`] to make the deletion visible to searches.
    pub fn delete_documents(&self, _py: Python, id: &str) -> PyResult<()> {
        let engine = self.engine.clone();
        let id = id.to_string();
        self.rt
            .block_on(engine.delete_documents(&id))
            .map_err(laurus_err)
    }

    /// Flush buffered writes and make all pending changes searchable.
    pub fn commit(&self, _py: Python) -> PyResult<()> {
        let engine = self.engine.clone();
        self.rt.block_on(engine.commit()).map_err(laurus_err)
    }

    // ── Search ────────────────────────────────────────────────────────────

    /// Search the index and return a list of [`SearchResult`] objects.
    ///
    /// `query` may be:
    ///   - A **DSL string** (e.g. `"title:hello"`, `"~\"memory safety\""`)
    ///   - A **lexical query** object (`TermQuery`, `BooleanQuery`, `GeoDistanceQuery`, …)
    ///   - A **vector query** object (`VectorQuery`, `VectorTextQuery`)
    ///   - A **[`SearchRequest`]** for full control (hybrid, filter, fusion)
    ///
    /// Args:
    ///     query: The query to execute.
    ///     limit: Maximum number of results to return (default 10).
    ///     offset: Pagination offset (default 0).
    ///
    /// Returns:
    ///     A list of [`SearchResult`] objects with `.id`, `.score`, `.document`.
    #[pyo3(signature = (query, *, limit=10, offset=0))]
    pub fn search(
        &self,
        py: Python,
        query: &Bound<PyAny>,
        limit: usize,
        offset: usize,
    ) -> PyResult<Vec<PySearchResult>> {
        let request = build_request_from_py(py, query, limit, offset)?;

        let engine = self.engine.clone();
        let results = self
            .rt
            .block_on(engine.search(request))
            .map_err(laurus_err)?;

        results
            .into_iter()
            .map(|r| to_py_search_result(py, r))
            .collect()
    }

    /// Execute multiple independent searches in one call.
    ///
    /// Each query in `queries` is dispatched in parallel on the underlying
    /// tokio runtime via `laurus::Engine::search_batch`. Each entry can
    /// be the same kind of value `search()` accepts: a DSL string, a
    /// `LexicalQuery` / `VectorQuery` / `VectorTextQuery` object, or a
    /// `SearchRequest`. The same `limit` and `offset` are applied to
    /// every query in the batch.
    ///
    /// Args:
    ///     queries: A list of queries to execute. Order is preserved in
    ///         the output.
    ///     limit: Maximum number of results to return per query
    ///         (default 10).
    ///     offset: Pagination offset applied to each query (default 0).
    ///
    /// Returns:
    ///     A list of lists: `results[i]` is the result list for
    ///     `queries[i]`. Empty input returns an empty list without
    ///     invoking the engine.
    ///
    /// Issue [#717](https://github.com/mosuka/laurus/issues/717)
    /// Phase 3b of [#648](https://github.com/mosuka/laurus/issues/648).
    #[pyo3(signature = (queries, *, limit=10, offset=0))]
    pub fn search_batch(
        &self,
        py: Python,
        queries: &Bound<PyAny>,
        limit: usize,
        offset: usize,
    ) -> PyResult<Vec<Vec<PySearchResult>>> {
        let queries_seq = queries.try_iter().map_err(|_| {
            PyRuntimeError::new_err(
                "search_batch: expected an iterable of queries (DSL string, Query object, or SearchRequest)",
            )
        })?;

        let mut requests = Vec::new();
        for item in queries_seq {
            let item = item?;
            requests.push(build_request_from_py(py, &item, limit, offset)?);
        }

        if requests.is_empty() {
            return Ok(Vec::new());
        }

        let engine = self.engine.clone();
        let batch_results = self
            .rt
            .block_on(engine.search_batch(requests))
            .map_err(laurus_err)?;

        batch_results
            .into_iter()
            .map(|per_query_results| {
                per_query_results
                    .into_iter()
                    .map(|r| to_py_search_result(py, r))
                    .collect::<PyResult<Vec<_>>>()
            })
            .collect()
    }

    // ── Schema & stats ────────────────────────────────────────────────────

    /// Return index statistics.
    ///
    /// Returns a dict with keys:
    ///   - `document_count` (int): total indexed documents.
    ///   - `vector_fields` (dict): per-field vector statistics.
    pub fn stats(&self, py: Python) -> PyResult<Py<PyAny>> {
        let engine = self.engine.clone();
        let stats: EngineStats = self
            .rt
            .block_on(async { engine.stats() })
            .map_err(laurus_err)?;
        let dict = PyDict::new(py);
        dict.set_item("document_count", stats.document_count)?;
        let vf = PyDict::new(py);
        for (field, field_stats) in &stats.vector_fields {
            let fd = PyDict::new(py);
            fd.set_item("count", field_stats.vector_count)?;
            fd.set_item("dimension", field_stats.dimension)?;
            vf.set_item(field, fd)?;
        }
        dict.set_item("vector_fields", vf)?;
        Ok(dict.into_any().unbind())
    }

    fn __repr__(&self) -> String {
        "Index()".to_string()
    }
}

// ---------------------------------------------------------------------------
// Storage factory helper
// ---------------------------------------------------------------------------

fn create_storage(path: Option<&str>) -> PyResult<Arc<dyn Storage>> {
    let config = match path {
        None => StorageConfig::Memory(Default::default()),
        Some(p) => {
            use laurus::storage::file::FileStorageConfig;
            StorageConfig::File(FileStorageConfig::new(Path::new(p)))
        }
    };
    StorageFactory::create(config).map_err(laurus_err)
}
