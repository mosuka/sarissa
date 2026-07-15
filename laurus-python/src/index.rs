//! Python-facing [`Index`] class — the primary entry point for the laurus binding.

use std::path::Path;
use std::sync::Arc;

use crate::convert::{dict_to_document, document_to_dict};
use crate::errors::laurus_err;
use crate::schema::PySchema;
use crate::search::{PySearchResult, build_request_from_py, to_py_search_result};
use laurus::{
    DEFAULT_GROUP_MAX_BYTES, DEFAULT_GROUP_MAX_RECORDS, Engine, EngineStats, Storage,
    StorageConfig, StorageFactory, WalSyncPolicy,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

// ---------------------------------------------------------------------------
// WalSyncPolicy
// ---------------------------------------------------------------------------

/// Durability policy that controls when Write-Ahead Log (WAL) appends are
/// flushed (`fsync`'d) to durable storage.
///
/// This is a value object wrapping the Rust [`laurus::WalSyncPolicy`]. It is
/// passed to [`Index`] at construction time via the `wal_sync_policy` keyword
/// argument. It trades the durability of an *individual* write against ingest
/// throughput; [`Index.commit`] is always a hard durability barrier regardless
/// of the policy in effect.
///
/// ## Constructing a policy
///
/// ```python
/// import laurus
///
/// # Per-record durability (the default): every append is fsync'd
/// # before it returns. Safest, lowest ingest throughput.
/// policy = laurus.WalSyncPolicy.per_record()
///
/// # Group-commit durability with default thresholds (batch fsyncs).
/// policy = laurus.WalSyncPolicy.group()
///
/// # Group-commit with explicit thresholds and a flush interval.
/// policy = laurus.WalSyncPolicy.group(
///     max_records=4096,
///     max_bytes=4 * 1024 * 1024,
///     max_interval_ms=1000,
/// )
///
/// index = laurus.Index(wal_sync_policy=policy)
/// ```
#[pyclass(name = "WalSyncPolicy", skip_from_py_object)]
#[derive(Clone)]
pub struct PyWalSyncPolicy {
    /// The wrapped Rust durability policy.
    pub inner: WalSyncPolicy,
}

#[pymethods]
impl PyWalSyncPolicy {
    /// Create a per-record durability policy.
    ///
    /// Every WAL append is fsync'd to durable storage before the write call
    /// returns. This is the safest policy and the default behaviour when no
    /// `wal_sync_policy` is supplied to [`Index`], at the cost of the lowest
    /// ingest throughput.
    ///
    /// Returns:
    ///     A `WalSyncPolicy` wrapping `WalSyncPolicy::PerRecord`.
    #[staticmethod]
    pub fn per_record() -> Self {
        Self {
            inner: WalSyncPolicy::PerRecord,
        }
    }

    /// Create a group-commit durability policy.
    ///
    /// WAL appends are batched and fsync'd together once any of the configured
    /// thresholds is reached, rather than one fsync per record. This increases
    /// ingest throughput at the cost of potentially losing the last unsynced
    /// batch on a crash. [`Index.commit`] remains a hard durability barrier,
    /// and [`Index.flush_wal`] forces a flush on demand.
    ///
    /// Args:
    ///     max_records: Flush after this many records accumulate since the last
    ///         flush. Defaults to laurus' built-in
    ///         `DEFAULT_GROUP_MAX_RECORDS` (1024) when `None`.
    ///     max_bytes: Flush after this many bytes accumulate since the last
    ///         flush. Defaults to laurus' built-in `DEFAULT_GROUP_MAX_BYTES`
    ///         (1 MiB) when `None`.
    ///     max_interval_ms: Optional time-based flush interval in milliseconds.
    ///         When set, a background timer flushes the WAL at least this often
    ///         even if the size thresholds have not been reached. When `None`
    ///         (the default) no time-based flushing occurs.
    ///
    /// Returns:
    ///     A `WalSyncPolicy` wrapping `WalSyncPolicy::Group { .. }`.
    #[staticmethod]
    #[pyo3(signature = (max_records=None, max_bytes=None, max_interval_ms=None))]
    pub fn group(
        max_records: Option<usize>,
        max_bytes: Option<usize>,
        max_interval_ms: Option<u64>,
    ) -> Self {
        Self {
            inner: WalSyncPolicy::Group {
                max_records: max_records.unwrap_or(DEFAULT_GROUP_MAX_RECORDS),
                max_bytes: max_bytes.unwrap_or(DEFAULT_GROUP_MAX_BYTES),
                max_interval: max_interval_ms.map(std::time::Duration::from_millis),
            },
        }
    }

    fn __repr__(&self) -> String {
        match self.inner {
            WalSyncPolicy::PerRecord => "WalSyncPolicy.per_record()".to_string(),
            WalSyncPolicy::Group {
                max_records,
                max_bytes,
                max_interval,
            } => format!(
                "WalSyncPolicy.group(max_records={}, max_bytes={}, max_interval_ms={})",
                max_records,
                max_bytes,
                match max_interval {
                    Some(d) => d.as_millis().to_string(),
                    None => "None".to_string(),
                }
            ),
        }
    }
}

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
    ///     wal_sync_policy: Optional [`WalSyncPolicy`] controlling when WAL
    ///           appends are fsync'd. When `None` (the default), laurus uses
    ///           per-record durability (every append is fsync'd before it
    ///           returns).
    #[new]
    #[pyo3(signature = (path=None, schema=None, wal_sync_policy=None))]
    pub fn new(
        path: Option<String>,
        schema: Option<&PySchema>,
        wal_sync_policy: Option<&PyWalSyncPolicy>,
    ) -> PyResult<Self> {
        let rt =
            tokio::runtime::Runtime::new().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let storage = create_storage(path.as_deref())?;
        let schema = schema.map(|s| s.inner.clone()).unwrap_or_default();

        let mut builder = Engine::builder(storage, schema);
        if let Some(p) = wal_sync_policy {
            builder = builder.wal_sync_policy(p.inner);
        }

        let engine = rt.block_on(builder.build()).map_err(laurus_err)?;

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

    /// Index many documents in one call, replacing existing documents by id.
    ///
    /// Batched form of [`put_document`]: the `(id, dict)` pairs are applied
    /// sequentially, in order, with a single WAL fsync for the whole batch.
    /// Duplicate ids within one batch deduplicate exactly like the same puts
    /// issued one by one (the last occurrence wins).
    ///
    /// Args:
    ///     docs: An iterable of `(id, dict)` pairs.
    ///
    /// Fails fast at the first document that cannot be indexed; the raised
    /// error names the failing position and id. Documents applied before the
    /// failure are **not** rolled back (retrying the batch is idempotent).
    /// Call [`commit`] to make the changes visible to searches.
    pub fn put_documents(&self, py: Python, docs: &Bound<PyAny>) -> PyResult<()> {
        let batch = pairs_to_documents(py, docs)?;
        if batch.is_empty() {
            return Ok(());
        }
        let engine = self.engine.clone();
        self.rt
            .block_on(engine.put_documents(batch))
            .map_err(laurus_err)
    }

    /// Append many document versions in one call, without removing existing
    /// versions.
    ///
    /// Batched form of [`add_document`]. Ordering, single-fsync durability,
    /// and fail-fast error semantics match [`put_documents`], but repeated
    /// ids accumulate as separate versions instead of deduplicating.
    ///
    /// Args:
    ///     docs: An iterable of `(id, dict)` pairs.
    pub fn add_documents(&self, py: Python, docs: &Bound<PyAny>) -> PyResult<()> {
        let batch = pairs_to_documents(py, docs)?;
        if batch.is_empty() {
            return Ok(());
        }
        let engine = self.engine.clone();
        self.rt
            .block_on(engine.add_documents(batch))
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

    /// Force any buffered Write-Ahead Log (WAL) appends to be flushed
    /// (`fsync`'d) to durable storage.
    ///
    /// This matters only under a group-commit [`WalSyncPolicy`]
    /// ([`WalSyncPolicy.group`]), where individual appends are batched and not
    /// fsync'd immediately. Under the default per-record policy
    /// ([`WalSyncPolicy.per_record`]) every append is already durable, so this
    /// call is effectively a no-op.
    ///
    /// Durability trade-off: with group commit, a crash can lose the most
    /// recent unsynced batch of appends. Call `flush_wal()` to bound that
    /// window on demand without paying for a full [`commit`], which would also
    /// materialize the in-memory index state. Use `flush_wal()` when you want
    /// the WAL durable but do not yet need the pending changes to be
    /// searchable; use [`commit`] when you need both.
    ///
    /// This call is synchronous and does not make any pending changes
    /// searchable; use [`commit`] for that.
    ///
    /// Raises:
    ///     An exception if the underlying WAL flush fails (for example, an I/O
    ///     error while fsync'ing).
    pub fn flush_wal(&self, _py: Python) -> PyResult<()> {
        self.engine.flush_wal().map_err(laurus_err)
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
// Batch-ingestion helper
// ---------------------------------------------------------------------------

/// Convert a Python iterable of `(id, dict)` pairs into the engine's
/// `(String, Document)` batch, naming the offending position on any entry
/// that is not a two-element `(str, dict)` pair.
fn pairs_to_documents(
    py: Python,
    docs: &Bound<PyAny>,
) -> PyResult<Vec<(String, laurus::Document)>> {
    let iter = docs.try_iter().map_err(|_| {
        PyRuntimeError::new_err(
            "expected an iterable of (id, dict) pairs, e.g. [(\"doc1\", {\"title\": \"...\"}), ...]",
        )
    })?;

    let mut batch = Vec::new();
    for (index, item) in iter.enumerate() {
        let item = item?;
        let (id, doc): (String, Bound<PyDict>) = item.extract().map_err(|_| {
            PyRuntimeError::new_err(format!(
                "documents[{index}]: expected a (id: str, doc: dict) pair"
            ))
        })?;
        batch.push((id, dict_to_document(py, &doc)?));
    }
    Ok(batch)
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
