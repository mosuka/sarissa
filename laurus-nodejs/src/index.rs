//! Node.js-facing [`Index`] class — the primary entry point for the laurus binding.

use std::path::Path;
use std::sync::Arc;

use crate::convert::{data_value_to_json, json_to_document};
use crate::errors::laurus_err;
use crate::query::{JsQuery, JsTermQuery, JsVectorQuery, JsVectorQueryInner, JsVectorTextQuery};
use crate::schema::JsSchema;
use crate::search::{
    JsSearchRequest, JsSearchResult, build_dsl_request, build_lexical_request,
    build_vector_request, to_js_search_result,
};
use laurus::{Engine, Storage, StorageConfig, StorageFactory};
use napi::bindgen_prelude::*;
use napi_derive::napi;
use serde_json::Value;

// ---------------------------------------------------------------------------
// Index
// ---------------------------------------------------------------------------

/// Laurus search index — the main entry point for the Node.js binding.
///
/// ## Creating an index
///
/// ```javascript
/// const { Index, Schema } = require("laurus-nodejs");
///
/// // In-memory (ephemeral, great for prototyping)
/// const index = await Index.create();
///
/// // File-based (persistent)
/// const schema = new Schema();
/// schema.addTextField("title");
/// schema.addTextField("body");
/// schema.addHnswField("embedding", 384);
/// const index = await Index.create("./myindex", schema);
/// ```
///
/// ## Adding documents
///
/// ```javascript
/// await index.putDocument("doc1", { title: "Hello", body: "World" });
/// await index.commit();
/// ```
///
/// ## Searching
///
/// ```javascript
/// // DSL string
/// const results = await index.search("title:hello");
///
/// // Term query
/// const results = await index.searchTerm("body", "rust");
///
/// // Via SearchRequest for full control
/// const req = new SearchRequest();
/// req.setVectorTextQuery("embedding", "concurrent");
/// req.setRrfFusion();
/// const results = await index.searchWithRequest(req);
/// ```
#[napi(js_name = "Index")]
pub struct JsIndex {
    engine: Arc<Engine>,
}

#[napi]
impl JsIndex {
    /// Create a new index.
    ///
    /// # Arguments
    ///
    /// * `path` - Directory path for persistent storage.
    ///     Pass `null` or omit for an ephemeral in-memory index.
    /// * `schema` - Schema definition. If omitted, an empty schema is used.
    ///
    /// # Returns
    ///
    /// A new `Index` instance.
    #[napi(factory)]
    pub async fn create(path: Option<String>, schema: Option<&JsSchema>) -> Result<Self> {
        let storage = create_storage(path.as_deref())?;
        let schema = schema.map(|s| s.inner.clone()).unwrap_or_default();

        let engine = Engine::new(storage, schema).await.map_err(laurus_err)?;

        Ok(Self {
            engine: Arc::new(engine),
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
    /// * `doc` - An object mapping field names to values.
    #[napi]
    pub async fn put_document(&self, id: String, doc: Value) -> Result<()> {
        let document = json_to_document(&doc)?;
        self.engine
            .put_document(&id, document)
            .await
            .map_err(laurus_err)
    }

    /// Append a document version without removing existing versions.
    ///
    /// Laurus supports multiple versions of the same id (chunk-per-document
    /// RAG pattern). Use `putDocument` to replace.
    ///
    /// # Arguments
    ///
    /// * `id` - External document identifier.
    /// * `doc` - An object mapping field names to values.
    #[napi]
    pub async fn add_document(&self, id: String, doc: Value) -> Result<()> {
        let document = json_to_document(&doc)?;
        self.engine
            .add_document(&id, document)
            .await
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
    /// A list of document objects (one per indexed version).
    #[napi]
    pub async fn get_documents(&self, id: String) -> Result<Vec<Value>> {
        let docs = self.engine.get_documents(&id).await.map_err(laurus_err)?;
        Ok(docs
            .iter()
            .map(|doc| {
                let mut map = serde_json::Map::new();
                for (field, value) in &doc.fields {
                    map.insert(field.clone(), data_value_to_json(value));
                }
                Value::Object(map)
            })
            .collect())
    }

    /// Delete all document versions stored under `id`.
    ///
    /// Call `commit()` to make the deletion visible to searches.
    ///
    /// # Arguments
    ///
    /// * `id` - External document identifier.
    #[napi]
    pub async fn delete_documents(&self, id: String) -> Result<()> {
        self.engine.delete_documents(&id).await.map_err(laurus_err)
    }

    /// Flush buffered writes and make all pending changes searchable.
    #[napi]
    pub async fn commit(&self) -> Result<()> {
        self.engine.commit().await.map_err(laurus_err)
    }

    // ── Search ────────────────────────────────────────────────────────────

    /// Search using a DSL string query.
    ///
    /// # Arguments
    ///
    /// * `query` - The query DSL string (e.g. `"title:hello"`, `"~\"memory safety\""`).
    /// * `limit` - Maximum number of results (default 10).
    /// * `offset` - Pagination offset (default 0).
    ///
    /// # Returns
    ///
    /// An array of SearchResult objects.
    #[napi]
    pub async fn search(
        &self,
        query: String,
        limit: Option<u32>,
        offset: Option<u32>,
    ) -> Result<Vec<JsSearchResult>> {
        let request = build_dsl_request(
            query,
            limit.unwrap_or(10) as usize,
            offset.unwrap_or(0) as usize,
        );
        let results = self.engine.search(request).await.map_err(laurus_err)?;
        Ok(results.into_iter().map(to_js_search_result).collect())
    }

    /// Search using a term query.
    ///
    /// # Arguments
    ///
    /// * `field` - The field to search in.
    /// * `term` - The exact term to match.
    /// * `limit` - Maximum number of results (default 10).
    /// * `offset` - Pagination offset (default 0).
    ///
    /// # Returns
    ///
    /// An array of SearchResult objects.
    #[napi]
    pub async fn search_term(
        &self,
        field: String,
        term: String,
        limit: Option<u32>,
        offset: Option<u32>,
    ) -> Result<Vec<JsSearchResult>> {
        let query = JsQuery::TermQuery(JsTermQuery { field, term });
        let request = build_lexical_request(
            &query,
            limit.unwrap_or(10) as usize,
            offset.unwrap_or(0) as usize,
        )?;
        let results = self.engine.search(request).await.map_err(laurus_err)?;
        Ok(results.into_iter().map(to_js_search_result).collect())
    }

    /// Search using a pre-computed embedding vector.
    ///
    /// # Arguments
    ///
    /// * `field` - The vector field name.
    /// * `vector` - The embedding vector as an array of numbers.
    /// * `limit` - Maximum number of results (default 10).
    /// * `offset` - Pagination offset (default 0).
    ///
    /// # Returns
    ///
    /// An array of SearchResult objects.
    #[napi]
    pub async fn search_vector(
        &self,
        field: String,
        vector: Vec<f64>,
        limit: Option<u32>,
        offset: Option<u32>,
    ) -> Result<Vec<JsSearchResult>> {
        let query = JsVectorQuery::VectorQuery(JsVectorQueryInner {
            field,
            vector: vector.into_iter().map(|v| v as f32).collect(),
        });
        let request = build_vector_request(
            &query,
            limit.unwrap_or(10) as usize,
            offset.unwrap_or(0) as usize,
        );
        let results = self.engine.search(request).await.map_err(laurus_err)?;
        Ok(results.into_iter().map(to_js_search_result).collect())
    }

    /// Search using a text-based vector query (embedded by the registered embedder).
    ///
    /// # Arguments
    ///
    /// * `field` - The vector field name.
    /// * `text` - The text to embed and search with.
    /// * `limit` - Maximum number of results (default 10).
    /// * `offset` - Pagination offset (default 0).
    ///
    /// # Returns
    ///
    /// An array of SearchResult objects.
    #[napi]
    pub async fn search_vector_text(
        &self,
        field: String,
        text: String,
        limit: Option<u32>,
        offset: Option<u32>,
    ) -> Result<Vec<JsSearchResult>> {
        let query = JsVectorQuery::VectorTextQuery(JsVectorTextQuery { field, text });
        let request = build_vector_request(
            &query,
            limit.unwrap_or(10) as usize,
            offset.unwrap_or(0) as usize,
        );
        let results = self.engine.search(request).await.map_err(laurus_err)?;
        Ok(results.into_iter().map(to_js_search_result).collect())
    }

    /// Search using a full SearchRequest for advanced control.
    ///
    /// # Arguments
    ///
    /// * `request` - A `SearchRequest` object.
    ///
    /// # Returns
    ///
    /// An array of SearchResult objects.
    #[napi]
    pub async fn search_with_request(
        &self,
        request: &JsSearchRequest,
    ) -> Result<Vec<JsSearchResult>> {
        let req = request.build()?;
        let results = self.engine.search(req).await.map_err(laurus_err)?;
        Ok(results.into_iter().map(to_js_search_result).collect())
    }

    /// Execute multiple independent searches in one call.
    ///
    /// Each query is dispatched in parallel on the underlying tokio
    /// runtime via `laurus::Engine::search_batch`. The same `limit`
    /// and `offset` are applied to every query in the batch.
    ///
    /// # Arguments
    ///
    /// * `queries` - An array of query DSL strings.
    /// * `limit` - Maximum number of results per query (default 10).
    /// * `offset` - Pagination offset applied to each query (default 0).
    ///
    /// # Returns
    ///
    /// An array of arrays: `results[i]` is the result list for
    /// `queries[i]`. Empty input returns `[]` without invoking the
    /// engine.
    ///
    /// Issue [#718](https://github.com/mosuka/laurus/issues/718)
    /// Phase 3c of [#648](https://github.com/mosuka/laurus/issues/648).
    #[napi]
    pub async fn search_batch(
        &self,
        queries: Vec<String>,
        limit: Option<u32>,
        offset: Option<u32>,
    ) -> Result<Vec<Vec<JsSearchResult>>> {
        if queries.is_empty() {
            return Ok(Vec::new());
        }

        let limit = limit.unwrap_or(10) as usize;
        let offset = offset.unwrap_or(0) as usize;
        let requests: Vec<_> = queries
            .into_iter()
            .map(|q| build_dsl_request(q, limit, offset))
            .collect();

        let batch_results = self
            .engine
            .search_batch(requests)
            .await
            .map_err(laurus_err)?;

        Ok(batch_results
            .into_iter()
            .map(|per_query_results| {
                per_query_results
                    .into_iter()
                    .map(to_js_search_result)
                    .collect()
            })
            .collect())
    }

    // ── Stats ─────────────────────────────────────────────────────────────

    /// Return index statistics.
    ///
    /// # Returns
    ///
    /// An object with:
    ///   - `documentCount` (number): total indexed documents.
    ///   - `vectorFields` (object): per-field vector statistics with `count` and `dimension`.
    #[napi]
    pub fn stats(&self) -> Result<Value> {
        let stats = self.engine.stats().map_err(laurus_err)?;
        let mut vector_fields = serde_json::Map::new();
        for (field, field_stats) in &stats.vector_fields {
            vector_fields.insert(
                field.clone(),
                serde_json::json!({
                    "count": field_stats.vector_count,
                    "dimension": field_stats.dimension,
                }),
            );
        }
        Ok(serde_json::json!({
            "documentCount": stats.document_count,
            "vectorFields": vector_fields,
        }))
    }
}

// ---------------------------------------------------------------------------
// Storage factory helper
// ---------------------------------------------------------------------------

fn create_storage(path: Option<&str>) -> Result<Arc<dyn Storage>> {
    let config = match path {
        None => StorageConfig::Memory(Default::default()),
        Some(p) => {
            use laurus::storage::file::FileStorageConfig;
            StorageConfig::File(FileStorageConfig::new(Path::new(p)))
        }
    };
    StorageFactory::create(config).map_err(laurus_err)
}
