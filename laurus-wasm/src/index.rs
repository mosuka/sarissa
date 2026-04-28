//! WASM-facing [`Index`] class — the primary entry point for the laurus-wasm binding.

use std::sync::Arc;

use crate::convert::{data_value_to_json, json_to_document};
use crate::errors::laurus_err;
use crate::query::{
    JsGeo3dBoundingBoxQuery, JsGeo3dDistanceQuery, JsGeo3dNearestQuery, JsQuery, JsTermQuery,
    JsVectorQuery, JsVectorQueryInner, JsVectorTextQuery,
};
use crate::schema::WasmSchema;
use crate::search::{build_dsl_request, build_lexical_request, build_vector_request};
use crate::storage::OpfsPersistence;
use laurus::embedding::embedder::Embedder;
use laurus::embedding::per_field::PerFieldEmbedder;
use laurus::embedding::precomputed::PrecomputedEmbedder;
use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::{Engine, EngineBuilder};
use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Serialize search results to JS via JSON.parse(JSON string).
///
/// This avoids issues with `serde_wasm_bindgen` not correctly handling
/// nested `serde_json::Value` types. Instead, we serialize the entire
/// result to a JSON string and then parse it in JS.
fn search_results_to_js(results: Vec<laurus::SearchResult>) -> Result<JsValue, JsValue> {
    let json_results: Vec<serde_json::Value> = results
        .into_iter()
        .map(|r| {
            let document = r.document.map(|doc| {
                let mut map = serde_json::Map::new();
                for (field, value) in doc.fields {
                    map.insert(field, data_value_to_json(&value));
                }
                serde_json::Value::Object(map)
            });
            serde_json::json!({
                "id": r.id,
                "score": r.score,
                "document": document,
            })
        })
        .collect();

    let json_str = serde_json::to_string(&json_results)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))?;
    js_sys::JSON::parse(&json_str)
}

/// Serialize documents to JS via JSON.parse(JSON string).
fn documents_to_js(docs: Vec<laurus::Document>) -> Result<JsValue, JsValue> {
    let json_docs: Vec<serde_json::Value> = docs
        .iter()
        .map(|doc| {
            let mut map = serde_json::Map::new();
            for (field, value) in &doc.fields {
                map.insert(field.clone(), data_value_to_json(value));
            }
            serde_json::Value::Object(map)
        })
        .collect();

    let json_str = serde_json::to_string(&json_docs)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))?;
    js_sys::JSON::parse(&json_str)
}

/// Build a [`PerFieldEmbedder`] from JS callback embedders and the schema.
///
/// Reads the schema to find which vector fields reference which embedder name,
/// then maps field names to the corresponding JS callback embedder.
fn build_per_field_embedder(
    schema: &laurus::Schema,
    js_embedders: std::collections::HashMap<String, crate::embedder::JsCallbackEmbedder>,
) -> Arc<dyn Embedder> {
    let default: Arc<dyn Embedder> = Arc::new(PrecomputedEmbedder::new());
    let per_field = PerFieldEmbedder::new(default);

    // Build a map: embedder_name -> Arc<dyn Embedder>
    let mut embedder_map: std::collections::HashMap<String, Arc<dyn Embedder>> =
        std::collections::HashMap::new();
    for (name, embedder) in js_embedders {
        embedder_map.insert(name, Arc::new(embedder));
    }

    // Map field_name -> embedder based on the schema's field → embedder_name mapping
    for (field_name, field_option) in &schema.fields {
        if let Some(embedder_name) = field_option.embedder_name()
            && let Some(emb) = embedder_map.get(embedder_name)
        {
            per_field.add_embedder(field_name, emb.clone());
        }
    }

    Arc::new(per_field)
}

// ---------------------------------------------------------------------------
// Index
// ---------------------------------------------------------------------------

/// Laurus search index — the main entry point for the WASM binding.
///
/// Supports two storage modes:
/// - **In-memory** (`Index.create(schema)`) — ephemeral, data lost on page reload
/// - **OPFS-persistent** (`Index.open(name, schema)`) — data survives page reloads
///
/// ```javascript
/// import { Index, Schema } from "laurus-wasm";
///
/// const schema = new Schema();
/// schema.addTextField("title");
/// schema.addTextField("body");
///
/// // In-memory (ephemeral)
/// const index = await Index.create(schema);
///
/// // OPFS-persistent (survives page reloads)
/// const index = await Index.open("my-index", schema);
///
/// await index.putDocument("doc1", { title: "Hello", body: "World" });
/// await index.commit(); // also persists to OPFS if opened with open()
///
/// const results = await index.search("title:hello");
/// ```
#[wasm_bindgen(js_name = "Index")]
pub struct WasmIndex {
    engine: Arc<Engine>,
    storage: Arc<MemoryStorage>,
    opfs: Option<OpfsPersistence>,
}

#[wasm_bindgen(js_class = "Index")]
impl WasmIndex {
    /// Create a new in-memory index (ephemeral, not persisted).
    ///
    /// # Arguments
    ///
    /// * `schema` - Schema definition.
    ///
    /// # Returns
    ///
    /// A new `Index` instance backed by in-memory storage.
    #[wasm_bindgen]
    pub async fn create(schema: WasmSchema) -> Result<WasmIndex, JsValue> {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let js_embedders = schema.js_embedders;
        let schema = schema.inner;

        // Build embedder BEFORE moving schema into EngineBuilder
        let embedder = if js_embedders.is_empty() {
            None
        } else {
            Some(build_per_field_embedder(&schema, js_embedders))
        };

        let mut builder = EngineBuilder::new(storage.clone() as Arc<dyn Storage>, schema);
        if let Some(emb) = embedder {
            builder = builder.embedder(emb);
        }

        let engine = builder.build().await.map_err(laurus_err)?;

        Ok(Self {
            engine: Arc::new(engine),
            storage,
            opfs: None,
        })
    }

    /// Open or create a persistent index backed by OPFS.
    ///
    /// If an index with the given name already exists in OPFS, its data is
    /// loaded into memory. Otherwise, a new empty index is created.
    ///
    /// Data is automatically persisted to OPFS on each `commit()` call.
    ///
    /// # Arguments
    ///
    /// * `name` - Index name (used as the OPFS subdirectory name).
    /// * `schema` - Schema definition.
    ///
    /// # Returns
    ///
    /// A new `Index` instance backed by OPFS-persistent storage.
    #[wasm_bindgen]
    pub async fn open(name: String, schema: WasmSchema) -> Result<WasmIndex, JsValue> {
        let opfs = OpfsPersistence::open(&name).await?;
        let storage = opfs.load().await?;
        let js_embedders = schema.js_embedders;
        let schema = schema.inner;

        let embedder = if js_embedders.is_empty() {
            None
        } else {
            Some(build_per_field_embedder(&schema, js_embedders))
        };

        let mut builder = EngineBuilder::new(storage.clone() as Arc<dyn Storage>, schema);
        if let Some(emb) = embedder {
            builder = builder.embedder(emb);
        }

        let engine = builder.build().await.map_err(laurus_err)?;

        Ok(Self {
            engine: Arc::new(engine),
            storage,
            opfs: Some(opfs),
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
    /// * `doc` - A JS object mapping field names to values.
    #[wasm_bindgen(js_name = "putDocument")]
    pub async fn put_document(&self, id: String, doc: JsValue) -> Result<(), JsValue> {
        let value: serde_json::Value = serde_wasm_bindgen::from_value(doc)
            .map_err(|e| JsValue::from_str(&format!("Invalid document: {e}")))?;
        let document = json_to_document(&value)?;
        self.engine
            .put_document(&id, document)
            .await
            .map_err(laurus_err)
    }

    /// Append a document version without removing existing versions.
    ///
    /// # Arguments
    ///
    /// * `id` - External document identifier.
    /// * `doc` - A JS object mapping field names to values.
    #[wasm_bindgen(js_name = "addDocument")]
    pub async fn add_document(&self, id: String, doc: JsValue) -> Result<(), JsValue> {
        let value: serde_json::Value = serde_wasm_bindgen::from_value(doc)
            .map_err(|e| JsValue::from_str(&format!("Invalid document: {e}")))?;
        let document = json_to_document(&value)?;
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
    /// A JS array of document objects.
    #[wasm_bindgen(js_name = "getDocuments")]
    pub async fn get_documents(&self, id: String) -> Result<JsValue, JsValue> {
        let docs = self.engine.get_documents(&id).await.map_err(laurus_err)?;
        documents_to_js(docs)
    }

    /// Delete all document versions stored under `id`.
    ///
    /// Call `commit()` to make the deletion visible to searches.
    ///
    /// # Arguments
    ///
    /// * `id` - External document identifier.
    #[wasm_bindgen(js_name = "deleteDocuments")]
    pub async fn delete_documents(&self, id: String) -> Result<(), JsValue> {
        self.engine.delete_documents(&id).await.map_err(laurus_err)
    }

    /// Flush buffered writes and make all pending changes searchable.
    ///
    /// If this index was opened with `Index.open()`, the data is also
    /// persisted to OPFS.
    #[wasm_bindgen]
    pub async fn commit(&self) -> Result<(), JsValue> {
        self.engine.commit().await.map_err(laurus_err)?;

        // Persist to OPFS if applicable
        if let Some(opfs) = &self.opfs {
            opfs.save(self.storage.as_ref()).await?;
        }

        Ok(())
    }

    // ── Search ────────────────────────────────────────────────────────────

    /// Search using a DSL string query.
    ///
    /// # Arguments
    ///
    /// * `query` - The query DSL string (e.g. `"title:hello"`).
    /// * `limit` - Maximum number of results (default 10).
    /// * `offset` - Pagination offset (default 0).
    ///
    /// # Returns
    ///
    /// A JS array of SearchResult objects `{ id, score, document }`.
    #[wasm_bindgen]
    pub async fn search(
        &self,
        query: String,
        limit: Option<u32>,
        offset: Option<u32>,
    ) -> Result<JsValue, JsValue> {
        let request = build_dsl_request(
            query,
            limit.unwrap_or(10) as usize,
            offset.unwrap_or(0) as usize,
        );
        let results = self.engine.search(request).await.map_err(laurus_err)?;
        search_results_to_js(results)
    }

    /// Search using a term query.
    ///
    /// # Arguments
    ///
    /// * `field` - The field to search in.
    /// * `term` - The exact term to match.
    /// * `limit` - Maximum number of results (default 10).
    /// * `offset` - Pagination offset (default 0).
    #[wasm_bindgen(js_name = "searchTerm")]
    pub async fn search_term(
        &self,
        field: String,
        term: String,
        limit: Option<u32>,
        offset: Option<u32>,
    ) -> Result<JsValue, JsValue> {
        let query = JsQuery::TermQuery(JsTermQuery { field, term });
        let request = build_lexical_request(
            &query,
            limit.unwrap_or(10) as usize,
            offset.unwrap_or(0) as usize,
        )?;
        let results = self.engine.search(request).await.map_err(laurus_err)?;
        search_results_to_js(results)
    }

    /// Search using a 3D ECEF distance (sphere) query.
    ///
    /// # Arguments
    ///
    /// * `field` - The Geo3d field name.
    /// * `x`, `y`, `z` - Sphere centre in ECEF meters.
    /// * `radius_m` - Sphere radius in meters.
    /// * `limit` - Maximum number of results (default 10).
    /// * `offset` - Pagination offset (default 0).
    #[wasm_bindgen(js_name = "searchGeo3dDistance")]
    #[allow(clippy::too_many_arguments)]
    pub async fn search_geo3d_distance(
        &self,
        field: String,
        x: f64,
        y: f64,
        z: f64,
        radius_m: f64,
        limit: Option<u32>,
        offset: Option<u32>,
    ) -> Result<JsValue, JsValue> {
        let query = JsQuery::Geo3dDistanceQuery(JsGeo3dDistanceQuery {
            field,
            x,
            y,
            z,
            radius_m,
        });
        let request = build_lexical_request(
            &query,
            limit.unwrap_or(10) as usize,
            offset.unwrap_or(0) as usize,
        )?;
        let results = self.engine.search(request).await.map_err(laurus_err)?;
        search_results_to_js(results)
    }

    /// Search using a 3D ECEF axis-aligned bounding-box query.
    ///
    /// # Arguments
    ///
    /// * `field` - The Geo3d field name.
    /// * `min_x`, `min_y`, `min_z` - Lower corner of the box.
    /// * `max_x`, `max_y`, `max_z` - Upper corner of the box.
    /// * `limit` - Maximum number of results (default 10).
    /// * `offset` - Pagination offset (default 0).
    #[wasm_bindgen(js_name = "searchGeo3dBoundingBox")]
    #[allow(clippy::too_many_arguments)]
    pub async fn search_geo3d_bounding_box(
        &self,
        field: String,
        min_x: f64,
        min_y: f64,
        min_z: f64,
        max_x: f64,
        max_y: f64,
        max_z: f64,
        limit: Option<u32>,
        offset: Option<u32>,
    ) -> Result<JsValue, JsValue> {
        let query = JsQuery::Geo3dBoundingBoxQuery(JsGeo3dBoundingBoxQuery {
            field,
            min_x,
            min_y,
            min_z,
            max_x,
            max_y,
            max_z,
        });
        let request = build_lexical_request(
            &query,
            limit.unwrap_or(10) as usize,
            offset.unwrap_or(0) as usize,
        )?;
        let results = self.engine.search(request).await.map_err(laurus_err)?;
        search_results_to_js(results)
    }

    /// Search using a 3D ECEF k-nearest-neighbours query.
    ///
    /// # Arguments
    ///
    /// * `field` - The Geo3d field name.
    /// * `x`, `y`, `z` - Centre coordinates in ECEF meters.
    /// * `k` - Number of nearest neighbours to return.
    /// * `limit` - Maximum number of results (default 10).
    /// * `offset` - Pagination offset (default 0).
    #[wasm_bindgen(js_name = "searchGeo3dNearest")]
    #[allow(clippy::too_many_arguments)]
    pub async fn search_geo3d_nearest(
        &self,
        field: String,
        x: f64,
        y: f64,
        z: f64,
        k: u32,
        limit: Option<u32>,
        offset: Option<u32>,
        initial_radius_m: Option<f64>,
        max_radius_m: Option<f64>,
    ) -> Result<JsValue, JsValue> {
        let query = JsQuery::Geo3dNearestQuery(JsGeo3dNearestQuery {
            field,
            x,
            y,
            z,
            k,
            initial_radius_m,
            max_radius_m,
        });
        let request = build_lexical_request(
            &query,
            limit.unwrap_or(10) as usize,
            offset.unwrap_or(0) as usize,
        )?;
        let results = self.engine.search(request).await.map_err(laurus_err)?;
        search_results_to_js(results)
    }

    /// Search using a pre-computed embedding vector.
    ///
    /// # Arguments
    ///
    /// * `field` - The vector field name.
    /// * `vector` - The embedding vector as a Float64Array or number[].
    /// * `limit` - Maximum number of results (default 10).
    /// * `offset` - Pagination offset (default 0).
    #[wasm_bindgen(js_name = "searchVector")]
    pub async fn search_vector(
        &self,
        field: String,
        vector: Vec<f64>,
        limit: Option<u32>,
        offset: Option<u32>,
    ) -> Result<JsValue, JsValue> {
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
        search_results_to_js(results)
    }

    /// Search using a text-based vector query (embedded by the registered embedder).
    ///
    /// # Arguments
    ///
    /// * `field` - The vector field name.
    /// * `text` - The text to embed and search with.
    /// * `limit` - Maximum number of results (default 10).
    /// * `offset` - Pagination offset (default 0).
    #[wasm_bindgen(js_name = "searchVectorText")]
    pub async fn search_vector_text(
        &self,
        field: String,
        text: String,
        limit: Option<u32>,
        offset: Option<u32>,
    ) -> Result<JsValue, JsValue> {
        let query = JsVectorQuery::VectorTextQuery(JsVectorTextQuery { field, text });
        let request = build_vector_request(
            &query,
            limit.unwrap_or(10) as usize,
            offset.unwrap_or(0) as usize,
        );
        let results = self.engine.search(request).await.map_err(laurus_err)?;
        search_results_to_js(results)
    }

    // ── Stats ─────────────────────────────────────────────────────────────

    /// Return index statistics.
    ///
    /// # Returns
    ///
    /// An object with `documentCount` and `vectorFields`.
    #[wasm_bindgen]
    pub fn stats(&self) -> Result<JsValue, JsValue> {
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
        let json = serde_json::json!({
            "documentCount": stats.document_count,
            "vectorFields": vector_fields,
        });
        let json_str = serde_json::to_string(&json)
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))?;
        js_sys::JSON::parse(&json_str)
    }
}
