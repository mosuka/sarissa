//! Node.js wrappers for search request/result and fusion algorithm types.

use crate::convert::data_value_to_json;
use crate::query::{
    JsGeo3dBoundingBoxQuery, JsGeo3dDistanceQuery, JsGeo3dNearestQuery, JsPhraseQuery, JsQuery,
    JsTermQuery, JsVectorQuery, JsVectorQueryInner, JsVectorTextQuery, extract_lexical_query,
    query_to_lexical_search_query, vector_query_to_search_query,
};
use laurus::{FusionAlgorithm, LexicalSearchQuery, SearchRequestBuilder, SearchResult};
use napi::bindgen_prelude::*;
use napi_derive::napi;

// ---------------------------------------------------------------------------
// Fusion algorithm types
// ---------------------------------------------------------------------------

/// Reciprocal Rank Fusion — rank-based result merging for hybrid search.
///
/// ## Example
///
/// ```javascript
/// const { RRF } = require("laurus-nodejs");
/// const fusion = new RRF(60.0);
/// ```
#[napi(js_name = "RRF")]
pub struct JsRRF {
    pub(crate) k: f64,
}

#[napi]
impl JsRRF {
    /// Create a new RRF fusion algorithm.
    ///
    /// # Arguments
    ///
    /// * `k` - The RRF parameter (default 60.0). Higher values reduce the impact of rank differences.
    #[napi(constructor)]
    pub fn new(k: Option<f64>) -> Self {
        Self {
            k: k.unwrap_or(60.0),
        }
    }
}

/// Weighted sum fusion — normalises lexical and vector scores then combines them.
///
/// ## Example
///
/// ```javascript
/// const { WeightedSum } = require("laurus-nodejs");
/// const fusion = new WeightedSum(0.3, 0.7);
/// ```
#[napi(js_name = "WeightedSum")]
pub struct JsWeightedSum {
    pub(crate) lexical_weight: f64,
    pub(crate) vector_weight: f64,
}

#[napi]
impl JsWeightedSum {
    /// Create a new weighted sum fusion algorithm.
    ///
    /// # Arguments
    ///
    /// * `lexical_weight` - Weight for lexical search scores (default 0.5).
    /// * `vector_weight` - Weight for vector search scores (default 0.5).
    #[napi(constructor)]
    pub fn new(lexical_weight: Option<f64>, vector_weight: Option<f64>) -> Self {
        Self {
            lexical_weight: lexical_weight.unwrap_or(0.5),
            vector_weight: vector_weight.unwrap_or(0.5),
        }
    }
}

// ---------------------------------------------------------------------------
// SearchResult
// ---------------------------------------------------------------------------

/// A single search result returned by `Index.search()`.
///
/// Properties:
///   - `id` (string): External document identifier.
///   - `score` (number): Relevance score (BM25, similarity, or fused).
///   - `document` (object | null): Retrieved document fields, or `null` if deleted.
#[napi(object)]
pub struct JsSearchResult {
    /// External document identifier.
    pub id: String,
    /// Relevance score.
    pub score: f64,
    /// Retrieved document fields as a key-value object, or `null`.
    pub document: Option<serde_json::Value>,
}

/// Convert a [`SearchResult`] from the engine into a serializable [`JsSearchResult`].
///
/// # Arguments
///
/// * `r` - The engine search result.
///
/// # Returns
///
/// A `JsSearchResult` with document fields converted to JSON.
pub fn to_js_search_result(r: SearchResult) -> JsSearchResult {
    let document = r.document.map(|doc| {
        let mut map = serde_json::Map::new();
        for (field, value) in doc.fields {
            map.insert(field, data_value_to_json(&value));
        }
        serde_json::Value::Object(map)
    });
    JsSearchResult {
        id: r.id,
        score: r.score as f64,
        document,
    }
}

// ---------------------------------------------------------------------------
// SearchRequest
// ---------------------------------------------------------------------------

/// Full-featured search request for advanced control over query, fusion, and
/// filtering.
///
/// ## Example — hybrid search with filter
///
/// ```javascript
/// const { SearchRequest, VectorTextQuery, TermQuery, RRF } = require("laurus-nodejs");
///
/// const request = new SearchRequest({
///     vectorQuery: new VectorTextQuery("text_vec", "type system"),
///     filterQuery: new TermQuery("category", "type-system"),
///     fusion: new RRF(60.0),
///     limit: 3,
/// });
/// const results = await index.search(request);
/// ```
#[napi(js_name = "SearchRequest")]
pub struct JsSearchRequest {
    /// A DSL string query.
    pub(crate) query_dsl: Option<String>,
    /// Lexical query component.
    pub(crate) lexical_query: Option<JsQuery>,
    /// Vector query component.
    pub(crate) vector_query: Option<JsVectorQuery>,
    /// Filter query applied after scoring.
    pub(crate) filter_query: Option<JsQuery>,
    /// Fusion algorithm.
    pub(crate) fusion: Option<FusionChoice>,
    pub(crate) limit: usize,
    pub(crate) offset: usize,
}

pub enum FusionChoice {
    RRF(f64),
    WeightedSum(f32, f32),
}

#[napi]
impl JsSearchRequest {
    /// Create a new search request.
    ///
    /// # Arguments
    ///
    /// * `limit` - Maximum number of results (default 10).
    /// * `offset` - Pagination offset (default 0).
    #[napi(constructor)]
    pub fn new(limit: Option<u32>, offset: Option<u32>) -> Self {
        Self {
            query_dsl: None,
            lexical_query: None,
            vector_query: None,
            filter_query: None,
            fusion: None,
            limit: limit.unwrap_or(10) as usize,
            offset: offset.unwrap_or(0) as usize,
        }
    }

    /// Set a DSL string query.
    ///
    /// # Arguments
    ///
    /// * `dsl` - The query DSL string (e.g. `"title:hello"`).
    #[napi]
    pub fn set_query_dsl(&mut self, dsl: String) {
        self.query_dsl = Some(dsl);
    }

    /// Set a lexical term query.
    ///
    /// # Arguments
    ///
    /// * `field` - The field name.
    /// * `term` - The term to match.
    #[napi]
    pub fn set_lexical_term_query(&mut self, field: String, term: String) {
        self.lexical_query = Some(JsQuery::TermQuery(JsTermQuery { field, term }));
    }

    /// Set a lexical phrase query.
    ///
    /// # Arguments
    ///
    /// * `field` - The field name.
    /// * `terms` - The ordered list of terms.
    #[napi]
    pub fn set_lexical_phrase_query(&mut self, field: String, terms: Vec<String>) {
        self.lexical_query = Some(JsQuery::PhraseQuery(JsPhraseQuery { field, terms }));
    }

    /// Set a 3D ECEF distance (sphere) query.
    ///
    /// # Arguments
    ///
    /// * `field` - The Geo3d field name.
    /// * `x`, `y`, `z` - Sphere centre in ECEF meters.
    /// * `radius_m` - Sphere radius in meters.
    #[napi(js_name = "setLexicalGeo3dDistanceQuery")]
    pub fn set_lexical_geo3d_distance_query(
        &mut self,
        field: String,
        x: f64,
        y: f64,
        z: f64,
        radius_m: f64,
    ) {
        self.lexical_query = Some(JsQuery::Geo3dDistanceQuery(JsGeo3dDistanceQuery {
            field,
            x,
            y,
            z,
            radius_m,
        }));
    }

    /// Set a 3D ECEF axis-aligned bounding-box query.
    ///
    /// # Arguments
    ///
    /// * `field` - The Geo3d field name.
    /// * `min_x`, `min_y`, `min_z` - Lower corner of the box.
    /// * `max_x`, `max_y`, `max_z` - Upper corner of the box.
    #[napi(js_name = "setLexicalGeo3dBoundingBoxQuery")]
    #[allow(clippy::too_many_arguments)]
    pub fn set_lexical_geo3d_bounding_box_query(
        &mut self,
        field: String,
        min_x: f64,
        min_y: f64,
        min_z: f64,
        max_x: f64,
        max_y: f64,
        max_z: f64,
    ) {
        self.lexical_query = Some(JsQuery::Geo3dBoundingBoxQuery(JsGeo3dBoundingBoxQuery {
            field,
            min_x,
            min_y,
            min_z,
            max_x,
            max_y,
            max_z,
        }));
    }

    /// Set a 3D ECEF k-nearest-neighbours query.
    ///
    /// # Arguments
    ///
    /// * `field` - The Geo3d field name.
    /// * `x`, `y`, `z` - Centre coordinates in ECEF meters.
    /// * `k` - Number of nearest neighbours to return.
    /// * `initial_radius_m` - Optional starting radius for the
    ///   expanding-radius search.
    /// * `max_radius_m` - Optional hard cap on the search radius.
    #[napi(js_name = "setLexicalGeo3dNearestQuery")]
    #[allow(clippy::too_many_arguments)]
    pub fn set_lexical_geo3d_nearest_query(
        &mut self,
        field: String,
        x: f64,
        y: f64,
        z: f64,
        k: u32,
        initial_radius_m: Option<f64>,
        max_radius_m: Option<f64>,
    ) {
        self.lexical_query = Some(JsQuery::Geo3dNearestQuery(JsGeo3dNearestQuery {
            field,
            x,
            y,
            z,
            k,
            initial_radius_m,
            max_radius_m,
        }));
    }

    /// Set a vector query with a pre-computed embedding.
    ///
    /// # Arguments
    ///
    /// * `field` - The vector field name.
    /// * `vector` - The embedding vector as an array of numbers.
    #[napi]
    pub fn set_vector_query(&mut self, field: String, vector: Vec<f64>) {
        self.vector_query = Some(JsVectorQuery::VectorQuery(JsVectorQueryInner {
            field,
            vector: vector.into_iter().map(|v| v as f32).collect(),
        }));
    }

    /// Set a vector text query (text will be embedded by the registered embedder).
    ///
    /// # Arguments
    ///
    /// * `field` - The vector field name.
    /// * `text` - The text to embed.
    #[napi]
    pub fn set_vector_text_query(&mut self, field: String, text: String) {
        self.vector_query = Some(JsVectorQuery::VectorTextQuery(JsVectorTextQuery {
            field,
            text,
        }));
    }

    /// Set a filter query (applied after scoring).
    ///
    /// # Arguments
    ///
    /// * `field` - The field name.
    /// * `term` - The term to filter on.
    #[napi]
    pub fn set_filter_query(&mut self, field: String, term: String) {
        self.filter_query = Some(JsQuery::TermQuery(JsTermQuery { field, term }));
    }

    /// Set RRF fusion algorithm.
    ///
    /// # Arguments
    ///
    /// * `k` - The RRF parameter (default 60.0).
    #[napi]
    pub fn set_rrf_fusion(&mut self, k: Option<f64>) {
        self.fusion = Some(FusionChoice::RRF(k.unwrap_or(60.0)));
    }

    /// Set weighted sum fusion algorithm.
    ///
    /// # Arguments
    ///
    /// * `lexical_weight` - Weight for lexical scores (default 0.5).
    /// * `vector_weight` - Weight for vector scores (default 0.5).
    #[napi]
    pub fn set_weighted_sum_fusion(
        &mut self,
        lexical_weight: Option<f64>,
        vector_weight: Option<f64>,
    ) {
        self.fusion = Some(FusionChoice::WeightedSum(
            lexical_weight.unwrap_or(0.5) as f32,
            vector_weight.unwrap_or(0.5) as f32,
        ));
    }
}

impl JsSearchRequest {
    /// Build the Laurus [`laurus::SearchRequest`] from this wrapper.
    pub fn build(&self) -> Result<laurus::SearchRequest> {
        let mut builder = SearchRequestBuilder::new()
            .limit(self.limit)
            .offset(self.offset);

        // Fusion algorithm
        if let Some(fusion) = &self.fusion {
            match fusion {
                FusionChoice::RRF(k) => {
                    builder = builder.fusion_algorithm(FusionAlgorithm::RRF { k: *k });
                }
                FusionChoice::WeightedSum(lw, vw) => {
                    builder = builder.fusion_algorithm(FusionAlgorithm::WeightedSum {
                        lexical_weight: *lw,
                        vector_weight: *vw,
                    });
                }
            }
        }

        // Filter query
        if let Some(fq) = &self.filter_query {
            builder = builder.filter_query(extract_lexical_query(fq)?);
        }

        // Explicit hybrid: lexical_query + vector_query both set
        if let (Some(lq), Some(vq)) = (&self.lexical_query, &self.vector_query) {
            builder = builder
                .lexical_query(query_to_lexical_search_query(lq)?)
                .vector_query(vector_query_to_search_query(vq));
            // Apply default RRF if no fusion specified
            if self.fusion.is_none() {
                builder = builder.fusion_algorithm(FusionAlgorithm::RRF { k: 60.0 });
            }
            return Ok(builder.build());
        }

        // Only lexical_query set
        if let Some(lq) = &self.lexical_query {
            builder = builder.lexical_query(query_to_lexical_search_query(lq)?);
            return Ok(builder.build());
        }

        // Only vector_query set
        if let Some(vq) = &self.vector_query {
            builder = builder.vector_query(vector_query_to_search_query(vq));
            return Ok(builder.build());
        }

        // DSL string
        if let Some(dsl) = &self.query_dsl {
            builder = builder.query_dsl(dsl.clone());
            return Ok(builder.build());
        }

        Ok(builder.build())
    }
}

// ---------------------------------------------------------------------------
// Helper: build a SearchRequest from index.search() arguments
// ---------------------------------------------------------------------------

/// Build a [`laurus::SearchRequest`] from a DSL string with limit/offset.
///
/// # Arguments
///
/// * `dsl` - The query DSL string.
/// * `limit` - Maximum results.
/// * `offset` - Pagination offset.
///
/// # Returns
///
/// A `SearchRequest` configured with the DSL query.
pub fn build_dsl_request(dsl: String, limit: usize, offset: usize) -> laurus::SearchRequest {
    SearchRequestBuilder::new()
        .limit(limit)
        .offset(offset)
        .query_dsl(dsl)
        .build()
}

/// Build a [`laurus::SearchRequest`] from a lexical query.
///
/// # Arguments
///
/// * `query` - The lexical query.
/// * `limit` - Maximum results.
/// * `offset` - Pagination offset.
///
/// # Returns
///
/// A `SearchRequest` configured with the lexical query.
pub fn build_lexical_request(
    query: &JsQuery,
    limit: usize,
    offset: usize,
) -> Result<laurus::SearchRequest> {
    Ok(SearchRequestBuilder::new()
        .limit(limit)
        .offset(offset)
        .lexical_query(LexicalSearchQuery::Obj(extract_lexical_query(query)?))
        .build())
}

/// Build a [`laurus::SearchRequest`] from a vector query.
///
/// # Arguments
///
/// * `query` - The vector query.
/// * `limit` - Maximum results.
/// * `offset` - Pagination offset.
///
/// # Returns
///
/// A `SearchRequest` configured with the vector query.
pub fn build_vector_request(
    query: &JsVectorQuery,
    limit: usize,
    offset: usize,
) -> laurus::SearchRequest {
    SearchRequestBuilder::new()
        .limit(limit)
        .offset(offset)
        .vector_query(vector_query_to_search_query(query))
        .build()
}
