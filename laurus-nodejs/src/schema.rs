//! Node.js wrapper for the Laurus [`Schema`] type.

use std::str::FromStr;

use laurus::{
    BooleanOption, BytesOption, DateTimeOption, DistanceMetric, DynamicFieldPolicy,
    EmbedderDefinition, FieldOption, FlatOption, FloatOption, Geo3dOption, GeoOption, HnswOption,
    IntegerOption, IvfOption, Schema, TextOption,
};
use napi::bindgen_prelude::*;
use napi_derive::napi;

/// Parse a distance metric string into [`DistanceMetric`].
///
/// # Arguments
///
/// * `s` - Distance metric name: "cosine", "euclidean", "dot_product"/"dot", "manhattan", "angular".
///
/// # Returns
///
/// The corresponding [`DistanceMetric`] variant.
fn parse_distance(s: &str) -> Result<DistanceMetric> {
    match s.to_lowercase().as_str() {
        "cosine" => Ok(DistanceMetric::Cosine),
        "euclidean" => Ok(DistanceMetric::Euclidean),
        "dot_product" | "dot" => Ok(DistanceMetric::DotProduct),
        "manhattan" => Ok(DistanceMetric::Manhattan),
        "angular" => Ok(DistanceMetric::Angular),
        other => Err(napi::Error::from_reason(format!(
            "Unknown distance metric: '{other}'. Valid: cosine, euclidean, dot_product, manhattan, angular"
        ))),
    }
}

/// Schema builder for defining index fields and embedders.
///
/// ## Example
///
/// ```javascript
/// const { Schema } = require("laurus-nodejs");
///
/// const schema = new Schema();
/// schema.addTextField("title");
/// schema.addHnswField("embedding", 384, { distance: "cosine" });
/// schema.addIntegerField("year");
/// schema.setDefaultFields(["title"]);
/// ```
#[napi(js_name = "Schema")]
pub struct JsSchema {
    pub(crate) inner: Schema,
}

#[napi]
impl JsSchema {
    /// Create a new empty schema.
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {
            inner: Schema::new(),
        }
    }

    /// Add a full-text searchable text field.
    ///
    /// # Arguments
    ///
    /// * `name` - Field name.
    /// * `stored` - Whether the original value is retrievable (default `true`).
    /// * `indexed` - Whether the field is searchable (default `true`).
    /// * `term_vectors` - Whether term position information is stored (default `false`).
    /// * `analyzer` - Optional named analyzer to use.
    #[napi]
    pub fn add_text_field(
        &mut self,
        name: String,
        stored: Option<bool>,
        indexed: Option<bool>,
        term_vectors: Option<bool>,
        analyzer: Option<String>,
    ) {
        self.inner.fields.insert(
            name,
            FieldOption::Text(TextOption {
                indexed: indexed.unwrap_or(true),
                stored: stored.unwrap_or(true),
                term_vectors: term_vectors.unwrap_or(false),
                analyzer,
            }),
        );
    }

    /// Add an integer (i64) field.
    ///
    /// # Arguments
    ///
    /// * `name` - Field name.
    /// * `stored` - Whether the value is retrievable (default `true`).
    /// * `indexed` - Whether the field is searchable (default `true`).
    /// * `multi_valued` - When `true`, the field accepts arrays of integers
    ///   and range queries match if any value satisfies the predicate
    ///   (Lucene-style "any match"). Default `false`.
    #[napi]
    pub fn add_integer_field(
        &mut self,
        name: String,
        stored: Option<bool>,
        indexed: Option<bool>,
        multi_valued: Option<bool>,
    ) {
        self.inner.fields.insert(
            name,
            FieldOption::Integer(IntegerOption {
                indexed: indexed.unwrap_or(true),
                stored: stored.unwrap_or(true),
                multi_valued: multi_valued.unwrap_or(false),
            }),
        );
    }

    /// Add a float (f64) field.
    ///
    /// # Arguments
    ///
    /// * `name` - Field name.
    /// * `stored` - Whether the value is retrievable (default `true`).
    /// * `indexed` - Whether the field is searchable (default `true`).
    /// * `multi_valued` - When `true`, the field accepts arrays of floats
    ///   and range queries match if any value satisfies the predicate
    ///   (Lucene-style "any match"). Default `false`.
    #[napi]
    pub fn add_float_field(
        &mut self,
        name: String,
        stored: Option<bool>,
        indexed: Option<bool>,
        multi_valued: Option<bool>,
    ) {
        self.inner.fields.insert(
            name,
            FieldOption::Float(FloatOption {
                indexed: indexed.unwrap_or(true),
                stored: stored.unwrap_or(true),
                multi_valued: multi_valued.unwrap_or(false),
            }),
        );
    }

    /// Add a boolean field.
    ///
    /// # Arguments
    ///
    /// * `name` - Field name.
    /// * `stored` - Whether the value is retrievable (default `true`).
    /// * `indexed` - Whether the field is searchable (default `true`).
    #[napi]
    pub fn add_boolean_field(&mut self, name: String, stored: Option<bool>, indexed: Option<bool>) {
        self.inner.fields.insert(
            name,
            FieldOption::Boolean(BooleanOption {
                indexed: indexed.unwrap_or(true),
                stored: stored.unwrap_or(true),
            }),
        );
    }

    /// Add a date/time field.
    ///
    /// # Arguments
    ///
    /// * `name` - Field name.
    /// * `stored` - Whether the value is retrievable (default `true`).
    /// * `indexed` - Whether the field is searchable (default `true`).
    #[napi]
    pub fn add_datetime_field(
        &mut self,
        name: String,
        stored: Option<bool>,
        indexed: Option<bool>,
    ) {
        self.inner.fields.insert(
            name,
            FieldOption::DateTime(DateTimeOption {
                indexed: indexed.unwrap_or(true),
                stored: stored.unwrap_or(true),
            }),
        );
    }

    /// Add a geographic coordinate field (latitude, longitude).
    ///
    /// # Arguments
    ///
    /// * `name` - Field name.
    /// * `stored` - Whether the value is retrievable (default `true`).
    /// * `indexed` - Whether the field is searchable (default `true`).
    #[napi]
    pub fn add_geo_field(&mut self, name: String, stored: Option<bool>, indexed: Option<bool>) {
        self.inner.fields.insert(
            name,
            FieldOption::Geo(GeoOption {
                indexed: indexed.unwrap_or(true),
                stored: stored.unwrap_or(true),
            }),
        );
    }

    /// Add a 3D ECEF Cartesian point field (x, y, z in meters).
    ///
    /// Values are submitted as a `{ x, y, z }` JSON object and are queryable
    /// via `Geo3dDistanceQuery`, `Geo3dBoundingBoxQuery`, and
    /// `Geo3dNearestQuery` (passed through `SearchRequest.setLexicalGeo3d*`
    /// setters). See the conceptual docs at `docs/src/concepts/geo3d.md`.
    ///
    /// # Arguments
    ///
    /// * `name` - Field name.
    /// * `stored` - Whether the value is retrievable (default `true`).
    /// * `indexed` - Whether the field is searchable (default `true`).
    #[napi(js_name = "addGeo3dField")]
    pub fn add_geo3d_field(&mut self, name: String, stored: Option<bool>, indexed: Option<bool>) {
        self.inner.fields.insert(
            name,
            FieldOption::Geo3d(Geo3dOption {
                indexed: indexed.unwrap_or(true),
                stored: stored.unwrap_or(true),
            }),
        );
    }

    /// Add a binary data field.
    ///
    /// # Arguments
    ///
    /// * `name` - Field name.
    /// * `stored` - Whether the value is retrievable (default `true`).
    #[napi]
    pub fn add_bytes_field(&mut self, name: String, stored: Option<bool>) {
        self.inner.fields.insert(
            name,
            FieldOption::Bytes(BytesOption {
                stored: stored.unwrap_or(true),
            }),
        );
    }

    /// Add an HNSW approximate nearest-neighbor vector index field.
    ///
    /// # Arguments
    ///
    /// * `name` - Field name.
    /// * `dimension` - Vector dimensionality.
    /// * `distance` - Distance metric — "cosine" (default), "euclidean", "dot_product".
    /// * `m` - HNSW branching factor (default 16).
    /// * `ef_construction` - Build-time expansion factor (default 200).
    /// * `embedder` - Optional embedder name registered via `addEmbedder`.
    #[napi]
    pub fn add_hnsw_field(
        &mut self,
        name: String,
        dimension: u32,
        distance: Option<String>,
        m: Option<u32>,
        ef_construction: Option<u32>,
        embedder: Option<String>,
    ) -> Result<()> {
        let opt = HnswOption {
            dimension: dimension as usize,
            distance: parse_distance(distance.as_deref().unwrap_or("cosine"))?,
            m: m.unwrap_or(16) as usize,
            ef_construction: ef_construction.unwrap_or(200) as usize,
            embedder,
            ..Default::default()
        };
        self.inner.fields.insert(name, FieldOption::Hnsw(opt));
        Ok(())
    }

    /// Add a flat (brute-force) vector index field.
    ///
    /// # Arguments
    ///
    /// * `name` - Field name.
    /// * `dimension` - Vector dimensionality.
    /// * `distance` - Distance metric — "cosine" (default), "euclidean", "dot_product".
    /// * `embedder` - Optional embedder name registered via `addEmbedder`.
    #[napi]
    pub fn add_flat_field(
        &mut self,
        name: String,
        dimension: u32,
        distance: Option<String>,
        embedder: Option<String>,
    ) -> Result<()> {
        let opt = FlatOption {
            dimension: dimension as usize,
            distance: parse_distance(distance.as_deref().unwrap_or("cosine"))?,
            embedder,
            ..Default::default()
        };
        self.inner.fields.insert(name, FieldOption::Flat(opt));
        Ok(())
    }

    /// Add an IVF (Inverted File Index) approximate nearest-neighbor vector field.
    ///
    /// # Arguments
    ///
    /// * `name` - Field name.
    /// * `dimension` - Vector dimensionality.
    /// * `distance` - Distance metric — "cosine" (default), "euclidean", "dot_product".
    /// * `n_clusters` - Number of Voronoi clusters (default 100).
    /// * `n_probe` - Number of clusters to probe at search time (default 1).
    /// * `embedder` - Optional embedder name registered via `addEmbedder`.
    #[napi]
    pub fn add_ivf_field(
        &mut self,
        name: String,
        dimension: u32,
        distance: Option<String>,
        n_clusters: Option<u32>,
        n_probe: Option<u32>,
        embedder: Option<String>,
    ) -> Result<()> {
        let opt = IvfOption {
            dimension: dimension as usize,
            distance: parse_distance(distance.as_deref().unwrap_or("cosine"))?,
            n_clusters: n_clusters.unwrap_or(100) as usize,
            n_probe: n_probe.unwrap_or(1) as usize,
            embedder,
            ..Default::default()
        };
        self.inner.fields.insert(name, FieldOption::Ivf(opt));
        Ok(())
    }

    /// Register a named embedder definition in the schema.
    ///
    /// The embedder can then be referenced by name from vector field options
    /// (e.g. `addHnswField("embedding", 384, { embedder: "my-bert" })`).
    ///
    /// The `config` object must have a `type` key selecting the backend:
    ///
    /// | type            | required keys | feature flag            |
    /// |-----------------|---------------|-------------------------|
    /// | "precomputed"   | —             | (always available)      |
    /// | "candle_bert"   | "model"       | `embeddings-candle`     |
    /// | "candle_clip"   | "model"       | `embeddings-multimodal` |
    /// | "openai"        | "model"       | `embeddings-openai`     |
    ///
    /// # Arguments
    ///
    /// * `name` - Unique embedder name referenced from vector fields.
    /// * `config` - Object describing the embedder, e.g.
    ///     `{ type: "candle_bert", model: "sentence-transformers/all-MiniLM-L6-v2" }`.
    #[napi]
    pub fn add_embedder(&mut self, name: String, config: serde_json::Value) -> Result<()> {
        let obj = config
            .as_object()
            .ok_or_else(|| napi::Error::from_reason("embedder config must be an object"))?;

        let embedder_type = obj
            .get("type")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                napi::Error::from_reason(
                    "embedder config must have a 'type' key (e.g. \"candle_bert\")",
                )
            })?
            .to_string();

        let get_model = |key: &str| -> Result<String> {
            obj.get("model")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .ok_or_else(|| {
                    napi::Error::from_reason(format!("{key} embedder requires a 'model' key"))
                })
        };

        let definition = match embedder_type.as_str() {
            "precomputed" => EmbedderDefinition::Precomputed,
            "candle_bert" => EmbedderDefinition::CandleBert {
                model: get_model("candle_bert")?,
            },
            "candle_clip" => EmbedderDefinition::CandleClip {
                model: get_model("candle_clip")?,
            },
            "openai" => EmbedderDefinition::Openai {
                model: get_model("openai")?,
            },
            other => {
                return Err(napi::Error::from_reason(format!(
                    "Unknown embedder type: '{other}'. Valid types: precomputed, candle_bert, candle_clip, openai"
                )));
            }
        };

        self.inner.embedders.insert(name, definition);
        Ok(())
    }

    /// Set the default fields used when no field is specified in a query.
    ///
    /// # Arguments
    ///
    /// * `fields` - List of field names.
    #[napi]
    pub fn set_default_fields(&mut self, fields: Vec<String>) {
        self.inner.default_fields = fields;
    }

    /// Set the policy for fields that are not declared in this schema.
    ///
    /// Behaviour:
    /// - `"strict"`: reject documents containing undeclared fields.
    /// - `"dynamic"` (default): infer a type for each undeclared field and
    ///   add it to the schema during ingestion. **Warning**: integer fields
    ///   silently truncate incoming float values (e.g. `3.14` → `3`).
    /// - `"ignore"`: silently drop undeclared fields.
    ///
    /// # Arguments
    ///
    /// * `policy` - One of `"strict"`, `"dynamic"`, `"ignore"`
    ///   (case-insensitive).
    ///
    /// # Errors
    ///
    /// Throws a JavaScript `Error` if `policy` is not one of the accepted names.
    #[napi]
    pub fn set_dynamic_field_policy(&mut self, policy: String) -> Result<()> {
        let parsed =
            DynamicFieldPolicy::from_str(&policy).map_err(|e| Error::from_reason(e.to_string()))?;
        self.inner.dynamic_field_policy = parsed;
        Ok(())
    }

    /// Return the currently configured dynamic field policy as a lowercase
    /// string (`"strict"` / `"dynamic"` / `"ignore"`).
    #[napi]
    pub fn dynamic_field_policy(&self) -> String {
        match self.inner.dynamic_field_policy {
            DynamicFieldPolicy::Strict => "strict".to_string(),
            DynamicFieldPolicy::Dynamic => "dynamic".to_string(),
            DynamicFieldPolicy::Ignore => "ignore".to_string(),
        }
    }

    /// Return the list of field names defined in this schema.
    ///
    /// # Returns
    ///
    /// An array of field name strings.
    #[napi]
    pub fn field_names(&self) -> Vec<String> {
        self.inner.fields.keys().cloned().collect()
    }

    /// Return a string representation of this schema, listing the declared
    /// field names. Convenient for `console.log` / `String(schema)`.
    #[napi(js_name = "toString")]
    pub fn to_string_repr(&self) -> String {
        format!(
            "Schema(fields={:?})",
            self.inner.fields.keys().collect::<Vec<_>>()
        )
    }
}
