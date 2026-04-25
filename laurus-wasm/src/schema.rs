//! WASM wrapper for the Laurus [`Schema`] type.

use std::collections::HashMap;
use std::str::FromStr;

use laurus::{
    BooleanOption, BytesOption, DateTimeOption, DistanceMetric, DynamicFieldPolicy,
    EmbedderDefinition, FieldOption, FlatOption, FloatOption, GeoOption, HnswOption, IntegerOption,
    IvfOption, Schema, TextOption,
};
use wasm_bindgen::prelude::*;

use crate::embedder::JsCallbackEmbedder;

/// Parse a distance metric string into [`DistanceMetric`].
///
/// # Arguments
///
/// * `s` - Distance metric name: "cosine", "euclidean", "dot_product"/"dot", "manhattan", "angular".
///
/// # Returns
///
/// The corresponding [`DistanceMetric`] variant.
fn parse_distance(s: &str) -> Result<DistanceMetric, JsValue> {
    match s.to_lowercase().as_str() {
        "cosine" => Ok(DistanceMetric::Cosine),
        "euclidean" => Ok(DistanceMetric::Euclidean),
        "dot_product" | "dot" => Ok(DistanceMetric::DotProduct),
        "manhattan" => Ok(DistanceMetric::Manhattan),
        "angular" => Ok(DistanceMetric::Angular),
        other => Err(JsValue::from_str(&format!(
            "Unknown distance metric: '{other}'. Valid: cosine, euclidean, dot_product, manhattan, angular"
        ))),
    }
}

/// Schema builder for defining index fields and embedders.
///
/// ```javascript
/// import { WasmSchema } from "laurus-wasm";
///
/// const schema = new WasmSchema();
/// schema.addTextField("title");
/// schema.addHnswField("embedding", 384);
/// schema.setDefaultFields(["title"]);
/// ```
#[wasm_bindgen(js_name = "Schema")]
pub struct WasmSchema {
    pub(crate) inner: Schema,
    /// JS callback embedders registered via `addEmbedder({ type: "callback" })`.
    /// Stored separately because they can't be serialized into `EmbedderDefinition`.
    pub(crate) js_embedders: HashMap<String, JsCallbackEmbedder>,
}

#[wasm_bindgen(js_class = "Schema")]
impl WasmSchema {
    /// Create a new empty schema.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: Schema::new(),
            js_embedders: HashMap::new(),
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
    #[wasm_bindgen(js_name = "addTextField")]
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
    #[wasm_bindgen(js_name = "addIntegerField")]
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
    #[wasm_bindgen(js_name = "addFloatField")]
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
    #[wasm_bindgen(js_name = "addBooleanField")]
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
    #[wasm_bindgen(js_name = "addDateTimeField")]
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
    #[wasm_bindgen(js_name = "addGeoField")]
    pub fn add_geo_field(&mut self, name: String, stored: Option<bool>, indexed: Option<bool>) {
        self.inner.fields.insert(
            name,
            FieldOption::Geo(GeoOption {
                indexed: indexed.unwrap_or(true),
                stored: stored.unwrap_or(true),
            }),
        );
    }

    /// Add a binary data field.
    #[wasm_bindgen(js_name = "addBytesField")]
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
    /// * `distance` - Distance metric (default "cosine").
    /// * `m` - HNSW branching factor (default 16).
    /// * `ef_construction` - Build-time expansion factor (default 200).
    /// * `embedder` - Optional embedder name registered via `addEmbedder`.
    #[wasm_bindgen(js_name = "addHnswField")]
    pub fn add_hnsw_field(
        &mut self,
        name: String,
        dimension: u32,
        distance: Option<String>,
        m: Option<u32>,
        ef_construction: Option<u32>,
        embedder: Option<String>,
    ) -> Result<(), JsValue> {
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
    #[wasm_bindgen(js_name = "addFlatField")]
    pub fn add_flat_field(
        &mut self,
        name: String,
        dimension: u32,
        distance: Option<String>,
        embedder: Option<String>,
    ) -> Result<(), JsValue> {
        let opt = FlatOption {
            dimension: dimension as usize,
            distance: parse_distance(distance.as_deref().unwrap_or("cosine"))?,
            embedder,
            ..Default::default()
        };
        self.inner.fields.insert(name, FieldOption::Flat(opt));
        Ok(())
    }

    /// Add an IVF approximate nearest-neighbor vector field.
    #[wasm_bindgen(js_name = "addIvfField")]
    pub fn add_ivf_field(
        &mut self,
        name: String,
        dimension: u32,
        distance: Option<String>,
        n_clusters: Option<u32>,
        n_probe: Option<u32>,
        embedder: Option<String>,
    ) -> Result<(), JsValue> {
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
    /// The `config` object must have a `type` key:
    ///
    /// | type           | extra keys | description                             |
    /// |----------------|------------|-----------------------------------------|
    /// | `"precomputed"` | —         | No embedding; vectors passed directly   |
    /// | `"callback"`   | `embed`    | JS function `(text) => Promise<number[]>` |
    ///
    /// ## Example — callback embedder with Transformers.js
    ///
    /// ```javascript
    /// const model = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    /// schema.addEmbedder("my-bert", {
    ///   type: "callback",
    ///   embed: async (text) => {
    ///     const out = await model(text, { pooling: 'mean', normalize: true });
    ///     return Array.from(out.data);
    ///   }
    /// });
    /// ```
    #[wasm_bindgen(js_name = "addEmbedder")]
    pub fn add_embedder(&mut self, name: String, config: JsValue) -> Result<(), JsValue> {
        // Read the "type" field from the config object
        let type_key = js_sys::Reflect::get(&config, &JsValue::from_str("type"))
            .map_err(|_| JsValue::from_str("Embedder config must have a 'type' key"))?;
        let embedder_type = type_key
            .as_string()
            .ok_or_else(|| JsValue::from_str("Embedder 'type' must be a string"))?;

        match embedder_type.as_str() {
            "precomputed" => {
                self.inner
                    .embedders
                    .insert(name, EmbedderDefinition::Precomputed);
            }
            "callback" => {
                // Extract the "embed" function from the config
                let embed_fn =
                    js_sys::Reflect::get(&config, &JsValue::from_str("embed")).map_err(|_| {
                        JsValue::from_str("Callback embedder config must have an 'embed' key")
                    })?;
                let func = embed_fn.dyn_into::<js_sys::Function>().map_err(|_| {
                    JsValue::from_str(
                        "'embed' must be a function: (text: string) => Promise<number[]>",
                    )
                })?;

                // Register as precomputed in the schema (so the engine creates the field)
                // and store the actual JS embedder separately.
                self.inner
                    .embedders
                    .insert(name.clone(), EmbedderDefinition::Precomputed);
                self.js_embedders
                    .insert(name.clone(), JsCallbackEmbedder::new(name, func));
            }
            other => {
                return Err(JsValue::from_str(&format!(
                    "Unsupported embedder type: '{other}'. Valid: 'precomputed', 'callback'"
                )));
            }
        }

        Ok(())
    }

    /// Set the default fields used when no field is specified in a query.
    #[wasm_bindgen(js_name = "setDefaultFields")]
    pub fn set_default_fields(&mut self, fields: Vec<String>) {
        self.inner.default_fields = fields;
    }

    /// Set the policy for fields that are not declared in this schema.
    ///
    /// Accepted values (case-insensitive): `"strict"`, `"dynamic"`,
    /// `"ignore"`. Behaviour:
    ///
    /// - `"strict"`: reject documents containing undeclared fields.
    /// - `"dynamic"` (default): infer a type for each undeclared field and
    ///   add it to the schema during ingestion. **Warning**: integer fields
    ///   silently truncate incoming float values (e.g. `3.14` → `3`).
    /// - `"ignore"`: silently drop undeclared fields.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error string if `policy` is not one of the
    /// accepted names.
    #[wasm_bindgen(js_name = "setDynamicFieldPolicy")]
    pub fn set_dynamic_field_policy(&mut self, policy: String) -> Result<(), JsValue> {
        let parsed =
            DynamicFieldPolicy::from_str(&policy).map_err(|e| JsValue::from_str(&e.to_string()))?;
        self.inner.dynamic_field_policy = parsed;
        Ok(())
    }

    /// Return the currently configured dynamic field policy as a lowercase
    /// string (`"strict"` / `"dynamic"` / `"ignore"`).
    #[wasm_bindgen(js_name = "dynamicFieldPolicy")]
    pub fn dynamic_field_policy(&self) -> String {
        match self.inner.dynamic_field_policy {
            DynamicFieldPolicy::Strict => "strict".into(),
            DynamicFieldPolicy::Dynamic => "dynamic".into(),
            DynamicFieldPolicy::Ignore => "ignore".into(),
        }
    }

    /// Return the list of field names defined in this schema.
    #[wasm_bindgen(js_name = "fieldNames")]
    pub fn field_names(&self) -> Vec<String> {
        self.inner.fields.keys().cloned().collect()
    }
}
