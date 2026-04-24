//! PHP wrapper for the Laurus [`Schema`] type.

use std::cell::RefCell;
use std::str::FromStr;

use ext_php_rs::convert::FromZval;
use ext_php_rs::prelude::*;
use ext_php_rs::types::ZendHashTable;
use laurus::{
    BooleanOption, BytesOption, DateTimeOption, DistanceMetric, DynamicFieldPolicy,
    EmbedderDefinition, FieldOption, FloatOption, GeoOption, HnswOption, IntegerOption, IvfOption,
    Schema, TextOption,
};

/// Parse a distance metric string into [`DistanceMetric`].
///
/// # Arguments
///
/// * `s` - Distance metric name (e.g. "cosine", "euclidean", "dot_product").
///
/// # Returns
///
/// The corresponding `DistanceMetric`.
fn parse_distance(s: &str) -> PhpResult<DistanceMetric> {
    match s.to_lowercase().as_str() {
        "cosine" => Ok(DistanceMetric::Cosine),
        "euclidean" => Ok(DistanceMetric::Euclidean),
        "dot_product" | "dot" => Ok(DistanceMetric::DotProduct),
        "manhattan" => Ok(DistanceMetric::Manhattan),
        "angular" => Ok(DistanceMetric::Angular),
        other => Err(format!(
            "Unknown distance metric: '{}'. Valid: cosine, euclidean, dot_product, manhattan, angular",
            other
        )
        .into()),
    }
}

/// Helper to extract a string from a [`ZendHashTable`] by key.
fn ht_get_string(ht: &ZendHashTable, key: &str) -> PhpResult<String> {
    let zv = ht.get(key).ok_or(format!("missing key '{key}'"))?;
    String::from_zval(zv).ok_or_else(|| format!("'{key}' must be a string").into())
}

/// PHP-facing schema builder (`Laurus\Schema`).
///
/// Uses `RefCell` for interior mutability since ext-php-rs methods receive `&self`.
#[php_class]
#[php(name = "Laurus\\Schema")]
pub struct PhpSchema {
    pub inner: RefCell<Schema>,
}

#[php_impl]
impl PhpSchema {
    /// Create a new empty schema.
    pub fn __construct() -> Self {
        Self {
            inner: RefCell::new(Schema::new()),
        }
    }

    /// Add a full-text searchable text field.
    ///
    /// # Arguments
    ///
    /// * `name` - Field name.
    /// * `stored` - Whether the original value is retrievable (default: true).
    /// * `indexed` - Whether the field is searchable (default: true).
    /// * `term_vectors` - Whether term position information is stored (default: false).
    /// * `analyzer` - Named analyzer to use (optional).
    #[php(defaults(stored = true, indexed = true, term_vectors = false))]
    pub fn add_text_field(
        &self,
        name: String,
        stored: bool,
        indexed: bool,
        term_vectors: bool,
        analyzer: Option<String>,
    ) {
        self.inner.borrow_mut().fields.insert(
            name,
            FieldOption::Text(TextOption {
                indexed,
                stored,
                term_vectors,
                analyzer,
            }),
        );
    }

    /// Add an integer (i64) field.
    ///
    /// # Arguments
    ///
    /// * `name` - Field name.
    /// * `stored` - Whether the value is retrievable (default: true).
    /// * `indexed` - Whether the field is searchable (default: true).
    #[php(defaults(stored = true, indexed = true))]
    pub fn add_integer_field(&self, name: String, stored: bool, indexed: bool) {
        self.inner.borrow_mut().fields.insert(
            name,
            FieldOption::Integer(IntegerOption { indexed, stored }),
        );
    }

    /// Add a float (f64) field.
    ///
    /// # Arguments
    ///
    /// * `name` - Field name.
    /// * `stored` - Whether the value is retrievable (default: true).
    /// * `indexed` - Whether the field is searchable (default: true).
    #[php(defaults(stored = true, indexed = true))]
    pub fn add_float_field(&self, name: String, stored: bool, indexed: bool) {
        self.inner
            .borrow_mut()
            .fields
            .insert(name, FieldOption::Float(FloatOption { indexed, stored }));
    }

    /// Add a boolean field.
    ///
    /// # Arguments
    ///
    /// * `name` - Field name.
    /// * `stored` - Whether the value is retrievable (default: true).
    /// * `indexed` - Whether the field is searchable (default: true).
    #[php(defaults(stored = true, indexed = true))]
    pub fn add_boolean_field(&self, name: String, stored: bool, indexed: bool) {
        self.inner.borrow_mut().fields.insert(
            name,
            FieldOption::Boolean(BooleanOption { indexed, stored }),
        );
    }

    /// Add a date/time field.
    ///
    /// # Arguments
    ///
    /// * `name` - Field name.
    /// * `stored` - Whether the value is retrievable (default: true).
    /// * `indexed` - Whether the field is searchable (default: true).
    #[php(defaults(stored = true, indexed = true))]
    pub fn add_datetime_field(&self, name: String, stored: bool, indexed: bool) {
        self.inner.borrow_mut().fields.insert(
            name,
            FieldOption::DateTime(DateTimeOption { indexed, stored }),
        );
    }

    /// Add a geographic coordinate field (latitude, longitude).
    ///
    /// # Arguments
    ///
    /// * `name` - Field name.
    /// * `stored` - Whether the value is retrievable (default: true).
    /// * `indexed` - Whether the field is searchable (default: true).
    #[php(defaults(stored = true, indexed = true))]
    pub fn add_geo_field(&self, name: String, stored: bool, indexed: bool) {
        self.inner
            .borrow_mut()
            .fields
            .insert(name, FieldOption::Geo(GeoOption { indexed, stored }));
    }

    /// Add a binary data field.
    ///
    /// # Arguments
    ///
    /// * `name` - Field name.
    /// * `stored` - Whether the value is retrievable (default: true).
    #[php(defaults(stored = true))]
    pub fn add_bytes_field(&self, name: String, stored: bool) {
        self.inner
            .borrow_mut()
            .fields
            .insert(name, FieldOption::Bytes(BytesOption { stored }));
    }

    /// Add an HNSW approximate nearest-neighbor vector index field.
    ///
    /// # Arguments
    ///
    /// * `name` - Field name.
    /// * `dimension` - Vector dimensionality.
    /// * `distance` - Distance metric (default: "cosine").
    /// * `m` - HNSW branching factor (default: 16).
    /// * `ef_construction` - Build-time expansion factor (default: 200).
    /// * `embedder` - Embedder name registered via `addEmbedder` (default: "" for none).
    #[php(defaults(m = 16, ef_construction = 200))]
    pub fn add_hnsw_field(
        &self,
        name: String,
        dimension: i64,
        distance: Option<String>,
        m: i64,
        ef_construction: i64,
        embedder: Option<String>,
    ) -> PhpResult<()> {
        let dist_str = distance.unwrap_or_else(|| "cosine".to_string());
        let opt = HnswOption {
            dimension: dimension as usize,
            distance: parse_distance(&dist_str)?,
            m: m as usize,
            ef_construction: ef_construction as usize,
            embedder,
            ..Default::default()
        };
        self.inner
            .borrow_mut()
            .fields
            .insert(name, FieldOption::Hnsw(opt));
        Ok(())
    }

    /// Add a flat (brute-force) vector index field.
    ///
    /// # Arguments
    ///
    /// * `name` - Field name.
    /// * `dimension` - Vector dimensionality.
    /// * `distance` - Distance metric (default: "cosine").
    /// * `embedder` - Embedder name registered via `addEmbedder` (default: "" for none).
    pub fn add_flat_field(
        &self,
        name: String,
        dimension: i64,
        distance: Option<String>,
        embedder: Option<String>,
    ) -> PhpResult<()> {
        let dist_str = distance.unwrap_or_else(|| "cosine".to_string());
        let opt = laurus::FlatOption {
            dimension: dimension as usize,
            distance: parse_distance(&dist_str)?,
            embedder,
            ..Default::default()
        };
        self.inner
            .borrow_mut()
            .fields
            .insert(name, FieldOption::Flat(opt));
        Ok(())
    }

    /// Add an IVF (Inverted File Index) approximate nearest-neighbor vector field.
    ///
    /// # Arguments
    ///
    /// * `name` - Field name.
    /// * `dimension` - Vector dimensionality.
    /// * `distance` - Distance metric (default: "cosine").
    /// * `n_clusters` - Number of Voronoi clusters (default: 100).
    /// * `n_probe` - Number of clusters to probe at search time (default: 1).
    /// * `embedder` - Embedder name registered via `addEmbedder` (default: "" for none).
    #[php(defaults(n_clusters = 100, n_probe = 1))]
    pub fn add_ivf_field(
        &self,
        name: String,
        dimension: i64,
        distance: Option<String>,
        n_clusters: i64,
        n_probe: i64,
        embedder: Option<String>,
    ) -> PhpResult<()> {
        let dist_str = distance.unwrap_or_else(|| "cosine".to_string());
        let opt = IvfOption {
            dimension: dimension as usize,
            distance: parse_distance(&dist_str)?,
            n_clusters: n_clusters as usize,
            n_probe: n_probe as usize,
            embedder,
            ..Default::default()
        };
        self.inner
            .borrow_mut()
            .fields
            .insert(name, FieldOption::Ivf(opt));
        Ok(())
    }

    /// Register a named embedder definition in the schema.
    ///
    /// The `config` array must have a `"type"` key selecting the backend:
    ///
    /// | type              | required keys | feature flag            |
    /// |-------------------|---------------|-------------------------|
    /// | `"precomputed"`   | —             | (always available)      |
    /// | `"candle_bert"`   | `"model"`     | `embeddings-candle`     |
    /// | `"candle_clip"`   | `"model"`     | `embeddings-multimodal` |
    /// | `"openai"`        | `"model"`     | `embeddings-openai`     |
    ///
    /// # Arguments
    ///
    /// * `name` - Unique embedder name referenced from vector fields.
    /// * `config` - Associative array describing the embedder.
    pub fn add_embedder(&self, name: String, config: &ZendHashTable) -> PhpResult<()> {
        let embedder_type = ht_get_string(config, "type")?;

        let definition = match embedder_type.as_str() {
            "precomputed" => EmbedderDefinition::Precomputed,
            "candle_bert" => {
                let model = ht_get_string(config, "model")?;
                EmbedderDefinition::CandleBert { model }
            }
            "candle_clip" => {
                let model = ht_get_string(config, "model")?;
                EmbedderDefinition::CandleClip { model }
            }
            "openai" => {
                let model = ht_get_string(config, "model")?;
                EmbedderDefinition::Openai { model }
            }
            other => {
                return Err(format!(
                    "Unknown embedder type: '{}'. Valid types: precomputed, candle_bert, candle_clip, openai",
                    other
                )
                .into());
            }
        };

        self.inner.borrow_mut().embedders.insert(name, definition);
        Ok(())
    }

    /// Set the default fields used when no field is specified in a query.
    ///
    /// # Arguments
    ///
    /// * `field_names` - Array of field name strings.
    pub fn set_default_fields(&self, field_names: Vec<String>) {
        self.inner.borrow_mut().default_fields = field_names;
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
    /// # Arguments
    ///
    /// * `policy` - One of `"strict"`, `"dynamic"`, `"ignore"`.
    ///
    /// # Errors
    ///
    /// Throws a PHP `Exception` if `policy` is not one of the accepted names.
    pub fn set_dynamic_field_policy(&self, policy: String) -> PhpResult<()> {
        let parsed = DynamicFieldPolicy::from_str(&policy)
            .map_err(|e| PhpException::default(e.to_string()))?;
        self.inner.borrow_mut().dynamic_field_policy = parsed;
        Ok(())
    }

    /// Return the currently configured dynamic field policy as a lowercase
    /// string (`"strict"` / `"dynamic"` / `"ignore"`).
    pub fn dynamic_field_policy(&self) -> String {
        match self.inner.borrow().dynamic_field_policy {
            DynamicFieldPolicy::Strict => "strict".to_string(),
            DynamicFieldPolicy::Dynamic => "dynamic".to_string(),
            DynamicFieldPolicy::Ignore => "ignore".to_string(),
        }
    }

    /// Return the list of field names defined in this schema.
    pub fn field_names(&self) -> Vec<String> {
        self.inner.borrow().fields.keys().cloned().collect()
    }

    /// Return a string representation of this schema.
    pub fn __to_string(&self) -> String {
        format!(
            "Schema(fields={:?})",
            self.inner.borrow().fields.keys().collect::<Vec<_>>()
        )
    }
}
