//! Python wrapper for the Laurus [`Schema`] type.

use std::str::FromStr;

use laurus::{
    AnalyzerSpec, BooleanOption, BuiltinAnalyzerSpec, BytesOption, DateTimeOption, DistanceMetric,
    DynamicFieldPolicy, EmbedderDefinition, FieldOption, FlatOption, FloatOption, Geo3dOption,
    GeoOption, HnswOption, IntegerOption, IvfOption, QuantizationMethod, RerankStorageKind, Schema,
    TextOption,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Convert a Python analyzer reference into an [`AnalyzerSpec`].
///
/// Accepts either a `str` (resolved as [`AnalyzerSpec::Named`]) or a
/// `dict` describing a parameterized built-in preset. The dict must
/// include a `"language"` key naming the preset (currently only
/// `"japanese"`) plus its required parameters (`"dict"` for Japanese,
/// optionally `"mode"` and `"user_dict"`).
fn analyzer_spec_from_py(py: Python<'_>, obj: Py<PyAny>) -> PyResult<AnalyzerSpec> {
    if let Ok(name) = obj.extract::<String>(py) {
        return Ok(AnalyzerSpec::Named(name));
    }
    let bound = obj.bind(py);
    if let Ok(dict) = bound.cast::<PyDict>() {
        let language = dict
            .get_item("language")?
            .ok_or_else(|| PyValueError::new_err("analyzer dict requires a 'language' key"))?
            .extract::<String>()?;
        match language.as_str() {
            "japanese" => {
                let dict_path = dict
                    .get_item("dict")?
                    .ok_or_else(|| {
                        PyValueError::new_err("japanese analyzer requires a 'dict' path")
                    })?
                    .extract::<String>()?;
                let mode = dict
                    .get_item("mode")?
                    .map(|v| v.extract::<String>())
                    .transpose()?
                    .unwrap_or_else(|| "normal".to_string());
                let user_dict = dict
                    .get_item("user_dict")?
                    .map(|v| v.extract::<String>())
                    .transpose()?;
                return Ok(AnalyzerSpec::Builtin(BuiltinAnalyzerSpec::Japanese {
                    mode,
                    dict: dict_path,
                    user_dict,
                }));
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "unsupported analyzer language: {other}"
                )));
            }
        }
    }
    Err(PyValueError::new_err(
        "analyzer must be a str or a dict (e.g. {'language': 'japanese', 'dict': '/path'})",
    ))
}

/// Parse a distance metric string into [`DistanceMetric`].
fn parse_distance(s: &str) -> PyResult<DistanceMetric> {
    match s.to_lowercase().as_str() {
        "cosine" => Ok(DistanceMetric::Cosine),
        "euclidean" => Ok(DistanceMetric::Euclidean),
        "dot_product" | "dot" => Ok(DistanceMetric::DotProduct),
        "manhattan" => Ok(DistanceMetric::Manhattan),
        "angular" => Ok(DistanceMetric::Angular),
        other => Err(PyValueError::new_err(format!(
            "Unknown distance metric: '{}'. Valid: cosine, euclidean, dot_product, manhattan, angular",
            other
        ))),
    }
}

/// Parse a quantizer name plus optional `subvector_count` into a
/// [`QuantizationMethod`].
///
/// Accepts `"scalar_8bit"` / `"scalar"` (the default when `name` is
/// `None`) and `"product_quantization"` / `"pq"`. Product quantization
/// requires a `subvector_count` (which must divide the field dimension —
/// the divisibility itself is validated by the core at index-build time);
/// supplying `subvector_count` for any other quantizer is rejected so an
/// incoherent configuration cannot silently reach the core.
fn parse_quantizer(
    name: Option<&str>,
    subvector_count: Option<usize>,
) -> PyResult<QuantizationMethod> {
    match name.map(|s| s.to_lowercase()).as_deref() {
        None | Some("scalar_8bit") | Some("scalar") => {
            if subvector_count.is_some() {
                return Err(PyValueError::new_err(
                    "subvector_count is only valid with quantizer='product_quantization'",
                ));
            }
            Ok(QuantizationMethod::Scalar8Bit)
        }
        Some("product_quantization") | Some("pq") => {
            let subvector_count = subvector_count.ok_or_else(|| {
                PyValueError::new_err(
                    "quantizer='product_quantization' requires subvector_count \
                     (must divide the field dimension)",
                )
            })?;
            Ok(QuantizationMethod::ProductQuantization { subvector_count })
        }
        Some(other) => Err(PyValueError::new_err(format!(
            "Unknown quantizer: '{other}'. Valid: scalar_8bit, product_quantization"
        ))),
    }
}

/// Parse a rerank-storage name into an optional [`RerankStorageKind`].
///
/// `None` (the default) keeps the Stage-1 int8-only segment; `"f32"`
/// enables the Stage-2 full-precision rerank sidecar (`*.hnsw.f32`).
fn parse_rerank_storage(name: Option<&str>) -> PyResult<Option<RerankStorageKind>> {
    match name.map(|s| s.to_lowercase()).as_deref() {
        None => Ok(None),
        Some("f32") => Ok(Some(RerankStorageKind::F32)),
        Some(other) => Err(PyValueError::new_err(format!(
            "Unknown rerank_storage: '{other}'. Valid: f32"
        ))),
    }
}

/// Python-facing schema builder.
///
/// ## Example
///
/// ```python
/// schema = laurus.Schema()
/// schema.add_text_field("title")
/// schema.add_hnsw_field("embedding", dimension=384, distance="cosine")
/// schema.add_integer_field("year")
/// schema.set_default_fields(["title"])
/// ```
#[pyclass(name = "Schema")]
pub struct PySchema {
    pub inner: Schema,
}

#[pymethods]
impl PySchema {
    /// Create a new empty schema.
    #[new]
    pub fn new() -> Self {
        Self {
            inner: Schema::new(),
        }
    }

    /// Add a full-text searchable text field.
    ///
    /// Args:
    ///     name: Field name.
    ///     stored: Whether the original value is retrievable (default True).
    ///     indexed: Whether the field is searchable (default True).
    ///     term_vectors: Whether term position information is stored
    ///         (default False).
    ///     analyzer: Either a string analyzer name (``"standard"``,
    ///         ``"english"``, ``"keyword"``, ``"simple"``, ``"noop"``, or
    ///         a custom name registered via ``add_analyzer``), or a dict
    ///         configuring a parameterized built-in preset such as
    ///         ``{"language": "japanese", "dict": "/var/lib/lindera/ipadic"}``.
    #[pyo3(signature = (name, *, stored=true, indexed=true, term_vectors=false, analyzer=None))]
    pub fn add_text_field(
        &mut self,
        py: Python<'_>,
        name: &str,
        stored: bool,
        indexed: bool,
        term_vectors: bool,
        analyzer: Option<Py<PyAny>>,
    ) -> PyResult<()> {
        let analyzer = analyzer
            .map(|obj| analyzer_spec_from_py(py, obj))
            .transpose()?;
        self.inner.fields.insert(
            name.to_string(),
            FieldOption::Text(TextOption {
                indexed,
                stored,
                term_vectors,
                analyzer,
            }),
        );
        Ok(())
    }

    /// Add an integer (i64) field.
    ///
    /// Args:
    ///     name: Field name.
    ///     stored: Whether the value is retrievable (default True).
    ///     indexed: Whether the field is searchable for range queries
    ///         (default True).
    ///     multi_valued: When True, the field accepts arrays of integers
    ///         and range queries match if any value satisfies the
    ///         predicate (Lucene-style "any match"). Default False.
    #[pyo3(signature = (name, *, stored=true, indexed=true, multi_valued=false))]
    pub fn add_integer_field(
        &mut self,
        name: &str,
        stored: bool,
        indexed: bool,
        multi_valued: bool,
    ) {
        self.inner.fields.insert(
            name.to_string(),
            FieldOption::Integer(IntegerOption {
                indexed,
                stored,
                multi_valued,
            }),
        );
    }

    /// Add a float (f64) field.
    ///
    /// Args:
    ///     name: Field name.
    ///     stored: Whether the value is retrievable (default True).
    ///     indexed: Whether the field is searchable for range queries
    ///         (default True).
    ///     multi_valued: When True, the field accepts arrays of floats
    ///         and range queries match if any value satisfies the
    ///         predicate (Lucene-style "any match"). Default False.
    #[pyo3(signature = (name, *, stored=true, indexed=true, multi_valued=false))]
    pub fn add_float_field(&mut self, name: &str, stored: bool, indexed: bool, multi_valued: bool) {
        self.inner.fields.insert(
            name.to_string(),
            FieldOption::Float(FloatOption {
                indexed,
                stored,
                multi_valued,
            }),
        );
    }

    /// Add a boolean field.
    #[pyo3(signature = (name, *, stored=true, indexed=true))]
    pub fn add_boolean_field(&mut self, name: &str, stored: bool, indexed: bool) {
        self.inner.fields.insert(
            name.to_string(),
            FieldOption::Boolean(BooleanOption { indexed, stored }),
        );
    }

    /// Add a date/time field.
    #[pyo3(signature = (name, *, stored=true, indexed=true))]
    pub fn add_datetime_field(&mut self, name: &str, stored: bool, indexed: bool) {
        self.inner.fields.insert(
            name.to_string(),
            FieldOption::DateTime(DateTimeOption { indexed, stored }),
        );
    }

    /// Add a geographic coordinate field (latitude, longitude).
    #[pyo3(signature = (name, *, stored=true, indexed=true))]
    pub fn add_geo_field(&mut self, name: &str, stored: bool, indexed: bool) {
        self.inner.fields.insert(
            name.to_string(),
            FieldOption::Geo(GeoOption { indexed, stored }),
        );
    }

    /// Add a 3D ECEF Cartesian point field (x, y, z in meters).
    ///
    /// Values are submitted as a 3-tuple `(x, y, z)` of floats and are
    /// queryable via `Geo3dDistanceQuery`, `Geo3dBoundingBoxQuery`, and
    /// `Geo3dNearestQuery`. See the conceptual docs at
    /// `docs/src/concepts/geo3d.md` for the coordinate system.
    #[pyo3(signature = (name, *, stored=true, indexed=true))]
    pub fn add_geo3d_field(&mut self, name: &str, stored: bool, indexed: bool) {
        self.inner.fields.insert(
            name.to_string(),
            FieldOption::Geo3d(Geo3dOption { indexed, stored }),
        );
    }

    /// Add a binary data field.
    #[pyo3(signature = (name, *, stored=true))]
    pub fn add_bytes_field(&mut self, name: &str, stored: bool) {
        self.inner
            .fields
            .insert(name.to_string(), FieldOption::Bytes(BytesOption { stored }));
    }

    /// Add an HNSW approximate nearest-neighbor vector index field.
    ///
    /// Args:
    ///     name: Field name.
    ///     dimension: Vector dimensionality.
    ///     distance: Distance metric — "cosine" (default), "euclidean", "dot_product".
    ///     m: HNSW branching factor (default 16).
    ///     ef_construction: Build-time expansion factor (default 200).
    ///     default_ef_search: Schema-level default for the search-time
    ///         `ef_search` candidate-list size (Issue #644). When unset,
    ///         the searcher uses an internal fallback of 50. Per-query
    ///         overrides via the search request still take precedence.
    ///     quantizer: Vector quantizer — "scalar_8bit" (default) or
    ///         "product_quantization". Product quantization requires
    ///         `subvector_count`.
    ///     subvector_count: Number of PQ sub-vectors. Required when
    ///         `quantizer="product_quantization"` and must divide
    ///         `dimension`; rejected for other quantizers.
    ///     rerank_storage: Stage-2 rerank sidecar — None (default) keeps
    ///         the int8-only segment, "f32" stores full-precision vectors
    ///         in a `*.hnsw.f32` sidecar for exact rerank distances.
    ///     embedder: Optional embedder name registered via `add_embedder`.
    ///         When set, text payloads are automatically embedded by the Rust engine.
    ///     pq_codebook_path: Storage-relative file name of a shared PQ
    ///         codebook (Issue #631), trained once via the
    ///         `laurus train pq-codebook` CLI command. Only meaningful with
    ///         `quantizer="product_quantization"`; commits then encode
    ///         against the pre-trained codebook instead of re-training
    ///         k-means per segment. None (default) keeps per-segment training.
    #[pyo3(signature = (name, dimension, *, distance="cosine", m=16, ef_construction=200, default_ef_search=None, quantizer=None, subvector_count=None, rerank_storage=None, embedder=None, pq_codebook_path=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn add_hnsw_field(
        &mut self,
        name: &str,
        dimension: usize,
        distance: &str,
        m: usize,
        ef_construction: usize,
        default_ef_search: Option<usize>,
        quantizer: Option<String>,
        subvector_count: Option<usize>,
        rerank_storage: Option<String>,
        embedder: Option<String>,
        pq_codebook_path: Option<String>,
    ) -> PyResult<()> {
        let opt = HnswOption {
            dimension,
            distance: parse_distance(distance)?,
            m,
            ef_construction,
            default_ef_search,
            quantizer: parse_quantizer(quantizer.as_deref(), subvector_count)?,
            rerank_storage: parse_rerank_storage(rerank_storage.as_deref())?,
            embedder,
            pq_codebook_path,
            ..Default::default()
        };
        self.inner
            .fields
            .insert(name.to_string(), FieldOption::Hnsw(opt));
        Ok(())
    }

    /// Add a flat (brute-force) vector index field.
    ///
    /// Args:
    ///     name: Field name.
    ///     dimension: Vector dimensionality.
    ///     distance: Distance metric — "cosine" (default), "euclidean", "dot_product".
    ///     embedder: Optional embedder name registered via `add_embedder`.
    ///         When set, text payloads are automatically embedded by the Rust engine.
    #[pyo3(signature = (name, dimension, *, distance="cosine", embedder=None))]
    pub fn add_flat_field(
        &mut self,
        name: &str,
        dimension: usize,
        distance: &str,
        embedder: Option<String>,
    ) -> PyResult<()> {
        let opt = FlatOption {
            dimension,
            distance: parse_distance(distance)?,
            embedder,
            ..Default::default()
        };
        self.inner
            .fields
            .insert(name.to_string(), FieldOption::Flat(opt));
        Ok(())
    }

    /// Add an IVF (Inverted File Index) approximate nearest-neighbor vector field.
    ///
    /// Args:
    ///     name: Field name.
    ///     dimension: Vector dimensionality.
    ///     distance: Distance metric — "cosine" (default), "euclidean", "dot_product".
    ///     n_clusters: Number of Voronoi clusters (default 100).
    ///     n_probe: Number of clusters to probe at search time (default 1).
    ///     embedder: Optional embedder name registered via `add_embedder`.
    ///         When set, text payloads are automatically embedded by the Rust engine.
    #[pyo3(signature = (name, dimension, *, distance="cosine", n_clusters=100, n_probe=1, embedder=None))]
    pub fn add_ivf_field(
        &mut self,
        name: &str,
        dimension: usize,
        distance: &str,
        n_clusters: usize,
        n_probe: usize,
        embedder: Option<String>,
    ) -> PyResult<()> {
        let opt = IvfOption {
            dimension,
            distance: parse_distance(distance)?,
            n_clusters,
            n_probe,
            embedder,
            ..Default::default()
        };
        self.inner
            .fields
            .insert(name.to_string(), FieldOption::Ivf(opt));
        Ok(())
    }

    /// Register a named embedder definition in the schema.
    ///
    /// The embedder can then be referenced by name from vector field options
    /// (e.g. `add_hnsw_field(..., embedder="my-bert")`).
    ///
    /// The `config` dict must have a `"type"` key selecting the backend:
    ///
    /// | type            | required keys | feature flag          |
    /// |-----------------|---------------|-----------------------|
    /// | `"precomputed"` | —             | (always available)    |
    /// | `"candle_bert"` | `"model"`     | `embeddings-candle`   |
    /// | `"candle_clip"` | `"model"`     | `embeddings-multimodal` |
    /// | `"openai"`      | `"model"`     | `embeddings-openai`   |
    ///
    /// Args:
    ///     name: Unique embedder name referenced from vector fields.
    ///     config: Dict describing the embedder, e.g.
    ///         `{"type": "candle_bert", "model": "sentence-transformers/all-MiniLM-L6-v2"}`.
    ///
    /// Example:
    ///     ```python
    ///     schema.add_embedder("bert", {"type": "candle_bert", "model": "sentence-transformers/all-MiniLM-L6-v2"})
    ///     schema.add_hnsw_field("embedding", dimension=384, embedder="bert")
    ///     ```
    pub fn add_embedder(&mut self, name: &str, config: &Bound<PyAny>) -> PyResult<()> {
        let dict = config.extract::<Bound<PyDict>>().map_err(|_| {
            PyValueError::new_err("embedder config must be a dict, e.g. {\"type\": \"candle_bert\", \"model\": \"...\"}")
        })?;
        let dict = &dict;

        let embedder_type: String = dict
            .get_item("type")?
            .ok_or_else(|| PyValueError::new_err("embedder config must have a 'type' key"))?
            .extract()?;

        let definition = match embedder_type.as_str() {
            "precomputed" => EmbedderDefinition::Precomputed,
            "candle_bert" => {
                let model: String = dict
                    .get_item("model")?
                    .ok_or_else(|| {
                        PyValueError::new_err("candle_bert embedder requires a 'model' key")
                    })?
                    .extract()?;
                EmbedderDefinition::CandleBert { model }
            }
            "candle_clip" => {
                let model: String = dict
                    .get_item("model")?
                    .ok_or_else(|| {
                        PyValueError::new_err("candle_clip embedder requires a 'model' key")
                    })?
                    .extract()?;
                EmbedderDefinition::CandleClip { model }
            }
            "openai" => {
                let model: String = dict
                    .get_item("model")?
                    .ok_or_else(|| PyValueError::new_err("openai embedder requires a 'model' key"))?
                    .extract()?;
                EmbedderDefinition::Openai { model }
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unknown embedder type: '{}'. Valid types: precomputed, candle_bert, candle_clip, openai",
                    other
                )));
            }
        };

        self.inner.embedders.insert(name.to_string(), definition);
        Ok(())
    }

    /// Set the default fields used when no field is specified in a query.
    pub fn set_default_fields(&mut self, fields: Vec<String>) {
        self.inner.default_fields = fields;
    }

    /// Set the policy for fields that are not declared in this schema.
    ///
    /// Args:
    ///     policy: One of ``"strict"``, ``"dynamic"`` (default), or
    ///         ``"ignore"``. Case-insensitive.
    ///
    /// Behaviour:
    ///     * ``"strict"``: reject documents containing undeclared fields.
    ///     * ``"dynamic"``: infer a type for each undeclared field and add
    ///       it to the schema during ingestion. **Warning**: integer fields
    ///       silently truncate incoming float values (e.g. ``3.14`` → ``3``).
    ///     * ``"ignore"``: silently drop undeclared fields.
    ///
    /// Raises:
    ///     ValueError: if ``policy`` is not one of the accepted names.
    pub fn set_dynamic_field_policy(&mut self, policy: &str) -> PyResult<()> {
        let parsed = DynamicFieldPolicy::from_str(policy)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        self.inner.dynamic_field_policy = parsed;
        Ok(())
    }

    /// Return the currently configured dynamic field policy as a lowercase
    /// string (``"strict"`` / ``"dynamic"`` / ``"ignore"``).
    pub fn dynamic_field_policy(&self) -> &'static str {
        match self.inner.dynamic_field_policy {
            DynamicFieldPolicy::Strict => "strict",
            DynamicFieldPolicy::Dynamic => "dynamic",
            DynamicFieldPolicy::Ignore => "ignore",
        }
    }

    /// Return the list of field names defined in this schema.
    pub fn field_names(&self) -> Vec<String> {
        self.inner.fields.keys().cloned().collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "Schema(fields={:?})",
            self.inner.fields.keys().collect::<Vec<_>>()
        )
    }
}
