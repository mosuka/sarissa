//! Vector field configuration options.
//!
//! This module defines options for configuring vector fields, including
//! index types and parameters for different algorithms (Flat, HNSW, IVF).

use serde::{Deserialize, Serialize};

use crate::vector::core::distance::DistanceMetric;
use crate::vector::core::quantization;
use crate::vector::core::rerank::RerankStorageKind;

fn default_dimension() -> usize {
    128
}

fn default_getting_m() -> usize {
    16
}

fn default_getting_ef_construction() -> usize {
    200
}

fn default_getting_n_clusters() -> usize {
    100
}

fn default_getting_n_probe() -> usize {
    1
}

/// Options for vector fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "options", rename_all = "snake_case")]
pub enum FieldOption {
    /// Flat index options.
    Flat(FlatOption),
    /// HNSW index options.
    Hnsw(HnswOption),
    /// IVF index options.
    Ivf(IvfOption),
}

impl Default for FieldOption {
    fn default() -> Self {
        FieldOption::Hnsw(HnswOption::default())
    }
}

impl FieldOption {
    /// Get the dimension of the vector field.
    pub fn dimension(&self) -> usize {
        match self {
            FieldOption::Flat(opt) => opt.dimension,
            FieldOption::Hnsw(opt) => opt.dimension,
            FieldOption::Ivf(opt) => opt.dimension,
        }
    }

    /// Get the distance metric.
    pub fn distance(&self) -> DistanceMetric {
        match self {
            FieldOption::Flat(opt) => opt.distance,
            FieldOption::Hnsw(opt) => opt.distance,
            FieldOption::Ivf(opt) => opt.distance,
        }
    }

    /// Get the base weight.
    pub fn base_weight(&self) -> f32 {
        match self {
            FieldOption::Flat(opt) => opt.base_weight,
            FieldOption::Hnsw(opt) => opt.base_weight,
            FieldOption::Ivf(opt) => opt.base_weight,
        }
    }

    /// Get the index kind.
    pub fn index_kind(&self) -> VectorIndexKind {
        match self {
            FieldOption::Flat(_) => VectorIndexKind::Flat,
            FieldOption::Hnsw(_) => VectorIndexKind::Hnsw,
            FieldOption::Ivf(_) => VectorIndexKind::Ivf,
        }
    }
}

/// Options for Flat vector index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlatOption {
    /// Number of dimensions for each vector. Defaults to `128`.
    #[serde(default = "default_dimension")]
    pub dimension: usize,
    /// Distance metric used for similarity computation. Defaults to [`DistanceMetric::Cosine`].
    #[serde(default = "default_distance_metric")]
    pub distance: DistanceMetric,
    /// Base weight applied to similarity scores from this field. Defaults to `1.0`.
    #[serde(default = "default_weight")]
    pub base_weight: f32,
    /// Quantization method used for the on-disk vector format.
    /// Defaults to [`quantization::QuantizationMethod::Scalar8Bit`]
    /// (Issue #481 Stage 1: int8 SQ is mandatory; the previous
    /// `Option::None` "no quantization" path no longer exists).
    #[serde(default)]
    pub quantizer: quantization::QuantizationMethod,
    /// Two-stage rerank storage backend (Issue #481 Stage 2).
    ///
    /// When `Some(RerankStorageKind::F32)`, an `*.{ext}.f32` sidecar
    /// is written alongside the int8 segment so the searcher can
    /// re-score the top `top_k * rerank_factor` candidates with the
    /// original f32 vectors. Costs ~4x extra disk and memory per
    /// vector but recovers Stage 1 recall close to the f32 baseline.
    ///
    /// `None` (the default) leaves the field on the Stage 1 int8-only
    /// path; queries that set `rerank_factor` against such a field
    /// silently fall back to Stage 1 ranking (the original
    /// information was discarded at index time).
    #[serde(default)]
    pub rerank_storage: Option<RerankStorageKind>,
    /// Embedder name for this vector field.
    /// When set, the engine automatically embeds input using the named embedder.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedder: Option<String>,
}

impl Default for FlatOption {
    fn default() -> Self {
        Self {
            dimension: 128,
            distance: default_distance_metric(),
            base_weight: default_weight(),
            quantizer: quantization::QuantizationMethod::Scalar8Bit,
            rerank_storage: None,
            embedder: None,
        }
    }
}

/// Options for HNSW vector index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswOption {
    /// Number of dimensions for each vector. Defaults to `128`.
    #[serde(default = "default_dimension")]
    pub dimension: usize,
    /// Distance metric used for similarity computation. Defaults to [`DistanceMetric::Cosine`].
    #[serde(default = "default_distance_metric")]
    pub distance: DistanceMetric,
    /// Maximum number of bi-directional links per node in the HNSW graph.
    /// Higher values improve recall but increase memory usage. Defaults to `16`.
    #[serde(default = "default_getting_m")]
    pub m: usize,
    /// Size of the dynamic candidate list during index construction.
    /// Higher values produce a higher-quality graph at the cost of slower
    /// build times. Defaults to `200`.
    #[serde(default = "default_getting_ef_construction")]
    pub ef_construction: usize,
    /// Base weight applied to similarity scores from this field. Defaults to `1.0`.
    #[serde(default = "default_weight")]
    pub base_weight: f32,
    /// Quantization method used for the on-disk vector format.
    /// Defaults to [`quantization::QuantizationMethod::Scalar8Bit`]
    /// (Issue #481 Stage 1: int8 SQ is mandatory; the previous
    /// `Option::None` "no quantization" path no longer exists).
    #[serde(default)]
    pub quantizer: quantization::QuantizationMethod,
    /// Two-stage rerank storage backend (Issue #481 Stage 2).
    ///
    /// When `Some(RerankStorageKind::F32)`, an `*.{ext}.f32` sidecar
    /// is written alongside the int8 segment so the searcher can
    /// re-score the top `top_k * rerank_factor` candidates with the
    /// original f32 vectors. Costs ~4x extra disk and memory per
    /// vector but recovers Stage 1 recall close to the f32 baseline.
    ///
    /// `None` (the default) leaves the field on the Stage 1 int8-only
    /// path; queries that set `rerank_factor` against such a field
    /// silently fall back to Stage 1 ranking (the original
    /// information was discarded at index time).
    #[serde(default)]
    pub rerank_storage: Option<RerankStorageKind>,
    /// Embedder name for this vector field.
    /// When set, the engine automatically embeds input using the named embedder.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedder: Option<String>,
}

impl Default for HnswOption {
    fn default() -> Self {
        Self {
            dimension: 128,
            distance: default_distance_metric(),
            m: default_getting_m(),
            ef_construction: default_getting_ef_construction(),
            base_weight: default_weight(),
            quantizer: quantization::QuantizationMethod::Scalar8Bit,
            rerank_storage: None,
            embedder: None,
        }
    }
}

/// Options for IVF vector index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IvfOption {
    /// Number of dimensions for each vector.
    pub dimension: usize,
    /// Distance metric used for similarity computation. Defaults to [`DistanceMetric::Cosine`].
    #[serde(default = "default_distance_metric")]
    pub distance: DistanceMetric,
    /// Number of Voronoi clusters used to partition the vector space.
    /// More clusters speed up search but increase build time. Defaults to `100`.
    #[serde(default = "default_getting_n_clusters")]
    pub n_clusters: usize,
    /// Number of clusters to probe during search.
    /// Higher values improve recall at the cost of query latency. Defaults to `1`.
    #[serde(default = "default_getting_n_probe")]
    pub n_probe: usize,
    /// Base weight applied to similarity scores from this field. Defaults to `1.0`.
    #[serde(default = "default_weight")]
    pub base_weight: f32,
    /// Quantization method used for the on-disk vector format.
    /// Defaults to [`quantization::QuantizationMethod::Scalar8Bit`]
    /// (Issue #481 Stage 1: int8 SQ is mandatory; the previous
    /// `Option::None` "no quantization" path no longer exists).
    #[serde(default)]
    pub quantizer: quantization::QuantizationMethod,
    /// Two-stage rerank storage backend (Issue #481 Stage 2).
    ///
    /// When `Some(RerankStorageKind::F32)`, an `*.{ext}.f32` sidecar
    /// is written alongside the int8 segment so the searcher can
    /// re-score the top `top_k * rerank_factor` candidates with the
    /// original f32 vectors. Costs ~4x extra disk and memory per
    /// vector but recovers Stage 1 recall close to the f32 baseline.
    ///
    /// `None` (the default) leaves the field on the Stage 1 int8-only
    /// path; queries that set `rerank_factor` against such a field
    /// silently fall back to Stage 1 ranking (the original
    /// information was discarded at index time).
    #[serde(default)]
    pub rerank_storage: Option<RerankStorageKind>,
    /// Embedder name for this vector field.
    /// When set, the engine automatically embeds input using the named embedder.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedder: Option<String>,
}

impl Default for IvfOption {
    fn default() -> Self {
        Self {
            dimension: 128,
            distance: default_distance_metric(),
            n_clusters: default_getting_n_clusters(),
            n_probe: default_getting_n_probe(),
            base_weight: default_weight(),
            quantizer: quantization::QuantizationMethod::Scalar8Bit,
            rerank_storage: None,
            embedder: None,
        }
    }
}

/// The type of vector index to use.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum VectorIndexKind {
    /// Flat (brute-force) index - exact but slower for large datasets.
    Flat,
    /// HNSW (Hierarchical Navigable Small World) - approximate but fast.
    Hnsw,
    /// IVF (Inverted File Index) - approximate with clustering.
    Ivf,
}

// From implementations for VectorOption
impl From<FlatOption> for FieldOption {
    fn from(opt: FlatOption) -> Self {
        FieldOption::Flat(opt)
    }
}

impl From<HnswOption> for FieldOption {
    fn from(opt: HnswOption) -> Self {
        FieldOption::Hnsw(opt)
    }
}

impl From<IvfOption> for FieldOption {
    fn from(opt: IvfOption) -> Self {
        FieldOption::Ivf(opt)
    }
}

// Builder pattern for FlatOption
impl FlatOption {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            ..Default::default()
        }
    }

    pub fn dimension(mut self, dimension: usize) -> Self {
        self.dimension = dimension;
        self
    }

    pub fn distance(mut self, distance: DistanceMetric) -> Self {
        self.distance = distance;
        self
    }

    pub fn base_weight(mut self, weight: f32) -> Self {
        self.base_weight = weight;
        self
    }

    pub fn quantizer(mut self, quantizer: quantization::QuantizationMethod) -> Self {
        self.quantizer = quantizer;
        self
    }

    /// Enable two-stage rerank (Issue #481 Stage 2) for this field.
    ///
    /// Storing rerank data writes a sidecar file at index time so the
    /// searcher can re-score the top `top_k * rerank_factor`
    /// candidates against the original full-precision vectors. Pass
    /// `None` (or omit) to stay on the Stage 1 int8-only path.
    pub fn rerank_storage(mut self, kind: RerankStorageKind) -> Self {
        self.rerank_storage = Some(kind);
        self
    }
}

// Builder pattern for HnswOption
impl HnswOption {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            ..Default::default()
        }
    }

    pub fn dimension(mut self, dimension: usize) -> Self {
        self.dimension = dimension;
        self
    }

    pub fn distance(mut self, distance: DistanceMetric) -> Self {
        self.distance = distance;
        self
    }

    pub fn m(mut self, m: usize) -> Self {
        self.m = m;
        self
    }

    pub fn ef_construction(mut self, ef: usize) -> Self {
        self.ef_construction = ef;
        self
    }

    pub fn base_weight(mut self, weight: f32) -> Self {
        self.base_weight = weight;
        self
    }

    pub fn quantizer(mut self, quantizer: quantization::QuantizationMethod) -> Self {
        self.quantizer = quantizer;
        self
    }

    /// Enable two-stage rerank (Issue #481 Stage 2) for this field.
    ///
    /// Storing rerank data writes a sidecar file at index time so the
    /// searcher can re-score the top `top_k * rerank_factor`
    /// candidates against the original full-precision vectors. Pass
    /// `None` (or omit) to stay on the Stage 1 int8-only path.
    pub fn rerank_storage(mut self, kind: RerankStorageKind) -> Self {
        self.rerank_storage = Some(kind);
        self
    }
}

// Builder pattern for IvfOption
impl IvfOption {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            ..Default::default()
        }
    }

    pub fn dimension(mut self, dimension: usize) -> Self {
        self.dimension = dimension;
        self
    }

    pub fn distance(mut self, distance: DistanceMetric) -> Self {
        self.distance = distance;
        self
    }

    pub fn n_clusters(mut self, n: usize) -> Self {
        self.n_clusters = n;
        self
    }

    pub fn n_probe(mut self, n: usize) -> Self {
        self.n_probe = n;
        self
    }

    pub fn base_weight(mut self, weight: f32) -> Self {
        self.base_weight = weight;
        self
    }

    pub fn quantizer(mut self, quantizer: quantization::QuantizationMethod) -> Self {
        self.quantizer = quantizer;
        self
    }

    /// Enable two-stage rerank (Issue #481 Stage 2) for this field.
    ///
    /// Storing rerank data writes a sidecar file at index time so the
    /// searcher can re-score the top `top_k * rerank_factor`
    /// candidates against the original full-precision vectors. Pass
    /// `None` (or omit) to stay on the Stage 1 int8-only path.
    pub fn rerank_storage(mut self, kind: RerankStorageKind) -> Self {
        self.rerank_storage = Some(kind);
        self
    }
}

// Helpers

fn default_distance_metric() -> DistanceMetric {
    DistanceMetric::Cosine
}

fn default_weight() -> f32 {
    1.0
}
