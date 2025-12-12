//! Configuration for lexical index types.
//!
//! This module provides configuration types for lexical indexes.
//!
//! # Configuration with Analyzer
//!
//! The recommended way to configure a LexicalEngine is to use the builder pattern
//! with an `Analyzer`, similar to how `Embedder` is used in `VectorEngine`.
//!
//! ```no_run
//! use platypus::lexical::index::config::LexicalIndexConfig;
//! use platypus::analysis::analyzer::standard::StandardAnalyzer;
//! use platypus::analysis::analyzer::per_field::PerFieldAnalyzer;
//! use std::sync::Arc;
//!
//! // Simple configuration with default analyzer
//! let config = LexicalIndexConfig::builder().build();
//!
//! // Configuration with custom analyzer
//! let analyzer = Arc::new(StandardAnalyzer::default());
//! let config = LexicalIndexConfig::builder()
//!     .analyzer(analyzer)
//!     .build();
//!
//! // Configuration with per-field analyzer
//! let default_analyzer = Arc::new(StandardAnalyzer::default());
//! let per_field = PerFieldAnalyzer::new(default_analyzer);
//! let config = LexicalIndexConfig::builder()
//!     .analyzer(Arc::new(per_field))
//!     .build();
//! ```

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::analysis::analyzer::analyzer::Analyzer;
use crate::analysis::analyzer::standard::StandardAnalyzer;

/// Configuration for lexical index types.
///
/// This enum provides type-safe configuration for different index implementations.
/// Each variant contains the configuration specific to that index type.
///
/// # Design Pattern
///
/// This follows an enum-based configuration pattern where:
/// - Each index type has its own dedicated config struct
/// - Pattern matching ensures exhaustive handling of all index types
/// - New index types can be added without breaking existing code
///
/// # Index Types
///
/// - **Inverted**: Traditional inverted index (default)
///   - Fast full-text search
///   - Good for keyword queries
///   - Supports boolean operations
///
/// Future index types that could be added:
/// - **ColumnStore**: Column-oriented index for aggregations
/// - **LSMTree**: Log-structured merge-tree for write-heavy workloads
///
/// # Example
///
/// ```no_run
/// use platypus::lexical::index::config::{LexicalIndexConfig, InvertedIndexConfig};
///
/// // Use default inverted index
/// let config = LexicalIndexConfig::default();
///
/// // Custom inverted index configuration
/// let mut inverted_config = InvertedIndexConfig::default();
/// inverted_config.max_docs_per_segment = 500_000;
/// inverted_config.compress_stored_fields = true;
/// let config = LexicalIndexConfig::Inverted(inverted_config);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum LexicalIndexConfig {
    /// Inverted index configuration
    Inverted(InvertedIndexConfig),
    // Future index types can be added here:
    // ColumnStore(ColumnStoreConfig),
    // LSMTree(LSMTreeConfig),
}

impl Default for LexicalIndexConfig {
    fn default() -> Self {
        LexicalIndexConfig::Inverted(InvertedIndexConfig::default())
    }
}

impl LexicalIndexConfig {
    /// Create a new builder for LexicalIndexConfig.
    ///
    /// This provides a fluent API for constructing configuration,
    /// consistent with `VectorIndexConfig::builder()`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use platypus::lexical::index::config::LexicalIndexConfig;
    /// use platypus::analysis::analyzer::standard::StandardAnalyzer;
    /// use std::sync::Arc;
    ///
    /// let config = LexicalIndexConfig::builder()
    ///     .analyzer(Arc::new(StandardAnalyzer::default()))
    ///     .max_docs_per_segment(500_000)
    ///     .compress_stored_fields(true)
    ///     .build();
    /// ```
    pub fn builder() -> LexicalIndexConfigBuilder {
        LexicalIndexConfigBuilder::new()
    }

    /// Get a human-readable name for the index type.
    pub fn index_type_name(&self) -> &str {
        match self {
            LexicalIndexConfig::Inverted(_) => "Inverted",
            // Future index types:
            // LexicalIndexConfig::ColumnStore(_) => "ColumnStore",
            // LexicalIndexConfig::LSMTree(_) => "LSMTree",
        }
    }

    /// Get the analyzer from the configuration.
    ///
    /// Returns the analyzer used for text tokenization.
    pub fn analyzer(&self) -> &Arc<dyn Analyzer> {
        match self {
            LexicalIndexConfig::Inverted(config) => &config.analyzer,
        }
    }
}

/// Configuration specific to inverted index.
///
/// These settings control the behavior of the inverted index implementation,
/// including segment management, buffering, compression, and term storage options.
#[derive(Clone, Serialize, Deserialize)]
pub struct InvertedIndexConfig {
    /// Maximum number of documents per segment.
    ///
    /// When a segment reaches this size, it will be considered for merging.
    /// Larger values reduce merge overhead but increase memory usage.
    pub max_docs_per_segment: u64,

    /// Buffer size for writing operations (in bytes).
    ///
    /// Controls how much data is buffered in memory before being flushed to disk.
    /// Larger buffers improve write performance but use more memory.
    pub write_buffer_size: usize,

    /// Whether to use compression for stored fields.
    ///
    /// Enabling compression reduces disk usage but increases CPU overhead
    /// for indexing and retrieval operations.
    pub compress_stored_fields: bool,

    /// Whether to store term vectors.
    ///
    /// Term vectors enable advanced features like highlighting and more-like-this
    /// queries, but increase index size and indexing time.
    pub store_term_vectors: bool,

    /// Merge factor for segment merging.
    ///
    /// Controls how many segments are merged at once. Higher values reduce
    /// the number of merge operations but create larger temporary segments.
    pub merge_factor: u32,

    /// Maximum number of segments before merging.
    ///
    /// When the number of segments exceeds this threshold, a merge operation
    /// will be triggered to consolidate them.
    pub max_segments: u32,

    /// Analyzer for text fields.
    ///
    /// This analyzer is used to tokenize text during indexing and querying.
    /// Can be a PerFieldAnalyzer to use different analyzers for different fields.
    #[serde(skip)]
    #[serde(default = "default_analyzer")]
    pub analyzer: Arc<dyn Analyzer>,
}

fn default_analyzer() -> Arc<dyn Analyzer> {
    Arc::new(StandardAnalyzer::new().expect("StandardAnalyzer should be creatable"))
}

impl Default for InvertedIndexConfig {
    fn default() -> Self {
        InvertedIndexConfig {
            max_docs_per_segment: 1000000,
            write_buffer_size: 1024 * 1024, // 1MB
            compress_stored_fields: false,
            store_term_vectors: false,
            merge_factor: 10,
            max_segments: 100,
            analyzer: std::sync::Arc::new(
                crate::analysis::analyzer::standard::StandardAnalyzer::new()
                    .expect("StandardAnalyzer should be creatable"),
            ),
        }
    }
}

impl std::fmt::Debug for InvertedIndexConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InvertedIndexConfig")
            .field("max_docs_per_segment", &self.max_docs_per_segment)
            .field("write_buffer_size", &self.write_buffer_size)
            .field("compress_stored_fields", &self.compress_stored_fields)
            .field("store_term_vectors", &self.store_term_vectors)
            .field("merge_factor", &self.merge_factor)
            .field("max_segments", &self.max_segments)
            .field("analyzer", &self.analyzer.name())
            .finish()
    }
}

/// Builder for LexicalIndexConfig.
///
/// Provides a fluent API for constructing LexicalIndexConfig,
/// consistent with `VectorIndexConfigBuilder`.
///
/// # Example
///
/// ```no_run
/// use platypus::lexical::index::config::LexicalIndexConfig;
/// use platypus::analysis::analyzer::standard::StandardAnalyzer;
/// use platypus::analysis::analyzer::per_field::PerFieldAnalyzer;
/// use std::sync::Arc;
///
/// // Simple usage with defaults
/// let config = LexicalIndexConfig::builder().build();
///
/// // With custom analyzer
/// let analyzer = Arc::new(StandardAnalyzer::default());
/// let config = LexicalIndexConfig::builder()
///     .analyzer(analyzer)
///     .build();
///
/// // With per-field analyzer (similar to VectorEngine's PerFieldEmbedder)
/// let default_analyzer = Arc::new(StandardAnalyzer::default());
/// let per_field = PerFieldAnalyzer::new(default_analyzer);
/// let config = LexicalIndexConfig::builder()
///     .analyzer(Arc::new(per_field))
///     .max_docs_per_segment(500_000)
///     .compress_stored_fields(true)
///     .build();
/// ```
#[derive(Default)]
pub struct LexicalIndexConfigBuilder {
    analyzer: Option<Arc<dyn Analyzer>>,
    max_docs_per_segment: Option<u64>,
    write_buffer_size: Option<usize>,
    compress_stored_fields: Option<bool>,
    store_term_vectors: Option<bool>,
    merge_factor: Option<u32>,
    max_segments: Option<u32>,
}

impl LexicalIndexConfigBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the analyzer for text fields.
    ///
    /// Use `PerFieldAnalyzer` for field-specific analyzers.
    /// This is analogous to `VectorIndexConfigBuilder::embedder()`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use platypus::lexical::index::config::LexicalIndexConfig;
    /// use platypus::analysis::analyzer::standard::StandardAnalyzer;
    /// use platypus::analysis::analyzer::per_field::PerFieldAnalyzer;
    /// use std::sync::Arc;
    ///
    /// // With a single analyzer
    /// let config = LexicalIndexConfig::builder()
    ///     .analyzer(Arc::new(StandardAnalyzer::default()))
    ///     .build();
    ///
    /// // With per-field analyzers
    /// let default_analyzer = Arc::new(StandardAnalyzer::default());
    /// let per_field = PerFieldAnalyzer::new(default_analyzer);
    /// let config = LexicalIndexConfig::builder()
    ///     .analyzer(Arc::new(per_field))
    ///     .build();
    /// ```
    pub fn analyzer(mut self, analyzer: Arc<dyn Analyzer>) -> Self {
        self.analyzer = Some(analyzer);
        self
    }

    /// Set the maximum number of documents per segment.
    ///
    /// When a segment reaches this size, it will be considered for merging.
    /// Larger values reduce merge overhead but increase memory usage.
    /// Default: 1,000,000
    pub fn max_docs_per_segment(mut self, max_docs: u64) -> Self {
        self.max_docs_per_segment = Some(max_docs);
        self
    }

    /// Set the buffer size for writing operations (in bytes).
    ///
    /// Controls how much data is buffered in memory before being flushed to disk.
    /// Larger buffers improve write performance but use more memory.
    /// Default: 1MB (1,048,576 bytes)
    pub fn write_buffer_size(mut self, size: usize) -> Self {
        self.write_buffer_size = Some(size);
        self
    }

    /// Enable or disable compression for stored fields.
    ///
    /// Enabling compression reduces disk usage but increases CPU overhead
    /// for indexing and retrieval operations.
    /// Default: false
    pub fn compress_stored_fields(mut self, compress: bool) -> Self {
        self.compress_stored_fields = Some(compress);
        self
    }

    /// Enable or disable term vector storage.
    ///
    /// Term vectors enable advanced features like highlighting and more-like-this
    /// queries, but increase index size and indexing time.
    /// Default: false
    pub fn store_term_vectors(mut self, store: bool) -> Self {
        self.store_term_vectors = Some(store);
        self
    }

    /// Set the merge factor for segment merging.
    ///
    /// Controls how many segments are merged at once. Higher values reduce
    /// the number of merge operations but create larger temporary segments.
    /// Default: 10
    pub fn merge_factor(mut self, factor: u32) -> Self {
        self.merge_factor = Some(factor);
        self
    }

    /// Set the maximum number of segments before merging.
    ///
    /// When the number of segments exceeds this threshold, a merge operation
    /// will be triggered to consolidate them.
    /// Default: 100
    pub fn max_segments(mut self, max: u32) -> Self {
        self.max_segments = Some(max);
        self
    }

    /// Build the configuration.
    ///
    /// Returns `LexicalIndexConfig::Inverted` with the configured settings.
    /// Any unset values will use defaults.
    pub fn build(self) -> LexicalIndexConfig {
        let mut config = InvertedIndexConfig::default();

        if let Some(analyzer) = self.analyzer {
            config.analyzer = analyzer;
        }
        if let Some(max_docs) = self.max_docs_per_segment {
            config.max_docs_per_segment = max_docs;
        }
        if let Some(size) = self.write_buffer_size {
            config.write_buffer_size = size;
        }
        if let Some(compress) = self.compress_stored_fields {
            config.compress_stored_fields = compress;
        }
        if let Some(store) = self.store_term_vectors {
            config.store_term_vectors = store;
        }
        if let Some(factor) = self.merge_factor {
            config.merge_factor = factor;
        }
        if let Some(max) = self.max_segments {
            config.max_segments = max;
        }

        LexicalIndexConfig::Inverted(config)
    }
}
