//! Configuration for the lexical engine.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::analysis::analyzer::analyzer::Analyzer;
use crate::lexical::index::config::InvertedIndexConfig;

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
/// use sarissa::lexical::engine::config::LexicalIndexConfig;
/// use sarissa::lexical::index::config::InvertedIndexConfig;
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
    /// use sarissa::lexical::engine::config::LexicalIndexConfig;
    /// use sarissa::analysis::analyzer::standard::StandardAnalyzer;
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

/// Builder for LexicalIndexConfig.
///
/// Provides a fluent API for constructing LexicalIndexConfig,
/// consistent with `VectorIndexConfigBuilder`.
///
/// # Example
///
/// ```no_run
/// use sarissa::lexical::engine::config::LexicalIndexConfig;
/// use sarissa::analysis::analyzer::standard::StandardAnalyzer;
/// use sarissa::analysis::analyzer::per_field::PerFieldAnalyzer;
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
pub struct LexicalIndexConfigBuilder {
    analyzer: Option<Arc<dyn Analyzer>>,
    max_docs_per_segment: Option<u64>,
    write_buffer_size: Option<usize>,
    compress_stored_fields: Option<bool>,
    store_term_vectors: Option<bool>,
    merge_factor: Option<u32>,
    max_segments: Option<u32>,
    default_fields: Vec<String>,
}

impl Default for LexicalIndexConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl LexicalIndexConfigBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            analyzer: None,
            max_docs_per_segment: None,
            write_buffer_size: None,
            compress_stored_fields: None,
            store_term_vectors: None,
            merge_factor: None,
            max_segments: None,
            default_fields: Vec::new(),
        }
    }

    /// Set the analyzer for text fields.
    ///
    /// Use `PerFieldAnalyzer` for field-specific analyzers.
    /// This is analogous to `VectorIndexConfigBuilder::embedder()`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use sarissa::lexical::engine::config::LexicalIndexConfig;
    /// use sarissa::analysis::analyzer::standard::StandardAnalyzer;
    /// use sarissa::analysis::analyzer::per_field::PerFieldAnalyzer;
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

    /// Add a default field to search when no field is specified.
    pub fn default_field(mut self, field: impl Into<String>) -> Self {
        let field = field.into();
        if !self.default_fields.contains(&field) {
            self.default_fields.push(field);
        }
        self
    }

    /// Set the default fields to search when no field is specified.
    ///
    /// This replaces any previously set default fields.
    pub fn default_fields(mut self, fields: Vec<String>) -> Self {
        self.default_fields = fields;
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
        if !self.default_fields.is_empty() {
            config.default_fields = self.default_fields;
        }

        LexicalIndexConfig::Inverted(config)
    }
}
