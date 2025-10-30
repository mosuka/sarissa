//! Configuration for lexical index types.

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
/// use yatagarasu::lexical::index::config::{LexicalIndexConfig, InvertedIndexConfig};
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
    /// Get a human-readable name for the index type.
    pub fn index_type_name(&self) -> &str {
        match self {
            LexicalIndexConfig::Inverted(_) => "Inverted",
            // Future index types:
            // LexicalIndexConfig::ColumnStore(_) => "ColumnStore",
            // LexicalIndexConfig::LSMTree(_) => "LSMTree",
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
