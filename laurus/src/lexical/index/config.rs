//! Configuration for lexical index types.
//!
//! This module provides configuration types for lexical indexes.

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::analysis::analyzer::analyzer::Analyzer;
use crate::analysis::analyzer::standard::StandardAnalyzer;
use crate::lexical::core::field::FieldOption;

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

    /// Default fields to search when no field is specified.
    #[serde(default)]
    pub default_fields: Vec<String>,

    /// Shard ID for the index.
    #[serde(default)]
    pub shard_id: u16,

    /// Field-specific configurations.
    /// If empty, all fields are indexed with default options (schema-less).
    #[serde(default)]
    pub fields: HashMap<String, FieldOption>,

    /// Maximum number of entries in the snapshot-scoped query / filter result
    /// cache (Issue #578).
    ///
    /// Repeated filter clauses (tenancy, category, status, …) are memoised as
    /// document-id sets so they are evaluated once per reader snapshot instead
    /// of on every request. `0` disables the cache. Defaults to `1024`.
    #[serde(default = "default_query_filter_cache_capacity")]
    pub query_filter_cache_capacity: usize,
}

fn default_analyzer() -> Arc<dyn Analyzer> {
    Arc::new(StandardAnalyzer::new().expect("StandardAnalyzer should be creatable"))
}

fn default_query_filter_cache_capacity() -> usize {
    1024
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
            default_fields: Vec::new(),
            shard_id: 0,
            fields: HashMap::new(),
            query_filter_cache_capacity: default_query_filter_cache_capacity(),
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
            .field("default_fields", &self.default_fields)
            .field(
                "query_filter_cache_capacity",
                &self.query_filter_cache_capacity,
            )
            .finish()
    }
}
