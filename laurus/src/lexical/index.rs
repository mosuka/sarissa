//! Lexical indexing module for building and maintaining lexical indexes.
//!
//! This module provides the [`LexicalIndex`] trait, an index factory, configuration,
//! the core inverted index implementation, and supporting data structures
//! (term dictionaries, posting lists, segments).
//!
//! # Module Structure
//!
//! - [`config`] - Index configuration (storage mode, analyzer, merge policy, etc.)
//! - [`factory`] - Index factory for creating and opening indexes
//! - [`inverted`] - Inverted index implementation (including segments and maintenance)
//! - [`structures`] - Low-level data structures (term dictionaries, posting lists, skip lists)

use std::sync::Arc;

use crate::error::Result;
use crate::lexical::index::inverted::InvertedIndexStats;
use crate::lexical::reader::LexicalIndexReader;
use crate::lexical::search::searcher::LexicalSearcher;
use crate::lexical::writer::LexicalIndexWriter;
use crate::storage::Storage;

/// Trait for lexical index implementations.
///
/// This trait defines the high-level interface for lexical indexes.
/// Different index types (Inverted, ColumnStore, LSMTree, etc.) implement this trait
/// to provide their specific functionality while maintaining a common interface.
pub trait LexicalIndex: Send + Sync + std::fmt::Debug {
    /// Get a reader for this index.
    ///
    /// Returns a reader that can be used to query the index.
    fn reader(&self) -> Result<Arc<dyn LexicalIndexReader>>;

    /// Get a writer for this index.
    ///
    /// Returns a writer that can be used to add documents to the index.
    fn writer(&self) -> Result<Box<dyn LexicalIndexWriter>>;

    /// Get the storage backend for this index.
    ///
    /// Returns a reference to the underlying storage.
    fn storage(&self) -> &Arc<dyn Storage>;

    /// Close the index and release resources.
    ///
    /// This should flush any pending writes and release all resources.
    /// Uses interior mutability for thread-safe access.
    fn close(&self) -> Result<()>;

    /// Check if the index is closed.
    ///
    /// Returns true if the index has been closed.
    fn is_closed(&self) -> bool;

    /// Get index statistics.
    ///
    /// Returns statistics about the index such as document count, term count, etc.
    fn stats(&self) -> Result<InvertedIndexStats>;

    /// Optimize the index (merge segments, etc.).
    ///
    /// Performs index optimization such as merging segments to improve query performance.
    /// Uses interior mutability for thread-safe access.
    /// Optimize the index (merge segments, etc.).
    ///
    /// Performs index optimization such as merging segments to improve query performance.
    /// Uses interior mutability for thread-safe access.
    fn optimize(&self) -> Result<()>;

    /// Auto-merge hook invoked after each commit (Issue #755).
    ///
    /// Implementations may opportunistically merge segments to keep their
    /// number bounded (e.g. when it exceeds a configured threshold), without a
    /// manual [`optimize()`](Self::optimize). The default is a no-op; the
    /// inverted index merges its smallest segments when the count exceeds
    /// `max_segments`. Must be cheap when no merge is needed.
    fn maybe_merge(&self) -> Result<()> {
        Ok(())
    }

    /// Refresh the index metadata from storage.
    ///
    /// Should be called after external writes (e.g. by a Writer) to ensure
    /// the index state (like document count) is up-to-date.
    fn refresh(&self) -> Result<()> {
        Ok(())
    }

    /// Create a searcher tailored for this index implementation.
    ///
    /// Returns a boxed [`LexicalSearcher`] capable of executing search/count operations.
    fn searcher(&self) -> Result<Box<dyn LexicalSearcher>>;

    /// Get the default fields configured for this index.
    fn default_fields(&self) -> Result<Vec<String>> {
        Ok(Vec::new())
    }

    /// Get the last processed WAL sequence number.
    fn last_wal_seq(&self) -> u64 {
        0
    }

    /// Set the last processed WAL sequence number.
    fn set_last_wal_seq(&self, _seq: u64) -> Result<()> {
        Ok(())
    }

    /// Dynamically add a new field to the index at runtime.
    ///
    /// After this call, subsequent writers created via [`writer()`](Self::writer)
    /// will include the new field in their configuration, enabling indexing of
    /// documents that contain this field.
    ///
    /// # Arguments
    ///
    /// * `name` - The field name
    /// * `option` - The field configuration (e.g., indexed, stored, term_vectors)
    ///
    /// # Errors
    ///
    /// Returns an error if the index implementation does not support dynamic field
    /// addition.
    fn add_field(
        &self,
        _name: &str,
        _option: crate::lexical::core::field::FieldOption,
    ) -> Result<()> {
        Err(crate::error::LaurusError::invalid_argument(
            "This index implementation does not support dynamic field addition",
        ))
    }

    /// Dynamically remove a field from the index at runtime.
    ///
    /// Only fields that were dynamically added via [`add_field`](Self::add_field)
    /// can be removed. Fields defined in the initial index configuration are not
    /// affected at the index level (though the engine-level schema will no longer
    /// list them).
    ///
    /// # Arguments
    ///
    /// * `name` - The field name to remove
    ///
    /// # Errors
    ///
    /// Returns an error if the index implementation does not support dynamic field
    /// deletion.
    fn delete_field(&self, _name: &str) -> Result<()> {
        Err(crate::error::LaurusError::invalid_argument(
            "This index implementation does not support dynamic field deletion",
        ))
    }
}

pub mod config;
pub mod factory;

pub mod inverted;
pub mod structures;
