//! Lexical indexing module for building and maintaining lexical indexes.
//!
//! This module handles all lexical index construction:
//! - Building inverted indexes
//! - Document indexing and analysis
//!
//! # Module Structure
//!
//! - `config`: Index configuration
//! - `factory`: Index factory for creating and opening indexes
//! - `inverted`: Inverted index implementation (including segments and maintenance)

use std::sync::Arc;

use crate::error::Result;
use crate::lexical::index::inverted::types::InvertedIndexStats;
use crate::lexical::reader::IndexReader;
use crate::lexical::writer::IndexWriter;
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
    fn reader(&self) -> Result<Box<dyn IndexReader>>;

    /// Get a writer for this index.
    ///
    /// Returns a writer that can be used to add or update documents.
    fn writer(&self) -> Result<Box<dyn IndexWriter>>;

    /// Get the storage backend for this index.
    ///
    /// Returns a reference to the underlying storage.
    fn storage(&self) -> &Arc<dyn Storage>;

    /// Close the index and release resources.
    ///
    /// This should flush any pending writes and release all resources.
    fn close(&mut self) -> Result<()>;

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
    fn optimize(&mut self) -> Result<()>;
}

pub mod config;
pub mod factory;

pub mod inverted;
