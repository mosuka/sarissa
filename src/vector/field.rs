//! Shared vector field traits bridging collection and index layers.

use std::any::Any;
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::vector::core::document::StoredVector;
use crate::vector::core::vector::Vector;
use crate::vector::engine::{QueryVector, VectorFieldConfig};

/// Represents a single logical vector field backed by an index implementation.
pub trait VectorField: Send + Sync + Debug {
    /// Returns the field (column) name.
    fn name(&self) -> &str;
    /// Returns the immutable configuration for this field.
    fn config(&self) -> &VectorFieldConfig;
    /// Returns the field writer that ingests vectors for this field.
    fn writer(&self) -> &dyn VectorFieldWriter;
    /// Returns the field reader that serves queries for this field.
    fn reader(&self) -> &dyn VectorFieldReader;
    /// Returns a cloneable writer handle for sharing across runtimes.
    fn writer_handle(&self) -> Arc<dyn VectorFieldWriter>;
    /// Returns a cloneable reader handle for sharing across runtimes.
    fn reader_handle(&self) -> Arc<dyn VectorFieldReader>;
    /// Returns a type-erased reference for downcasting to concrete implementations.
    fn as_any(&self) -> &dyn Any;
}

/// Writer interface for ingesting doc-centric vectors into a single field index.
pub trait VectorFieldWriter: Send + Sync + Debug {
    /// Add or replace a vector for the given document and field version.
    fn add_stored_vector(&self, doc_id: u64, vector: &StoredVector, version: u64) -> Result<()>;
    /// Delete the vectors associated with the provided document id.
    fn delete_document(&self, doc_id: u64, version: u64) -> Result<()>;

    /// Check if the writer has storage configured.
    fn has_storage(&self) -> bool;

    /// Get access to the stored vectors with field names.
    fn vectors(&self) -> Vec<(u64, String, Vector)>;

    /// Rebuild the index with the provided vectors, effectively replacing the current content.
    fn rebuild(&self, vectors: Vec<(u64, String, Vector)>) -> Result<()>;

    /// Flush any buffered data to durable storage.
    fn flush(&self) -> Result<()>;
}

/// Reader interface that exposes field-local search/statistics.
pub trait VectorFieldReader: Send + Sync + Debug {
    /// Execute a field-scoped ANN search.
    fn search(&self, request: FieldSearchInput) -> Result<FieldSearchResults>;
    /// Return the latest field statistics (vector count, dimension, ...).
    fn stats(&self) -> Result<VectorFieldStats>;
}

/// Query parameters passed to field-level searchers.
#[derive(Debug, Clone)]
pub struct FieldSearchInput {
    pub field: String,
    pub query_vectors: Vec<QueryVector>,
    pub limit: usize,
}

/// Field-level hits returned by an index.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FieldSearchResults {
    #[serde(default)]
    pub hits: Vec<FieldHit>,
}

/// A single hit originating from a concrete field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldHit {
    pub doc_id: u64,
    pub field: String,
    pub score: f32,
    pub distance: f32,
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

/// Basic statistics collected per field.
#[derive(Debug, Clone, Copy, Default)]
pub struct VectorFieldStats {
    pub vector_count: usize,
    pub dimension: usize,
}
