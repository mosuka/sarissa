//! Base query trait and common query functionality.

use std::any::Any;
use std::fmt::Debug;

use crate::error::Result;
use crate::full_text::reader::IndexReader;
use crate::query::matcher::Matcher;
use crate::query::scorer::Scorer;

/// Trait for search queries.
pub trait Query: Send + Sync + Debug {
    /// Create a matcher for this query.
    fn matcher(&self, reader: &dyn IndexReader) -> Result<Box<dyn Matcher>>;

    /// Create a scorer for this query.
    fn scorer(&self, reader: &dyn IndexReader) -> Result<Box<dyn Scorer>>;

    /// Get the boost factor for this query.
    fn boost(&self) -> f32;

    /// Set the boost factor for this query.
    fn set_boost(&mut self, boost: f32);

    /// Get a human-readable description of this query.
    fn description(&self) -> String;

    /// Clone this query.
    fn clone_box(&self) -> Box<dyn Query>;

    /// Check if this query matches any documents.
    fn is_empty(&self, reader: &dyn IndexReader) -> Result<bool>;

    /// Get the estimated cost of executing this query.
    fn cost(&self, reader: &dyn IndexReader) -> Result<u64>;

    /// Get this query as Any for downcasting.
    fn as_any(&self) -> &dyn Any;

    /// Get the field name this query searches in, if applicable.
    /// Returns None for queries that don't target a specific field (e.g., BooleanQuery).
    fn field(&self) -> Option<&str> {
        None
    }
}
