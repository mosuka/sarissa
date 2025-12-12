//! No-operation embedder for pre-computed vectors.
//!
//! This module provides a [`NoOpEmbedder`] that does not perform any actual
//! embedding. It is used when documents already contain pre-computed vectors
//! and no embedding is needed during indexing or querying.
//!
//! # Usage
//!
//! Use `NoOpEmbedder` when:
//! - Documents already contain pre-computed vectors
//! - No embedding is needed during indexing or querying
//! - You want to explicitly indicate that embedding is not supported
//!
//! # Example
//!
//! ```rust,ignore
//! use std::sync::Arc;
//! use platypus::embedding::noop::NoOpEmbedder;
//! use platypus::vector::index::config::VectorIndexConfig;
//!
//! let config = VectorIndexConfig::builder()
//!     .embedder(Arc::new(NoOpEmbedder::new()))
//!     .field("body_vector", field_config)
//!     .build()?;
//! ```
//!
//! # Behavior
//!
//! - [`NoOpEmbedder::get_text_embedder()`] always returns `None`
//! - [`NoOpEmbedder::get_image_embedder()`] always returns `None`
//! - [`NoOpEmbedder::supports_text()`] always returns `false`
//! - [`NoOpEmbedder::supports_image()`] always returns `false`
//!
//! If you attempt to index a document with text or image segments (not
//! pre-computed vectors), an error will be returned because no embedder
//! is available.

use std::any::Any;
use std::sync::Arc;

use crate::embedding::embedder::Embedder;
use crate::embedding::image_embedder::ImageEmbedder;
use crate::embedding::text_embedder::TextEmbedder;

/// A no-operation embedder that does not support text or image embedding.
///
/// This embedder is used when documents already contain pre-computed vectors
/// and no embedding is needed. It implements the Null Object Pattern, providing
/// a concrete implementation that explicitly does nothing.
///
/// # Example
///
/// ```rust
/// use std::sync::Arc;
/// use platypus::embedding::noop::NoOpEmbedder;
/// use platypus::embedding::embedder::Embedder;
///
/// let embedder = Arc::new(NoOpEmbedder::new());
/// assert!(!embedder.supports_text("any_field"));
/// assert!(!embedder.supports_image("any_field"));
/// ```
///
/// # When to Use
///
/// - **Pre-computed vectors**: When vectors are computed externally before indexing
/// - **Testing**: When embedding functionality is not needed in tests
/// - **Default placeholder**: As a default value when embedder is required but not used
#[derive(Debug, Clone, Default)]
pub struct NoOpEmbedder;

impl NoOpEmbedder {
    /// Creates a new `NoOpEmbedder`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use platypus::embedding::noop::NoOpEmbedder;
    ///
    /// let embedder = NoOpEmbedder::new();
    /// ```
    pub fn new() -> Self {
        Self
    }
}

impl Embedder for NoOpEmbedder {
    /// Returns `None` for any field, as this embedder does not support text embedding.
    fn get_text_embedder(&self, _field: &str) -> Option<Arc<dyn TextEmbedder>> {
        None
    }

    /// Returns `None` for any field, as this embedder does not support image embedding.
    fn get_image_embedder(&self, _field: &str) -> Option<Arc<dyn ImageEmbedder>> {
        None
    }

    /// Returns `None` for any field, as this embedder has no defined text dimension.
    fn text_dimension(&self, _field: &str) -> Option<usize> {
        None
    }

    /// Returns `None` for any field, as this embedder has no defined image dimension.
    fn image_dimension(&self, _field: &str) -> Option<usize> {
        None
    }

    /// Returns `false` for any field, as this embedder does not support text embedding.
    fn supports_text(&self, _field: &str) -> bool {
        false
    }

    /// Returns `false` for any field, as this embedder does not support image embedding.
    fn supports_image(&self, _field: &str) -> bool {
        false
    }

    /// Returns a reference to self as `&dyn Any` for downcasting.
    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let embedder = NoOpEmbedder::new();
        assert!(embedder.get_text_embedder("any").is_none());
        assert!(embedder.get_image_embedder("any").is_none());
    }

    #[test]
    fn test_default() {
        let embedder = NoOpEmbedder::default();
        assert!(embedder.get_text_embedder("field").is_none());
    }

    #[test]
    fn test_supports_text() {
        let embedder = NoOpEmbedder::new();
        assert!(!embedder.supports_text("any_field"));
        assert!(!embedder.supports_text(""));
        assert!(!embedder.supports_text("body"));
    }

    #[test]
    fn test_supports_image() {
        let embedder = NoOpEmbedder::new();
        assert!(!embedder.supports_image("any_field"));
        assert!(!embedder.supports_image(""));
        assert!(!embedder.supports_image("image"));
    }

    #[test]
    fn test_text_dimension() {
        let embedder = NoOpEmbedder::new();
        assert!(embedder.text_dimension("any").is_none());
    }

    #[test]
    fn test_image_dimension() {
        let embedder = NoOpEmbedder::new();
        assert!(embedder.image_dimension("any").is_none());
    }

    #[test]
    fn test_as_any() {
        let embedder = NoOpEmbedder::new();
        let any = embedder.as_any();
        assert!(any.downcast_ref::<NoOpEmbedder>().is_some());
    }

    #[test]
    fn test_clone() {
        let embedder1 = NoOpEmbedder::new();
        let embedder2 = embedder1.clone();
        assert!(!embedder2.supports_text("field"));
    }

    #[test]
    fn test_debug() {
        let embedder = NoOpEmbedder::new();
        let debug_str = format!("{:?}", embedder);
        assert_eq!(debug_str, "NoOpEmbedder");
    }

    #[test]
    fn test_arc_embedder() {
        let embedder: Arc<dyn Embedder> = Arc::new(NoOpEmbedder::new());
        assert!(!embedder.supports_text("field"));
        assert!(!embedder.supports_image("field"));
        assert!(embedder.get_text_embedder("field").is_none());
        assert!(embedder.get_image_embedder("field").is_none());
    }
}
