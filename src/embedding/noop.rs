//! No-operation embedder for pre-computed vectors.
//!
//! This module provides a [`NoOpEmbedder`] that does not perform any actual
//! embedding. It is used when documents already contain pre-computed vectors
//! and no embedding is needed during indexing or querying.
//!
//! This is analogous to `NoOpAnalyzer` in the lexical module.
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
//!     .embedder(NoOpEmbedder::new())
//!     .field("body_vector", field_config)
//!     .build()?;
//! ```
//!
//! # Behavior
//!
//! - [`NoOpEmbedder::embed()`] returns an error for any input
//! - [`NoOpEmbedder::supports_text()`] always returns `false`
//! - [`NoOpEmbedder::supports_image()`] always returns `false`
//!
//! If you attempt to embed text or images, an error will be returned.

use std::any::Any;

use async_trait::async_trait;

use crate::embedding::embedder::{EmbedInput, EmbedInputType, Embedder};
use crate::error::{PlatypusError, Result};
use crate::vector::core::vector::Vector;

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
/// let embedder: Arc<dyn Embedder> = Arc::new(NoOpEmbedder::new());
/// assert!(!embedder.supports_text());
/// assert!(!embedder.supports_image());
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

#[async_trait]
impl Embedder for NoOpEmbedder {
    /// Returns an error for any input, as this embedder does not support embedding.
    async fn embed(&self, _input: &EmbedInput<'_>) -> Result<Vector> {
        Err(PlatypusError::invalid_argument(
            "NoOpEmbedder does not support embedding - use pre-computed vectors",
        ))
    }

    /// Returns an empty list as this embedder does not support any input types.
    fn supported_input_types(&self) -> Vec<EmbedInputType> {
        vec![]
    }

    /// Returns `false` as this embedder does not support text input.
    fn supports_text(&self) -> bool {
        false
    }

    /// Returns `false` as this embedder does not support image input.
    fn supports_image(&self) -> bool {
        false
    }

    fn name(&self) -> &str {
        "NoOpEmbedder"
    }

    /// Returns a reference to self as `&dyn Any` for downcasting.
    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_new() {
        let embedder = NoOpEmbedder::new();
        assert!(!embedder.supports_text());
        assert!(!embedder.supports_image());
    }

    #[test]
    fn test_supports_text() {
        let embedder = NoOpEmbedder::new();
        assert!(!embedder.supports_text());
    }

    #[test]
    fn test_supports_image() {
        let embedder = NoOpEmbedder::new();
        assert!(!embedder.supports_image());
    }

    #[test]
    fn test_supported_input_types() {
        let embedder = NoOpEmbedder::new();
        assert!(embedder.supported_input_types().is_empty());
    }

    #[test]
    fn test_is_multimodal() {
        let embedder = NoOpEmbedder::new();
        assert!(!embedder.is_multimodal());
    }

    #[tokio::test]
    async fn test_embed_returns_error() {
        let embedder = NoOpEmbedder::new();

        let result = embedder.embed(&EmbedInput::Text("hello")).await;
        assert!(result.is_err());

        let result = embedder.embed(&EmbedInput::ImagePath("/path")).await;
        assert!(result.is_err());
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
        assert!(!embedder2.supports_text());
    }

    #[test]
    fn test_debug() {
        let embedder = NoOpEmbedder::new();
        let debug_str = format!("{:?}", embedder);
        assert_eq!(debug_str, "NoOpEmbedder");
    }

    #[test]
    fn test_name() {
        let embedder = NoOpEmbedder::new();
        assert_eq!(embedder.name(), "NoOpEmbedder");
    }

    #[test]
    fn test_arc_embedder() {
        let embedder: Arc<dyn Embedder> = Arc::new(NoOpEmbedder::new());
        assert!(!embedder.supports_text());
        assert!(!embedder.supports_image());
    }
}
