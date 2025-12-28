//! Embedder for pre-computed vectors.
//!
//! This module provides a [`PrecomputedEmbedder`] that does not perform any actual
//! embedding. It is used when documents already contain pre-computed vectors
//! and no embedding is needed during indexing or querying.
//!
//! This is analogous to `NoOpAnalyzer` in the lexical module.
//!
//! # Usage
//!
//! Use `PrecomputedEmbedder` when:
//! - Documents already contain pre-computed vectors
//! - No embedding is needed during indexing or querying
//! - You want to explicitly indicate that embedding is not supported
//!
//! # Example
//!
//! ```rust,ignore
//! use std::sync::Arc;
//! use sarissa::embedding::precomputed::PrecomputedEmbedder;
//! use sarissa::vector::index::config::VectorIndexConfig;
//!
//! let config = VectorIndexConfig::builder()
//!     .embedder(PrecomputedEmbedder::new())
//!     .field("body_vector", field_config)
//!     .build()?;
//! ```
//!
//! # Behavior
//!
//! - [`PrecomputedEmbedder::embed()`] returns an error for any input
//! - [`PrecomputedEmbedder::supports_text()`] always returns `false`
//! - [`PrecomputedEmbedder::supports_image()`] always returns `false`
//!
//! If you attempt to embed text or images, an error will be returned.

use std::any::Any;

use async_trait::async_trait;

use crate::embedding::embedder::{EmbedInput, EmbedInputType, Embedder};
use crate::error::{Result, SarissaError};
use crate::vector::core::vector::Vector;

/// An embedder that does not support text or image embedding, used for pre-computed vectors.
///
/// This embedder is used when documents already contain pre-computed vectors
/// and no embedding is needed. It implements the Null Object Pattern, providing
/// a concrete implementation that explicitly does nothing.
///
/// # Example
///
/// ```rust
/// use std::sync::Arc;
/// use sarissa::embedding::precomputed::PrecomputedEmbedder;
/// use sarissa::embedding::embedder::Embedder;
///
/// let embedder: Arc<dyn Embedder> = Arc::new(PrecomputedEmbedder::new());
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
pub struct PrecomputedEmbedder;

impl PrecomputedEmbedder {
    /// Creates a new `PrecomputedEmbedder`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use sarissa::embedding::precomputed::PrecomputedEmbedder;
    ///
    /// let embedder = PrecomputedEmbedder::new();
    /// ```
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Embedder for PrecomputedEmbedder {
    /// Returns an error for any input, as this embedder does not support embedding.
    async fn embed(&self, _input: &EmbedInput<'_>) -> Result<Vector> {
        Err(SarissaError::invalid_argument(
            "PrecomputedEmbedder does not support embedding - use pre-computed vectors",
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
        "PrecomputedEmbedder"
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
        let embedder = PrecomputedEmbedder::new();
        assert!(!embedder.supports_text());
        assert!(!embedder.supports_image());
    }

    #[test]
    fn test_supports_text() {
        let embedder = PrecomputedEmbedder::new();
        assert!(!embedder.supports_text());
    }

    #[test]
    fn test_supports_image() {
        let embedder = PrecomputedEmbedder::new();
        assert!(!embedder.supports_image());
    }

    #[test]
    fn test_supported_input_types() {
        let embedder = PrecomputedEmbedder::new();
        assert!(embedder.supported_input_types().is_empty());
    }

    #[test]
    fn test_is_multimodal() {
        let embedder = PrecomputedEmbedder::new();
        assert!(!embedder.is_multimodal());
    }

    #[tokio::test]
    async fn test_embed_returns_error() {
        let embedder = PrecomputedEmbedder::new();

        let result = embedder.embed(&EmbedInput::Text("hello")).await;
        assert!(result.is_err());

        let result = embedder.embed(&EmbedInput::ImagePath("/path")).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_as_any() {
        let embedder = PrecomputedEmbedder::new();
        let any = embedder.as_any();
        assert!(any.downcast_ref::<PrecomputedEmbedder>().is_some());
    }

    #[test]
    fn test_clone() {
        let embedder1 = PrecomputedEmbedder::new();
        let embedder2 = embedder1.clone();
        assert!(!embedder2.supports_text());
    }

    #[test]
    fn test_debug() {
        let embedder = PrecomputedEmbedder::new();
        let debug_str = format!("{:?}", embedder);
        assert_eq!(debug_str, "PrecomputedEmbedder");
    }

    #[test]
    fn test_name() {
        let embedder = PrecomputedEmbedder::new();
        assert_eq!(embedder.name(), "PrecomputedEmbedder");
    }

    #[test]
    fn test_arc_embedder() {
        let embedder: Arc<dyn Embedder> = Arc::new(PrecomputedEmbedder::new());
        assert!(!embedder.supports_text());
        assert!(!embedder.supports_image());
    }
}
