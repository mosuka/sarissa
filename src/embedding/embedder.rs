//! Unified embedder trait for vector indexing.
//!
//! This module provides the `Embedder` trait, which serves as a common interface
//! for embedders used with `VectorIndex`. It is analogous to how `Analyzer` is
//! used with `LexicalIndex` in the lexical module.
//!
//! # Design
//!
//! The `Embedder` trait abstracts over different embedding strategies:
//! - Text-only embedders (e.g., BERT, sentence-transformers)
//! - Image-only embedders
//! - Multimodal embedders (e.g., CLIP) that handle both text and images
//! - Per-field embedders that route to different underlying embedders based on field name
//!
//! # Example
//!
//! ```no_run
//! # #[cfg(feature = "embeddings-candle")]
//! # {
//! use platypus::embedding::embedder::Embedder;
//! use platypus::embedding::per_field::PerFieldEmbedder;
//! use platypus::embedding::candle_text_embedder::CandleTextEmbedder;
//! use platypus::embedding::text_embedder::TextEmbedder;
//! use std::sync::Arc;
//!
//! # fn example() -> platypus::error::Result<()> {
//! let text_embedder: Arc<dyn TextEmbedder> = Arc::new(
//!     CandleTextEmbedder::new("sentence-transformers/all-MiniLM-L6-v2")?
//! );
//!
//! let mut per_field = PerFieldEmbedder::with_default_text(text_embedder);
//!
//! // Use per_field as an Embedder
//! let embedder: Arc<dyn Embedder> = Arc::new(per_field);
//! # Ok(())
//! # }
//! # }
//! ```

use std::any::Any;
use std::fmt::Debug;
use std::sync::Arc;

use crate::embedding::image_embedder::ImageEmbedder;
use crate::embedding::text_embedder::TextEmbedder;

/// Unified embedder trait for vector indexing.
///
/// This trait provides a common interface for embedders that can be used
/// with `VectorIndex`, similar to how `Analyzer` is used with `LexicalIndex`.
///
/// Implementations can support text embedding, image embedding, or both.
/// The `PerFieldEmbedder` implementation allows routing to different embedders
/// based on field names.
///
/// # Field-Based Routing
///
/// The primary use case is field-based routing, where different vector fields
/// may use different embedding models:
///
/// - `title_embedding` might use a lightweight model for short text
/// - `body_embedding` might use a larger model for longer content
/// - `image_embedding` might use CLIP or another vision model
///
/// # Thread Safety
///
/// Implementations must be `Send + Sync` to support concurrent embedding
/// operations across multiple threads.
pub trait Embedder: Send + Sync + Debug {
    /// Get the text embedder for a specific field.
    ///
    /// Returns `None` if the field does not support text embedding.
    ///
    /// # Arguments
    ///
    /// * `field` - The field name
    fn get_text_embedder(&self, field: &str) -> Option<Arc<dyn TextEmbedder>>;

    /// Get the image embedder for a specific field.
    ///
    /// Returns `None` if the field does not support image embedding.
    ///
    /// # Arguments
    ///
    /// * `field` - The field name
    fn get_image_embedder(&self, field: &str) -> Option<Arc<dyn ImageEmbedder>>;

    /// Get the embedding dimension for text in a specific field.
    ///
    /// Returns `None` if the field does not support text embedding.
    ///
    /// # Arguments
    ///
    /// * `field` - The field name
    fn text_dimension(&self, field: &str) -> Option<usize> {
        self.get_text_embedder(field).map(|e| e.dimension())
    }

    /// Get the embedding dimension for images in a specific field.
    ///
    /// Returns `None` if the field does not support image embedding.
    ///
    /// # Arguments
    ///
    /// * `field` - The field name
    fn image_dimension(&self, field: &str) -> Option<usize> {
        self.get_image_embedder(field).map(|e| e.dimension())
    }

    /// Check if a field supports text embedding.
    ///
    /// # Arguments
    ///
    /// * `field` - The field name
    fn supports_text(&self, field: &str) -> bool {
        self.get_text_embedder(field).is_some()
    }

    /// Check if a field supports image embedding.
    ///
    /// # Arguments
    ///
    /// * `field` - The field name
    fn supports_image(&self, field: &str) -> bool {
        self.get_image_embedder(field).is_some()
    }

    /// Downcast support for runtime type identification.
    ///
    /// This enables downcasting to concrete types when needed.
    fn as_any(&self) -> &dyn Any;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Result;
    use crate::vector::core::vector::Vector;
    use async_trait::async_trait;

    #[derive(Debug)]
    struct MockTextEmbedder {
        dimension: usize,
    }

    #[async_trait]
    impl TextEmbedder for MockTextEmbedder {
        async fn embed(&self, _text: &str) -> Result<Vector> {
            Ok(Vector::new(vec![0.0; self.dimension]))
        }

        fn dimension(&self) -> usize {
            self.dimension
        }

        fn name(&self) -> &str {
            "mock-text"
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    struct MockEmbedder {
        text_embedder: Arc<dyn TextEmbedder>,
    }

    impl std::fmt::Debug for MockEmbedder {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("MockEmbedder")
                .field("text_embedder_dimension", &self.text_embedder.dimension())
                .finish()
        }
    }

    impl Embedder for MockEmbedder {
        fn get_text_embedder(&self, _field: &str) -> Option<Arc<dyn TextEmbedder>> {
            Some(Arc::clone(&self.text_embedder))
        }

        fn get_image_embedder(&self, _field: &str) -> Option<Arc<dyn ImageEmbedder>> {
            None
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    #[test]
    fn test_embedder_trait() {
        let text_embedder: Arc<dyn TextEmbedder> = Arc::new(MockTextEmbedder { dimension: 384 });
        let embedder = MockEmbedder { text_embedder };

        assert!(embedder.supports_text("any_field"));
        assert!(!embedder.supports_image("any_field"));
        assert_eq!(embedder.text_dimension("any_field"), Some(384));
        assert_eq!(embedder.image_dimension("any_field"), None);
    }
}
