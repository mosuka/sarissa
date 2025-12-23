//! Unified embedder trait for vector indexing.
//!
//! This module provides the `Embedder` trait, which serves as a unified interface
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
//! # Symmetry with Analyzer
//!
//! This design mirrors `Analyzer` in the lexical module:
//!
//! | Lexical | Vector |
//! |---------|--------|
//! | `Analyzer` | `Embedder` |
//! | `PerFieldAnalyzer` | `PerFieldEmbedder` |
//! | `NoOpAnalyzer` | `NoOpEmbedder` |
//! | `analyze(text) -> TokenStream` | `embed(input) -> Vector` |
//!
//! # Example
//!
//! ```no_run
//! # #[cfg(feature = "embeddings-candle")]
//! # {
//! use platypus::embedding::embedder::{Embedder, EmbedInput};
//! use platypus::embedding::per_field::PerFieldEmbedder;
//! use platypus::embedding::candle_text_embedder::CandleTextEmbedder;
//! use std::sync::Arc;
//!
//! # async fn example() -> platypus::error::Result<()> {
//! let text_embedder = Arc::new(
//!     CandleTextEmbedder::new("sentence-transformers/all-MiniLM-L6-v2")?
//! );
//!
//! let per_field = PerFieldEmbedder::new(text_embedder);
//!
//! // Use per_field as an Embedder
//! let embedder: Arc<dyn Embedder> = Arc::new(per_field);
//!
//! // Embed text
//! let vector = embedder.embed(&EmbedInput::Text("Hello, world!")).await?;
//! # Ok(())
//! # }
//! # }
//! ```

use std::any::Any;
use std::fmt::Debug;
use std::io::Write;

use async_trait::async_trait;
use tempfile::NamedTempFile;

use crate::error::{PlatypusError, Result};
use crate::vector::core::vector::Vector;

/// Input types for embedding operations.
///
/// This enum represents the different types of input that can be embedded.
/// Embedders declare which input types they support via `supported_input_types()`.
#[derive(Debug, Clone)]
pub enum EmbedInput<'a> {
    /// Text input for text embedding.
    Text(&'a str),

    /// Image file path for image embedding.
    ImagePath(&'a str),

    /// Raw image bytes for image embedding.
    /// The optional string is a MIME type hint (e.g., "image/png").
    ImageBytes(&'a [u8], Option<&'a str>),

    /// URI reference (file:// or path) for image embedding.
    ImageUri(&'a str),
}

impl<'a> EmbedInput<'a> {
    /// Get the input type of this input.
    pub fn input_type(&self) -> EmbedInputType {
        match self {
            EmbedInput::Text(_) => EmbedInputType::Text,
            EmbedInput::ImagePath(_) | EmbedInput::ImageBytes(_, _) | EmbedInput::ImageUri(_) => {
                EmbedInputType::Image
            }
        }
    }

    /// Check if this is a text input.
    pub fn is_text(&self) -> bool {
        matches!(self, EmbedInput::Text(_))
    }

    /// Check if this is an image input.
    pub fn is_image(&self) -> bool {
        matches!(
            self,
            EmbedInput::ImagePath(_) | EmbedInput::ImageBytes(_, _) | EmbedInput::ImageUri(_)
        )
    }

    /// Get the text content if this is a text input.
    pub fn as_text(&self) -> Option<&'a str> {
        match self {
            EmbedInput::Text(text) => Some(text),
            _ => None,
        }
    }

    /// Get the image path if this is an image path input.
    pub fn as_image_path(&self) -> Option<&'a str> {
        match self {
            EmbedInput::ImagePath(path) => Some(path),
            _ => None,
        }
    }
}

/// Types of input that an embedder can support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EmbedInputType {
    /// Text input support.
    Text,

    /// Image input support.
    Image,
}

/// Unified embedder trait for vector indexing.
///
/// This trait provides a common interface for embedders that can be used
/// with `VectorIndex`, similar to how `Analyzer` is used with `LexicalIndex`.
///
/// # Supported Input Types
///
/// Embedders declare which input types they support:
/// - Text-only embedders: `[EmbedInputType::Text]`
/// - Image-only embedders: `[EmbedInputType::Image]`
/// - Multimodal embedders (e.g., CLIP): `[EmbedInputType::Text, EmbedInputType::Image]`
///
/// # Thread Safety
///
/// Implementations must be `Send + Sync` to support concurrent embedding
/// operations across multiple threads.
///
/// # Example
///
/// ```
/// use async_trait::async_trait;
/// use platypus::embedding::embedder::{Embedder, EmbedInput, EmbedInputType};
/// use platypus::error::{PlatypusError, Result};
/// use platypus::vector::core::vector::Vector;
///
/// #[derive(Debug)]
/// struct MyTextEmbedder {
///     dimension: usize,
/// }
///
/// #[async_trait]
/// impl Embedder for MyTextEmbedder {
///     async fn embed(&self, input: &EmbedInput<'_>) -> Result<Vector> {
///         match input {
///             EmbedInput::Text(text) => {
///                 // Generate embedding from text
///                 Ok(Vector::new(vec![0.0; self.dimension]))
///             }
///             _ => Err(PlatypusError::invalid_argument(
///                 "this embedder only supports text input"
///             )),
///         }
///     }
///     fn supported_input_types(&self) -> Vec<EmbedInputType> {
///         vec![EmbedInputType::Text]
///     }
///
///     fn name(&self) -> &str {
///         "my-text-embedder"
///     }
///
///     fn as_any(&self) -> &dyn std::any::Any {
///         self
///     }
/// }
/// ```
#[async_trait]
pub trait Embedder: Send + Sync + Debug {
    /// Generate an embedding vector for the given input.
    ///
    /// # Arguments
    ///
    /// * `input` - The input to embed (text, image path, image bytes, etc.)
    ///
    /// # Returns
    ///
    /// A vector representation of the input, or an error if the input type
    /// is not supported or embedding fails.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The input type is not supported by this embedder
    /// - The embedding operation fails (e.g., model error, file not found)
    async fn embed(&self, input: &EmbedInput<'_>) -> Result<Vector>;

    /// Generate embeddings for multiple inputs in batch.
    ///
    /// The default implementation calls `embed` sequentially.
    /// Override this method for better performance with batch processing.
    ///
    /// # Arguments
    ///
    /// * `inputs` - A slice of inputs to embed
    ///
    /// # Returns
    ///
    /// A vector of embeddings, one for each input
    async fn embed_batch(&self, inputs: &[EmbedInput<'_>]) -> Result<Vec<Vector>> {
        let mut results = Vec::with_capacity(inputs.len());
        for input in inputs {
            results.push(self.embed(input).await?);
        }
        Ok(results)
    }

    /// Get the input types supported by this embedder.
    ///
    /// # Returns
    ///
    /// A vector of supported input types
    fn supported_input_types(&self) -> Vec<EmbedInputType>;

    /// Check if this embedder supports the given input type.
    ///
    /// # Arguments
    ///
    /// * `input_type` - The input type to check
    ///
    /// # Returns
    ///
    /// `true` if the embedder supports this input type
    fn supports(&self, input_type: EmbedInputType) -> bool {
        self.supported_input_types().contains(&input_type)
    }

    /// Check if this embedder supports text input.
    fn supports_text(&self) -> bool {
        self.supports(EmbedInputType::Text)
    }

    /// Check if this embedder supports image input.
    fn supports_image(&self) -> bool {
        self.supports(EmbedInputType::Image)
    }

    /// Check if this embedder is multimodal (supports both text and image).
    fn is_multimodal(&self) -> bool {
        self.supports_text() && self.supports_image()
    }

    /// Get the name/identifier of this embedder.
    ///
    /// This is useful for logging and debugging purposes.
    ///
    /// # Returns
    ///
    /// A string identifying the embedder (e.g., model name)
    fn name(&self) -> &str {
        "unknown"
    }

    /// Downcast support for runtime type identification.
    ///
    /// This enables downcasting to concrete types when needed.
    fn as_any(&self) -> &dyn Any;
}

// =============================================================================
// Helper functions for embedder implementations
// =============================================================================

/// Helper function to embed image bytes by writing to a temporary file.
///
/// This is useful for embedders that only support file path input.
pub async fn embed_image_bytes_via_temp_file<E>(
    embedder: &E,
    bytes: &[u8],
    _mime: Option<&str>,
) -> Result<Vector>
where
    E: Embedder + ?Sized,
{
    let mut temp_file = NamedTempFile::new().map_err(|err| {
        PlatypusError::internal(format!(
            "failed to create temporary file for image embedding: {err}"
        ))
    })?;
    temp_file.write_all(bytes).map_err(|err| {
        PlatypusError::internal(format!("failed to write temporary image payload: {err}"))
    })?;
    let temp_path = temp_file.into_temp_path();
    let path_buf = temp_path.to_path_buf();
    let path_string = path_buf.to_str().ok_or_else(|| {
        PlatypusError::invalid_argument("temporary image path contains invalid UTF-8 characters")
    })?;
    let vector = embedder.embed(&EmbedInput::ImagePath(path_string)).await?;
    drop(temp_path);
    Ok(vector)
}

/// Helper function to embed an image URI.
///
/// This handles `file://` URIs and plain file paths.
pub async fn embed_image_uri<E>(embedder: &E, uri: &str) -> Result<Vector>
where
    E: Embedder + ?Sized,
{
    if uri.starts_with("http://") || uri.starts_with("https://") {
        return Err(PlatypusError::invalid_argument(
            "remote HTTP(S) URIs are not supported yet—download to disk first",
        ));
    }
    let path = uri.strip_prefix("file://").unwrap_or(uri);
    embedder.embed(&EmbedInput::ImagePath(path)).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct MockTextEmbedder {
        dimension: usize,
    }

    #[async_trait]
    impl Embedder for MockTextEmbedder {
        async fn embed(&self, input: &EmbedInput<'_>) -> Result<Vector> {
            match input {
                EmbedInput::Text(_) => Ok(Vector::new(vec![0.0; self.dimension])),
                _ => Err(PlatypusError::invalid_argument(
                    "this embedder only supports text input",
                )),
            }
        }

        fn supported_input_types(&self) -> Vec<EmbedInputType> {
            vec![EmbedInputType::Text]
        }

        fn name(&self) -> &str {
            "mock-text"
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    #[derive(Debug)]
    struct MockMultimodalEmbedder;

    #[async_trait]
    impl Embedder for MockMultimodalEmbedder {
        async fn embed(&self, input: &EmbedInput<'_>) -> Result<Vector> {
            match input {
                EmbedInput::Text(_) | EmbedInput::ImagePath(_) => Ok(Vector::new(vec![0.0; 3])),
                _ => Err(PlatypusError::invalid_argument("unsupported input type")),
            }
        }

        fn supported_input_types(&self) -> Vec<EmbedInputType> {
            vec![EmbedInputType::Text, EmbedInputType::Image]
        }

        fn name(&self) -> &str {
            "mock-multimodal"
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    #[test]
    fn test_embed_input_type() {
        assert_eq!(EmbedInput::Text("hello").input_type(), EmbedInputType::Text);
        assert_eq!(
            EmbedInput::ImagePath("/path/to/image.jpg").input_type(),
            EmbedInputType::Image
        );
        assert_eq!(
            EmbedInput::ImageBytes(&[0, 1, 2], None).input_type(),
            EmbedInputType::Image
        );
    }

    #[test]
    fn test_text_embedder_supports() {
        let embedder = MockTextEmbedder { dimension: 384 };

        assert!(embedder.supports_text());
        assert!(!embedder.supports_image());
        assert!(!embedder.is_multimodal());
    }

    #[test]
    fn test_multimodal_embedder_supports() {
        let embedder = MockMultimodalEmbedder;

        assert!(embedder.supports_text());
        assert!(embedder.supports_image());
        assert!(embedder.is_multimodal());
    }

    #[tokio::test]
    async fn test_text_embedder_embed() {
        let embedder = MockTextEmbedder { dimension: 384 };

        // Text input should work
        let result = embedder.embed(&EmbedInput::Text("hello")).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().data.len(), 384);

        // Image input should fail
        let result = embedder.embed(&EmbedInput::ImagePath("/path")).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_multimodal_embedder_embed() {
        let embedder = MockMultimodalEmbedder;

        // Both text and image should work
        let text_result = embedder.embed(&EmbedInput::Text("hello")).await;
        assert!(text_result.is_ok());

        let image_result = embedder.embed(&EmbedInput::ImagePath("/path")).await;
        assert!(image_result.is_ok());
    }
}
