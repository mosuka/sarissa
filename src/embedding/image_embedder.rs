//! Image embedding trait for Platypus's multimodal vector search.

use std::io::Write;

use async_trait::async_trait;
use tempfile::NamedTempFile;

use crate::error::{PlatypusError, Result};
use crate::vector::core::vector::Vector;

/// Trait for converting images to vector embeddings.
///
/// This trait provides a common interface for various image embedding methods
/// (neural models, API-based services, etc.) to integrate with Platypus's
/// multimodal vector search pipeline.
///
/// # Examples
///
/// ## Using Candle CLIP embedder (requires `embeddings-multimodal` feature)
///
/// ```no_run
/// # #[cfg(feature = "embeddings-multimodal")]
/// # {
/// use platypus::embedding::image_embedder::ImageEmbedder;
/// use platypus::embedding::candle_multimodal_embedder::CandleMultimodalEmbedder;
///
/// # async fn example() -> platypus::error::Result<()> {
/// let embedder = CandleMultimodalEmbedder::new(
///     "openai/clip-vit-base-patch32"
/// )?;
///
/// let vector = embedder.embed("path/to/image.jpg").await?;
/// println!("Dimension: {}", ImageEmbedder::dimension(&embedder));
/// # Ok(())
/// # }
/// # }
/// ```
///
/// ## Custom implementation
///
/// ```
/// use async_trait::async_trait;
/// use platypus::embedding::image_embedder::ImageEmbedder;
/// use platypus::error::Result;
/// use platypus::vector::core::vector::Vector;
///
/// struct MyCustomImageEmbedder {
///     dimension: usize,
/// }
///
/// #[async_trait]
/// impl ImageEmbedder for MyCustomImageEmbedder {
///     async fn embed(&self, image_path: &str) -> Result<Vector> {
///         // Your custom implementation
///         let embedding = vec![0.0; self.dimension];
///         Ok(Vector::new(embedding))
///     }
///
///     fn dimension(&self) -> usize {
///         self.dimension
///     }
/// }
/// ```
#[async_trait]
pub trait ImageEmbedder: Send + Sync {
    /// Generate an embedding vector for the given image.
    ///
    /// # Arguments
    ///
    /// * `image_path` - Path to the image file to embed
    ///
    /// # Returns
    ///
    /// A vector representation of the input image
    async fn embed(&self, image_path: &str) -> Result<Vector>;

    /// Generate embeddings for multiple images in batch.
    ///
    /// The default implementation calls `embed` sequentially.
    /// Override this method for better performance with batch processing.
    ///
    /// # Arguments
    ///
    /// * `image_paths` - A slice of image file paths to embed
    ///
    /// # Returns
    ///
    /// A vector of embeddings, one for each input image
    async fn embed_batch(&self, image_paths: &[&str]) -> Result<Vec<Vector>> {
        let mut results = Vec::with_capacity(image_paths.len());
        for path in image_paths {
            results.push(self.embed(path).await?);
        }
        Ok(results)
    }

    /// Get the dimension of generated embeddings.
    ///
    /// # Returns
    ///
    /// The number of dimensions in the embedding vectors
    fn dimension(&self) -> usize;

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

    /// Embed an in-memory binary payload by writing it to a temporary file.
    async fn embed_bytes(&self, bytes: &[u8], _mime: Option<&str>) -> Result<Vector> {
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
        let path_string = path_buf
            .to_str()
            .ok_or_else(|| {
                PlatypusError::invalid_argument(
                    "temporary image path contains invalid UTF-8 characters",
                )
            })?
            .to_string();
        let vector = self.embed(path_string.as_str()).await?;
        drop(temp_path);
        Ok(vector)
    }

    /// Embed a resource identified by URI. Currently supports file paths and `file://` URIs.
    async fn embed_uri(&self, uri: &str, _media_hint: Option<&str>) -> Result<Vector> {
        if uri.starts_with("http://") || uri.starts_with("https://") {
            return Err(PlatypusError::invalid_argument(
                "remote HTTP(S) URIs are not supported yet—download to disk first",
            ));
        }
        let path = uri.strip_prefix("file://").unwrap_or(uri);
        self.embed(path).await
    }
}
