//! Image embedding trait for Yatagarasu's multimodal vector search.

use async_trait::async_trait;

use crate::error::Result;
use crate::vector::core::vector::Vector;

/// Trait for converting images to vector embeddings.
///
/// This trait provides a common interface for various image embedding methods
/// (neural models, API-based services, etc.) to integrate with Yatagarasu's
/// multimodal vector search pipeline.
///
/// # Examples
///
/// ## Using Candle CLIP embedder (requires `embeddings-multimodal` feature)
///
/// ```no_run
/// use yatagarasu::embedding::image_embedder::ImageEmbedder;
/// use yatagarasu::embedding::candle_multimodal_embedder::CandleMultimodalEmbedder;
///
/// # async fn example() -> yatagarasu::error::Result<()> {
/// let embedder = CandleMultimodalEmbedder::new(
///     "openai/clip-vit-base-patch32"
/// )?;
///
/// let vector = embedder.embed("path/to/image.jpg").await?;
/// println!("Dimension: {}", ImageEmbedder::dimension(&embedder));
/// # Ok(())
/// # }
/// ```
///
/// ## Custom implementation
///
/// ```
/// use async_trait::async_trait;
/// use yatagarasu::embedding::image_embedder::ImageEmbedder;
/// use yatagarasu::error::Result;
/// use yatagarasu::vector::core::vector::Vector;
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
}
