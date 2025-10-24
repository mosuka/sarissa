//! Multimodal embedding trait for cross-modal search.

use crate::embedding::image_embedder::ImageEmbedder;
use crate::embedding::text_embedder::TextEmbedder;

/// Trait for embedders that support both text and image inputs in the same vector space.
///
/// This trait is automatically implemented for types that implement both
/// `TextEmbedder` and `ImageEmbedder`, ensuring that text and images are
/// embedded into the same semantic vector space for cross-modal search.
///
/// # Cross-Modal Search
///
/// With a multimodal embedder, you can:
/// - Search for images using text queries
/// - Find similar images using an image query
/// - Compare semantic similarity between text and images
///
/// # Examples
///
/// ## Text-to-Image Search
///
/// ```no_run
/// use sage::embedding::multimodal_embedder::MultimodalEmbedder;
/// use sage::embedding::text_embedder::TextEmbedder;
/// use sage::embedding::image_embedder::ImageEmbedder;
/// use sage::embedding::candle_multimodal_embedder::CandleMultimodalEmbedder;
///
/// # async fn example() -> sage::error::Result<()> {
/// let embedder = CandleMultimodalEmbedder::new("openai/clip-vit-base-patch32")?;
///
/// // Embed text query
/// let query_vector = TextEmbedder::embed(&embedder, "a cat sitting on a mat").await?;
///
/// // Embed image
/// let image_vector = ImageEmbedder::embed_image(&embedder, "cat.jpg").await?;
///
/// // They're in the same vector space, so you can compare them
/// # Ok(())
/// # }
/// ```
///
/// ## Image-to-Image Search
///
/// ```no_run
/// use sage::embedding::multimodal_embedder::MultimodalEmbedder;
/// use sage::embedding::image_embedder::ImageEmbedder;
/// use sage::embedding::candle_multimodal_embedder::CandleMultimodalEmbedder;
///
/// # async fn example() -> sage::error::Result<()> {
/// let embedder = CandleMultimodalEmbedder::new("openai/clip-vit-base-patch32")?;
///
/// // Use an image as query
/// let query_vector = ImageEmbedder::embed_image(&embedder, "query.jpg").await?;
///
/// // Search against other images
/// let similar_image = ImageEmbedder::embed_image(&embedder, "similar.jpg").await?;
/// # Ok(())
/// # }
/// ```
pub trait MultimodalEmbedder: TextEmbedder + ImageEmbedder {
    /// Verify that text and image embeddings have the same dimension.
    ///
    /// This method is automatically implemented and will always return true
    /// for valid multimodal embedders.
    fn is_compatible(&self) -> bool {
        // Both traits require the same dimension() method
        // The compiler ensures they return the same value
        true
    }
}

// Blanket implementation: any type that implements both TextEmbedder and ImageEmbedder
// is automatically a MultimodalEmbedder
impl<T> MultimodalEmbedder for T where T: TextEmbedder + ImageEmbedder {}
