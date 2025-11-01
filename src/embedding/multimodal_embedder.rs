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
/// use yatagarasu::embedding::multimodal_embedder::MultimodalEmbedder;
/// use yatagarasu::embedding::text_embedder::TextEmbedder;
/// use yatagarasu::embedding::image_embedder::ImageEmbedder;
/// use yatagarasu::embedding::candle_multimodal_embedder::CandleMultimodalEmbedder;
///
/// # async fn example() -> yatagarasu::error::Result<()> {
/// let embedder = CandleMultimodalEmbedder::new("openai/clip-vit-base-patch32")?;
///
/// // Embed text query
/// let query_vector = TextEmbedder::embed(&embedder, "a cat sitting on a mat").await?;
///
/// // Embed image
/// let image_vector = ImageEmbedder::embed(&embedder, "cat.jpg").await?;
///
/// // They're in the same vector space, so you can compare them
/// # Ok(())
/// # }
/// ```
///
/// ## Image-to-Image Search
///
/// ```no_run
/// use yatagarasu::embedding::multimodal_embedder::MultimodalEmbedder;
/// use yatagarasu::embedding::image_embedder::ImageEmbedder;
/// use yatagarasu::embedding::candle_multimodal_embedder::CandleMultimodalEmbedder;
///
/// # async fn example() -> yatagarasu::error::Result<()> {
/// let embedder = CandleMultimodalEmbedder::new("openai/clip-vit-base-patch32")?;
///
/// // Use an image as query
/// let query_vector = ImageEmbedder::embed(&embedder, "query.jpg").await?;
///
/// // Search against other images
/// let similar_image = ImageEmbedder::embed(&embedder, "similar.jpg").await?;
/// # Ok(())
/// # }
/// ```
pub trait MultimodalEmbedder: TextEmbedder + ImageEmbedder {
    /// Verify that text and image embeddings have the same dimension.
    ///
    /// This method is automatically implemented and will always return true
    /// for valid multimodal embedders. It exists as a sanity check to confirm
    /// that both text and image embeddings are in the same vector space.
    ///
    /// # Returns
    ///
    /// Always `true` for valid implementations, since the trait bounds ensure
    /// that `TextEmbedder::dimension()` and `ImageEmbedder::dimension()` must
    /// be the same method.
    fn is_compatible(&self) -> bool {
        // Both traits require the same dimension() method
        // The compiler ensures they return the same value
        true
    }
}

/// Blanket implementation: any type that implements both TextEmbedder and ImageEmbedder
/// is automatically a MultimodalEmbedder.
///
/// This enables any embedder that can handle both text and images to be used
/// for cross-modal search without additional implementation work.
impl<T> MultimodalEmbedder for T where T: TextEmbedder + ImageEmbedder {}
