//! Text embedding trait for vector search integration.

use async_trait::async_trait;

use crate::error::Result;
use crate::vector::Vector;

/// Trait for converting text to vector embeddings.
///
/// This trait provides a common interface for various embedding methods
/// (neural models, API-based services, etc.) to work with Sage's vector search.
///
/// # Examples
///
/// ## Using Candle embedder (requires `embeddings-candle` feature)
///
/// ```no_run
/// # #[cfg(feature = "embeddings-candle")]
/// # {
/// use yatagarasu::embedding::text_embedder::TextEmbedder;
/// use yatagarasu::embedding::candle_text_embedder::CandleTextEmbedder;
///
/// # async fn example() -> yatagarasu::error::Result<()> {
/// let embedder = CandleTextEmbedder::new(
///     "sentence-transformers/all-MiniLM-L6-v2"
/// )?;
///
/// let vector = embedder.embed("Hello, world!").await?;
/// println!("Dimension: {}", embedder.dimension());
/// # Ok(())
/// # }
/// # }
/// ```
///
/// ## Using OpenAI embedder (requires `embeddings-openai` feature)
///
/// ```no_run
/// # #[cfg(feature = "embeddings-openai")]
/// # {
/// use yatagarasu::embedding::text_embedder::TextEmbedder;
/// use yatagarasu::embedding::openai_text_embedder::OpenAITextEmbedder;
///
/// # async fn example() -> yatagarasu::error::Result<()> {
/// let embedder = OpenAITextEmbedder::new(
///     "your-api-key".to_string(),
///     "text-embedding-3-small".to_string()
/// )?;
///
/// let vector = embedder.embed("Hello, world!").await?;
/// # Ok(())
/// # }
/// # }
/// ```
///
/// ## Custom implementation
///
/// ```
/// use async_trait::async_trait;
/// use yatagarasu::embedding::text_embedder::TextEmbedder;
/// use yatagarasu::error::Result;
/// use yatagarasu::vector::Vector;
///
/// struct MyCustomEmbedder {
///     dimension: usize,
/// }
///
/// #[async_trait]
/// impl TextEmbedder for MyCustomEmbedder {
///     async fn embed(&self, text: &str) -> Result<Vector> {
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
pub trait TextEmbedder: Send + Sync {
    /// Generate an embedding vector for the given text.
    ///
    /// # Arguments
    ///
    /// * `text` - The text to embed
    ///
    /// # Returns
    ///
    /// A vector representation of the input text
    async fn embed(&self, text: &str) -> Result<Vector>;

    /// Generate embeddings for multiple texts in batch.
    ///
    /// The default implementation calls `embed` sequentially.
    /// Override this method for better performance with batch processing.
    ///
    /// # Arguments
    ///
    /// * `texts` - A slice of text strings to embed
    ///
    /// # Returns
    ///
    /// A vector of embeddings, one for each input text
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vector>> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.embed(text).await?);
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
