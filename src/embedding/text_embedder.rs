//! Text embedding trait for Platypus's semantic search pipeline.

use async_trait::async_trait;

use crate::error::Result;
use crate::vector::core::vector::Vector;

/// Trait for converting text to vector embeddings.
///
/// This trait provides a common interface for various embedding methods
/// (local neural models, API-based services, etc.) to plug into Platypus's
/// vector and hybrid search layers.
///
/// # Examples
///
/// ## Using Candle embedder (requires `embeddings-candle` feature)
///
/// ```no_run
/// # #[cfg(feature = "embeddings-candle")]
/// # {
/// use platypus::embedding::text_embedder::TextEmbedder;
/// use platypus::embedding::candle_text_embedder::CandleTextEmbedder;
///
/// # async fn example() -> platypus::error::Result<()> {
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
/// use platypus::embedding::text_embedder::TextEmbedder;
/// use platypus::embedding::openai_text_embedder::OpenAITextEmbedder;
///
/// # async fn example() -> platypus::error::Result<()> {
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
/// use platypus::embedding::text_embedder::TextEmbedder;
/// use platypus::error::Result;
/// use platypus::vector::core::vector::Vector;
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

    /// Generate an embedding vector for the given text with field context.
    ///
    /// This method allows embedders to use field information when generating embeddings.
    /// For example, `PerFieldEmbedder` uses this to select the appropriate embedder
    /// for each field.
    ///
    /// The default implementation ignores the field name and calls `embed`.
    ///
    /// # Arguments
    ///
    /// * `text` - The text to embed
    /// * `field_name` - The name of the field being embedded
    ///
    /// # Returns
    ///
    /// A vector representation of the input text
    async fn embed_with_field(&self, text: &str, _field_name: &str) -> Result<Vector> {
        self.embed(text).await
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

    /// Downcast to Any for dynamic type checking.
    ///
    /// This method enables downcasting from a trait object to a concrete type.
    /// It is primarily used by PerFieldEmbedder and similar wrapper types.
    ///
    /// # Returns
    ///
    /// A reference to self as an Any trait object
    fn as_any(&self) -> &dyn std::any::Any {
        // Default implementation that panics - concrete types should override
        panic!("as_any not implemented for this embedder")
    }
}
