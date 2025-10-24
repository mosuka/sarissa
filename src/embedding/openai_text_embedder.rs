//! OpenAI API-based text embedder implementation.
//!
//! This module provides a text embedder using OpenAI's Embeddings API.
//! Requires the `embeddings-openai` feature to be enabled.

#[cfg(feature = "embeddings-openai")]
use async_trait::async_trait;
#[cfg(feature = "embeddings-openai")]
use reqwest::Client;
#[cfg(feature = "embeddings-openai")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "embeddings-openai")]
use crate::embedding::text_embedder::TextEmbedder;
#[cfg(feature = "embeddings-openai")]
use crate::error::{Result, SageError};
#[cfg(feature = "embeddings-openai")]
use crate::vector::Vector;

#[cfg(feature = "embeddings-openai")]
#[derive(Debug, Serialize)]
struct EmbeddingRequest {
    model: String,
    input: Vec<String>,
}

#[cfg(feature = "embeddings-openai")]
#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[cfg(feature = "embeddings-openai")]
#[derive(Debug, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

/// OpenAI API-based text embedder.
///
/// This embedder uses OpenAI's Embeddings API to generate high-quality
/// text embeddings. Requires an API key and internet connection.
///
/// # Features
///
/// - State-of-the-art embedding quality
/// - Multiple model options
/// - Batch processing support
/// - Easy to use (no model management)
///
/// # Cost Considerations
///
/// OpenAI charges per token processed. Consider using `CandleTextEmbedder`
/// for cost-sensitive applications or when processing large volumes.
///
/// # Examples
///
/// ```no_run
/// use sage::embedding::text_embedder::TextEmbedder;
/// use sage::embedding::openai_text_embedder::OpenAITextEmbedder;
///
/// # async fn example() -> sage::error::Result<()> {
/// // Create embedder with API key
/// let embedder = OpenAITextEmbedder::new(
///     std::env::var("OPENAI_API_KEY").unwrap(),
///     "text-embedding-3-small".to_string()
/// )?;
///
/// // Generate embedding
/// let vector = embedder.embed("Rust is awesome!").await?;
/// println!("Embedding dimension: {}", embedder.dimension());
///
/// // Batch processing (more efficient)
/// let texts = vec!["Hello", "World"];
/// let vectors = embedder.embed_batch(&texts).await?;
/// # Ok(())
/// # }
/// ```
#[cfg(feature = "embeddings-openai")]
pub struct OpenAITextEmbedder {
    client: Client,
    api_key: String,
    model: String,
    dimension: usize,
}

#[cfg(feature = "embeddings-openai")]
impl OpenAITextEmbedder {
    /// Create a new OpenAI embedder.
    ///
    /// # Arguments
    ///
    /// * `api_key` - OpenAI API key (get from https://platform.openai.com/api-keys)
    /// * `model` - Model name to use
    ///
    /// # Supported Models
    ///
    /// - `text-embedding-3-small` - 1536 dimensions, fast and cost-effective
    /// - `text-embedding-3-large` - 3072 dimensions, highest quality
    /// - `text-embedding-ada-002` - 1536 dimensions, legacy model
    ///
    /// # Returns
    ///
    /// A new `OpenAITextEmbedder` instance
    ///
    /// # Errors
    ///
    /// Returns an error if the model name is not recognized
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use sage::embedding::openai_text_embedder::OpenAITextEmbedder;
    ///
    /// # fn example() -> sage::error::Result<()> {
    /// // Small model (recommended for most use cases)
    /// let embedder = OpenAITextEmbedder::new(
    ///     "sk-...".to_string(),
    ///     "text-embedding-3-small".to_string()
    /// )?;
    ///
    /// // Large model (best quality)
    /// let embedder = OpenAITextEmbedder::new(
    ///     "sk-...".to_string(),
    ///     "text-embedding-3-large".to_string()
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(api_key: String, model: String) -> Result<Self> {
        let dimension = match model.as_str() {
            "text-embedding-3-small" => 1536,
            "text-embedding-3-large" => 3072,
            "text-embedding-ada-002" => 1536,
            _ => {
                return Err(SageError::InvalidOperation(format!(
                    "Unknown OpenAI embedding model: {}. Supported models: \
                     text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002",
                    model
                )));
            }
        };

        Ok(Self {
            client: Client::new(),
            api_key,
            model,
            dimension,
        })
    }

    /// Create an embedder with a custom dimension.
    ///
    /// OpenAI's newer models support custom dimensions for reduced cost and storage.
    ///
    /// # Arguments
    ///
    /// * `api_key` - OpenAI API key
    /// * `model` - Model name
    /// * `dimension` - Custom dimension size (must be supported by the model)
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use sage::embedding::openai_text_embedder::OpenAITextEmbedder;
    ///
    /// # fn example() -> sage::error::Result<()> {
    /// // Use smaller dimension for cost savings
    /// let embedder = OpenAITextEmbedder::with_dimension(
    ///     "sk-...".to_string(),
    ///     "text-embedding-3-small".to_string(),
    ///     512  // Reduced from 1536
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_dimension(api_key: String, model: String, dimension: usize) -> Result<Self> {
        Ok(Self {
            client: Client::new(),
            api_key,
            model,
            dimension,
        })
    }
}

#[cfg(feature = "embeddings-openai")]
#[async_trait]
impl TextEmbedder for OpenAITextEmbedder {
    async fn embed(&self, text: &str) -> Result<Vector> {
        let request = EmbeddingRequest {
            model: self.model.clone(),
            input: vec![text.to_string()],
        };

        let response = self
            .client
            .post("https://api.openai.com/v1/embeddings")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| SageError::InvalidOperation(format!("OpenAI API request failed: {}", e)))?
            .json::<EmbeddingResponse>()
            .await
            .map_err(|e| {
                SageError::InvalidOperation(format!("Failed to parse OpenAI response: {}", e))
            })?;

        let embedding = response
            .data
            .into_iter()
            .next()
            .ok_or_else(|| SageError::InvalidOperation("No embedding in response".to_string()))?
            .embedding;

        Ok(Vector::new(embedding))
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vector>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let request = EmbeddingRequest {
            model: self.model.clone(),
            input: texts.iter().map(|s| s.to_string()).collect(),
        };

        let response = self
            .client
            .post("https://api.openai.com/v1/embeddings")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| SageError::InvalidOperation(format!("OpenAI API request failed: {}", e)))?
            .json::<EmbeddingResponse>()
            .await
            .map_err(|e| {
                SageError::InvalidOperation(format!("Failed to parse OpenAI response: {}", e))
            })?;

        Ok(response
            .data
            .into_iter()
            .map(|d| Vector::new(d.embedding))
            .collect())
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn name(&self) -> &str {
        &self.model
    }
}
