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
use crate::error::{Result, YatagarasuError};
#[cfg(feature = "embeddings-openai")]
use crate::vector::Vector;

/// Request structure for OpenAI Embeddings API.
#[cfg(feature = "embeddings-openai")]
#[derive(Debug, Serialize)]
struct EmbeddingRequest {
    /// Model identifier to use for embeddings.
    model: String,
    /// Input texts to embed (batch).
    input: Vec<String>,
    /// Optional custom dimension (only for newer models).
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<usize>,
}

/// Response structure from OpenAI Embeddings API.
#[cfg(feature = "embeddings-openai")]
#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    /// List of embedding data objects.
    data: Vec<EmbeddingData>,
}

/// Individual embedding data from API response.
#[cfg(feature = "embeddings-openai")]
#[derive(Debug, Deserialize)]
struct EmbeddingData {
    /// The embedding vector.
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
/// use yatagarasu::embedding::text_embedder::TextEmbedder;
/// use yatagarasu::embedding::openai_text_embedder::OpenAITextEmbedder;
///
/// # async fn example() -> yatagarasu::error::Result<()> {
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
    /// HTTP client for making API requests.
    client: Client,
    /// OpenAI API key for authentication.
    api_key: String,
    /// OpenAI model name (e.g., "text-embedding-3-small").
    model: String,
    /// Dimension of the output embeddings.
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
    /// use yatagarasu::embedding::openai_text_embedder::OpenAITextEmbedder;
    ///
    /// # fn example() -> yatagarasu::error::Result<()> {
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
        // Validate model
        match model.as_str() {
            "text-embedding-3-small" | "text-embedding-3-large" | "text-embedding-ada-002" => {}
            _ => {
                return Err(YatagarasuError::InvalidOperation(format!(
                    "Unknown OpenAI embedding model: {}. Supported models: \
                     text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002",
                    model
                )));
            }
        }

        let dimension = Self::default_dimension(&model);

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
    /// use yatagarasu::embedding::openai_text_embedder::OpenAITextEmbedder;
    ///
    /// # fn example() -> yatagarasu::error::Result<()> {
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

    /// Get the default dimension for a given model.
    ///
    /// Returns the standard embedding dimension for each OpenAI model.
    ///
    /// # Arguments
    ///
    /// * `model` - The model name
    ///
    /// # Returns
    ///
    /// Default dimension: 1536 for small/ada-002, 3072 for large
    fn default_dimension(model: &str) -> usize {
        match model {
            "text-embedding-3-small" => 1536,
            "text-embedding-3-large" => 3072,
            "text-embedding-ada-002" => 1536,
            _ => 1536, // fallback
        }
    }
}

#[cfg(feature = "embeddings-openai")]
#[async_trait]
impl TextEmbedder for OpenAITextEmbedder {
    /// Generate an embedding vector for the given text using OpenAI API.
    ///
    /// Makes a single API request to OpenAI's embeddings endpoint.
    ///
    /// # Arguments
    ///
    /// * `text` - The text to embed
    ///
    /// # Returns
    ///
    /// A vector representation from OpenAI's model
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - API request fails
    /// - Authentication fails
    /// - Response parsing fails
    async fn embed(&self, text: &str) -> Result<Vector> {
        let dimensions = if self.dimension == Self::default_dimension(&self.model) {
            None
        } else {
            Some(self.dimension)
        };

        let request = EmbeddingRequest {
            model: self.model.clone(),
            input: vec![text.to_string()],
            dimensions,
        };

        let http_response = self
            .client
            .post("https://api.openai.com/v1/embeddings")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| {
                YatagarasuError::InvalidOperation(format!("OpenAI API request failed: {}", e))
            })?;

        let status = http_response.status();
        let response_text = http_response.text().await.map_err(|e| {
            YatagarasuError::InvalidOperation(format!("Failed to read response text: {}", e))
        })?;

        if !status.is_success() {
            return Err(YatagarasuError::InvalidOperation(format!(
                "OpenAI API error (status {}): {}",
                status, response_text
            )));
        }

        let response: EmbeddingResponse = serde_json::from_str(&response_text).map_err(|e| {
            YatagarasuError::InvalidOperation(format!(
                "Failed to parse OpenAI response: {}. Response text: {}",
                e, response_text
            ))
        })?;

        let embedding = response
            .data
            .into_iter()
            .next()
            .ok_or_else(|| YatagarasuError::InvalidOperation("No embedding in response".to_string()))?
            .embedding;

        Ok(Vector::new(embedding))
    }

    /// Generate embeddings for multiple texts in a single batch request.
    ///
    /// This is more efficient than calling `embed` multiple times, as it makes
    /// a single API request for all texts.
    ///
    /// # Arguments
    ///
    /// * `texts` - A slice of text strings to embed
    ///
    /// # Returns
    ///
    /// A vector of embeddings, one for each input text
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - API request fails
    /// - Authentication fails
    /// - Response parsing fails
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vector>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let dimensions = if self.dimension == Self::default_dimension(&self.model) {
            None
        } else {
            Some(self.dimension)
        };

        let request = EmbeddingRequest {
            model: self.model.clone(),
            input: texts.iter().map(|s| s.to_string()).collect(),
            dimensions,
        };

        let http_response = self
            .client
            .post("https://api.openai.com/v1/embeddings")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| {
                YatagarasuError::InvalidOperation(format!("OpenAI API request failed: {}", e))
            })?;

        let status = http_response.status();
        let response_text = http_response.text().await.map_err(|e| {
            YatagarasuError::InvalidOperation(format!("Failed to read response text: {}", e))
        })?;

        if !status.is_success() {
            return Err(YatagarasuError::InvalidOperation(format!(
                "OpenAI API error (status {}): {}",
                status, response_text
            )));
        }

        let response: EmbeddingResponse = serde_json::from_str(&response_text).map_err(|e| {
            YatagarasuError::InvalidOperation(format!(
                "Failed to parse OpenAI response: {}. Response text: {}",
                e, response_text
            ))
        })?;

        Ok(response
            .data
            .into_iter()
            .map(|d| Vector::new(d.embedding))
            .collect())
    }

    /// Get the dimension of generated embeddings.
    ///
    /// Returns the configured dimension for this embedder, which may be
    /// customized using `with_dimension` or defaults to the model's standard size.
    ///
    /// # Returns
    ///
    /// The number of dimensions in the embedding vectors
    fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the name/identifier of this embedder.
    ///
    /// Returns the OpenAI model identifier.
    ///
    /// # Returns
    ///
    /// The model name (e.g., "text-embedding-3-small")
    fn name(&self) -> &str {
        &self.model
    }
}
