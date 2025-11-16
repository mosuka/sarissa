//! Candle-based text embedder implementation.
//!
//! This module provides a text embedder using HuggingFace Candle framework.
//! Requires the `embeddings-candle` feature to be enabled.

#[cfg(feature = "embeddings-candle")]
use async_trait::async_trait;
#[cfg(feature = "embeddings-candle")]
use candle_core::{DType, Device, Tensor};
#[cfg(feature = "embeddings-candle")]
use candle_nn::VarBuilder;
#[cfg(feature = "embeddings-candle")]
use candle_transformers::models::bert::{BertModel, Config};
#[cfg(feature = "embeddings-candle")]
use hf_hub::api::sync::ApiBuilder;
#[cfg(feature = "embeddings-candle")]
use tokenizers::Tokenizer;

#[cfg(feature = "embeddings-candle")]
use crate::embedding::text_embedder::TextEmbedder;
#[cfg(feature = "embeddings-candle")]
use crate::error::{PlatypusError, Result};
#[cfg(feature = "embeddings-candle")]
use crate::vector::core::vector::Vector;

/// Candle-based text embedder using BERT models from HuggingFace.
///
/// This embedder uses the Candle framework to run BERT models locally,
/// providing high-quality embeddings without external API dependencies.
///
/// # Features
///
/// - Offline inference (no API calls)
/// - GPU acceleration support
/// - Multiple BERT model support
/// - Fast inference with Rust performance
///
/// # Examples
///
/// ```no_run
/// use platypus::embedding::text_embedder::TextEmbedder;
/// use platypus::embedding::candle_text_embedder::CandleTextEmbedder;
///
/// # async fn example() -> platypus::error::Result<()> {
/// // Create embedder with a sentence-transformers model
/// let embedder = CandleTextEmbedder::new(
///     "sentence-transformers/all-MiniLM-L6-v2"
/// )?;
///
/// // Generate embedding
/// let vector = embedder.embed("Rust is awesome!").await?;
/// println!("Embedding dimension: {}", embedder.dimension());
///
/// // Batch processing
/// let texts = vec!["Hello", "World"];
/// let vectors = embedder.embed_batch(&texts).await?;
/// # Ok(())
/// # }
/// ```
#[cfg(feature = "embeddings-candle")]
pub struct CandleTextEmbedder {
    /// The BERT model for generating embeddings.
    model: BertModel,
    /// Tokenizer for converting text to token IDs.
    tokenizer: Tokenizer,
    /// Device to run the model on (CPU or GPU).
    device: Device,
    /// Dimension of the output embeddings.
    dimension: usize,
    /// Name of the HuggingFace model.
    model_name: String,
}

#[cfg(feature = "embeddings-candle")]
impl CandleTextEmbedder {
    /// Create a new Candle-based embedder from a HuggingFace model.
    ///
    /// The model will be automatically downloaded from HuggingFace Hub if not cached.
    ///
    /// # Arguments
    ///
    /// * `model_name` - HuggingFace model identifier (e.g., "sentence-transformers/all-MiniLM-L6-v2")
    ///
    /// # Returns
    ///
    /// A new `CandleTextEmbedder` instance
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Model download fails
    /// - Model loading fails
    /// - Device initialization fails
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use platypus::embedding::candle_text_embedder::CandleTextEmbedder;
    ///
    /// # fn example() -> platypus::error::Result<()> {
    /// // Small and fast model
    /// let embedder = CandleTextEmbedder::new(
    ///     "sentence-transformers/all-MiniLM-L6-v2"
    /// )?;
    ///
    /// // Larger, more accurate model
    /// let embedder = CandleTextEmbedder::new(
    ///     "sentence-transformers/all-mpnet-base-v2"
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(model_name: &str) -> Result<Self> {
        // Setup device (prefer GPU if available)
        let device = Device::cuda_if_available(0)
            .map_err(|e| PlatypusError::InvalidOperation(format!("Device setup failed: {}", e)))?;

        // Download model from HuggingFace Hub with proper cache directory
        let cache_dir = std::env::var("HF_HOME")
            .or_else(|_| std::env::var("HOME").map(|home| format!("{}/.cache/huggingface", home)))
            .unwrap_or_else(|_| "/tmp/huggingface".to_string());

        let api = ApiBuilder::new()
            .with_cache_dir(cache_dir.into())
            .build()
            .map_err(|e| {
                PlatypusError::InvalidOperation(format!("HF API initialization failed: {}", e))
            })?;
        let repo = api.model(model_name.to_string());

        // Load config
        let config_filename = repo.get("config.json").map_err(|e| {
            PlatypusError::InvalidOperation(format!("Config download failed: {}", e))
        })?;
        let config_str = std::fs::read_to_string(config_filename)
            .map_err(|e| PlatypusError::InvalidOperation(format!("Config read failed: {}", e)))?;
        let config: Config = serde_json::from_str(&config_str)
            .map_err(|e| PlatypusError::InvalidOperation(format!("Config parse failed: {}", e)))?;

        // Load weights
        let weights_filename = repo.get("model.safetensors").map_err(|e| {
            PlatypusError::InvalidOperation(format!("Weights download failed: {}", e))
        })?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_filename], DType::F32, &device).map_err(
                |e| PlatypusError::InvalidOperation(format!("VarBuilder creation failed: {}", e)),
            )?
        };

        // Load model
        let model = BertModel::load(vb, &config)
            .map_err(|e| PlatypusError::InvalidOperation(format!("Model load failed: {}", e)))?;

        // Load tokenizer
        let tokenizer_filename = repo.get("tokenizer.json").map_err(|e| {
            PlatypusError::InvalidOperation(format!("Tokenizer download failed: {}", e))
        })?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(|e| {
            PlatypusError::InvalidOperation(format!("Tokenizer load failed: {}", e))
        })?;

        let dimension = config.hidden_size;

        Ok(Self {
            model,
            tokenizer,
            device,
            dimension,
            model_name: model_name.to_string(),
        })
    }

    /// Perform mean pooling over token embeddings.
    ///
    /// This method averages the token embeddings using the attention mask to ignore
    /// padding tokens. Mean pooling is a common technique for generating sentence
    /// embeddings from token-level BERT outputs.
    ///
    /// # Arguments
    ///
    /// * `embeddings` - Token embeddings from BERT model (shape: [batch_size, seq_len, hidden_dim])
    /// * `attention_mask` - Attention mask indicating valid tokens (shape: [batch_size, seq_len])
    ///
    /// # Returns
    ///
    /// Mean-pooled embeddings (shape: [batch_size, hidden_dim])
    ///
    /// # How it works
    ///
    /// 1. Expand attention mask to match embedding dimensions
    /// 2. Multiply embeddings by mask to zero out padding tokens
    /// 3. Sum embeddings across sequence dimension
    /// 4. Divide by sum of mask values to get mean
    fn mean_pool(&self, embeddings: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        // Expand attention mask to match embedding dimensions
        let mask_expanded = attention_mask
            .unsqueeze(2)
            .map_err(|e| PlatypusError::InvalidOperation(e.to_string()))?
            .expand(embeddings.shape())
            .map_err(|e| PlatypusError::InvalidOperation(e.to_string()))?
            .to_dtype(embeddings.dtype())
            .map_err(|e| PlatypusError::InvalidOperation(e.to_string()))?;

        // Multiply embeddings by mask
        let masked_embeddings = embeddings
            .mul(&mask_expanded)
            .map_err(|e| PlatypusError::InvalidOperation(e.to_string()))?;

        // Sum across sequence dimension
        let sum_embeddings = masked_embeddings
            .sum(1)
            .map_err(|e| PlatypusError::InvalidOperation(e.to_string()))?;

        // Sum mask values
        let sum_mask = mask_expanded
            .sum(1)
            .map_err(|e| PlatypusError::InvalidOperation(e.to_string()))?;

        // Divide to get mean
        let mean = sum_embeddings
            .div(&sum_mask)
            .map_err(|e| PlatypusError::InvalidOperation(e.to_string()))?;

        Ok(mean)
    }
}

#[cfg(feature = "embeddings-candle")]
#[async_trait]
impl TextEmbedder for CandleTextEmbedder {
    /// Generate an embedding vector for the given text.
    ///
    /// This method performs the following steps:
    /// 1. Tokenize the input text
    /// 2. Convert tokens to tensors
    /// 3. Run BERT forward pass
    /// 4. Apply mean pooling over token embeddings
    /// 5. L2 normalize the result
    ///
    /// # Arguments
    ///
    /// * `text` - The text to embed
    ///
    /// # Returns
    ///
    /// A normalized vector representation of the input text
    async fn embed(&self, text: &str) -> Result<Vector> {
        // Tokenize
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| PlatypusError::InvalidOperation(format!("Tokenization failed: {}", e)))?;

        let token_ids = encoding.get_ids();
        let attention_mask = encoding.get_attention_mask();

        // Convert to tensors
        let token_ids_tensor = Tensor::new(token_ids, &self.device)
            .map_err(|e| PlatypusError::InvalidOperation(format!("Tensor creation failed: {}", e)))?
            .unsqueeze(0)
            .map_err(|e| PlatypusError::InvalidOperation(e.to_string()))?;

        let attention_mask_tensor = Tensor::new(attention_mask, &self.device)
            .map_err(|e| PlatypusError::InvalidOperation(e.to_string()))?
            .unsqueeze(0)
            .map_err(|e| PlatypusError::InvalidOperation(e.to_string()))?;

        // Forward pass
        let embeddings = self
            .model
            .forward(&token_ids_tensor, &attention_mask_tensor, None)
            .map_err(|e| PlatypusError::InvalidOperation(format!("Model forward failed: {}", e)))?;

        // Mean pooling
        let pooled = self.mean_pool(&embeddings, &attention_mask_tensor)?;

        // Normalize (L2 normalization)
        // Calculate L2 norm: sqrt(sum(x^2))
        let norm = pooled
            .sqr()
            .map_err(|e| PlatypusError::InvalidOperation(e.to_string()))?
            .sum_all()
            .map_err(|e| PlatypusError::InvalidOperation(e.to_string()))?
            .sqrt()
            .map_err(|e| PlatypusError::InvalidOperation(e.to_string()))?
            .to_scalar::<f32>()
            .map_err(|e| PlatypusError::InvalidOperation(e.to_string()))?;

        // Divide by norm to normalize
        let normalized = pooled
            .affine((1.0 / norm) as f64, 0.0)
            .map_err(|e| PlatypusError::InvalidOperation(e.to_string()))?;

        // Convert to Vector
        let vector_data: Vec<f32> = normalized
            .squeeze(0)
            .map_err(|e| PlatypusError::InvalidOperation(e.to_string()))?
            .to_vec1()
            .map_err(|e| PlatypusError::InvalidOperation(e.to_string()))?;

        Ok(Vector::new(vector_data))
    }

    /// Get the dimension of generated embeddings.
    ///
    /// Returns the hidden size of the BERT model, which is typically
    /// 384 for MiniLM models and 768 for base BERT models.
    ///
    /// # Returns
    ///
    /// The number of dimensions in the embedding vectors
    fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the name/identifier of this embedder.
    ///
    /// Returns the HuggingFace model identifier used to create this embedder.
    ///
    /// # Returns
    ///
    /// The model name (e.g., "sentence-transformers/all-MiniLM-L6-v2")
    fn name(&self) -> &str {
        &self.model_name
    }
}
