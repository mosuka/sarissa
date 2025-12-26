//! Candle-based multimodal embedder implementation using CLIP.
//!
//! This module provides a multimodal embedder using HuggingFace Candle framework
//! with CLIP (Contrastive Language-Image Pre-Training) models.
//! Requires the `embeddings-multimodal` feature to be enabled.

#[cfg(feature = "embeddings-multimodal")]
use std::any::Any;

#[cfg(feature = "embeddings-multimodal")]
use async_trait::async_trait;
#[cfg(feature = "embeddings-multimodal")]
use candle_core::{DType, Device, Module, Tensor};
#[cfg(feature = "embeddings-multimodal")]
use candle_nn::{Linear, VarBuilder};
#[cfg(feature = "embeddings-multimodal")]
use candle_transformers::models::clip;
#[cfg(feature = "embeddings-multimodal")]
use hf_hub::api::sync::ApiBuilder;
#[cfg(feature = "embeddings-multimodal")]
use tokenizers::Tokenizer;

#[cfg(feature = "embeddings-multimodal")]
use crate::embedding::embedder::{EmbedInput, EmbedInputType, Embedder};
#[cfg(feature = "embeddings-multimodal")]
use crate::error::{SarissaError, Result};
#[cfg(feature = "embeddings-multimodal")]
use crate::vector::core::vector::Vector;

/// Candle-based multimodal embedder using CLIP models from HuggingFace.
///
/// This embedder uses the Candle framework to run CLIP models locally,
/// providing embeddings for both text and images in the same vector space.
///
/// # Features
///
/// - Offline inference (no API calls)
/// - GPU acceleration support
/// - Cross-modal search (text-to-image, image-to-image)
/// - Multiple CLIP model variants support
///
/// # Supported Models
///
/// Note: Use models from the HuggingFace model hub that have CLIP architecture.
/// The default config currently supports `vit-base-patch32` architecture.
///
/// Example repositories:
/// - Models with pre-trained CLIP weights compatible with Candle
/// - Custom fine-tuned CLIP models
///
/// # Examples
///
/// ## Text-to-Image Search
///
/// ```no_run
/// use sarissa::embedding::embedder::{Embedder, EmbedInput};
/// use sarissa::embedding::candle_multimodal_embedder::CandleMultimodalEmbedder;
///
/// # async fn example() -> sarissa::error::Result<()> {
/// let embedder = CandleMultimodalEmbedder::new(
///     "openai/clip-vit-base-patch32"
/// )?;
///
/// // Embed text query
/// let text_vec = embedder.embed(&EmbedInput::Text("a photo of a cat")).await?;
///
/// // Embed images
/// let img1 = embedder.embed(&EmbedInput::ImagePath("cat.jpg")).await?;
/// let img2 = embedder.embed(&EmbedInput::ImagePath("dog.jpg")).await?;
///
/// // Text and images are in the same vector space
/// # Ok(())
/// # }
/// ```
///
/// ## Image-to-Image Search
///
/// ```no_run
/// use sarissa::embedding::embedder::{Embedder, EmbedInput};
/// use sarissa::embedding::candle_multimodal_embedder::CandleMultimodalEmbedder;
///
/// # async fn example() -> sarissa::error::Result<()> {
/// let embedder = CandleMultimodalEmbedder::new(
///     "openai/clip-vit-base-patch32"
/// )?;
///
/// // Find similar images
/// let query = embedder.embed(&EmbedInput::ImagePath("query.jpg")).await?;
/// let inputs = vec![
///     EmbedInput::ImagePath("img1.jpg"),
///     EmbedInput::ImagePath("img2.jpg"),
///     EmbedInput::ImagePath("img3.jpg"),
/// ];
/// let images = embedder.embed_batch(&inputs).await?;
/// # Ok(())
/// # }
/// ```
#[cfg(feature = "embeddings-multimodal")]
pub struct CandleMultimodalEmbedder {
    /// CLIP text transformer model.
    text_model: clip::text_model::ClipTextTransformer,
    /// CLIP vision transformer model.
    vision_model: clip::vision_model::ClipVisionTransformer,
    /// Linear projection layer for text embeddings.
    text_projection: Linear,
    /// Linear projection layer for vision embeddings.
    vision_projection: Linear,
    /// Tokenizer for text input.
    tokenizer: Tokenizer,
    /// Device to run models on (CPU or GPU).
    device: Device,
    /// Dimension of the shared embedding space.
    dimension: usize,
    /// Name of the HuggingFace CLIP model.
    model_name: String,
    /// Expected image size (width/height in pixels).
    image_size: usize,
}

#[cfg(feature = "embeddings-multimodal")]
impl std::fmt::Debug for CandleMultimodalEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CandleMultimodalEmbedder")
            .field("model_name", &self.model_name)
            .field("dimension", &self.dimension)
            .field("image_size", &self.image_size)
            .finish()
    }
}

#[cfg(feature = "embeddings-multimodal")]
impl CandleMultimodalEmbedder {
    /// Create a new Candle-based multimodal embedder from a HuggingFace CLIP model.
    ///
    /// The model will be automatically downloaded from HuggingFace Hub if not cached.
    ///
    /// # Arguments
    ///
    /// * `model_name` - HuggingFace CLIP model identifier (e.g., "openai/clip-vit-base-patch32")
    ///
    /// # Returns
    ///
    /// A new `CandleMultimodalEmbedder` instance
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
    /// use sarissa::embedding::candle_multimodal_embedder::CandleMultimodalEmbedder;
    ///
    /// # fn example() -> sarissa::error::Result<()> {
    /// // Fast and efficient
    /// let embedder = CandleMultimodalEmbedder::new(
    ///     "openai/clip-vit-base-patch32"
    /// )?;
    ///
    /// // Higher quality
    /// let embedder = CandleMultimodalEmbedder::new(
    ///     "openai/clip-vit-large-patch14"
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(model_name: &str) -> Result<Self> {
        // Setup device (prefer GPU if available)
        let device = Device::cuda_if_available(0)
            .map_err(|e| SarissaError::InvalidOperation(format!("Device setup failed: {}", e)))?;

        // Download model from HuggingFace Hub
        let cache_dir = std::env::var("HF_HOME")
            .or_else(|_| std::env::var("HOME").map(|home| format!("{}/.cache/huggingface", home)))
            .unwrap_or_else(|_| "/tmp/huggingface".to_string());

        let api = ApiBuilder::new()
            .with_cache_dir(cache_dir.into())
            .build()
            .map_err(|e| {
                SarissaError::InvalidOperation(format!("HF API initialization failed: {}", e))
            })?;
        let repo = api.model(model_name.to_string());

        // Load config
        // Note: Using default vit_base_patch32 config
        // TODO: Parse config.json for model-specific settings
        let config = clip::ClipConfig::vit_base_patch32();

        // Load weights - try safetensors first, fall back to pytorch
        let weights_filename = repo
            .get("model.safetensors")
            .or_else(|_| repo.get("pytorch_model.bin"))
            .map_err(|e| {
                SarissaError::InvalidOperation(format!("Weights download failed: {}", e))
            })?;

        let vb = if weights_filename.to_string_lossy().ends_with(".safetensors") {
            unsafe {
                VarBuilder::from_mmaped_safetensors(&[weights_filename], DType::F32, &device)
                    .map_err(|e| {
                        SarissaError::InvalidOperation(format!(
                            "VarBuilder creation failed: {}",
                            e
                        ))
                    })?
            }
        } else {
            VarBuilder::from_pth(&weights_filename, DType::F32, &device).map_err(|e| {
                SarissaError::InvalidOperation(format!("VarBuilder creation failed: {}", e))
            })?
        };

        // Load text model
        let text_model =
            clip::text_model::ClipTextTransformer::new(vb.pp("text_model"), &config.text_config)
                .map_err(|e| {
                    SarissaError::InvalidOperation(format!("Text model load failed: {}", e))
                })?;

        // Load vision model
        let vision_model = clip::vision_model::ClipVisionTransformer::new(
            vb.pp("vision_model"),
            &config.vision_config,
        )
        .map_err(|e| SarissaError::InvalidOperation(format!("Vision model load failed: {}", e)))?;

        // Load projection layers
        let projection_dim = config.text_config.projection_dim;

        // CLIP models use linear layers without bias
        let text_projection = candle_nn::linear_no_bias(
            config.text_config.embed_dim,
            projection_dim,
            vb.pp("text_projection"),
        )
        .map_err(|e| {
            SarissaError::InvalidOperation(format!("Text projection load failed: {}", e))
        })?;

        let vision_projection = candle_nn::linear_no_bias(
            config.vision_config.embed_dim,
            projection_dim,
            vb.pp("visual_projection"),
        )
        .map_err(|e| {
            SarissaError::InvalidOperation(format!("Vision projection load failed: {}", e))
        })?;

        // Load tokenizer
        let tokenizer_filename = repo.get("tokenizer.json").map_err(|e| {
            SarissaError::InvalidOperation(format!("Tokenizer download failed: {}", e))
        })?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(|e| {
            SarissaError::InvalidOperation(format!("Tokenizer load failed: {}", e))
        })?;

        let dimension = projection_dim;
        let image_size = config.vision_config.image_size;

        Ok(Self {
            text_model,
            vision_model,
            text_projection,
            vision_projection,
            tokenizer,
            device,
            dimension,
            model_name: model_name.to_string(),
            image_size,
        })
    }

    /// Embed text using CLIP text encoder.
    async fn embed_text(&self, text: &str) -> Result<Vector> {
        // Tokenize
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| SarissaError::InvalidOperation(format!("Tokenization failed: {}", e)))?;

        let token_ids = encoding.get_ids();

        // Convert to tensor
        let token_ids_tensor = Tensor::new(token_ids, &self.device)
            .map_err(|e| SarissaError::InvalidOperation(format!("Tensor creation failed: {}", e)))?
            .unsqueeze(0)
            .map_err(|e| SarissaError::InvalidOperation(e.to_string()))?;

        // Forward pass through text model
        let text_features = self.text_model.forward(&token_ids_tensor).map_err(|e| {
            SarissaError::InvalidOperation(format!("Text model forward failed: {}", e))
        })?;

        // Project to common embedding space
        let projected = self.text_projection.forward(&text_features).map_err(|e| {
            SarissaError::InvalidOperation(format!("Text projection failed: {}", e))
        })?;

        // Normalize
        let normalized = self.normalize(&projected)?;

        // Convert to Vector
        let vector_data: Vec<f32> = normalized
            .squeeze(0)
            .map_err(|e| SarissaError::InvalidOperation(e.to_string()))?
            .to_vec1()
            .map_err(|e| SarissaError::InvalidOperation(e.to_string()))?;

        Ok(Vector::new(vector_data))
    }

    /// Embed image using CLIP vision encoder.
    async fn embed_image(&self, image_path: &str) -> Result<Vector> {
        // Preprocess image
        let image_tensor = self.preprocess_image(image_path)?;

        // Forward pass through vision model
        let vision_features = self.vision_model.forward(&image_tensor).map_err(|e| {
            SarissaError::InvalidOperation(format!("Vision model forward failed: {}", e))
        })?;

        // Project to common embedding space
        let projected = self
            .vision_projection
            .forward(&vision_features)
            .map_err(|e| {
                SarissaError::InvalidOperation(format!("Vision projection failed: {}", e))
            })?;

        // Normalize
        let normalized = self.normalize(&projected)?;

        // Convert to Vector
        let vector_data: Vec<f32> = normalized
            .squeeze(0)
            .map_err(|e| SarissaError::InvalidOperation(e.to_string()))?
            .to_vec1()
            .map_err(|e| SarissaError::InvalidOperation(e.to_string()))?;

        Ok(Vector::new(vector_data))
    }

    /// Preprocess image to the format expected by CLIP vision model.
    ///
    /// This method performs standard CLIP image preprocessing:
    /// 1. Load image from file
    /// 2. Resize to expected size (e.g., 224x224)
    /// 3. Convert to RGB
    /// 4. Normalize using ImageNet mean/std
    /// 5. Convert to tensor format (C, H, W)
    ///
    /// # Arguments
    ///
    /// * `image_path` - Path to the image file
    ///
    /// # Returns
    ///
    /// Preprocessed image tensor ready for CLIP vision model
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Image file cannot be opened
    /// - Image cannot be decoded
    /// - Tensor creation fails
    fn preprocess_image(&self, image_path: &str) -> Result<Tensor> {
        use image::{DynamicImage, ImageReader};

        // Load image
        let img_reader = ImageReader::open(image_path)
            .map_err(|e| SarissaError::InvalidOperation(format!("Image open failed: {}", e)))?
            .with_guessed_format()
            .map_err(|e| {
                SarissaError::InvalidOperation(format!("Image format guess failed: {}", e))
            })?;

        let img = img_reader
            .decode()
            .map_err(|e| SarissaError::InvalidOperation(format!("Image decode failed: {}", e)))?;

        // Resize to model's expected size
        let img = img.resize_exact(
            self.image_size as u32,
            self.image_size as u32,
            image::imageops::FilterType::Triangle,
        );

        // Convert to RGB
        let img = match img {
            DynamicImage::ImageRgb8(img) => img,
            img => img.to_rgb8(),
        };

        // Convert to tensor (C, H, W format)
        let img_data = img.into_raw();
        let img_tensor = Tensor::from_vec(
            img_data,
            (self.image_size, self.image_size, 3),
            &self.device,
        )
        .map_err(|e| SarissaError::InvalidOperation(format!("Tensor creation failed: {}", e)))?;

        // Normalize: (pixel / 255.0 - mean) / std
        // CLIP uses ImageNet normalization
        let mean = Tensor::new(&[0.48145466f32, 0.4578275, 0.40821073], &self.device)
            .map_err(|e| SarissaError::InvalidOperation(e.to_string()))?
            .reshape((1, 1, 3))
            .map_err(|e| SarissaError::InvalidOperation(e.to_string()))?;
        let std = Tensor::new(&[0.2686295_f32, 0.2613026, 0.2757771], &self.device)
            .map_err(|e| SarissaError::InvalidOperation(e.to_string()))?
            .reshape((1, 1, 3))
            .map_err(|e| SarissaError::InvalidOperation(e.to_string()))?;

        // Scale to [0, 1] and normalize
        let normalized = img_tensor
            .to_dtype(DType::F32)
            .map_err(|e| SarissaError::InvalidOperation(e.to_string()))?
            .affine(1.0 / 255.0, 0.0)
            .map_err(|e| SarissaError::InvalidOperation(e.to_string()))?
            .broadcast_sub(&mean)
            .map_err(|e| SarissaError::InvalidOperation(e.to_string()))?
            .broadcast_div(&std)
            .map_err(|e| SarissaError::InvalidOperation(e.to_string()))?;

        // Permute to (C, H, W)
        let normalized = normalized
            .permute((2, 0, 1))
            .map_err(|e| SarissaError::InvalidOperation(e.to_string()))?
            .unsqueeze(0)
            .map_err(|e| SarissaError::InvalidOperation(e.to_string()))?;

        Ok(normalized)
    }

    /// Normalize embeddings using L2 normalization.
    ///
    /// Divides each embedding vector by its L2 norm, ensuring all vectors
    /// have unit length. This is standard for CLIP embeddings to enable
    /// cosine similarity via dot product.
    ///
    /// # Arguments
    ///
    /// * `tensor` - Embeddings to normalize (shape: [batch_size, dimension])
    ///
    /// # Returns
    ///
    /// L2-normalized embeddings with the same shape
    fn normalize(&self, tensor: &Tensor) -> Result<Tensor> {
        let norm = tensor
            .sqr()
            .map_err(|e| SarissaError::InvalidOperation(e.to_string()))?
            .sum_keepdim(1)
            .map_err(|e| SarissaError::InvalidOperation(e.to_string()))?
            .sqrt()
            .map_err(|e| SarissaError::InvalidOperation(e.to_string()))?;

        tensor
            .broadcast_div(&norm)
            .map_err(|e| SarissaError::InvalidOperation(e.to_string()))
    }
}

#[cfg(feature = "embeddings-multimodal")]
#[async_trait]
impl Embedder for CandleMultimodalEmbedder {
    /// Generate an embedding vector for the given input.
    ///
    /// Supports both text and image inputs. Text and images are embedded into
    /// the same vector space, enabling cross-modal similarity search.
    ///
    /// # Arguments
    ///
    /// * `input` - The input to embed (text or image)
    ///
    /// # Returns
    ///
    /// A normalized vector representation in CLIP's shared embedding space
    async fn embed(&self, input: &EmbedInput<'_>) -> Result<Vector> {
        match input {
            EmbedInput::Text(text) => self.embed_text(text).await,
            EmbedInput::ImagePath(path) => self.embed_image(path).await,
            EmbedInput::ImageBytes(_, _) => Err(SarissaError::invalid_argument(
                "CandleMultimodalEmbedder does not yet support image bytes input",
            )),
            EmbedInput::ImageUri(_) => Err(SarissaError::invalid_argument(
                "CandleMultimodalEmbedder does not support image URI input",
            )),
        }
    }

    /// Get the supported input types.
    ///
    /// CandleMultimodalEmbedder supports both text and image inputs.
    fn supported_input_types(&self) -> Vec<EmbedInputType> {
        vec![EmbedInputType::Text, EmbedInputType::Image]
    }

    /// Get the name/identifier of this embedder.
    ///
    /// Returns the HuggingFace CLIP model identifier.
    fn name(&self) -> &str {
        &self.model_name
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
