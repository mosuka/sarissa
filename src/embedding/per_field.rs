//! Per-field embedder for applying different embedders to different fields.
//!
//! This module provides `PerFieldEmbedder`, which allows specifying different
//! embedders for different vector fields. It supports both text and image
//! embeddings, making it suitable for multimodal search applications.
//!
//! This is analogous to `PerFieldAnalyzer` in the lexical module.

use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

use crate::embedding::embedder::Embedder;
use crate::embedding::image_embedder::ImageEmbedder;
use crate::embedding::text_embedder::TextEmbedder;
use crate::error::Result;
use crate::vector::core::vector::Vector;

/// A per-field embedder that supports both text and image embeddings.
///
/// Similar to `PerFieldAnalyzer` in the lexical module, this allows specifying
/// different embedders for different vector fields. It supports:
///
/// - Text-only embedders (e.g., BERT, sentence-transformers)
/// - Image-only embedders
/// - Multimodal embedders (e.g., CLIP) that handle both text and images
///
/// # Memory Efficiency
///
/// When using the same embedder for multiple fields, reuse a single instance
/// with `Arc::clone` to save memory. This is especially important for large
/// embedding models.
///
/// # Examples
///
/// ## Text-only embedder
///
/// ```no_run
/// # #[cfg(feature = "embeddings-candle")]
/// # {
/// use platypus::embedding::per_field::PerFieldEmbedder;
/// use platypus::embedding::candle_text_embedder::CandleTextEmbedder;
/// use platypus::embedding::text_embedder::TextEmbedder;
/// use std::sync::Arc;
///
/// # fn example() -> platypus::error::Result<()> {
/// let default_embedder: Arc<dyn TextEmbedder> = Arc::new(
///     CandleTextEmbedder::new("sentence-transformers/all-MiniLM-L6-v2")?
/// );
///
/// let mut per_field = PerFieldEmbedder::with_default_text(default_embedder.clone());
///
/// // Use a specialized embedder for title field
/// let title_embedder: Arc<dyn TextEmbedder> = Arc::new(
///     CandleTextEmbedder::new("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")?
/// );
/// per_field.add_text_embedder("title_embedding", title_embedder);
///
/// // "content_embedding" will use default_embedder
/// // "title_embedding" will use the specialized title_embedder
/// # Ok(())
/// # }
/// # }
/// ```
///
/// ## Multimodal embedder
///
/// ```no_run
/// # #[cfg(feature = "embeddings-multimodal")]
/// # {
/// use platypus::embedding::per_field::PerFieldEmbedder;
/// use platypus::embedding::candle_multimodal_embedder::CandleMultimodalEmbedder;
/// use platypus::embedding::text_embedder::TextEmbedder;
/// use platypus::embedding::image_embedder::ImageEmbedder;
/// use std::sync::Arc;
///
/// # fn example() -> platypus::error::Result<()> {
/// let clip_embedder = Arc::new(
///     CandleMultimodalEmbedder::new("openai/clip-vit-base-patch32")?
/// );
///
/// let mut per_field = PerFieldEmbedder::new();
///
/// // Register CLIP for both text and image on a field
/// per_field.add_multimodal_embedder("content_embedding", clip_embedder);
/// # Ok(())
/// # }
/// # }
/// ```
#[derive(Clone, Default)]
pub struct PerFieldEmbedder {
    /// Text embedders per field.
    text_embedders: HashMap<String, Arc<dyn TextEmbedder>>,

    /// Image embedders per field.
    image_embedders: HashMap<String, Arc<dyn ImageEmbedder>>,

    /// Default text embedder for fields not explicitly configured.
    default_text_embedder: Option<Arc<dyn TextEmbedder>>,

    /// Default image embedder for fields not explicitly configured.
    default_image_embedder: Option<Arc<dyn ImageEmbedder>>,
}

impl std::fmt::Debug for PerFieldEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PerFieldEmbedder")
            .field(
                "text_fields",
                &self.text_embedders.keys().collect::<Vec<_>>(),
            )
            .field(
                "image_fields",
                &self.image_embedders.keys().collect::<Vec<_>>(),
            )
            .field("has_default_text", &self.default_text_embedder.is_some())
            .field("has_default_image", &self.default_image_embedder.is_some())
            .finish()
    }
}

impl PerFieldEmbedder {
    /// Create a new empty per-field embedder.
    ///
    /// Use this when you want to explicitly configure each field without defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a per-field embedder with a default text embedder.
    ///
    /// Fields not explicitly configured will use this default for text embedding.
    ///
    /// # Arguments
    ///
    /// * `embedder` - The default text embedder
    pub fn with_default_text(embedder: Arc<dyn TextEmbedder>) -> Self {
        Self {
            default_text_embedder: Some(embedder),
            ..Default::default()
        }
    }

    /// Create a per-field embedder with a default image embedder.
    ///
    /// Fields not explicitly configured will use this default for image embedding.
    ///
    /// # Arguments
    ///
    /// * `embedder` - The default image embedder
    pub fn with_default_image(embedder: Arc<dyn ImageEmbedder>) -> Self {
        Self {
            default_image_embedder: Some(embedder),
            ..Default::default()
        }
    }

    /// Create a per-field embedder with default embedders for both text and image.
    ///
    /// # Arguments
    ///
    /// * `text_embedder` - The default text embedder
    /// * `image_embedder` - The default image embedder
    pub fn with_defaults(
        text_embedder: Arc<dyn TextEmbedder>,
        image_embedder: Arc<dyn ImageEmbedder>,
    ) -> Self {
        Self {
            default_text_embedder: Some(text_embedder),
            default_image_embedder: Some(image_embedder),
            ..Default::default()
        }
    }

    /// Set the default text embedder.
    ///
    /// # Arguments
    ///
    /// * `embedder` - The default text embedder
    pub fn set_default_text_embedder(&mut self, embedder: Arc<dyn TextEmbedder>) {
        self.default_text_embedder = Some(embedder);
    }

    /// Set the default image embedder.
    ///
    /// # Arguments
    ///
    /// * `embedder` - The default image embedder
    pub fn set_default_image_embedder(&mut self, embedder: Arc<dyn ImageEmbedder>) {
        self.default_image_embedder = Some(embedder);
    }

    /// Add a text embedder for a specific field.
    ///
    /// # Arguments
    ///
    /// * `field` - The field name
    /// * `embedder` - The text embedder to use for this field
    pub fn add_text_embedder(&mut self, field: impl Into<String>, embedder: Arc<dyn TextEmbedder>) {
        self.text_embedders.insert(field.into(), embedder);
    }

    /// Add an image embedder for a specific field.
    ///
    /// # Arguments
    ///
    /// * `field` - The field name
    /// * `embedder` - The image embedder to use for this field
    pub fn add_image_embedder(
        &mut self,
        field: impl Into<String>,
        embedder: Arc<dyn ImageEmbedder>,
    ) {
        self.image_embedders.insert(field.into(), embedder);
    }

    /// Add a multimodal embedder for a specific field.
    ///
    /// This registers the embedder as both a text and image embedder for the field.
    /// Use this with embedders like CLIP that support both modalities.
    ///
    /// # Arguments
    ///
    /// * `field` - The field name
    /// * `embedder` - The multimodal embedder (must implement both TextEmbedder and ImageEmbedder)
    pub fn add_multimodal_embedder<E>(&mut self, field: impl Into<String>, embedder: Arc<E>)
    where
        E: TextEmbedder + ImageEmbedder + 'static,
    {
        let field = field.into();
        self.text_embedders
            .insert(field.clone(), embedder.clone() as Arc<dyn TextEmbedder>);
        self.image_embedders
            .insert(field, embedder as Arc<dyn ImageEmbedder>);
    }

    /// Get the text embedder for a specific field.
    ///
    /// Returns the field-specific embedder if configured, otherwise returns the default.
    /// Returns `None` if no embedder is available for the field.
    ///
    /// # Arguments
    ///
    /// * `field` - The field name
    pub fn get_text_embedder(&self, field: &str) -> Option<Arc<dyn TextEmbedder>> {
        self.text_embedders
            .get(field)
            .cloned()
            .or_else(|| self.default_text_embedder.clone())
    }

    /// Get the image embedder for a specific field.
    ///
    /// Returns the field-specific embedder if configured, otherwise returns the default.
    /// Returns `None` if no embedder is available for the field.
    ///
    /// # Arguments
    ///
    /// * `field` - The field name
    pub fn get_image_embedder(&self, field: &str) -> Option<Arc<dyn ImageEmbedder>> {
        self.image_embedders
            .get(field)
            .cloned()
            .or_else(|| self.default_image_embedder.clone())
    }

    /// Get the default text embedder.
    pub fn default_text_embedder(&self) -> Option<&Arc<dyn TextEmbedder>> {
        self.default_text_embedder.as_ref()
    }

    /// Get the default image embedder.
    pub fn default_image_embedder(&self) -> Option<&Arc<dyn ImageEmbedder>> {
        self.default_image_embedder.as_ref()
    }

    /// Embed text with the embedder for the given field.
    ///
    /// # Arguments
    ///
    /// * `field` - The field name to determine which embedder to use
    /// * `text` - The text to embed
    ///
    /// # Returns
    ///
    /// The embedding vector for the text, or an error if no text embedder is available.
    pub async fn embed_text(&self, field: &str, text: &str) -> Result<Vector> {
        let embedder = self.get_text_embedder(field).ok_or_else(|| {
            crate::error::PlatypusError::invalid_config(format!(
                "no text embedder configured for field '{}'",
                field
            ))
        })?;
        embedder.embed(text).await
    }

    /// Embed an image with the embedder for the given field.
    ///
    /// # Arguments
    ///
    /// * `field` - The field name to determine which embedder to use
    /// * `image_path` - The path to the image file
    ///
    /// # Returns
    ///
    /// The embedding vector for the image, or an error if no image embedder is available.
    pub async fn embed_image(&self, field: &str, image_path: &str) -> Result<Vector> {
        let embedder = self.get_image_embedder(field).ok_or_else(|| {
            crate::error::PlatypusError::invalid_config(format!(
                "no image embedder configured for field '{}'",
                field
            ))
        })?;
        embedder.embed(image_path).await
    }

    /// Get the text embedding dimension for a specific field.
    ///
    /// # Arguments
    ///
    /// * `field` - The field name
    ///
    /// # Returns
    ///
    /// The embedding dimension, or `None` if no text embedder is available.
    pub fn text_dimension(&self, field: &str) -> Option<usize> {
        self.get_text_embedder(field).map(|e| e.dimension())
    }

    /// Get the image embedding dimension for a specific field.
    ///
    /// # Arguments
    ///
    /// * `field` - The field name
    ///
    /// # Returns
    ///
    /// The embedding dimension, or `None` if no image embedder is available.
    pub fn image_dimension(&self, field: &str) -> Option<usize> {
        self.get_image_embedder(field).map(|e| e.dimension())
    }

    /// List all configured text field names.
    pub fn text_fields(&self) -> Vec<&str> {
        self.text_embedders.keys().map(|s| s.as_str()).collect()
    }

    /// List all configured image field names.
    pub fn image_fields(&self) -> Vec<&str> {
        self.image_embedders.keys().map(|s| s.as_str()).collect()
    }

    /// List all configured field names (both text and image).
    pub fn configured_fields(&self) -> Vec<&str> {
        let mut fields: Vec<&str> = self.text_embedders.keys().map(|s| s.as_str()).collect();
        for field in self.image_embedders.keys() {
            if !fields.contains(&field.as_str()) {
                fields.push(field.as_str());
            }
        }
        fields
    }

    // Legacy compatibility methods

    /// Legacy: Create a per-field embedder with a default embedder.
    ///
    /// This method is provided for backward compatibility with code that used
    /// the old `PerFieldEmbedder::new(default_embedder)` signature.
    #[deprecated(since = "0.2.0", note = "Use with_default_text() instead")]
    pub fn with_default(default_embedder: Arc<dyn TextEmbedder>) -> Self {
        Self::with_default_text(default_embedder)
    }

    /// Legacy: Add a field-specific embedder (text only).
    ///
    /// This method is provided for backward compatibility.
    #[deprecated(since = "0.2.0", note = "Use add_text_embedder() instead")]
    pub fn add_embedder(&mut self, field: impl Into<String>, embedder: Arc<dyn TextEmbedder>) {
        self.add_text_embedder(field, embedder);
    }

    /// Legacy: Get the embedder for a specific field (text only).
    ///
    /// This method is provided for backward compatibility.
    #[deprecated(since = "0.2.0", note = "Use get_text_embedder() instead")]
    pub fn get_embedder(&self, field: &str) -> Option<&Arc<dyn TextEmbedder>> {
        self.text_embedders
            .get(field)
            .or(self.default_text_embedder.as_ref())
    }

    /// Legacy: Embed text with the embedder for the given field.
    ///
    /// This method is provided for backward compatibility.
    #[deprecated(since = "0.2.0", note = "Use embed_text() instead")]
    pub async fn embed_field(&self, field: &str, text: &str) -> Result<Vector> {
        self.embed_text(field, text).await
    }

    /// Legacy: Get the dimension for a specific field (text only).
    ///
    /// This method is provided for backward compatibility.
    #[deprecated(since = "0.2.0", note = "Use text_dimension() instead")]
    pub fn field_dimension(&self, field: &str) -> usize {
        self.text_dimension(field).unwrap_or(0)
    }
}

// Implement Embedder trait for PerFieldEmbedder
impl Embedder for PerFieldEmbedder {
    fn get_text_embedder(&self, field: &str) -> Option<Arc<dyn TextEmbedder>> {
        PerFieldEmbedder::get_text_embedder(self, field)
    }

    fn get_image_embedder(&self, field: &str) -> Option<Arc<dyn ImageEmbedder>> {
        PerFieldEmbedder::get_image_embedder(self, field)
    }

    fn text_dimension(&self, field: &str) -> Option<usize> {
        PerFieldEmbedder::text_dimension(self, field)
    }

    fn image_dimension(&self, field: &str) -> Option<usize> {
        PerFieldEmbedder::image_dimension(self, field)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// Implement TextEmbedder for PerFieldEmbedder (backward compatibility)
#[async_trait::async_trait]
impl TextEmbedder for PerFieldEmbedder {
    /// Embed text using the default embedder.
    ///
    /// Note: When using PerFieldEmbedder, it's recommended to use `embed_text()`
    /// to explicitly specify which field's embedder to use.
    async fn embed(&self, text: &str) -> Result<Vector> {
        let embedder = self.default_text_embedder.as_ref().ok_or_else(|| {
            crate::error::PlatypusError::invalid_config(
                "no default text embedder configured for PerFieldEmbedder",
            )
        })?;
        embedder.embed(text).await
    }

    /// Embed text with field context - selects embedder based on field name.
    async fn embed_with_field(&self, text: &str, field_name: &str) -> Result<Vector> {
        self.embed_text(field_name, text).await
    }

    /// Returns the dimension of the default embedder.
    ///
    /// Note: Different fields may have different dimensions.
    /// Use `text_dimension()` to get the dimension for a specific field.
    fn dimension(&self) -> usize {
        self.default_text_embedder
            .as_ref()
            .map(|e| e.dimension())
            .unwrap_or(0)
    }

    fn name(&self) -> &str {
        self.default_text_embedder
            .as_ref()
            .map(|e| e.name())
            .unwrap_or("per-field-embedder")
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;

    // Mock text embedder for testing
    #[derive(Debug, Clone)]
    struct MockTextEmbedder {
        name: String,
        dimension: usize,
    }

    #[async_trait]
    impl TextEmbedder for MockTextEmbedder {
        async fn embed(&self, _text: &str) -> Result<Vector> {
            Ok(Vector::new(vec![0.0; self.dimension]))
        }

        fn dimension(&self) -> usize {
            self.dimension
        }

        fn name(&self) -> &str {
            &self.name
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    // Mock image embedder for testing
    #[derive(Debug, Clone)]
    struct MockImageEmbedder {
        name: String,
        dimension: usize,
    }

    #[async_trait]
    impl ImageEmbedder for MockImageEmbedder {
        async fn embed(&self, _image_path: &str) -> Result<Vector> {
            Ok(Vector::new(vec![0.0; self.dimension]))
        }

        fn dimension(&self) -> usize {
            self.dimension
        }

        fn name(&self) -> &str {
            &self.name
        }
    }

    // Mock multimodal embedder for testing
    #[derive(Debug, Clone)]
    struct MockMultimodalEmbedder {
        dimension: usize,
    }

    #[async_trait]
    impl TextEmbedder for MockMultimodalEmbedder {
        async fn embed(&self, _text: &str) -> Result<Vector> {
            Ok(Vector::new(vec![1.0; self.dimension]))
        }

        fn dimension(&self) -> usize {
            self.dimension
        }

        fn name(&self) -> &str {
            "mock-multimodal"
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    #[async_trait]
    impl ImageEmbedder for MockMultimodalEmbedder {
        async fn embed(&self, _image_path: &str) -> Result<Vector> {
            Ok(Vector::new(vec![2.0; self.dimension]))
        }

        fn dimension(&self) -> usize {
            self.dimension
        }

        fn name(&self) -> &str {
            "mock-multimodal"
        }
    }

    #[test]
    fn test_new_empty() {
        let per_field = PerFieldEmbedder::new();
        assert!(per_field.default_text_embedder().is_none());
        assert!(per_field.default_image_embedder().is_none());
        assert!(per_field.text_fields().is_empty());
        assert!(per_field.image_fields().is_empty());
    }

    #[test]
    fn test_with_default_text() {
        let default: Arc<dyn TextEmbedder> = Arc::new(MockTextEmbedder {
            name: "default".into(),
            dimension: 384,
        });
        let per_field = PerFieldEmbedder::with_default_text(default);

        assert!(per_field.default_text_embedder().is_some());
        assert!(per_field.default_image_embedder().is_none());
        assert_eq!(per_field.text_dimension("any_field"), Some(384));
    }

    #[test]
    fn test_add_text_embedder() {
        let default: Arc<dyn TextEmbedder> = Arc::new(MockTextEmbedder {
            name: "default".into(),
            dimension: 384,
        });
        let mut per_field = PerFieldEmbedder::with_default_text(default);

        let title_embedder: Arc<dyn TextEmbedder> = Arc::new(MockTextEmbedder {
            name: "title".into(),
            dimension: 768,
        });
        per_field.add_text_embedder("title_embedding", title_embedder);

        // Default field uses default embedder
        assert_eq!(per_field.text_dimension("content_embedding"), Some(384));

        // Configured field uses specific embedder
        assert_eq!(per_field.text_dimension("title_embedding"), Some(768));
    }

    #[test]
    fn test_add_image_embedder() {
        let mut per_field = PerFieldEmbedder::new();

        let image_embedder: Arc<dyn ImageEmbedder> = Arc::new(MockImageEmbedder {
            name: "image".into(),
            dimension: 512,
        });
        per_field.add_image_embedder("image_embedding", image_embedder);

        assert_eq!(per_field.image_dimension("image_embedding"), Some(512));
        assert!(per_field.text_dimension("image_embedding").is_none());
    }

    #[test]
    fn test_add_multimodal_embedder() {
        let mut per_field = PerFieldEmbedder::new();

        let clip_embedder = Arc::new(MockMultimodalEmbedder { dimension: 512 });
        per_field.add_multimodal_embedder("content_embedding", clip_embedder);

        // Both text and image should be available
        assert_eq!(per_field.text_dimension("content_embedding"), Some(512));
        assert_eq!(per_field.image_dimension("content_embedding"), Some(512));
    }

    #[tokio::test]
    async fn test_embed_text() -> Result<()> {
        let default: Arc<dyn TextEmbedder> = Arc::new(MockTextEmbedder {
            name: "default".into(),
            dimension: 384,
        });
        let per_field = PerFieldEmbedder::with_default_text(default);

        let vec = per_field.embed_text("any_field", "test text").await?;
        assert_eq!(vec.dimension(), 384);

        Ok(())
    }

    #[tokio::test]
    async fn test_embed_text_no_embedder() {
        let per_field = PerFieldEmbedder::new();
        let result = per_field.embed_text("any_field", "test").await;
        assert!(result.is_err());
    }

    #[test]
    fn test_embedder_trait_implementation() {
        let default: Arc<dyn TextEmbedder> = Arc::new(MockTextEmbedder {
            name: "default".into(),
            dimension: 384,
        });
        let per_field = PerFieldEmbedder::with_default_text(default);

        // Test via Embedder trait
        let embedder: &dyn Embedder = &per_field;
        assert!(embedder.supports_text("any_field"));
        assert!(!embedder.supports_image("any_field"));
        assert_eq!(embedder.text_dimension("any_field"), Some(384));
    }

    #[test]
    fn test_configured_fields() {
        let mut per_field = PerFieldEmbedder::new();

        let text_embedder: Arc<dyn TextEmbedder> = Arc::new(MockTextEmbedder {
            name: "text".into(),
            dimension: 384,
        });
        per_field.add_text_embedder("text_field", text_embedder);

        let image_embedder: Arc<dyn ImageEmbedder> = Arc::new(MockImageEmbedder {
            name: "image".into(),
            dimension: 512,
        });
        per_field.add_image_embedder("image_field", image_embedder);

        let fields = per_field.configured_fields();
        assert!(fields.contains(&"text_field"));
        assert!(fields.contains(&"image_field"));
    }
}
