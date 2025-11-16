//! Per-field embedder for applying different embedders to different fields.

use std::collections::HashMap;
use std::sync::Arc;

use crate::embedding::text_embedder::TextEmbedder;
use crate::error::Result;
use crate::vector::core::vector::Vector;

/// A per-field embedder that applies different embedders to different fields.
///
/// This is similar to `PerFieldAnalyzer` in the lexical module. It allows you to specify
/// a different embedder for each vector field, with a default embedder for fields not
/// explicitly configured.
///
/// # Memory Efficiency
///
/// When using the same embedder for multiple fields, reuse a single instance
/// with `Arc::clone` to save memory. This is especially important for large embedding
/// models.
///
/// # Examples
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
/// // Default embedder for most fields
/// let default_embedder: Arc<dyn TextEmbedder> = Arc::new(
///     CandleTextEmbedder::new("sentence-transformers/all-MiniLM-L6-v2")?
/// );
///
/// let mut per_field = PerFieldEmbedder::new(default_embedder.clone());
///
/// // Use a specialized embedder for title field
/// let title_embedder: Arc<dyn TextEmbedder> = Arc::new(
///     CandleTextEmbedder::new("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")?
/// );
/// per_field.add_embedder("title_embedding", title_embedder);
///
/// // "content_embedding" and "summary_embedding" will use default_embedder
/// // "title_embedding" will use the specialized title_embedder
/// # Ok(())
/// # }
/// # }
/// ```
#[derive(Clone)]
pub struct PerFieldEmbedder {
    /// Default embedder for fields not in the map.
    default_embedder: Arc<dyn TextEmbedder>,

    /// Map of field names to their specific embedders.
    field_embedders: HashMap<String, Arc<dyn TextEmbedder>>,
}

impl std::fmt::Debug for PerFieldEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PerFieldEmbedder")
            .field("default_embedder", &self.default_embedder.name())
            .field("configured_fields", &self.configured_fields())
            .finish()
    }
}

impl PerFieldEmbedder {
    /// Create a new per-field embedder with a default embedder.
    ///
    /// # Arguments
    ///
    /// * `default_embedder` - The embedder to use for fields not explicitly configured
    pub fn new(default_embedder: Arc<dyn TextEmbedder>) -> Self {
        Self {
            default_embedder,
            field_embedders: HashMap::new(),
        }
    }

    /// Add a field-specific embedder.
    ///
    /// # Arguments
    ///
    /// * `field` - The field name
    /// * `embedder` - The embedder to use for this field
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # #[cfg(feature = "embeddings-candle")]
    /// # {
    /// use platypus::embedding::per_field::PerFieldEmbedder;
    /// use platypus::embedding::candle_text_embedder::CandleTextEmbedder;
    /// use std::sync::Arc;
    ///
    /// # fn example() -> platypus::error::Result<()> {
    /// let default_embedder: Arc<dyn platypus::embedding::text_embedder::TextEmbedder> = Arc::new(
    ///     CandleTextEmbedder::new("sentence-transformers/all-MiniLM-L6-v2")?
    /// );
    /// let mut per_field = PerFieldEmbedder::new(Arc::clone(&default_embedder));
    ///
    /// let title_embedder = Arc::new(
    ///     CandleTextEmbedder::new("another-model")?
    /// );
    /// per_field.add_embedder("title_embedding", title_embedder);
    /// # Ok(())
    /// # }
    /// # }
    /// ```
    pub fn add_embedder(&mut self, field: impl Into<String>, embedder: Arc<dyn TextEmbedder>) {
        self.field_embedders.insert(field.into(), embedder);
    }

    /// Get the embedder for a specific field.
    ///
    /// Returns the field-specific embedder if configured, otherwise returns the default embedder.
    ///
    /// # Arguments
    ///
    /// * `field` - The field name
    pub fn get_embedder(&self, field: &str) -> &Arc<dyn TextEmbedder> {
        self.field_embedders
            .get(field)
            .unwrap_or(&self.default_embedder)
    }

    /// Get the default embedder.
    pub fn default_embedder(&self) -> &Arc<dyn TextEmbedder> {
        &self.default_embedder
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
    /// The embedding vector for the text using the appropriate embedder
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # #[cfg(feature = "embeddings-candle")]
    /// # {
    /// use platypus::embedding::per_field::PerFieldEmbedder;
    /// use platypus::embedding::candle_text_embedder::CandleTextEmbedder;
    /// use std::sync::Arc;
    ///
    /// # async fn example() -> platypus::error::Result<()> {
    /// let default_embedder: Arc<dyn platypus::embedding::text_embedder::TextEmbedder> = Arc::new(
    ///     CandleTextEmbedder::new("sentence-transformers/all-MiniLM-L6-v2")?
    /// );
    /// let per_field = PerFieldEmbedder::new(Arc::clone(&default_embedder));
    ///
    /// // This will use the default embedder
    /// let vec1 = per_field.embed_field("content_embedding", "Some text").await?;
    ///
    /// // This will use the title-specific embedder if configured
    /// let vec2 = per_field.embed_field("title_embedding", "Title text").await?;
    /// # Ok(())
    /// # }
    /// # }
    /// ```
    pub async fn embed_field(&self, field: &str, text: &str) -> Result<Vector> {
        let embedder = self.get_embedder(field);
        embedder.embed(text).await
    }

    /// Get the dimension for a specific field.
    ///
    /// # Arguments
    ///
    /// * `field` - The field name
    ///
    /// # Returns
    ///
    /// The embedding dimension for the given field
    pub fn field_dimension(&self, field: &str) -> usize {
        self.get_embedder(field).dimension()
    }

    /// List all configured field names (excluding the default).
    pub fn configured_fields(&self) -> Vec<&str> {
        self.field_embedders.keys().map(|s| s.as_str()).collect()
    }
}

// Implement TextEmbedder for PerFieldEmbedder
// This allows PerFieldEmbedder to be used wherever TextEmbedder is expected
#[async_trait::async_trait]
impl crate::embedding::text_embedder::TextEmbedder for PerFieldEmbedder {
    /// Embed text using the default embedder.
    ///
    /// Note: When using PerFieldEmbedder, it's recommended to use `embed_field()`
    /// to explicitly specify which field's embedder to use.
    async fn embed(&self, text: &str) -> Result<Vector> {
        self.default_embedder.embed(text).await
    }

    /// Embed text with field context - selects embedder based on field name.
    ///
    /// This is the primary method for PerFieldEmbedder, allowing automatic
    /// selection of the appropriate embedder based on the field name.
    async fn embed_with_field(&self, text: &str, field_name: &str) -> Result<Vector> {
        self.embed_field(field_name, text).await
    }

    /// Returns the dimension of the default embedder.
    ///
    /// Note: Different fields may have different dimensions.
    /// Use `field_dimension()` to get the dimension for a specific field.
    fn dimension(&self) -> usize {
        self.default_embedder.dimension()
    }

    /// Returns the name of the default embedder.
    fn name(&self) -> &str {
        self.default_embedder.name()
    }

    /// Enable downcasting to PerFieldEmbedder.
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::text_embedder::TextEmbedder;
    use async_trait::async_trait;

    // Mock embedder for testing
    #[derive(Debug)]
    struct MockEmbedder {
        name: String,
        dimension: usize,
    }

    #[async_trait]
    impl TextEmbedder for MockEmbedder {
        async fn embed(&self, _text: &str) -> Result<Vector> {
            Ok(Vector::new(vec![0.0; self.dimension]))
        }

        fn dimension(&self) -> usize {
            self.dimension
        }

        fn name(&self) -> &str {
            &self.name
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    #[tokio::test]
    async fn test_per_field_embedder() -> Result<()> {
        let default_embedder: Arc<dyn TextEmbedder> = Arc::new(MockEmbedder {
            name: "default".to_string(),
            dimension: 384,
        });

        let mut per_field = PerFieldEmbedder::new(default_embedder);

        let title_embedder: Arc<dyn TextEmbedder> = Arc::new(MockEmbedder {
            name: "title".to_string(),
            dimension: 768,
        });
        per_field.add_embedder("title_embedding", title_embedder);

        // Default embedder
        assert_eq!(per_field.field_dimension("content_embedding"), 384);

        // Field-specific embedder
        assert_eq!(per_field.field_dimension("title_embedding"), 768);

        // Embed with field-specific embedder
        let vec = per_field.embed_field("title_embedding", "test").await?;
        assert_eq!(vec.dimension(), 768);

        Ok(())
    }
}
