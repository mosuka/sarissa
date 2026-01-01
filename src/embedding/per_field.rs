//! Per-field embedder for applying different embedders to different fields.
//!
//! This module provides `PerFieldEmbedder`, which allows specifying different
//! embedders for different vector fields. This is analogous to `PerFieldAnalyzer`
//! in the lexical module.
//!
//! # Design Symmetry with PerFieldAnalyzer
//!
//! | PerFieldAnalyzer | PerFieldEmbedder |
//! |-----------------|------------------|
//! | `new(default_analyzer)` | `new(default_embedder)` |
//! | `add_analyzer(field, analyzer)` | `add_embedder(field, embedder)` |
//! | `get_analyzer(field)` | `get_embedder(field)` |
//! | `default_analyzer()` | `default_embedder()` |
//! | `analyze_field(field, text)` | `embed_field(field, input)` |
//!
//! # Example
//!
//! ```no_run
//! # #[cfg(feature = "embeddings-candle")]
//! # {
//! use sarissa::embedding::per_field::PerFieldEmbedder;
//! use sarissa::embedding::embedder::{Embedder, EmbedInput};
//! use sarissa::embedding::candle_bert_embedder::CandleBertEmbedder;
//! use std::sync::Arc;
//!
//! # async fn example() -> sarissa::error::Result<()> {
//! // Create default embedder
//! let default_embedder: Arc<dyn Embedder> = Arc::new(
//!     CandleBertEmbedder::new("sentence-transformers/all-MiniLM-L6-v2")?
//! );
//!
//! // Create per-field embedder with default
//! let mut per_field = PerFieldEmbedder::new(default_embedder);
//!
//! // Add specialized embedder for title field
//! let title_embedder: Arc<dyn Embedder> = Arc::new(
//!     CandleBertEmbedder::new("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")?
//! );
//! per_field.add_embedder("title_embedding", Arc::clone(&title_embedder));
//!
//! // "content_embedding" will use default_embedder
//! // "title_embedding" will use the specialized title_embedder
//!
//! // Embed with field context
//! let input = EmbedInput::Text("Hello, world!");
//! let vector = per_field.embed_field("title_embedding", &input).await?;
//! # Ok(())
//! # }
//! # }
//! ```

use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

use crate::embedding::embedder::{EmbedInput, EmbedInputType, Embedder};
use crate::error::Result;
use crate::vector::core::vector::Vector;

/// A per-field embedder that applies different embedders to different fields.
///
/// This is similar to `PerFieldAnalyzer` in the lexical module. It allows you
/// to specify a different embedder for each field, with a default embedder
/// for fields not explicitly configured.
///
/// # Memory Efficiency
///
/// When using the same embedder for multiple fields, reuse a single instance
/// with `Arc::clone` to save memory. This is especially important for embedders
/// with large models.
///
/// # Example
///
/// ```no_run
/// # #[cfg(feature = "embeddings-candle")]
/// # {
/// use sarissa::embedding::embedder::Embedder;
/// use sarissa::embedding::per_field::PerFieldEmbedder;
/// use sarissa::embedding::candle_bert_embedder::CandleBertEmbedder;
/// use std::sync::Arc;
///
/// # fn example() -> sarissa::error::Result<()> {
/// // Create default embedder
/// let default_embedder: Arc<dyn Embedder> = Arc::new(
///     CandleBertEmbedder::new("sentence-transformers/all-MiniLM-L6-v2")?
/// );
///
/// // Create per-field embedder
/// let mut per_field = PerFieldEmbedder::new(default_embedder);
///
/// // Reuse embedder instances to save memory
/// let keyword_embedder: Arc<dyn Embedder> = Arc::new(
///     CandleBertEmbedder::new("sentence-transformers/all-MiniLM-L6-v2")?
/// );
/// per_field.add_embedder("id", Arc::clone(&keyword_embedder));
/// per_field.add_embedder("category", Arc::clone(&keyword_embedder));
/// # Ok(())
/// # }
/// # }
/// ```
#[derive(Clone)]
pub struct PerFieldEmbedder {
    /// Default embedder for fields not in the map.
    default_embedder: Arc<dyn Embedder>,

    /// Map of field names to their specific embedders.
    field_embedders: HashMap<String, Arc<dyn Embedder>>,
}

impl std::fmt::Debug for PerFieldEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PerFieldEmbedder")
            .field("default_embedder", &self.default_embedder.name())
            .field("fields", &self.field_embedders.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl PerFieldEmbedder {
    /// Create a new per-field embedder with a default embedder.
    ///
    /// # Arguments
    ///
    /// * `default_embedder` - The embedder to use for fields not explicitly configured
    ///
    /// # Example
    ///
    /// ```no_run
    /// # #[cfg(feature = "embeddings-candle")]
    /// # {
    /// use sarissa::embedding::per_field::PerFieldEmbedder;
    /// use sarissa::embedding::embedder::Embedder;
    /// use sarissa::embedding::candle_bert_embedder::CandleBertEmbedder;
    /// use std::sync::Arc;
    ///
    /// # fn example() -> sarissa::error::Result<()> {
    /// let default: Arc<dyn Embedder> = Arc::new(
    ///     CandleBertEmbedder::new("sentence-transformers/all-MiniLM-L6-v2")?
    /// );
    /// let per_field = PerFieldEmbedder::new(default);
    /// # Ok(())
    /// # }
    /// # }
    /// ```
    pub fn new(default_embedder: Arc<dyn Embedder>) -> Self {
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
    pub fn add_embedder(&mut self, field: impl Into<String>, embedder: Arc<dyn Embedder>) {
        self.field_embedders.insert(field.into(), embedder);
    }

    /// Get the embedder for a specific field.
    ///
    /// Returns the field-specific embedder if configured, otherwise returns the default.
    ///
    /// # Arguments
    ///
    /// * `field` - The field name
    pub fn get_embedder(&self, field: &str) -> &Arc<dyn Embedder> {
        self.field_embedders
            .get(field)
            .unwrap_or(&self.default_embedder)
    }

    /// Get the default embedder.
    pub fn default_embedder(&self) -> &Arc<dyn Embedder> {
        &self.default_embedder
    }

    /// Embed with the embedder for the given field.
    ///
    /// # Arguments
    ///
    /// * `field` - The field name to determine which embedder to use
    /// * `input` - The input to embed
    ///
    /// # Returns
    ///
    /// The embedding vector for the input.
    pub async fn embed_field(&self, field: &str, input: &EmbedInput<'_>) -> Result<Vector> {
        self.get_embedder(field).embed(input).await
    }

    /// List all configured field names.
    pub fn configured_fields(&self) -> Vec<&str> {
        self.field_embedders.keys().map(|s| s.as_str()).collect()
    }

    /// Check if a specific field supports the given input type.
    ///
    /// # Arguments
    ///
    /// * `field` - The field name
    /// * `input_type` - The input type to check
    pub fn field_supports(&self, field: &str, input_type: EmbedInputType) -> bool {
        self.get_embedder(field).supports(input_type)
    }
}

#[async_trait]
impl Embedder for PerFieldEmbedder {
    /// Embed with the default embedder.
    ///
    /// Note: When using PerFieldEmbedder, it's recommended to use `embed_field()`
    /// to explicitly specify which field's embedder to use.
    async fn embed(&self, input: &EmbedInput<'_>) -> Result<Vector> {
        self.default_embedder.embed(input).await
    }

    async fn embed_batch(&self, inputs: &[EmbedInput<'_>]) -> Result<Vec<Vector>> {
        self.default_embedder.embed_batch(inputs).await
    }

    /// Returns the supported input types of the default embedder.
    fn supported_input_types(&self) -> Vec<EmbedInputType> {
        self.default_embedder.supported_input_types()
    }

    fn name(&self) -> &str {
        "PerFieldEmbedder"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::SarissaError;

    #[derive(Debug)]
    struct MockEmbedder {
        name: String,
        dim: usize,
    }

    #[async_trait]
    impl Embedder for MockEmbedder {
        async fn embed(&self, input: &EmbedInput<'_>) -> Result<Vector> {
            match input {
                EmbedInput::Text(_) => Ok(Vector::new(vec![0.0; self.dim])),
                _ => Err(SarissaError::invalid_argument("only text supported")),
            }
        }

        fn supported_input_types(&self) -> Vec<EmbedInputType> {
            vec![EmbedInputType::Text]
        }

        fn name(&self) -> &str {
            &self.name
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    #[tokio::test]
    async fn test_per_field_embedder() {
        let default: Arc<dyn Embedder> = Arc::new(MockEmbedder {
            name: "default".into(),
            dim: 384,
        });
        let mut per_field = PerFieldEmbedder::new(default);

        let title_embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder {
            name: "title".into(),
            dim: 768,
        });
        per_field.add_embedder("title", Arc::clone(&title_embedder));
        per_field.add_embedder("description", title_embedder);

        let input = EmbedInput::Text("hello");
        let title_vec = per_field.embed_field("title", &input).await.unwrap();
        assert_eq!(title_vec.dimension(), 768);

        let desc_vec = per_field.embed_field("description", &input).await.unwrap();
        assert_eq!(desc_vec.dimension(), 768);

        let content_vec = per_field.embed_field("content", &input).await.unwrap();
        assert_eq!(content_vec.dimension(), 384);
    }

    #[tokio::test]
    async fn test_default_embedder_when_field_not_configured() {
        let default: Arc<dyn Embedder> = Arc::new(MockEmbedder {
            name: "default".into(),
            dim: 384,
        });
        let per_field = PerFieldEmbedder::new(default);

        let input = EmbedInput::Text("hello");
        let vec = per_field
            .embed_field("unknown_field", &input)
            .await
            .unwrap();
        assert_eq!(vec.dimension(), 384);
        assert_eq!(per_field.get_embedder("unknown_field").name(), "default");
    }

    #[tokio::test]
    async fn test_as_embedder_trait() {
        let default: Arc<dyn Embedder> = Arc::new(MockEmbedder {
            name: "default".into(),
            dim: 384,
        });
        let per_field = PerFieldEmbedder::new(default);

        let embedder: &dyn Embedder = &per_field;
        assert!(embedder.supports_text());

        let vec = embedder.embed(&EmbedInput::Text("hello")).await.unwrap();
        assert_eq!(vec.dimension(), 384);
    }

    #[tokio::test]
    async fn test_embed_field() {
        let default: Arc<dyn Embedder> = Arc::new(MockEmbedder {
            name: "default".into(),
            dim: 384,
        });
        let mut per_field = PerFieldEmbedder::new(default);

        let title_embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder {
            name: "title".into(),
            dim: 768,
        });
        per_field.add_embedder("title", title_embedder);

        // Embed with specific field
        let input = EmbedInput::Text("hello");
        let vec = per_field.embed_field("title", &input).await.unwrap();
        assert_eq!(vec.dimension(), 768);

        // Embed with default field
        let vec = per_field.embed_field("unknown", &input).await.unwrap();
        assert_eq!(vec.dimension(), 384);
    }

    #[test]
    fn test_configured_fields() {
        let default: Arc<dyn Embedder> = Arc::new(MockEmbedder {
            name: "default".into(),
            dim: 384,
        });
        let mut per_field = PerFieldEmbedder::new(default);

        let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder {
            name: "special".into(),
            dim: 512,
        });
        per_field.add_embedder("title", Arc::clone(&embedder));
        per_field.add_embedder("body", embedder);

        let fields = per_field.configured_fields();
        assert!(fields.contains(&"title"));
        assert!(fields.contains(&"body"));
        assert!(!fields.contains(&"unknown"));
    }

    #[test]
    fn test_field_supports() {
        let default: Arc<dyn Embedder> = Arc::new(MockEmbedder {
            name: "default".into(),
            dim: 384,
        });
        let per_field = PerFieldEmbedder::new(default);

        assert!(per_field.field_supports("any", EmbedInputType::Text));
        assert!(!per_field.field_supports("any", EmbedInputType::Image));
    }
}
