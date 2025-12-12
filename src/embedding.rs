//! Text and multimodal embedding support for Platypus's vector search.
//!
//! The core traits live here while concrete embedders are compiled in via feature
//! flags. Keeping implementations optional lets projects pick local Candle models,
//! hosted OpenAI APIs, or custom logic without bloating default builds.
//!
//! # Architecture
//!
//! The embedding module follows the same pattern as the analysis module:
//! - [`Embedder`](embedder::Embedder) trait - Core unified embedding interface (analogous to `Analyzer`)
//! - [`PerFieldEmbedder`](per_field::PerFieldEmbedder) - Field-specific embedders (analogous to `PerFieldAnalyzer`)
//! - [`NoOpEmbedder`](noop::NoOpEmbedder) - No-operation embedder for pre-computed vectors (analogous to `NoOpAnalyzer`)
//!
//! # Feature Flags
//!
//! Enable embedding implementations using Cargo features:
//!
//! - `embeddings-candle` - HuggingFace Candle implementation (local inference)
//! - `embeddings-openai` - OpenAI API implementation (cloud-based)
//! - `embeddings-multimodal` - Multimodal (text + image) embedding support
//! - `embeddings-all` - All embedding implementations
//!
//! # Usage
//!
//! ## Using Candle (Recommended for Production)
//!
//! Add to your `Cargo.toml`:
//! ```toml
//! [dependencies]
//! platypus = { version = "0.1", features = ["embeddings-candle"] }
//! ```
//!
//! Then use:
//! ```no_run
//! # #[cfg(feature = "embeddings-candle")]
//! # {
//! use platypus::embedding::embedder::{Embedder, EmbedInput};
//! use platypus::embedding::candle_text_embedder::CandleTextEmbedder;
//!
//! # async fn example() -> platypus::error::Result<()> {
//! let embedder = CandleTextEmbedder::new(
//!     "sentence-transformers/all-MiniLM-L6-v2"
//! )?;
//!
//! let vector = embedder.embed(&EmbedInput::Text("Hello, world!")).await?;
//! println!("Dimension: {}", embedder.dimension());
//! # Ok(())
//! # }
//! # }
//! ```
//!
//! ## Using OpenAI (Recommended for Prototyping)
//!
//! Add to your `Cargo.toml`:
//! ```toml
//! [dependencies]
//! platypus = { version = "0.1", features = ["embeddings-openai"] }
//! ```
//!
//! Then use:
//! ```no_run
//! # #[cfg(feature = "embeddings-openai")]
//! # {
//! use platypus::embedding::embedder::{Embedder, EmbedInput};
//! use platypus::embedding::openai_text_embedder::OpenAITextEmbedder;
//!
//! # async fn example() -> platypus::error::Result<()> {
//! let embedder = OpenAITextEmbedder::new(
//!     std::env::var("OPENAI_API_KEY").unwrap(),
//!     "text-embedding-3-small".to_string()
//! )?;
//!
//! let vector = embedder.embed(&EmbedInput::Text("Hello, world!")).await?;
//! # Ok(())
//! # }
//! # }
//! ```
//!
//! ## Per-Field Embedders
//!
//! Similar to `PerFieldAnalyzer`, you can use different embedders for different fields:
//!
//! ```no_run
//! # #[cfg(feature = "embeddings-candle")]
//! # {
//! use std::sync::Arc;
//! use platypus::embedding::embedder::{Embedder, EmbedInput};
//! use platypus::embedding::per_field::PerFieldEmbedder;
//! use platypus::embedding::candle_text_embedder::CandleTextEmbedder;
//!
//! # async fn example() -> platypus::error::Result<()> {
//! // Create embedders
//! let title_embedder: Arc<dyn Embedder> = Arc::new(
//!     CandleTextEmbedder::new("sentence-transformers/all-MiniLM-L6-v2")?
//! );
//! let body_embedder: Arc<dyn Embedder> = Arc::new(
//!     CandleTextEmbedder::new("sentence-transformers/all-mpnet-base-v2")?
//! );
//!
//! // Create per-field embedder
//! let mut embedder = PerFieldEmbedder::new(title_embedder);
//! embedder.add_embedder("body", body_embedder);
//!
//! // Embed for specific fields
//! let title_vec = embedder.embed_field("title", &EmbedInput::Text("Article Title")).await?;
//! let body_vec = embedder.embed_field("body", &EmbedInput::Text("Article body text...")).await?;
//! # Ok(())
//! # }
//! # }
//! ```
//!
//! ## Custom Implementation
//!
//! You can implement your own embedder by implementing the `Embedder` trait:
//!
//! ```
//! use std::any::Any;
//! use async_trait::async_trait;
//! use platypus::embedding::embedder::{Embedder, EmbedInput, EmbedInputType};
//! use platypus::error::Result;
//! use platypus::vector::Vector;
//!
//! #[derive(Debug)]
//! struct MyEmbedder {
//!     dimension: usize,
//! }
//!
//! #[async_trait]
//! impl Embedder for MyEmbedder {
//!     async fn embed(&self, input: &EmbedInput<'_>) -> Result<Vector> {
//!         // Your custom implementation
//!         Ok(Vector::new(vec![0.0; self.dimension]))
//!     }
//!
//!     fn dimension(&self) -> usize {
//!         self.dimension
//!     }
//!
//!     fn supported_input_types(&self) -> Vec<EmbedInputType> {
//!         vec![EmbedInputType::Text]
//!     }
//!
//!     fn as_any(&self) -> &dyn Any {
//!         self
//!     }
//! }
//! ```
//!
//! # Dynamic Switching
//!
//! You can switch between embedders at runtime using trait objects:
//!
//! ```no_run
//! use platypus::embedding::embedder::{Embedder, EmbedInput};
//! use std::sync::Arc;
//!
//! # async fn example() -> platypus::error::Result<()> {
//! #[cfg(all(feature = "embeddings-candle", feature = "embeddings-openai"))]
//! {
//!     use platypus::embedding::candle_text_embedder::CandleTextEmbedder;
//!     use platypus::embedding::openai_text_embedder::OpenAITextEmbedder;
//!
//!     let embedder: Arc<dyn Embedder> = if std::env::var("USE_OPENAI").is_ok() {
//!         Arc::new(OpenAITextEmbedder::new(
//!             std::env::var("OPENAI_API_KEY").unwrap(),
//!             "text-embedding-3-small".to_string()
//!         )?)
//!     } else {
//!         Arc::new(CandleTextEmbedder::new(
//!             "sentence-transformers/all-MiniLM-L6-v2"
//!         )?)
//!     };
//!
//!     let vector = embedder.embed(&EmbedInput::Text("Hello!")).await?;
//! }
//! # Ok(())
//! # }
//! ```

// Unified embedder trait (analogous to Analyzer in lexical module)
pub mod embedder;

// Per-field embedder support (analogous to PerFieldAnalyzer)
pub mod per_field;

// No-operation embedder for pre-computed vectors (analogous to NoOpAnalyzer)
pub mod noop;

// Candle implementation (requires feature flag)
#[cfg(feature = "embeddings-candle")]
pub mod candle_text_embedder;

// OpenAI implementation (requires feature flag)
#[cfg(feature = "embeddings-openai")]
pub mod openai_text_embedder;

// Multimodal embedding (requires embeddings-multimodal feature flag)
#[cfg(feature = "embeddings-multimodal")]
pub mod candle_multimodal_embedder;
