//! Text and multimodal embedding support for Platypus's vector search.
//!
//! The core traits live here while concrete embedders are compiled in via feature
//! flags. Keeping implementations optional lets projects pick local Candle models,
//! hosted OpenAI APIs, or custom logic without bloating default builds.
//!
//! # Feature Flags
//!
//! Enable embedding implementations using Cargo features:
//!
//! - `embeddings-candle` - HuggingFace Candle implementation (local inference)
//! - `embeddings-openai` - OpenAI API implementation (cloud-based)
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
//! use platypus::embedding::text_embedder::TextEmbedder;
//! use platypus::embedding::candle_text_embedder::CandleTextEmbedder;
//!
//! # async fn example() -> platypus::error::Result<()> {
//! let embedder = CandleTextEmbedder::new(
//!     "sentence-transformers/all-MiniLM-L6-v2"
//! )?;
//!
//! let vector = embedder.embed("Hello, world!").await?;
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
//! use platypus::embedding::text_embedder::TextEmbedder;
//! use platypus::embedding::openai_text_embedder::OpenAITextEmbedder;
//!
//! # async fn example() -> platypus::error::Result<()> {
//! let embedder = OpenAITextEmbedder::new(
//!     std::env::var("OPENAI_API_KEY").unwrap(),
//!     "text-embedding-3-small".to_string()
//! )?;
//!
//! let vector = embedder.embed("Hello, world!").await?;
//! # Ok(())
//! # }
//! # }
//! ```
//!
//! ## Custom Implementation
//!
//! You can implement your own embedder by implementing the `TextEmbedder` trait:
//!
//! ```
//! use async_trait::async_trait;
//! use platypus::embedding::text_embedder::TextEmbedder;
//! use platypus::error::Result;
//! use platypus::vector::Vector;
//!
//! struct MyEmbedder {
//!     dimension: usize,
//! }
//!
//! #[async_trait]
//! impl TextEmbedder for MyEmbedder {
//!     async fn embed(&self, text: &str) -> Result<Vector> {
//!         // Your custom implementation
//!         Ok(Vector::new(vec![0.0; self.dimension]))
//!     }
//!
//!     fn dimension(&self) -> usize {
//!         self.dimension
//!     }
//! }
//! ```
//!
//! # Dynamic Switching
//!
//! You can switch between embedders at runtime using trait objects:
//!
//! ```no_run
//! use platypus::embedding::text_embedder::TextEmbedder;
//! use std::sync::Arc;
//!
//! # async fn example() -> platypus::error::Result<()> {
//! #[cfg(all(feature = "embeddings-candle", feature = "embeddings-openai"))]
//! {
//!     use platypus::embedding::candle_text_embedder::CandleTextEmbedder;
//!     use platypus::embedding::openai_text_embedder::OpenAITextEmbedder;
//!
//!     let embedder: Arc<dyn TextEmbedder> = if std::env::var("USE_OPENAI").is_ok() {
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
//!     let vector = embedder.embed("Hello!").await?;
//! }
//! # Ok(())
//! # }
//! ```

pub mod text_embedder;

// Unified embedder trait (analogous to Analyzer in lexical module)
pub mod embedder;

// Per-field embedder support
pub mod per_field;

// No-operation embedder for pre-computed vectors
pub mod noop;

// Candle implementation (requires feature flag)
#[cfg(feature = "embeddings-candle")]
pub mod candle_text_embedder;

// OpenAI implementation (requires feature flag)
#[cfg(feature = "embeddings-openai")]
pub mod openai_text_embedder;

// Image and multimodal embeddings (requires embeddings-multimodal feature flag for implementations)
pub mod image_embedder;

#[cfg(feature = "embeddings-multimodal")]
pub mod multimodal_embedder;

#[cfg(feature = "embeddings-multimodal")]
pub mod candle_multimodal_embedder;
