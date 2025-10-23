//! Text embedding support for vector search.
//!
//! This module provides a trait-based interface for converting text to vector embeddings.
//! Sage does not include built-in embedding implementations by default to:
//!
//! - Keep dependencies minimal
//! - Allow users to choose their preferred embedding method
//! - Stay focused on search (not ML model management)
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
//! sage = { version = "0.1", features = ["embeddings-candle"] }
//! ```
//!
//! Then use:
//! ```no_run
//! # #[cfg(feature = "embeddings-candle")]
//! # {
//! use sage::embedding::{TextEmbedder, CandleTextEmbedder};
//!
//! # async fn example() -> sage::error::Result<()> {
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
//! sage = { version = "0.1", features = ["embeddings-openai"] }
//! ```
//!
//! Then use:
//! ```no_run
//! # #[cfg(feature = "embeddings-openai")]
//! # {
//! use sage::embedding::{TextEmbedder, OpenAITextEmbedder};
//!
//! # async fn example() -> sage::error::Result<()> {
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
//! use sage::embedding::TextEmbedder;
//! use sage::error::Result;
//! use sage::vector::Vector;
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
//! use sage::embedding::TextEmbedder;
//! use std::sync::Arc;
//!
//! # async fn example() -> sage::error::Result<()> {
//! #[cfg(all(feature = "embeddings-candle", feature = "embeddings-openai"))]
//! {
//!     use sage::embedding::{CandleTextEmbedder, OpenAITextEmbedder};
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

// Candle implementation (requires feature flag)
#[cfg(feature = "embeddings-candle")]
pub mod candle_text_embedder;

// OpenAI implementation (requires feature flag)
#[cfg(feature = "embeddings-openai")]
pub mod openai_text_embedder;

// Image and multimodal embeddings (requires embeddings-multimodal feature flag)
#[cfg(feature = "embeddings-multimodal")]
pub mod image_embedder;

#[cfg(feature = "embeddings-multimodal")]
pub mod multimodal_embedder;

#[cfg(feature = "embeddings-multimodal")]
pub mod candle_multimodal_embedder;
