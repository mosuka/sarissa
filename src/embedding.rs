//! Text and multimodal embedding support for Sarissa vector search.
//!
//! - Core traits: `Embedder`, `PerFieldEmbedder`, `NoOpEmbedder`
//! - Feature flags: `embeddings-candle`, `embeddings-openai`, `embeddings-multimodal`, `embeddings-all`
//! - Vector 次元はフィールド定義で明示し、embedder から推定しない
//!
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
