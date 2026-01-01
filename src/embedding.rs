//! Text and multimodal embedding support for Sarissa vector search.
//!
//! - Core traits: `Embedder`, `PerFieldEmbedder`, `PrecomputedEmbedder`
//! - Feature flags: `embeddings-candle`, `embeddings-openai`, `embeddings-multimodal`, `embeddings-all`
//! - Vector 次元はフィールド定義で明示し、embedder から推定しない
//!
// Unified embedder trait (analogous to Analyzer in lexical module)
pub mod embedder;

// Per-field embedder support (analogous to PerFieldAnalyzer)
pub mod per_field;

// Embedder for pre-computed vectors (analogous to NoOpAnalyzer)
pub mod precomputed;

// Candle implementation (requires feature flag)
#[cfg(feature = "embeddings-candle")]
pub mod candle_bert_embedder;

// OpenAI implementation (requires feature flag)
#[cfg(feature = "embeddings-openai")]
pub mod openai_text_embedder;

// Multimodal embedding (requires embeddings-multimodal feature flag)
#[cfg(feature = "embeddings-multimodal")]
pub mod candle_clip_embedder;
