//! Analyzer implementations that combine tokenizers and filters.

#[allow(clippy::module_inception)]
mod analyzer;
mod keyword;
mod noop;
mod per_field;
mod pipeline;
mod simple;
mod standard;

pub use analyzer::Analyzer;
pub use keyword::KeywordAnalyzer;
pub use noop::NoOpAnalyzer;
pub use per_field::PerFieldAnalyzer;
pub use pipeline::PipelineAnalyzer;
pub use simple::SimpleAnalyzer;
pub use standard::StandardAnalyzer;
