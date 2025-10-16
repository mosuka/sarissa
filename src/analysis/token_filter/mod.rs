//! Token filter implementations for token transformation.

use crate::analysis::token::TokenStream;
use crate::error::Result;

/// Trait for filters that transform token streams.
pub trait Filter: Send + Sync {
    /// Apply this filter to a token stream.
    fn filter(&self, tokens: TokenStream) -> Result<TokenStream>;

    /// Get the name of this filter (for debugging and configuration).
    fn name(&self) -> &'static str;
}

// Individual filter modules
pub mod boost;
pub mod flatten_graph;
pub mod limit;
pub mod lowercase;
pub mod remove_empty;
pub mod stem;
pub mod stop;
pub mod strip;
pub mod synonym_graph;
