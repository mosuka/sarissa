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
pub mod limit;
pub mod lowercase;
pub mod remove_empty;
pub mod stem;
pub mod stop;
pub mod strip;

// Re-export all filters for convenient access
pub use boost::BoostFilter;
pub use limit::LimitFilter;
pub use lowercase::LowercaseFilter;
pub use remove_empty::RemoveEmptyFilter;
pub use stem::{IdentityStemmer, PorterStemmer, SimpleStemmer, StemFilter, Stemmer};
pub use stop::StopFilter;
pub use strip::StripFilter;
