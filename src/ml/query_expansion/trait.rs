//! Query expander trait definition.

use crate::error::Result;
use crate::ml::MLContext;

use super::types::ExpandedQueryClause;

/// Query expansion strategy trait.
///
/// Implementations of this trait provide different methods for expanding
/// query terms to improve search recall.
pub trait QueryExpander: Send + Sync {
    /// Expand query tokens into additional search terms.
    ///
    /// # Arguments
    /// * `tokens` - The original query tokens to expand
    /// * `field` - The field name for the expanded queries
    /// * `context` - ML context containing user session and search history
    ///
    /// # Returns
    /// A vector of expanded query clauses with confidence scores
    fn expand(
        &self,
        tokens: &[String],
        field: &str,
        context: &MLContext,
    ) -> Result<Vec<ExpandedQueryClause>>;

    /// Get the name of this expander for debugging and logging.
    fn name(&self) -> &str;
}
