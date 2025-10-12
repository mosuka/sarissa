//! Intent classifier trait definition.

use anyhow::Result;

use crate::ml::query_expansion::QueryIntent;

/// Intent classifier trait.
///
/// Implementations of this trait provide different methods for classifying
/// query intent to improve search relevance.
pub trait IntentClassifier: Send + Sync {
    /// Predict the intent for a given query.
    ///
    /// # Arguments
    /// * `query` - The query string to classify
    ///
    /// # Returns
    /// The predicted `QueryIntent`
    fn predict(&self, query: &str) -> Result<QueryIntent>;

    /// Get the name of this classifier for debugging and logging.
    fn name(&self) -> &str;
}
