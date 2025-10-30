//! Multi-term query support.
//!
//! This module provides traits and utilities for queries that match multiple terms,
//! similar to Lucene's MultiTermQuery.

use crate::error::Result;
use crate::lexical::reader::IndexReader;
use crate::lexical::core::terms::TermsEnum;
use crate::query::query::Query;

/// A query that matches multiple terms based on some pattern or criteria.
///
/// This is the base trait for queries like:
/// - `FuzzyQuery`: matches terms within edit distance
/// - `PrefixQuery`: matches terms with a common prefix
/// - `WildcardQuery`: matches terms matching a wildcard pattern
/// - `RegexpQuery`: matches terms matching a regular expression
///
/// Similar to Lucene's MultiTermQuery, this trait provides a common interface
/// for queries that need to enumerate and match against the term dictionary.
///
/// # Design
///
/// MultiTermQuery implementations should:
/// 1. Enumerate matching terms from the index's term dictionary
/// 2. Apply rewrite strategies to convert to simpler queries (e.g., BooleanQuery)
/// 3. Limit the number of expanded terms to prevent resource exhaustion
///
/// # Example (conceptual - not fully implemented yet)
///
/// ```ignore
/// use yatagarasu::query::multi_term::MultiTermQuery;
/// use yatagarasu::query::fuzzy::FuzzyQuery;
///
/// let fuzzy_query = FuzzyQuery::new("content", "hello").max_edits(2);
///
/// // The query will enumerate terms from the index that match within edit distance 2
/// let reader = index.reader()?;
/// let matching_terms = fuzzy_query.enumerate_terms(&reader)?;
///
/// // Results are limited by max_expansions (default 50)
/// println!("Found {} matching terms", matching_terms.len());
/// ```
pub trait MultiTermQuery: Query {
    /// Get the field name this query searches in.
    fn field(&self) -> &str;

    /// Enumerate terms from the index that match this query's criteria.
    ///
    /// This method should:
    /// 1. Access the term dictionary for the field
    /// 2. Iterate over terms that potentially match
    /// 3. Filter terms based on the query's criteria (e.g., edit distance, pattern)
    /// 4. Limit results to max_expansions
    /// 5. Return terms sorted by relevance/score
    ///
    /// # Arguments
    ///
    /// * `reader` - The index reader to enumerate terms from
    ///
    /// # Returns
    ///
    /// A vector of tuples containing:
    /// - `term`: The matching term text
    /// - `doc_freq`: Number of documents containing this term
    /// - `boost`: Optional boost factor for this term (default 1.0)
    ///
    /// # Performance
    ///
    /// Implementations should use efficient term dictionary enumeration rather than
    /// scanning all documents. See the `TermsEnum` trait for the proper API.
    fn enumerate_terms(&self, reader: &dyn IndexReader) -> Result<Vec<(String, u64, f32)>>;

    /// Get the maximum number of terms this query will expand to.
    ///
    /// This prevents queries from matching too many terms and consuming
    /// excessive resources. Default should be 50, same as Lucene.
    fn max_expansions(&self) -> usize {
        50
    }

    /// Create a TermsEnum that filters terms according to this query's criteria.
    ///
    /// This is the preferred method for implementing efficient multi-term queries.
    /// Instead of enumerating all terms and filtering in memory, this creates
    /// a TermsEnum that only yields matching terms.
    ///
    /// # Example (conceptual)
    ///
    /// ```ignore
    /// // For a FuzzyQuery:
    /// fn get_terms_enum(&self, reader: &dyn IndexReader) -> Result<Box<dyn TermsEnum>> {
    ///     let terms = reader.terms(self.field)?;
    ///     let automaton = LevenshteinAutomaton::build(&self.term, self.max_edits);
    ///     Ok(Box::new(AutomatonTermsEnum::new(terms, automaton)))
    /// }
    /// ```
    fn get_terms_enum(&self, _reader: &dyn IndexReader) -> Result<Option<Box<dyn TermsEnum>>> {
        // Default implementation returns None, indicating that enumerate_terms()
        // should be used instead. Implementations should override this for better performance.
        Ok(None)
    }
}

/// Rewrite strategies for multi-term queries.
///
/// Similar to Lucene's RewriteMethod, these strategies determine how a
/// multi-term query is converted into a simpler form for execution.
///
/// # Strategies
///
/// - **TopTermsRewrite**: Collect the top N scoring terms (default)
/// - **ConstantScoreRewrite**: All matching terms get the same score
/// - **BooleanRewrite**: Convert to BooleanQuery with all matching terms
///
/// The choice of strategy affects both performance and scoring behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RewriteMethod {
    /// Collect the top N terms by score, then create a BooleanQuery.
    /// This is the default and most performant for scoring queries.
    ///
    /// Similar to Lucene's TopTermsScoringBooleanQueryRewrite.
    TopTermsScoring { max_expansions: usize },

    /// Collect the top N terms by document frequency, assign constant score.
    /// Good for filtering without needing accurate scores.
    ///
    /// Similar to Lucene's TopTermsBlendedFreqScoringRewrite.
    TopTermsBlended { max_expansions: usize },

    /// All matching terms get a constant score equal to the query boost.
    /// Most efficient when you don't need term-specific scoring.
    ///
    /// Similar to Lucene's CONSTANT_SCORE_REWRITE.
    ConstantScore,

    /// Convert to BooleanQuery with all matching terms.
    /// May hit max clause count limits for queries matching many terms.
    ///
    /// Similar to Lucene's SCORING_BOOLEAN_REWRITE.
    BooleanQuery,
}

impl Default for RewriteMethod {
    fn default() -> Self {
        // Use TopTermsBlended as default, same as Lucene's FuzzyQuery
        RewriteMethod::TopTermsBlended {
            max_expansions: 50,
        }
    }
}

impl RewriteMethod {
    /// Get the maximum number of terms to expand to, if applicable.
    pub fn max_expansions(&self) -> Option<usize> {
        match self {
            RewriteMethod::TopTermsScoring { max_expansions } => Some(*max_expansions),
            RewriteMethod::TopTermsBlended { max_expansions } => Some(*max_expansions),
            RewriteMethod::ConstantScore => None,
            RewriteMethod::BooleanQuery => None,
        }
    }

    /// Check if this rewrite method uses constant scoring.
    pub fn is_constant_score(&self) -> bool {
        matches!(self, RewriteMethod::ConstantScore)
    }

    /// Check if this rewrite method limits the number of expanded terms.
    pub fn is_top_terms(&self) -> bool {
        matches!(
            self,
            RewriteMethod::TopTermsScoring { .. } | RewriteMethod::TopTermsBlended { .. }
        )
    }
}

// TODO: Implement MultiTermQuery for FuzzyQuery
// TODO: Implement PrefixQuery
// TODO: Implement WildcardQuery (already exists, needs to implement this trait)
// TODO: Implement RegexpQuery
// TODO: Add rewrite() method to Query trait
// TODO: Implement AutomatonTermsEnum for efficient term filtering
