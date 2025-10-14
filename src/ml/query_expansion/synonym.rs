//! Synonym-based query expansion.
//!
//! This module uses the SynonymGraphFilter from the analysis layer to perform
//! token-graph-aware synonym expansion for queries.

use crate::analysis::token::Token;
use crate::analysis::token_filter::Filter;
use crate::analysis::token_filter::synonym_graph::{SynonymDictionary, SynonymGraphFilter};
use crate::error::Result;
use crate::ml::MLContext;
use crate::query::{BooleanQuery, BooleanQueryBuilder, Query, TermQuery};

use super::expander::QueryExpander;
use super::types::{ExpandedQueryClause, ExpansionType};

/// Synonym-based query expander.
///
/// This expander uses the SynonymGraphFilter to expand query terms using a dictionary of synonyms.
/// It properly handles multi-word synonyms by processing tokens as a graph.
pub struct SynonymQueryExpander {
    filter: SynonymGraphFilter,
    weight: f64,
}

impl SynonymQueryExpander {
    /// Create a new synonym expander.
    ///
    /// # Arguments
    /// * `dict_path` - Optional path to JSON synonym dictionary file
    /// * `weight` - Weight multiplier for expanded terms (0.0-1.0)
    pub fn new(dict_path: Option<&str>, weight: f64) -> Result<Self> {
        let dictionary = SynonymDictionary::new(dict_path)?;
        let filter = SynonymGraphFilter::new(dictionary, true); // keep_original = true

        Ok(Self { filter, weight })
    }

    /// Extract query paths from graph tokens.
    ///
    /// This processes tokens with position_increment and position_length attributes
    /// to find all possible paths through the synonym graph.
    fn extract_query_paths(&self, graph_tokens: &[Token]) -> Vec<Vec<Token>> {
        if graph_tokens.is_empty() {
            return vec![];
        }

        let mut paths: Vec<Vec<Token>> = vec![];

        // Group tokens by position
        let mut position_groups: Vec<Vec<Token>> = vec![];
        let mut current_group: Vec<Token> = vec![];

        for token in graph_tokens {
            if token.position_increment > 0 && !current_group.is_empty() {
                position_groups.push(current_group.clone());
                current_group.clear();
            }
            current_group.push(token.clone());
        }

        if !current_group.is_empty() {
            position_groups.push(current_group);
        }

        // For simplicity, we generate paths by taking alternatives at each position
        // A more sophisticated implementation would do proper graph traversal

        // For now, return the tokens that are not synonyms (original path)
        // and separate paths for each synonym variant
        let mut original_path = vec![];
        let mut synonym_paths: Vec<Vec<Token>> = vec![];

        for group in &position_groups {
            // Find original token (not a synonym)
            let original = group
                .iter()
                .find(|t| {
                    t.metadata
                        .as_ref()
                        .and_then(|m| m.token_type)
                        .map(|tt| tt != crate::analysis::token::TokenType::Synonym)
                        .unwrap_or(true)
                })
                .or_else(|| group.first());

            if let Some(orig) = original {
                original_path.push(orig.clone());
            }

            // Find synonym alternatives
            for token in group {
                if let Some(metadata) = &token.metadata
                    && metadata.token_type == Some(crate::analysis::token::TokenType::Synonym)
                {
                    // This is a synonym token
                    // For multi-token synonyms, collect the sequence
                    if token.position_length > 1 {
                        // Single-word synonym covering multiple positions
                        let mut syn_path = original_path[..original_path.len() - 1].to_vec();
                        syn_path.push(token.clone());
                        synonym_paths.push(syn_path);
                    } else {
                        // Single position synonym
                        let mut syn_path = original_path[..original_path.len() - 1].to_vec();
                        syn_path.push(token.clone());
                        synonym_paths.push(syn_path);
                    }
                }
            }
        }

        paths.push(original_path);
        paths.extend(synonym_paths);

        // Filter out empty paths
        paths.into_iter().filter(|p| !p.is_empty()).collect()
    }

    /// Build a query from a path of tokens.
    fn build_query_from_path(&self, tokens: &[Token], field: &str) -> Box<dyn Query> {
        if tokens.is_empty() {
            return Box::new(BooleanQuery::new());
        }

        if tokens.len() == 1 {
            let token = &tokens[0];
            let mut query = Box::new(TermQuery::new(field, token.text.clone())) as Box<dyn Query>;

            // Check if it's a multi-word synonym (position_length > 1)
            // In that case, don't add additional boost here
            let base_boost = if token.position_length > 1 {
                0.9 // Slightly lower for multi-word matches
            } else {
                0.8
            };

            query.set_boost((base_boost * self.weight) as f32);
            return query;
        }

        // Multiple tokens - could be original phrase or multi-word synonym sequence
        // Use BooleanQuery with MUST clauses
        let mut builder = BooleanQueryBuilder::new();
        for token in tokens {
            let term_query = Box::new(TermQuery::new(field, token.text.clone())) as Box<dyn Query>;
            builder = builder.must(term_query);
        }

        let mut query = Box::new(builder.build()) as Box<dyn Query>;
        query.set_boost((0.8 * self.weight) as f32);
        query
    }
}

impl QueryExpander for SynonymQueryExpander {
    fn expand(
        &self,
        tokens: &[String],
        field: &str,
        _context: &MLContext,
    ) -> Result<Vec<ExpandedQueryClause>> {
        // Convert string tokens to Token objects
        let token_objs: Vec<Token> = tokens
            .iter()
            .enumerate()
            .map(|(i, text)| Token::new(text, i))
            .collect();

        // Apply the synonym graph filter
        let graph_tokens: Vec<Token> = self
            .filter
            .filter(Box::new(token_objs.into_iter()))?
            .collect();

        // Extract paths from the graph
        let paths = self.extract_query_paths(&graph_tokens);

        // Convert paths to ExpandedQueryClause
        let mut expansions = Vec::new();

        for (path_idx, path) in paths.iter().enumerate() {
            // Skip the first path (original query)
            if path_idx == 0 {
                continue;
            }

            let query = self.build_query_from_path(path, field);
            let source_term = tokens.join(" ");

            expansions.push(ExpandedQueryClause {
                query,
                confidence: 0.8,
                expansion_type: ExpansionType::Synonym,
                source_term,
            });
        }

        Ok(expansions)
    }

    fn name(&self) -> &str {
        "synonym"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synonym_query_expander_basic() {
        let expander = SynonymQueryExpander::new(Some("resource/ml/synonyms.json"), 0.8).unwrap();
        let ml_context = MLContext::default();

        let result = expander
            .expand(&["ml".to_string()], "content", &ml_context)
            .unwrap();

        // Should have expansions for "machine learning", etc.
        assert!(!result.is_empty());
    }

    #[test]
    fn test_synonym_query_expander_multi_word() {
        let expander = SynonymQueryExpander::new(Some("resource/ml/synonyms.json"), 0.8).unwrap();
        let ml_context = MLContext::default();

        let result = expander
            .expand(
                &["data".to_string(), "science".to_string()],
                "content",
                &ml_context,
            )
            .unwrap();

        // Should have expansions
        assert!(!result.is_empty());
    }
}
