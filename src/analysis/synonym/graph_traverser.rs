//! Synonym graph traversal for extracting paths from synonym graphs.
//!
//! This module provides utilities for traversing token graphs created by synonym expansion,
//! extracting all possible paths through the graph based on position_increment and position_length.

use crate::analysis::token::{Token, TokenType};

/// Traverses synonym graphs to extract all possible paths.
pub struct SynonymGraphTraverser;

impl SynonymGraphTraverser {
    /// Extract all query paths from graph tokens.
    ///
    /// This processes tokens with position_increment and position_length attributes
    /// to find all possible paths through the synonym graph.
    ///
    /// # Returns
    /// A vector of paths, where each path is a vector of tokens.
    /// The first path is typically the original (non-synonym) path.
    pub fn extract_paths(graph_tokens: &[Token]) -> Vec<Vec<Token>> {
        if graph_tokens.is_empty() {
            return vec![];
        }

        let mut paths: Vec<Vec<Token>> = vec![];

        // Group tokens by position
        let position_groups = Self::group_by_position(graph_tokens);

        // Extract original path and synonym paths
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
                        .map(|tt| tt != TokenType::Synonym)
                        .unwrap_or(true)
                })
                .or_else(|| group.first());

            if let Some(orig) = original {
                original_path.push(orig.clone());
            }

            // Find synonym alternatives
            for token in group {
                if let Some(metadata) = &token.metadata
                    && metadata.token_type == Some(TokenType::Synonym)
                {
                    // This is a synonym token
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

    /// Group tokens by their position.
    ///
    /// Uses position_increment to determine when tokens should be grouped together
    /// (i.e., when they occupy the same position in the token graph).
    ///
    /// # Returns
    /// A vector of token groups, where each group represents tokens at the same position.
    pub fn group_by_position(graph_tokens: &[Token]) -> Vec<Vec<Token>> {
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

        position_groups
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::token::TokenMetadata;

    #[test]
    fn test_group_by_position() {
        let tokens = vec![
            Token::new("ml", 0),
            {
                let mut t = Token::new("machine", 0);
                t.position_increment = 0;
                t
            },
            Token::new("and", 1),
        ];

        let groups = SynonymGraphTraverser::group_by_position(&tokens);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].len(), 2); // ml, machine
        assert_eq!(groups[1].len(), 1); // and
    }

    #[test]
    fn test_extract_paths_simple() {
        let tokens = vec![
            Token::new("ml", 0),
            {
                let mut t = Token::new("machine", 0);
                t.position_increment = 0;
                t.metadata = Some(TokenMetadata {
                    original_text: None,
                    token_type: Some(TokenType::Synonym),
                    language: None,
                    attributes: std::collections::HashMap::new(),
                });
                t
            },
            Token::new("tutorial", 1),
        ];

        let paths = SynonymGraphTraverser::extract_paths(&tokens);
        assert!(paths.len() >= 2);

        // First path should be original
        assert_eq!(paths[0][0].text, "ml");
        assert_eq!(paths[0][1].text, "tutorial");

        // Should have a synonym path with "machine"
        let has_machine_path = paths.iter().any(|p| p.iter().any(|t| t.text == "machine"));
        assert!(has_machine_path);
    }

    #[test]
    fn test_extract_paths_with_position_length() {
        let tokens = vec![Token::new("machine", 0), Token::new("learning", 1), {
            let mut t = Token::new("ml", 0);
            t.position_increment = 0;
            t.position_length = 2;
            t.metadata = Some(TokenMetadata {
                original_text: None,
                token_type: Some(TokenType::Synonym),
                language: None,
                attributes: std::collections::HashMap::new(),
            });
            t
        }];

        let paths = SynonymGraphTraverser::extract_paths(&tokens);
        assert!(paths.len() >= 2);

        // Should have path with "ml" (position_length=2)
        let ml_path = paths.iter().find(|p| p.iter().any(|t| t.text == "ml"));
        assert!(ml_path.is_some());
    }
}
