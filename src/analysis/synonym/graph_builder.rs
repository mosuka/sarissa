//! Token graph builder for constructing synonym graphs.
//!
//! This module provides the core logic for building token graphs with proper
//! position_increment and position_length attributes for synonym expansion.

use crate::analysis::token::{Token, TokenType};
use crate::analysis::tokenizer::Tokenizer;

use super::dictionary::SynonymDictionary;

/// Builds token graphs with synonym expansion.
pub struct SynonymGraphBuilder {
    dictionary: SynonymDictionary,
    tokenizer: Option<Box<dyn Tokenizer>>,
    keep_original: bool,
    /// Boost multiplier for synonym tokens (None means no boost adjustment)
    synonym_boost: Option<f32>,
}

impl SynonymGraphBuilder {
    /// Create a new graph builder.
    ///
    /// # Arguments
    /// * `dictionary` - The synonym dictionary to use
    /// * `keep_original` - Whether to keep original tokens alongside synonyms
    pub fn new(dictionary: SynonymDictionary, keep_original: bool) -> Self {
        Self {
            dictionary,
            tokenizer: None,
            keep_original,
            synonym_boost: None,
        }
    }

    /// Create a new graph builder with a tokenizer.
    ///
    /// The tokenizer will be used to split multi-word synonyms into individual tokens.
    ///
    /// # Arguments
    /// * `dictionary` - The synonym dictionary to use
    /// * `tokenizer` - Tokenizer for splitting synonym terms
    /// * `keep_original` - Whether to keep original tokens alongside synonyms
    pub fn with_tokenizer(
        dictionary: SynonymDictionary,
        tokenizer: Box<dyn Tokenizer>,
        keep_original: bool,
    ) -> Self {
        Self {
            dictionary,
            tokenizer: Some(tokenizer),
            keep_original,
            synonym_boost: None,
        }
    }

    /// Set the boost multiplier for synonym tokens.
    ///
    /// # Arguments
    /// * `boost` - Boost multiplier (e.g., 0.8 to reduce synonym weight to 80%)
    ///
    /// # Example
    /// ```
    /// use sarissa::analysis::synonym::{SynonymDictionary, SynonymGraphBuilder};
    ///
    /// let mut dict = SynonymDictionary::new(None).unwrap();
    /// dict.add_synonym_group(vec!["ml".to_string(), "machine learning".to_string()]);
    ///
    /// let builder = SynonymGraphBuilder::new(dict, true)
    ///     .with_boost(0.8); // Synonyms get 80% of original weight
    /// ```
    pub fn with_boost(mut self, boost: f32) -> Self {
        self.synonym_boost = Some(boost);
        self
    }

    /// Build graph tokens from matched synonyms.
    ///
    /// This creates tokens with proper position_increment and position_length attributes
    /// to represent the synonym graph structure.
    pub fn build_graph_tokens(
        &self,
        original_tokens: &[Token],
        match_start: usize,
        match_length: usize,
        synonyms: &[String],
    ) -> Vec<Token> {
        let mut result = Vec::new();
        let match_start_offset = original_tokens[match_start].start_offset;
        let match_end_offset = original_tokens[match_start + match_length - 1].end_offset;

        // Add original tokens if keep_original is true
        if self.keep_original {
            for (i, original) in original_tokens[match_start..match_start + match_length]
                .iter()
                .enumerate()
            {
                let mut token = original.clone();
                token.position_increment = if i == 0 { 1 } else { 0 };
                token.position_length = 1;
                result.push(token);
            }
        }

        // Add synonym tokens
        for synonym in synonyms {
            // Tokenize the synonym using the tokenizer if available
            let syn_tokens = if let Some(tokenizer) = &self.tokenizer {
                // Use tokenizer to split synonym
                match tokenizer.tokenize(synonym) {
                    Ok(tokens) => tokens.map(|t| t.text).collect::<Vec<_>>(),
                    Err(_) => {
                        // Fallback to whitespace splitting on error
                        synonym.split_whitespace().map(|s| s.to_string()).collect()
                    }
                }
            } else {
                // Default: split by whitespace
                synonym.split_whitespace().map(|s| s.to_string()).collect()
            };

            if syn_tokens.len() == 1 {
                // Single-word synonym
                let mut token = Token::new(&syn_tokens[0], original_tokens[match_start].position);
                token.position_increment = if self.keep_original { 0 } else { 1 };
                token.position_length = match_length; // Spans the original phrase length
                token.start_offset = match_start_offset;
                token.end_offset = match_end_offset;
                token = token.with_token_type(TokenType::Synonym);

                // Apply boost if configured
                if let Some(boost) = self.synonym_boost {
                    // Single-word synonyms spanning multiple positions get slightly higher boost
                    let base_boost = if match_length > 1 { 0.9 } else { 0.8 };
                    token = token.with_boost(base_boost * boost);
                }

                result.push(token);
            } else {
                // Multi-word synonym
                for (i, syn_word) in syn_tokens.iter().enumerate() {
                    let mut token = Token::new(syn_word, original_tokens[match_start].position + i);
                    token.position_increment = if i == 0 {
                        if self.keep_original { 0 } else { 1 }
                    } else {
                        1
                    };
                    // First token spans the entire synonym phrase length
                    token.position_length = if i == 0 { syn_tokens.len() } else { 1 };
                    token.start_offset = match_start_offset;
                    token.end_offset = match_end_offset;
                    token = token.with_token_type(TokenType::Synonym);

                    // Apply boost if configured
                    if let Some(boost) = self.synonym_boost {
                        // Multi-word synonyms: first token gets slightly higher boost
                        let base_boost = if i == 0 { 0.9 } else { 0.8 };
                        token = token.with_boost(base_boost * boost);
                    }

                    result.push(token);
                }
            }
        }

        result
    }

    /// Try to match a synonym starting at the given position in the token buffer.
    ///
    /// Returns (matched_phrase, matched_length, synonyms) if a match is found.
    pub fn try_match_synonym(
        &self,
        tokens: &[Token],
        start: usize,
    ) -> Option<(String, usize, Vec<String>)> {
        let max_len = (tokens.len() - start).min(self.dictionary.max_phrase_length());

        // Try longest match first (greedy matching)
        for len in (1..=max_len).rev() {
            // Check byte offset continuity for multi-token phrases
            if len > 1 {
                // Check if all tokens are ALPHANUM type
                let all_alphanum = tokens[start..start + len].iter().all(|t| {
                    matches!(
                        t.metadata.as_ref().and_then(|m| m.token_type),
                        Some(TokenType::Alphanum) | Some(TokenType::Num)
                    )
                });

                // If not all ALPHANUM, check byte offset continuity
                if !all_alphanum {
                    let mut is_continuous = true;
                    for i in 0..len - 1 {
                        let current = &tokens[start + i];
                        let next = &tokens[start + i + 1];
                        if current.end_offset != next.start_offset {
                            is_continuous = false;
                            break;
                        }
                    }
                    // Skip this length if tokens are not continuous
                    if !is_continuous {
                        continue;
                    }
                }
            }

            let token_texts: Vec<&str> = tokens[start..start + len]
                .iter()
                .map(|t| t.text.as_str())
                .collect();

            // Try with space separator first (for English, etc.)
            let phrase_with_space = token_texts.join(" ");
            if let Some(synonyms) = self.dictionary.get_synonyms(&phrase_with_space) {
                return Some((phrase_with_space, len, synonyms.clone()));
            }

            // Try without space separator (for Japanese, Chinese, etc.)
            // Only do this for multi-token phrases
            if len > 1 {
                let phrase_no_space = token_texts.join("");
                if let Some(synonyms) = self.dictionary.get_synonyms(&phrase_no_space) {
                    return Some((phrase_no_space, len, synonyms.clone()));
                }
            }
        }

        None
    }

    /// Get a reference to the dictionary.
    pub fn dictionary(&self) -> &SynonymDictionary {
        &self.dictionary
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_graph_tokens_single_word_synonym() {
        let mut dict = SynonymDictionary::new(None).unwrap();
        dict.add_synonym_group(vec!["big".to_string(), "large".to_string()]);

        let builder = SynonymGraphBuilder::new(dict, true);

        let original_tokens = vec![Token::new("big", 0)];
        let synonyms = vec!["large".to_string()];

        let result = builder.build_graph_tokens(&original_tokens, 0, 1, &synonyms);

        // Should have original + synonym
        assert!(result.len() >= 2);

        // Find synonym token
        let large_token = result.iter().find(|t| t.text == "large");
        assert!(large_token.is_some());
        let large = large_token.unwrap();
        assert_eq!(large.position_increment, 0);
        assert_eq!(large.position_length, 1);
    }

    #[test]
    fn test_build_graph_tokens_multi_word_synonym() {
        use crate::analysis::tokenizer::WhitespaceTokenizer;

        let mut dict = SynonymDictionary::new(None).unwrap();
        dict.add_synonym_group(vec!["ml".to_string(), "machine learning".to_string()]);

        let tokenizer = Box::new(WhitespaceTokenizer);
        let builder = SynonymGraphBuilder::with_tokenizer(dict, tokenizer, true);

        let original_tokens = vec![Token::new("ml", 0)];
        let synonyms = vec!["machine learning".to_string()];

        let result = builder.build_graph_tokens(&original_tokens, 0, 1, &synonyms);

        // Find "machine" token (first of multi-word synonym)
        let machine_token = result.iter().find(|t| t.text == "machine");
        assert!(machine_token.is_some());
        let machine = machine_token.unwrap();
        assert_eq!(machine.position_increment, 0);
        assert_eq!(machine.position_length, 2); // Spans 2 positions

        // Find "learning" token (second of multi-word synonym)
        let learning_token = result.iter().find(|t| t.text == "learning");
        assert!(learning_token.is_some());
        let learning = learning_token.unwrap();
        assert_eq!(learning.position_increment, 1);
        assert_eq!(learning.position_length, 1);
    }

    #[test]
    fn test_try_match_synonym() {
        let mut dict = SynonymDictionary::new(None).unwrap();
        dict.add_synonym_group(vec!["ml".to_string(), "machine learning".to_string()]);

        let builder = SynonymGraphBuilder::new(dict, true);

        let tokens = vec![Token::new("ml", 0), Token::new("tutorial", 1)];

        let result = builder.try_match_synonym(&tokens, 0);
        assert!(result.is_some());
        let (phrase, len, synonyms) = result.unwrap();
        assert_eq!(phrase, "ml");
        assert_eq!(len, 1);
        assert!(synonyms.contains(&"machine learning".to_string()));
    }

    #[test]
    fn test_boost_single_word_synonym() {
        let mut dict = SynonymDictionary::new(None).unwrap();
        dict.add_synonym_group(vec!["big".to_string(), "large".to_string()]);

        let builder = SynonymGraphBuilder::new(dict, true).with_boost(0.8);

        let original_tokens = vec![Token::new("big", 0)];
        let synonyms = vec!["large".to_string()];

        let result = builder.build_graph_tokens(&original_tokens, 0, 1, &synonyms);

        // Find synonym token
        let large_token = result.iter().find(|t| t.text == "large");
        assert!(large_token.is_some());
        let large = large_token.unwrap();

        // Boost should be applied: 0.8 (base) * 0.8 (multiplier) = 0.64
        assert!((large.boost - 0.64).abs() < 0.001);
    }

    #[test]
    fn test_boost_multi_word_synonym() {
        use crate::analysis::tokenizer::WhitespaceTokenizer;

        let mut dict = SynonymDictionary::new(None).unwrap();
        dict.add_synonym_group(vec!["ml".to_string(), "machine learning".to_string()]);

        let tokenizer = Box::new(WhitespaceTokenizer);
        let builder = SynonymGraphBuilder::with_tokenizer(dict, tokenizer, true).with_boost(0.8);

        let original_tokens = vec![Token::new("ml", 0)];
        let synonyms = vec!["machine learning".to_string()];

        let result = builder.build_graph_tokens(&original_tokens, 0, 1, &synonyms);

        // Find "machine" token (first of multi-word synonym)
        let machine_token = result.iter().find(|t| t.text == "machine");
        assert!(machine_token.is_some());
        let machine = machine_token.unwrap();

        // First token boost: 0.9 (base) * 0.8 (multiplier) = 0.72
        assert!((machine.boost - 0.72).abs() < 0.001);

        // Find "learning" token (second of multi-word synonym)
        let learning_token = result.iter().find(|t| t.text == "learning");
        assert!(learning_token.is_some());
        let learning = learning_token.unwrap();

        // Second token boost: 0.8 (base) * 0.8 (multiplier) = 0.64
        assert!((learning.boost - 0.64).abs() < 0.001);
    }

    #[test]
    fn test_boost_not_applied_when_not_configured() {
        let mut dict = SynonymDictionary::new(None).unwrap();
        dict.add_synonym_group(vec!["big".to_string(), "large".to_string()]);

        // No boost configured
        let builder = SynonymGraphBuilder::new(dict, true);

        let original_tokens = vec![Token::new("big", 0)];
        let synonyms = vec!["large".to_string()];

        let result = builder.build_graph_tokens(&original_tokens, 0, 1, &synonyms);

        // Find synonym token
        let large_token = result.iter().find(|t| t.text == "large");
        assert!(large_token.is_some());
        let large = large_token.unwrap();

        // Default boost should be 1.0
        assert_eq!(large.boost, 1.0);
    }
}
