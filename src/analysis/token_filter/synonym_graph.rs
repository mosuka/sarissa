//! Synonym graph filter for token expansion with multi-word synonym support.
//!
//! This filter applies synonyms from a dictionary to an incoming token stream,
//! producing a correct graph output similar to Lucene's SynonymGraphFilter.

use std::sync::Arc;

use fst::{Map, MapBuilder, Streamer};

use crate::analysis::token::{Token, TokenStream, TokenType};
use crate::analysis::tokenizer::Tokenizer;
use crate::error::Result;

use super::Filter;

/// Synonym dictionary for token expansion.
///
/// Maps terms to their synonyms using FST (Finite State Transducer) for memory efficiency.
/// FST provides dramatic memory savings (10-100x) for large dictionaries (100k+ entries)
/// while maintaining fast lookup performance.
#[derive(Debug, Clone)]
pub struct SynonymDictionary {
    /// FST map: term -> index into synonym_lists
    fst_map: Arc<Map<Arc<[u8]>>>,
    /// Actual synonym lists indexed by FST values
    synonym_lists: Arc<Vec<Vec<String>>>,
    /// Maximum number of tokens to look ahead for multi-word synonym matching
    max_phrase_length: usize,
}

impl Default for SynonymDictionary {
    fn default() -> Self {
        Self::new(None).unwrap()
    }
}

impl SynonymDictionary {
    /// Create a new synonym dictionary.
    ///
    /// If `path` is provided, loads synonyms from the specified JSON file.
    /// If `path` is `None`, creates an empty dictionary.
    pub fn new(path: Option<&str>) -> Result<Self> {
        match path {
            Some(file_path) => Self::load_from_file(file_path),
            None => {
                // Create empty FST
                let builder = MapBuilder::memory();
                let fst_bytes = builder.into_inner().unwrap();
                let fst_map = Map::new(Arc::from(fst_bytes)).unwrap();

                Ok(Self {
                    fst_map: Arc::new(fst_map),
                    synonym_lists: Arc::new(Vec::new()),
                    max_phrase_length: 1,
                })
            }
        }
    }

    /// Load synonym dictionary from a JSON file.
    ///
    /// The JSON file should contain an array of synonym groups, where each group
    /// is an array of terms that are synonyms of each other.
    ///
    /// Example format:
    /// ```json
    /// [
    ///   ["ml", "machine learning", "machine-learning"],
    ///   ["ai", "artificial intelligence"]
    /// ]
    /// ```
    pub fn load_from_file(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            crate::error::SarissaError::storage(format!(
                "Failed to read synonym dictionary file '{}': {}",
                path, e
            ))
        })?;

        let synonym_groups: Vec<Vec<String>> = serde_json::from_str(&content).map_err(|e| {
            crate::error::SarissaError::parse(format!(
                "Failed to parse synonym dictionary JSON from '{}': {}",
                path, e
            ))
        })?;

        Self::from_synonym_groups(synonym_groups)
    }

    /// Build a synonym dictionary from synonym groups.
    fn from_synonym_groups(synonym_groups: Vec<Vec<String>>) -> Result<Self> {
        use std::collections::HashMap;

        // First, build all synonym mappings
        let mut term_to_synonyms: HashMap<String, Vec<String>> = HashMap::new();
        let mut max_phrase_length = 1;

        for group in synonym_groups {
            if group.is_empty() {
                continue;
            }

            // Calculate max phrase length for this group
            let max_words = group
                .iter()
                .map(|t| {
                    let word_count = t.split_whitespace().count();
                    if word_count == 1 {
                        let has_ascii = t.chars().any(|c| c.is_ascii_alphanumeric());
                        let char_count = t.chars().count();
                        if !has_ascii && char_count > 3 {
                            char_count.div_ceil(2)
                        } else {
                            1
                        }
                    } else {
                        word_count
                    }
                })
                .max()
                .unwrap_or(1);
            max_phrase_length = max_phrase_length.max(max_words);

            // Create bidirectional mappings
            for (i, term) in group.iter().enumerate() {
                let mut synonyms = Vec::new();
                for (j, other_term) in group.iter().enumerate() {
                    if i != j {
                        synonyms.push(other_term.clone());
                    }
                }
                term_to_synonyms.insert(term.clone(), synonyms);
            }
        }

        // Build FST from sorted keys
        let mut synonym_lists = Vec::new();
        let mut sorted_terms: Vec<_> = term_to_synonyms.keys().cloned().collect();
        sorted_terms.sort();

        let mut builder = MapBuilder::memory();
        for term in sorted_terms {
            let synonyms = term_to_synonyms.remove(&term).unwrap();
            let index = synonym_lists.len() as u64;
            synonym_lists.push(synonyms);
            builder.insert(term.as_bytes(), index).map_err(|e| {
                crate::error::SarissaError::parse(format!("FST build error: {}", e))
            })?;
        }

        let fst_bytes = builder
            .into_inner()
            .map_err(|e| crate::error::SarissaError::parse(format!("FST finalize error: {}", e)))?;
        let fst_map = Map::new(Arc::from(fst_bytes))
            .map_err(|e| crate::error::SarissaError::parse(format!("FST creation error: {}", e)))?;

        Ok(Self {
            fst_map: Arc::new(fst_map),
            synonym_lists: Arc::new(synonym_lists),
            max_phrase_length,
        })
    }

    /// Get synonyms for a given term or phrase.
    pub fn get_synonyms(&self, term: &str) -> Option<&Vec<String>> {
        let index = self.fst_map.get(term.as_bytes())? as usize;
        self.synonym_lists.get(index)
    }

    /// Add a synonym group where all terms are synonyms of each other.
    ///
    /// Note: This method rebuilds the entire FST, so it's inefficient for adding
    /// many groups one at a time. Prefer using `from_synonym_groups` or `load_from_file`
    /// for bulk loading.
    ///
    /// For example, adding `["big", "large", "huge"]` will create:
    /// - "big" -> ["large", "huge"]
    /// - "large" -> ["big", "huge"]
    /// - "huge" -> ["big", "large"]
    pub fn add_synonym_group(&mut self, terms: Vec<String>) {
        // Extract existing mappings from FST
        let mut all_groups = Vec::new();
        let mut processed_terms = std::collections::HashSet::new();

        // Collect existing synonym groups using FST streamer
        let mut stream = self.fst_map.stream();
        while let Some((key, value)) = stream.next() {
            let term = String::from_utf8_lossy(key).to_string();
            if processed_terms.contains(&term) {
                continue;
            }

            let index = value as usize;
            if let Some(synonyms) = self.synonym_lists.get(index) {
                let mut group = vec![term.clone()];
                group.extend(synonyms.clone());
                processed_terms.insert(term);
                for syn in synonyms {
                    processed_terms.insert(syn.clone());
                }
                all_groups.push(group);
            }
        }

        // Add new group
        all_groups.push(terms);

        // Rebuild FST
        *self = Self::from_synonym_groups(all_groups).unwrap();
    }

    /// Get the maximum phrase length in the dictionary.
    pub fn max_phrase_length(&self) -> usize {
        self.max_phrase_length
    }
}

/// Synonym graph filter that applies synonyms while maintaining correct token graph structure.
///
/// This filter produces a token graph with proper `position_increment` and `position_length`
/// attributes, enabling correct phrase query matching with multi-word synonyms.
pub struct SynonymGraphFilter {
    dictionary: SynonymDictionary,
    /// Whether to keep the original tokens when synonyms are found
    keep_original: bool,
    /// Optional tokenizer for tokenizing multi-word synonyms
    tokenizer: Option<Box<dyn Tokenizer>>,
}

impl SynonymGraphFilter {
    /// Create a new synonym graph filter.
    ///
    /// # Arguments
    /// * `dictionary` - The synonym dictionary to use
    /// * `keep_original` - If true, keep original tokens alongside synonyms
    pub fn new(dictionary: SynonymDictionary, keep_original: bool) -> Self {
        Self {
            dictionary,
            keep_original,
            tokenizer: None,
        }
    }

    /// Create a new synonym graph filter with a tokenizer.
    ///
    /// The tokenizer will be used to tokenize multi-word synonyms, which is essential
    /// for languages like Japanese where words are not separated by whitespace.
    ///
    /// # Arguments
    /// * `dictionary` - The synonym dictionary to use
    /// * `tokenizer` - Tokenizer to split synonym terms (e.g., for "機械学習" → ["機械", "学習"])
    /// * `keep_original` - If true, keep original tokens alongside synonyms
    pub fn with_tokenizer(
        dictionary: SynonymDictionary,
        tokenizer: Box<dyn Tokenizer>,
        keep_original: bool,
    ) -> Self {
        Self {
            dictionary,
            keep_original,
            tokenizer: Some(tokenizer),
        }
    }

    /// Create a synonym graph filter from a dictionary file.
    pub fn from_file(path: &str, keep_original: bool) -> Result<Self> {
        let dictionary = SynonymDictionary::load_from_file(path)?;
        Ok(Self::new(dictionary, keep_original))
    }

    /// Create a synonym graph filter from a dictionary file with a tokenizer.
    pub fn from_file_with_tokenizer(
        path: &str,
        tokenizer: Box<dyn Tokenizer>,
        keep_original: bool,
    ) -> Result<Self> {
        let dictionary = SynonymDictionary::load_from_file(path)?;
        Ok(Self::with_tokenizer(dictionary, tokenizer, keep_original))
    }

    /// Try to match a synonym starting at the given position in the token buffer.
    ///
    /// Returns (matched_phrase, matched_length, synonyms) if a match is found.
    fn try_match_synonym(
        &self,
        tokens: &[Token],
        start: usize,
    ) -> Option<(String, usize, Vec<String>)> {
        let max_len = (tokens.len() - start).min(self.dictionary.max_phrase_length());

        // Try longest match first (greedy matching)
        for len in (1..=max_len).rev() {
            // Check byte offset continuity for multi-token phrases
            // Rules:
            // 1. If all tokens are ALPHANUM: skip continuity check (allow space-separated phrases)
            // 2. If any token is CJK or other: check byte offset continuity (compound words only)
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

    /// Build graph tokens from matched synonyms.
    fn build_graph_tokens(
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
                // Use tokenizer to split synonym (e.g., "機械学習" → ["機械", "学習"])
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

                // Mark as synonym in metadata
                token = token.with_token_type(TokenType::Synonym);

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
                    token.position_length = 1;
                    token.start_offset = match_start_offset;
                    token.end_offset = match_end_offset;
                    token = token.with_token_type(TokenType::Synonym);

                    result.push(token);
                }
            }
        }

        result
    }
}

impl Filter for SynonymGraphFilter {
    fn filter(&self, tokens: TokenStream) -> Result<TokenStream> {
        let input_tokens: Vec<Token> = tokens.collect();
        let mut output_tokens = Vec::new();
        let mut i = 0;

        while i < input_tokens.len() {
            // Try to match a synonym at the current position
            if let Some((_, match_length, synonyms)) = self.try_match_synonym(&input_tokens, i) {
                // Build and add graph tokens
                let graph_tokens =
                    self.build_graph_tokens(&input_tokens, i, match_length, &synonyms);
                output_tokens.extend(graph_tokens);
                i += match_length;
            } else {
                // No match, add the original token
                output_tokens.push(input_tokens[i].clone());
                i += 1;
            }
        }

        Ok(Box::new(output_tokens.into_iter()))
    }

    fn name(&self) -> &'static str {
        "synonym_graph"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synonym_dictionary_basic() {
        let mut dict = SynonymDictionary::new(None).unwrap();
        dict.add_synonym_group(vec![
            "big".to_string(),
            "large".to_string(),
            "huge".to_string(),
        ]);

        let synonyms = dict.get_synonyms("big").unwrap();
        assert!(synonyms.contains(&"large".to_string()));
        assert!(synonyms.contains(&"huge".to_string()));
        assert!(!synonyms.contains(&"big".to_string()));
    }

    #[test]
    fn test_synonym_dictionary_multi_word() {
        let mut dict = SynonymDictionary::new(None).unwrap();
        dict.add_synonym_group(vec![
            "ml".to_string(),
            "machine learning".to_string(),
            "machine-learning".to_string(),
        ]);

        assert_eq!(dict.max_phrase_length(), 2);

        let synonyms = dict.get_synonyms("machine learning").unwrap();
        assert!(synonyms.contains(&"ml".to_string()));
        assert!(synonyms.contains(&"machine-learning".to_string()));
    }

    #[test]
    fn test_synonym_graph_filter_single_word() {
        let mut dict = SynonymDictionary::new(None).unwrap();
        dict.add_synonym_group(vec!["big".to_string(), "large".to_string()]);

        let filter = SynonymGraphFilter::new(dict, true);

        let tokens = vec![
            Token::new("the", 0),
            Token::new("big", 1),
            Token::new("cat", 2),
        ];

        let result = filter
            .filter(Box::new(tokens.into_iter()))
            .unwrap()
            .collect::<Vec<_>>();

        // Should have: "the", "big" (original), "large" (synonym), "cat"
        assert!(result.len() >= 4);
        assert_eq!(result[0].text, "the");
        assert_eq!(result[1].text, "big");
        assert_eq!(result[2].text, "large");
        assert_eq!(result[2].position_increment, 0); // Same position as "big"
        assert_eq!(result[3].text, "cat");
    }

    #[test]
    fn test_synonym_graph_filter_multi_word() {
        let mut dict = SynonymDictionary::new(None).unwrap();
        dict.add_synonym_group(vec![
            "data science".to_string(),
            "ds".to_string(),
            "dataset".to_string(),
        ]);

        let filter = SynonymGraphFilter::new(dict, true);

        let tokens = vec![
            Token::new("data", 0),
            Token::new("science", 1),
            Token::new("intro", 2),
        ];

        let result = filter
            .filter(Box::new(tokens.into_iter()))
            .unwrap()
            .collect::<Vec<_>>();

        // Should include original "data", "science" and synonyms "ds", "dataset"
        assert!(result.len() >= 4);

        // Find the "ds" token
        let ds_token = result.iter().find(|t| t.text == "ds");
        assert!(ds_token.is_some());
        let ds = ds_token.unwrap();
        assert_eq!(ds.position_increment, 0); // Stacked on position 0
        assert_eq!(ds.position_length, 2); // Spans 2 positions
    }

    #[test]
    fn test_synonym_dictionary_load_from_file() {
        let dict = SynonymDictionary::load_from_file("resource/ml/synonyms.json").unwrap();

        // Test English synonyms
        let ml_synonyms = dict.get_synonyms("ml");
        assert!(ml_synonyms.is_some());
        let ml_synonyms = ml_synonyms.unwrap();
        assert!(ml_synonyms.contains(&"machine learning".to_string()));

        // Test Japanese synonyms
        let learning_synonyms = dict.get_synonyms("学習");
        assert!(learning_synonyms.is_some());
    }

    #[test]
    fn test_synonym_graph_filter_with_tokenizer() {
        use crate::analysis::tokenizer::WhitespaceTokenizer;

        // Create a simple synonym dictionary
        let mut dict = SynonymDictionary::new(None).unwrap();
        dict.add_synonym_group(vec!["ml".to_string(), "machine learning".to_string()]);

        // Create filter with tokenizer
        let tokenizer = Box::new(WhitespaceTokenizer);
        let filter = SynonymGraphFilter::with_tokenizer(dict, tokenizer, true);

        let tokens = vec![
            Token::new("ml", 0),
            Token::new("is", 1),
            Token::new("cool", 2),
        ];

        let result = filter
            .filter(Box::new(tokens.into_iter()))
            .unwrap()
            .collect::<Vec<_>>();

        // Should have: "ml", "machine", "learning", "is", "cool"
        assert!(result.len() >= 4);

        // Find "machine" and "learning" tokens
        let has_machine = result.iter().any(|t| t.text == "machine");
        let has_learning = result.iter().any(|t| t.text == "learning");

        assert!(has_machine, "Expected 'machine' token");
        assert!(has_learning, "Expected 'learning' token");
    }

    #[test]
    fn test_synonym_graph_filter_japanese_with_mock_tokenizer() {
        // Mock tokenizer that splits Japanese properly
        struct JapaneseTokenizer;
        impl crate::analysis::tokenizer::Tokenizer for JapaneseTokenizer {
            fn tokenize(
                &self,
                text: &str,
            ) -> crate::error::Result<crate::analysis::token::TokenStream> {
                // Simple mock: split "機械学習" into "機械" and "学習"
                let tokens: Vec<Token> = if text == "機械学習" {
                    vec![Token::new("機械", 0), Token::new("学習", 1)]
                } else {
                    vec![Token::new(text, 0)]
                };
                Ok(Box::new(tokens.into_iter()))
            }

            fn name(&self) -> &'static str {
                "japanese_tokenizer"
            }
        }

        // Create synonym dictionary with Japanese
        let mut dict = SynonymDictionary::new(None).unwrap();
        dict.add_synonym_group(vec!["ml".to_string(), "機械学習".to_string()]);

        // Create filter with Japanese tokenizer
        let tokenizer = Box::new(JapaneseTokenizer);
        let filter = SynonymGraphFilter::with_tokenizer(dict, tokenizer, true);

        let tokens = vec![Token::new("ml", 0)];

        let result = filter
            .filter(Box::new(tokens.into_iter()))
            .unwrap()
            .collect::<Vec<_>>();

        // Should have: "ml", "機械", "学習"
        assert!(result.len() >= 3);

        // Find Japanese tokens
        let has_kikai = result.iter().any(|t| t.text == "機械");
        let has_gakushu = result.iter().any(|t| t.text == "学習");

        assert!(has_kikai, "Expected '機械' token from tokenized '機械学習'");
        assert!(
            has_gakushu,
            "Expected '学習' token from tokenized '機械学習'"
        );
    }

    #[test]
    fn test_synonym_match_without_spaces() {
        // Mock tokenizer that splits Japanese
        struct JapaneseTokenizer;
        impl crate::analysis::tokenizer::Tokenizer for JapaneseTokenizer {
            fn tokenize(
                &self,
                text: &str,
            ) -> crate::error::Result<crate::analysis::token::TokenStream> {
                let tokens: Vec<Token> = match text {
                    "機械学習" => vec![Token::new("機械", 0), Token::new("学習", 1)],
                    "マシンラーニング" => {
                        vec![Token::new("マシン", 0), Token::new("ラーニング", 1)]
                    }
                    other => vec![Token::new(other, 0)],
                };
                Ok(Box::new(tokens.into_iter()))
            }

            fn name(&self) -> &'static str {
                "japanese_tokenizer"
            }
        }

        // Dictionary with Japanese entry (no spaces)
        let mut dict = SynonymDictionary::new(None).unwrap();
        dict.add_synonym_group(vec!["機械学習".to_string(), "マシンラーニング".to_string()]);

        let tokenizer = Box::new(JapaneseTokenizer);
        let filter = SynonymGraphFilter::with_tokenizer(dict, tokenizer, true);

        // Input: tokens that were already tokenized by Japanese tokenizer
        // ["機械", "学習"] should match "機械学習" in dictionary
        let tokens = vec![Token::new("機械", 0), Token::new("学習", 1)];

        let result = filter
            .filter(Box::new(tokens.into_iter()))
            .unwrap()
            .collect::<Vec<_>>();

        // Should have: "機械", "学習" (original) and "マシン", "ラーニング" (synonym)
        assert!(
            result.len() >= 4,
            "Expected at least 4 tokens, got {}",
            result.len()
        );

        let has_machine = result.iter().any(|t| t.text == "マシン");
        let has_learning = result.iter().any(|t| t.text == "ラーニング");

        assert!(
            has_machine,
            "Expected 'マシン' token from synonym 'マシンラーニング'"
        );
        assert!(
            has_learning,
            "Expected 'ラーニング' token from synonym 'マシンラーニング'"
        );

        // Verify that the synonym tokens are at the same position as original
        let machine_token = result.iter().find(|t| t.text == "マシン").unwrap();
        assert_eq!(
            machine_token.position_increment, 0,
            "Synonym should be at same position"
        );
    }

    #[test]
    fn test_byte_offset_continuity_check() {
        // Test that tokens separated by whitespace in original text are not matched together
        let mut dict = SynonymDictionary::new(None).unwrap();
        dict.add_synonym_group(vec!["入門 本".to_string(), "beginner book".to_string()]);

        let filter = SynonymGraphFilter::new(dict, true);

        // Create tokens with non-continuous byte offsets (simulating "入門 本" with space)
        let mut token1 = Token::new("入門", 0);
        token1.start_offset = 15;
        token1.end_offset = 21;

        let mut token2 = Token::new("本", 1);
        token2.start_offset = 22; // Not continuous (21 != 22, there's a space)
        token2.end_offset = 25;

        let tokens = vec![token1, token2];

        let result = filter
            .filter(Box::new(tokens.into_iter()))
            .unwrap()
            .collect::<Vec<_>>();

        // Should NOT match "入門 本" as a synonym because byte offsets are not continuous
        // Should only have the original 2 tokens
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].text, "入門");
        assert_eq!(result[1].text, "本");

        // Verify no synonym tokens were added
        let has_beginner = result.iter().any(|t| t.text == "beginner");
        assert!(!has_beginner, "Should not match non-continuous tokens");
    }

    #[test]
    fn test_byte_offset_continuous_tokens_do_match() {
        // Test that tokens that ARE continuous in original text do match
        // Mock tokenizer for splitting Japanese synonyms
        struct JapaneseTokenizer;
        impl crate::analysis::tokenizer::Tokenizer for JapaneseTokenizer {
            fn tokenize(
                &self,
                text: &str,
            ) -> crate::error::Result<crate::analysis::token::TokenStream> {
                let tokens: Vec<Token> = if text == "マシンラーニング" {
                    vec![Token::new("マシン", 0), Token::new("ラーニング", 1)]
                } else {
                    vec![Token::new(text, 0)]
                };
                Ok(Box::new(tokens.into_iter()))
            }

            fn name(&self) -> &'static str {
                "japanese_tokenizer"
            }
        }

        let mut dict = SynonymDictionary::new(None).unwrap();
        dict.add_synonym_group(vec!["機械学習".to_string(), "マシンラーニング".to_string()]);

        let tokenizer = Box::new(JapaneseTokenizer);
        let filter = SynonymGraphFilter::with_tokenizer(dict, tokenizer, true);

        // Create tokens with continuous byte offsets (no space between)
        let mut token1 = Token::new("機械", 0);
        token1.start_offset = 0;
        token1.end_offset = 6;

        let mut token2 = Token::new("学習", 1);
        token2.start_offset = 6; // Continuous (6 == 6)
        token2.end_offset = 12;

        let tokens = vec![token1, token2];

        let result = filter
            .filter(Box::new(tokens.into_iter()))
            .unwrap()
            .collect::<Vec<_>>();

        // Should match "機械学習" and add synonym tokens
        assert!(
            result.len() >= 4,
            "Expected at least 4 tokens (2 original + 2 synonym), got {}",
            result.len()
        );

        // Verify synonym tokens were added
        let has_machine = result.iter().any(|t| t.text == "マシン");
        let has_learning = result.iter().any(|t| t.text == "ラーニング");
        assert!(
            has_machine,
            "Should match continuous tokens and add synonyms"
        );
        assert!(
            has_learning,
            "Should match continuous tokens and add synonyms"
        );
    }

    #[test]
    fn test_alphanum_with_spaces() {
        // Test that ALPHANUM tokens with spaces between them still match
        use crate::analysis::token::{Token, TokenMetadata, TokenType};

        let mut dict = SynonymDictionary::new(None).unwrap();
        dict.add_synonym_group(vec!["machine learning".to_string(), "ml".to_string()]);

        let filter = SynonymGraphFilter::new(dict, true);

        // Create tokens with ALPHANUM type and spaces between them
        let mut token1 = Token::new("machine", 0);
        token1.start_offset = 0;
        token1.end_offset = 7;
        token1.metadata = Some(TokenMetadata {
            original_text: None,
            token_type: Some(TokenType::Alphanum),
            language: None,
            attributes: std::collections::HashMap::new(),
        });

        let mut token2 = Token::new("learning", 1);
        token2.start_offset = 8; // Not continuous (space at position 7)
        token2.end_offset = 16;
        token2.metadata = Some(TokenMetadata {
            original_text: None,
            token_type: Some(TokenType::Alphanum),
            language: None,
            attributes: std::collections::HashMap::new(),
        });

        let tokens = vec![token1, token2];

        let result = filter
            .filter(Box::new(tokens.into_iter()))
            .unwrap()
            .collect::<Vec<_>>();

        // Should match "machine learning" even though there's a space (both ALPHANUM)
        let has_ml = result.iter().any(|t| t.text == "ml");
        assert!(has_ml, "ALPHANUM tokens with spaces should still match");
    }

    #[test]
    fn test_mixed_alphanum_cjk_continuous() {
        // Test mixed ALPHANUM + CJK tokens that are continuous
        use crate::analysis::token::{Token, TokenMetadata};

        let mut dict = SynonymDictionary::new(None).unwrap();
        dict.add_synonym_group(vec![
            "100円ショップ".to_string(),
            "hundred yen shop".to_string(),
        ]);

        let filter = SynonymGraphFilter::new(dict, true);

        // Create tokens: "100" (ALPHANUM) + "円" (CJK) + "ショップ" (CJK)
        let mut token1 = Token::new("100", 0);
        token1.start_offset = 0;
        token1.end_offset = 3;
        token1.metadata = Some(TokenMetadata {
            original_text: None,
            token_type: Some(TokenType::Num),
            language: None,
            attributes: std::collections::HashMap::new(),
        });

        let mut token2 = Token::new("円", 1);
        token2.start_offset = 3; // Continuous
        token2.end_offset = 6;
        token2.metadata = Some(TokenMetadata {
            original_text: None,
            token_type: Some(TokenType::Cjk),
            language: None,
            attributes: std::collections::HashMap::new(),
        });

        let mut token3 = Token::new("ショップ", 2);
        token3.start_offset = 6; // Continuous
        token3.end_offset = 18;
        token3.metadata = Some(TokenMetadata {
            original_text: None,
            token_type: Some(TokenType::Cjk),
            language: None,
            attributes: std::collections::HashMap::new(),
        });

        let tokens = vec![token1, token2, token3];

        let result = filter
            .filter(Box::new(tokens.into_iter()))
            .unwrap()
            .collect::<Vec<_>>();

        // Should match "100円ショップ" because all are continuous
        let has_synonym = result.iter().any(|t| t.text == "hundred");
        assert!(
            has_synonym,
            "Mixed ALPHANUM+CJK continuous tokens should match"
        );
    }

    #[test]
    fn test_mixed_alphanum_cjk_non_continuous() {
        // Test mixed ALPHANUM + CJK tokens that are NOT continuous
        use crate::analysis::token::{Token, TokenMetadata};

        let mut dict = SynonymDictionary::new(None).unwrap();
        dict.add_synonym_group(vec!["machine 学習".to_string(), "test synonym".to_string()]);

        let filter = SynonymGraphFilter::new(dict, true);

        // Create tokens: "machine" (ALPHANUM) + "学習" (CJK) with space
        let mut token1 = Token::new("machine", 0);
        token1.start_offset = 0;
        token1.end_offset = 7;
        token1.metadata = Some(TokenMetadata {
            original_text: None,
            token_type: Some(TokenType::Alphanum),
            language: None,
            attributes: std::collections::HashMap::new(),
        });

        let mut token2 = Token::new("学習", 1);
        token2.start_offset = 8; // Not continuous (space at position 7)
        token2.end_offset = 14;
        token2.metadata = Some(TokenMetadata {
            original_text: None,
            token_type: Some(TokenType::Cjk),
            language: None,
            attributes: std::collections::HashMap::new(),
        });

        let tokens = vec![token1, token2];

        let result = filter
            .filter(Box::new(tokens.into_iter()))
            .unwrap()
            .collect::<Vec<_>>();

        // Should NOT match because mixed types with space
        let has_synonym = result.iter().any(|t| t.text == "test");
        assert!(
            !has_synonym,
            "Mixed ALPHANUM+CJK with space should NOT match"
        );
    }

    #[test]
    fn test_cjk_non_continuous() {
        // Test CJK tokens that are NOT continuous (space between)
        use crate::analysis::token::{Token, TokenMetadata};

        let mut dict = SynonymDictionary::new(None).unwrap();
        dict.add_synonym_group(vec!["入門 本".to_string(), "beginner book".to_string()]);

        let filter = SynonymGraphFilter::new(dict, true);

        // Create tokens: "入門" + "本" with space
        let mut token1 = Token::new("入門", 0);
        token1.start_offset = 0;
        token1.end_offset = 6;
        token1.metadata = Some(TokenMetadata {
            original_text: None,
            token_type: Some(TokenType::Cjk),
            language: None,
            attributes: std::collections::HashMap::new(),
        });

        let mut token2 = Token::new("本", 1);
        token2.start_offset = 7; // Not continuous (space at position 6)
        token2.end_offset = 10;
        token2.metadata = Some(TokenMetadata {
            original_text: None,
            token_type: Some(TokenType::Cjk),
            language: None,
            attributes: std::collections::HashMap::new(),
        });

        let tokens = vec![token1, token2];

        let result = filter
            .filter(Box::new(tokens.into_iter()))
            .unwrap()
            .collect::<Vec<_>>();

        // Should NOT match because CJK with space
        let has_synonym = result.iter().any(|t| t.text == "beginner");
        assert!(!has_synonym, "CJK tokens with space should NOT match");
    }
}
