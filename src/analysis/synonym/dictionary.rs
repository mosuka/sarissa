//! Synonym dictionary for mapping terms to their synonyms.
//!
//! Uses FST (Finite State Transducer) for memory-efficient storage and fast lookup.

use std::sync::Arc;

use fst::{Map, MapBuilder, Streamer};

use crate::error::{PlatypusError, Result};

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
            PlatypusError::storage(format!(
                "Failed to read synonym dictionary file '{}': {}",
                path, e
            ))
        })?;

        let synonym_groups: Vec<Vec<String>> = serde_json::from_str(&content).map_err(|e| {
            PlatypusError::parse(format!(
                "Failed to parse synonym dictionary JSON from '{}': {}",
                path, e
            ))
        })?;

        Self::from_synonym_groups(synonym_groups)
    }

    /// Build a synonym dictionary from synonym groups.
    pub fn from_synonym_groups(synonym_groups: Vec<Vec<String>>) -> Result<Self> {
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
            builder
                .insert(term.as_bytes(), index)
                .map_err(|e| PlatypusError::parse(format!("FST build error: {}", e)))?;
        }

        let fst_bytes = builder
            .into_inner()
            .map_err(|e| PlatypusError::parse(format!("FST finalize error: {}", e)))?;
        let fst_map = Map::new(Arc::from(fst_bytes))
            .map_err(|e| PlatypusError::parse(format!("FST creation error: {}", e)))?;

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
    fn test_synonym_dictionary_load_from_file() {
        let dict = SynonymDictionary::load_from_file("resources/ml/synonyms.json").unwrap();

        // Test English synonyms
        let ml_synonyms = dict.get_synonyms("ml");
        assert!(ml_synonyms.is_some());
        let ml_synonyms = ml_synonyms.unwrap();
        assert!(ml_synonyms.contains(&"machine learning".to_string()));

        // Test Japanese synonyms
        let learning_synonyms = dict.get_synonyms("学習");
        assert!(learning_synonyms.is_some());
    }
}
