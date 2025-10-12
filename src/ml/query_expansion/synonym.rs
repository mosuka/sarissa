//! Synonym-based query expansion.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::ml::MLContext;
use crate::query::{Query, TermQuery};

use super::expander::QueryExpander;
use super::types::{ExpandedQueryClause, ExpansionType};

/// Synonym dictionary for term expansion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynonymDictionary {
    synonyms: HashMap<String, Vec<String>>,
}

impl Default for SynonymDictionary {
    fn default() -> Self {
        Self::new(None).unwrap()
    }
}

impl SynonymDictionary {
    /// Create a new synonym dictionary.
    /// If `path` is provided, loads synonyms from the specified JSON file.
    /// If `path` is `None`, creates an empty dictionary.
    pub fn new(path: Option<&str>) -> Result<Self> {
        match path {
            Some(file_path) => Self::load_from_file(file_path),
            None => Ok(Self {
                synonyms: HashMap::new(),
            }),
        }
    }

    /// Load synonym dictionary from a JSON file.
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

        let mut dict = Self::new(None)?;
        for group in synonym_groups {
            if !group.is_empty() {
                dict.add_synonym_group(group);
            }
        }

        Ok(dict)
    }

    pub fn get_synonyms(&self, term: &str) -> Option<&Vec<String>> {
        self.synonyms.get(term)
    }

    pub fn add_synonym_group(&mut self, terms: Vec<String>) {
        for (i, term) in terms.iter().enumerate() {
            let mut synonyms = Vec::new();
            for (j, other_term) in terms.iter().enumerate() {
                if i != j {
                    synonyms.push(other_term.clone());
                }
            }
            self.synonyms.insert(term.clone(), synonyms);
        }
    }
}

/// Synonym-based query expander.
///
/// Expands query terms using a dictionary of synonyms.
pub struct SynonymQueryExpander {
    dictionary: SynonymDictionary,
    weight: f64,
}

impl SynonymQueryExpander {
    /// Create a new synonym expander.
    ///
    /// # Arguments
    /// * `dict_path` - Optional path to JSON synonym dictionary file
    /// * `weight` - Weight multiplier for expanded terms (0.0-1.0)
    pub fn new(dict_path: Option<&str>, weight: f64) -> Result<Self> {
        Ok(Self {
            dictionary: SynonymDictionary::new(dict_path)?,
            weight,
        })
    }
}

impl QueryExpander for SynonymQueryExpander {
    fn expand(
        &self,
        tokens: &[String],
        field: &str,
        _context: &MLContext,
    ) -> Result<Vec<ExpandedQueryClause>> {
        let mut expansions = Vec::new();

        for term in tokens {
            if let Some(synonyms) = self.dictionary.get_synonyms(term) {
                for synonym in synonyms {
                    let mut query =
                        Box::new(TermQuery::new(field, synonym.clone())) as Box<dyn Query>;
                    let confidence = 0.8;
                    query.set_boost((confidence * self.weight) as f32);

                    expansions.push(ExpandedQueryClause {
                        query,
                        confidence,
                        expansion_type: ExpansionType::Synonym,
                        source_term: term.clone(),
                    });
                }
            }
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
    fn test_synonym_dictionary() {
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
    fn test_synonym_dictionary_load_from_file() {
        let dict = SynonymDictionary::load_from_file("resource/ml/synonyms.json").unwrap();

        // Test English synonyms
        let ml_synonyms = dict.get_synonyms("ml");
        assert!(ml_synonyms.is_some());
        let ml_synonyms = ml_synonyms.unwrap();
        assert!(ml_synonyms.contains(&"machine learning".to_string()));
        assert!(ml_synonyms.contains(&"machine-learning".to_string()));

        // Test Japanese synonyms
        let learning_synonyms = dict.get_synonyms("学習");
        assert!(learning_synonyms.is_some());
        let learning_synonyms = learning_synonyms.unwrap();
        assert!(learning_synonyms.contains(&"勉強".to_string()));
        assert!(learning_synonyms.contains(&"習得".to_string()));
    }

    #[test]
    fn test_synonym_dictionary_with_path() {
        let dict = SynonymDictionary::new(Some("resource/ml/synonyms.json")).unwrap();

        // Verify that synonyms are loaded from file
        assert!(dict.get_synonyms("ml").is_some());
        assert!(dict.get_synonyms("ai").is_some());
    }
}
