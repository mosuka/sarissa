//! Text embedding generation for vector indexing.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::{Result, SarissaError};
use crate::vector::Vector;

/// Configuration for embedding generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Output vector dimension.
    pub dimension: usize,
    /// Embedding generation method.
    pub method: EmbeddingMethod,
    /// Whether to normalize embeddings.
    pub normalize: bool,
    /// Minimum term frequency for inclusion.
    pub min_term_freq: usize,
    /// Maximum vocabulary size.
    pub max_vocab_size: usize,
    /// Whether to use parallel processing.
    pub parallel: bool,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            dimension: 128,
            method: EmbeddingMethod::TfIdf,
            normalize: true,
            min_term_freq: 2,
            max_vocab_size: 10000,
            parallel: true,
        }
    }
}

/// Methods for generating embeddings from text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmbeddingMethod {
    /// Term frequency-inverse document frequency.
    TfIdf,
    /// Simple bag-of-words with binary features.
    BagOfWords,
    /// N-gram based features.
    NGram { n: usize },
}

/// Engine for generating text embeddings.
pub struct EmbeddingEngine {
    config: EmbeddingConfig,
    vocabulary: HashMap<String, usize>,
    document_frequencies: HashMap<String, usize>,
    total_documents: usize,
    is_trained: bool,
}

impl EmbeddingEngine {
    /// Create a new embedding engine.
    pub fn new(config: EmbeddingConfig) -> Result<Self> {
        Ok(Self {
            config,
            vocabulary: HashMap::new(),
            document_frequencies: HashMap::new(),
            total_documents: 0,
            is_trained: false,
        })
    }

    /// Train the embedding engine on a corpus of documents.
    pub async fn train(&mut self, documents: &[&str]) -> Result<()> {
        self.total_documents = documents.len();

        // Build vocabulary and document frequencies
        for document in documents {
            let tokens = self.tokenize(document);
            let mut seen_terms = std::collections::HashSet::new();

            for token in tokens {
                // Add to vocabulary
                let vocab_index = self.vocabulary.len();
                self.vocabulary.entry(token.clone()).or_insert(vocab_index);

                // Count document frequency (once per document)
                if seen_terms.insert(token.clone()) {
                    *self.document_frequencies.entry(token).or_insert(0) += 1;
                }
            }
        }

        // Filter vocabulary by minimum frequency
        self.filter_vocabulary();

        // Truncate to max vocab size
        self.truncate_vocabulary();

        self.is_trained = true;
        Ok(())
    }

    /// Generate an embedding for a text document.
    pub fn embed(&self, text: &str) -> Result<Vector> {
        if !self.is_trained {
            return Err(SarissaError::InvalidOperation(
                "Embedding engine must be trained before use".to_string(),
            ));
        }

        let tokens = self.tokenize(text);
        let mut vector_data = vec![0.0; self.vocabulary.len()];

        match self.config.method {
            EmbeddingMethod::TfIdf => {
                self.compute_tfidf(&tokens, &mut vector_data)?;
            }
            EmbeddingMethod::BagOfWords => {
                self.compute_bow(&tokens, &mut vector_data)?;
            }
            EmbeddingMethod::NGram { n } => {
                self.compute_ngram(&tokens, n, &mut vector_data)?;
            }
        }

        let mut vector = Vector::new(vector_data);

        if self.config.normalize {
            vector.normalize();
        }

        Ok(vector)
    }

    /// Tokenize text into terms.
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split_whitespace()
            .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|s| !s.is_empty() && s.len() > 2)
            .map(|s| s.to_string())
            .collect()
    }

    /// Compute TF-IDF vector representation.
    fn compute_tfidf(&self, tokens: &[String], vector_data: &mut [f32]) -> Result<()> {
        // Count term frequencies
        let mut term_counts = HashMap::new();
        for token in tokens {
            *term_counts.entry(token.clone()).or_insert(0) += 1;
        }

        let total_tokens = tokens.len() as f32;

        for (term, &vocab_index) in &self.vocabulary {
            if let Some(&term_freq) = term_counts.get(term) {
                let tf = term_freq as f32 / total_tokens;
                let df = self.document_frequencies.get(term).unwrap_or(&1);
                let idf = (self.total_documents as f32 / *df as f32).ln();

                if vocab_index < vector_data.len() {
                    vector_data[vocab_index] = tf * idf;
                }
            }
        }

        Ok(())
    }

    /// Compute bag-of-words vector representation.
    fn compute_bow(&self, tokens: &[String], vector_data: &mut [f32]) -> Result<()> {
        let mut seen_terms = std::collections::HashSet::new();

        for token in tokens {
            if let Some(&vocab_index) = self.vocabulary.get(token) {
                if seen_terms.insert(token.clone()) && vocab_index < vector_data.len() {
                    vector_data[vocab_index] = 1.0;
                }
            }
        }

        Ok(())
    }

    /// Compute n-gram vector representation.
    fn compute_ngram(&self, tokens: &[String], n: usize, vector_data: &mut [f32]) -> Result<()> {
        if tokens.len() < n {
            return Ok(());
        }

        let mut ngram_counts = HashMap::new();

        for window in tokens.windows(n) {
            let ngram = window.join(" ");
            *ngram_counts.entry(ngram).or_insert(0) += 1;
        }

        let total_ngrams = ngram_counts.len() as f32;

        for (ngram, count) in ngram_counts {
            if let Some(&vocab_index) = self.vocabulary.get(&ngram) {
                if vocab_index < vector_data.len() {
                    vector_data[vocab_index] = count as f32 / total_ngrams;
                }
            }
        }

        Ok(())
    }

    /// Filter vocabulary by minimum frequency.
    fn filter_vocabulary(&mut self) {
        let min_freq = self.config.min_term_freq;
        self.vocabulary
            .retain(|term, _| self.document_frequencies.get(term).unwrap_or(&0) >= &min_freq);

        // Rebuild vocabulary indices
        let mut new_vocabulary = HashMap::new();
        for (index, term) in self.vocabulary.keys().enumerate() {
            new_vocabulary.insert(term.clone(), index);
        }
        self.vocabulary = new_vocabulary;
    }

    /// Truncate vocabulary to maximum size.
    fn truncate_vocabulary(&mut self) {
        if self.vocabulary.len() <= self.config.max_vocab_size {
            return;
        }

        // Sort terms by document frequency (descending)
        let mut terms: Vec<_> = self
            .vocabulary
            .keys()
            .map(|term| {
                (
                    term.clone(),
                    self.document_frequencies.get(term).unwrap_or(&0),
                )
            })
            .collect();

        terms.sort_by(|a, b| b.1.cmp(a.1));
        terms.truncate(self.config.max_vocab_size);

        // Rebuild vocabulary with top terms
        let mut new_vocabulary = HashMap::new();
        for (index, (term, _)) in terms.iter().enumerate() {
            new_vocabulary.insert(term.clone(), index);
        }
        self.vocabulary = new_vocabulary;
    }

    /// Get vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocabulary.len()
    }

    /// Check if the engine is trained.
    pub fn is_trained(&self) -> bool {
        self.is_trained
    }

    /// Get the configuration.
    pub fn config(&self) -> &EmbeddingConfig {
        &self.config
    }
}
