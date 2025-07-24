//! Text embedding generation for semantic vector search.
//!
//! This module provides functionality to convert text into dense vector representations
//! that capture semantic meaning. It supports multiple embedding methods:
//! - TF-IDF based embeddings for traditional sparse-to-dense conversion
//! - Word2Vec style embeddings using skipgram or CBOW models
//! - Simple bag-of-words embeddings for baseline comparisons
//! - Integration points for external embedding models (OpenAI, Sentence Transformers, etc.)

use crate::analysis::{Analyzer, StandardAnalyzer};
use crate::error::{Result, SarissaError};
use crate::vector::Vector;
use ahash::AHashMap;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, RwLock};

/// Configuration for text embedding generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Dimensionality of the generated embeddings.
    pub dimension: usize,
    /// Embedding method to use.
    pub method: EmbeddingMethod,
    /// Whether to normalize embeddings to unit length.
    pub normalize: bool,
    /// Minimum term frequency to include in vocabulary.
    pub min_term_freq: usize,
    /// Maximum vocabulary size.
    pub max_vocab_size: usize,
    /// Whether to use subword information.
    pub use_subwords: bool,
    /// Subword minimum length.
    pub subword_min_len: usize,
    /// Subword maximum length.
    pub subword_max_len: usize,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            dimension: 128,
            method: EmbeddingMethod::TfIdf,
            normalize: true,
            min_term_freq: 2,
            max_vocab_size: 50000,
            use_subwords: false,
            subword_min_len: 3,
            subword_max_len: 6,
        }
    }
}

/// Different methods for generating embeddings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmbeddingMethod {
    /// TF-IDF based embeddings.
    TfIdf,
    /// Simple bag-of-words counting.
    BagOfWords,
    /// Random embeddings (for testing/baseline).
    Random,
    /// Word2Vec style skipgram model.
    Word2Vec,
    /// External model integration placeholder.
    External,
}

impl EmbeddingMethod {
    /// Get the name of this embedding method.
    pub fn name(&self) -> &'static str {
        match self {
            EmbeddingMethod::TfIdf => "tf_idf",
            EmbeddingMethod::BagOfWords => "bag_of_words",
            EmbeddingMethod::Random => "random",
            EmbeddingMethod::Word2Vec => "word2vec",
            EmbeddingMethod::External => "external",
        }
    }

    /// Parse an embedding method from a string.
    pub fn parse_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "tf_idf" | "tfidf" => Ok(EmbeddingMethod::TfIdf),
            "bag_of_words" | "bow" => Ok(EmbeddingMethod::BagOfWords),
            "random" => Ok(EmbeddingMethod::Random),
            "word2vec" | "w2v" => Ok(EmbeddingMethod::Word2Vec),
            "external" => Ok(EmbeddingMethod::External),
            _ => Err(SarissaError::InvalidOperation(format!(
                "Unknown embedding method: {s}"
            ))),
        }
    }
}

/// Statistics about term frequencies for vocabulary building.
#[derive(Debug, Clone)]
pub struct TermStats {
    /// Total frequency of this term across all documents.
    pub term_freq: usize,
    /// Number of documents containing this term.
    pub doc_freq: usize,
    /// Index of this term in the vocabulary.
    pub index: usize,
}

/// Vocabulary for embedding generation.
#[derive(Debug, Clone)]
pub struct Vocabulary {
    /// Map from term to its statistics.
    pub term_to_stats: AHashMap<String, TermStats>,
    /// Map from term index to term string.
    pub index_to_term: Vec<String>,
    /// Total number of documents in the corpus.
    pub total_docs: usize,
    /// Total number of terms processed.
    pub total_terms: usize,
}

impl Vocabulary {
    /// Create a new empty vocabulary.
    pub fn new() -> Self {
        Self {
            term_to_stats: AHashMap::new(),
            index_to_term: Vec::new(),
            total_docs: 0,
            total_terms: 0,
        }
    }

    /// Build vocabulary from a collection of documents.
    pub fn build_from_documents(
        documents: &[String],
        analyzer: &dyn Analyzer,
        config: &EmbeddingConfig,
    ) -> Result<Self> {
        let mut term_counts: AHashMap<String, usize> = AHashMap::new();
        let mut doc_counts: AHashMap<String, usize> = AHashMap::new();
        let mut total_terms = 0;

        // First pass: count term and document frequencies
        for document in documents {
            let tokens = analyzer.analyze(document)?;
            let mut doc_terms = std::collections::HashSet::new();

            for token in tokens {
                let term = token.text.clone();

                // Add subwords if enabled
                if config.use_subwords {
                    for subword in Self::generate_subwords(
                        &term,
                        config.subword_min_len,
                        config.subword_max_len,
                    ) {
                        *term_counts.entry(subword.clone()).or_insert(0) += 1;
                        doc_terms.insert(subword);
                        total_terms += 1;
                    }
                } else {
                    *term_counts.entry(term.clone()).or_insert(0) += 1;
                    doc_terms.insert(term);
                    total_terms += 1;
                }
            }

            // Count document frequencies
            for term in doc_terms {
                *doc_counts.entry(term).or_insert(0) += 1;
            }
        }

        // Second pass: filter and build vocabulary
        let mut filtered_terms: Vec<_> = term_counts
            .iter()
            .filter(|&(_, &freq)| freq >= config.min_term_freq)
            .collect();

        // Sort by frequency (descending) and take top terms
        filtered_terms.sort_by(|a, b| b.1.cmp(a.1));
        if filtered_terms.len() > config.max_vocab_size {
            filtered_terms.truncate(config.max_vocab_size);
        }

        // Build final vocabulary
        let mut vocabulary = Vocabulary::new();
        vocabulary.total_docs = documents.len();
        vocabulary.total_terms = total_terms;

        for (index, &(ref term, &term_freq)) in filtered_terms.iter().enumerate() {
            let doc_freq = doc_counts.get(*term).copied().unwrap_or(0);

            vocabulary.term_to_stats.insert(
                term.to_string(),
                TermStats {
                    term_freq,
                    doc_freq,
                    index,
                },
            );

            vocabulary.index_to_term.push(term.to_string());
        }

        Ok(vocabulary)
    }

    /// Generate subwords for a term using character n-grams.
    fn generate_subwords(term: &str, min_len: usize, max_len: usize) -> Vec<String> {
        let mut subwords = Vec::new();
        let chars: Vec<char> = term.chars().collect();

        // Add the full term
        subwords.push(term.to_string());

        // Generate character n-grams
        for len in min_len..=max_len.min(chars.len()) {
            for start in 0..=(chars.len() - len) {
                let subword: String = chars[start..start + len].iter().collect();
                subwords.push(format!("<{subword}>")); // Mark as subword
            }
        }

        subwords
    }

    /// Get the size of the vocabulary.
    pub fn size(&self) -> usize {
        self.index_to_term.len()
    }

    /// Get term statistics by term string.
    pub fn get_term_stats(&self, term: &str) -> Option<&TermStats> {
        self.term_to_stats.get(term)
    }

    /// Get term by index.
    pub fn get_term_by_index(&self, index: usize) -> Option<&String> {
        self.index_to_term.get(index)
    }

    /// Calculate IDF (Inverse Document Frequency) for a term.
    pub fn calculate_idf(&self, term: &str) -> f32 {
        if let Some(stats) = self.get_term_stats(term) {
            if stats.doc_freq > 0 {
                (self.total_docs as f32 / stats.doc_freq as f32).ln()
            } else {
                0.0
            }
        } else {
            0.0
        }
    }
}

impl Default for Vocabulary {
    fn default() -> Self {
        Self::new()
    }
}

/// Text embedding generator that converts text to dense vectors.
pub struct TextEmbedder {
    /// Configuration for embedding generation.
    config: EmbeddingConfig,
    /// Vocabulary for term mapping.
    vocabulary: Arc<RwLock<Vocabulary>>,
    /// Text analyzer for tokenization.
    analyzer: Box<dyn Analyzer>,
    /// Whether the embedder has been trained.
    is_trained: bool,
}

impl TextEmbedder {
    /// Create a new text embedder with the given configuration.
    pub fn new(config: EmbeddingConfig) -> Result<Self> {
        let analyzer = Box::new(StandardAnalyzer::new()?);

        Ok(Self {
            config,
            vocabulary: Arc::new(RwLock::new(Vocabulary::new())),
            analyzer,
            is_trained: false,
        })
    }

    /// Create a text embedder with custom analyzer.
    pub fn with_analyzer(config: EmbeddingConfig, analyzer: Box<dyn Analyzer>) -> Self {
        Self {
            config,
            vocabulary: Arc::new(RwLock::new(Vocabulary::new())),
            analyzer,
            is_trained: false,
        }
    }

    /// Train the embedder on a collection of documents.
    pub fn train(&mut self, documents: &[String]) -> Result<()> {
        let vocabulary =
            Vocabulary::build_from_documents(documents, self.analyzer.as_ref(), &self.config)?;

        {
            let mut vocab = self.vocabulary.write().unwrap();
            *vocab = vocabulary;
        }

        self.is_trained = true;
        Ok(())
    }

    /// Check if the embedder has been trained.
    pub fn is_trained(&self) -> bool {
        self.is_trained
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        let vocab = self.vocabulary.read().unwrap();
        vocab.size()
    }

    /// Convert text to an embedding vector.
    pub fn embed_text(&self, text: &str) -> Result<Vector> {
        if !self.is_trained {
            return Err(SarissaError::InvalidOperation(
                "Embedder must be trained before generating embeddings".to_string(),
            ));
        }

        match self.config.method {
            EmbeddingMethod::TfIdf => self.embed_tfidf(text),
            EmbeddingMethod::BagOfWords => self.embed_bow(text),
            EmbeddingMethod::Random => self.embed_random(text),
            EmbeddingMethod::Word2Vec => self.embed_word2vec(text),
            EmbeddingMethod::External => Err(SarissaError::InvalidOperation(
                "External embeddings not yet implemented".to_string(),
            )),
        }
    }

    /// Convert multiple texts to embedding vectors.
    pub fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vector>> {
        texts.iter().map(|text| self.embed_text(text)).collect()
    }

    /// Generate TF-IDF based embedding.
    fn embed_tfidf(&self, text: &str) -> Result<Vector> {
        let tokens = self.analyzer.analyze(text)?;
        let vocab = self.vocabulary.read().unwrap();

        // Count term frequencies in the document
        let mut term_freqs: AHashMap<String, usize> = AHashMap::new();
        let mut total_terms = 0;

        for token in tokens {
            let term = token.text;

            if self.config.use_subwords {
                for subword in Vocabulary::generate_subwords(
                    &term,
                    self.config.subword_min_len,
                    self.config.subword_max_len,
                ) {
                    *term_freqs.entry(subword).or_insert(0) += 1;
                    total_terms += 1;
                }
            } else {
                *term_freqs.entry(term).or_insert(0) += 1;
                total_terms += 1;
            }
        }

        // Create embedding vector
        let mut embedding = vec![0.0_f32; self.config.dimension.min(vocab.size())];

        for (term, tf) in term_freqs {
            if let Some(stats) = vocab.get_term_stats(&term) {
                if stats.index < embedding.len() {
                    let tf_norm = tf as f32 / total_terms as f32;
                    let idf = vocab.calculate_idf(&term);
                    embedding[stats.index] = tf_norm * idf;
                }
            }
        }

        // If dimension is larger than vocabulary, pad with zeros
        if self.config.dimension > vocab.size() {
            embedding.resize(self.config.dimension, 0.0);
        }

        let mut vector = Vector::new(embedding);

        if self.config.normalize {
            vector.normalize();
        }

        Ok(vector)
    }

    /// Generate bag-of-words based embedding.
    fn embed_bow(&self, text: &str) -> Result<Vector> {
        let tokens = self.analyzer.analyze(text)?;
        let vocab = self.vocabulary.read().unwrap();

        let mut embedding = vec![0.0_f32; self.config.dimension.min(vocab.size())];

        for token in tokens {
            let term = token.text;

            if self.config.use_subwords {
                for subword in Vocabulary::generate_subwords(
                    &term,
                    self.config.subword_min_len,
                    self.config.subword_max_len,
                ) {
                    if let Some(stats) = vocab.get_term_stats(&subword) {
                        if stats.index < embedding.len() {
                            embedding[stats.index] += 1.0;
                        }
                    }
                }
            } else if let Some(stats) = vocab.get_term_stats(&term) {
                if stats.index < embedding.len() {
                    embedding[stats.index] += 1.0;
                }
            }
        }

        // Pad if necessary
        if self.config.dimension > vocab.size() {
            embedding.resize(self.config.dimension, 0.0);
        }

        let mut vector = Vector::new(embedding);

        if self.config.normalize {
            vector.normalize();
        }

        Ok(vector)
    }

    /// Generate random embedding (for testing/baseline).
    fn embed_random(&self, _text: &str) -> Result<Vector> {
        use rand::prelude::*;
        let mut rng = rand::rng();

        let embedding: Vec<f32> = (0..self.config.dimension)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();

        let mut vector = Vector::new(embedding);

        if self.config.normalize {
            vector.normalize();
        }

        Ok(vector)
    }

    /// Generate Word2Vec style embedding (simplified implementation).
    fn embed_word2vec(&self, text: &str) -> Result<Vector> {
        // For now, use a simplified approach that averages term embeddings
        // In a full implementation, this would use trained word embeddings
        let tokens = self.analyzer.analyze(text)?;
        let vocab = self.vocabulary.read().unwrap();

        let mut embedding = vec![0.0_f32; self.config.dimension];
        let mut count = 0;

        for token in tokens {
            let term = token.text;

            if let Some(_stats) = vocab.get_term_stats(&term) {
                // Generate a pseudo-embedding based on term index and frequency
                let term_hash = {
                    use std::collections::hash_map::DefaultHasher;
                    use std::hash::{Hash, Hasher};
                    let mut hasher = DefaultHasher::new();
                    term.hash(&mut hasher);
                    hasher.finish()
                };

                for (i, embedding_val) in
                    embedding.iter_mut().enumerate().take(self.config.dimension)
                {
                    let seed = term_hash.wrapping_add(i as u64);
                    let value = ((seed as f64 * 0.00001) % 2.0) - 1.0; // Range [-1, 1]
                    *embedding_val += value as f32;
                }
                count += 1;
            }
        }

        // Average the embeddings
        if count > 0 {
            for value in &mut embedding {
                *value /= count as f32;
            }
        }

        let mut vector = Vector::new(embedding);

        if self.config.normalize {
            vector.normalize();
        }

        Ok(vector)
    }

    /// Get vocabulary information.
    pub fn get_vocabulary(&self) -> Arc<RwLock<Vocabulary>> {
        Arc::clone(&self.vocabulary)
    }

    /// Get embedding configuration.
    pub fn get_config(&self) -> &EmbeddingConfig {
        &self.config
    }
}

/// Builder for creating text embedders with fluent API.
pub struct TextEmbedderBuilder {
    config: EmbeddingConfig,
    analyzer: Option<Box<dyn Analyzer>>,
}

impl TextEmbedderBuilder {
    /// Create a new embedder builder.
    pub fn new() -> Self {
        Self {
            config: EmbeddingConfig::default(),
            analyzer: None,
        }
    }

    /// Set the embedding dimension.
    pub fn dimension(mut self, dimension: usize) -> Self {
        self.config.dimension = dimension;
        self
    }

    /// Set the embedding method.
    pub fn method(mut self, method: EmbeddingMethod) -> Self {
        self.config.method = method;
        self
    }

    /// Enable or disable normalization.
    pub fn normalize(mut self, normalize: bool) -> Self {
        self.config.normalize = normalize;
        self
    }

    /// Set minimum term frequency.
    pub fn min_term_freq(mut self, freq: usize) -> Self {
        self.config.min_term_freq = freq;
        self
    }

    /// Set maximum vocabulary size.
    pub fn max_vocab_size(mut self, size: usize) -> Self {
        self.config.max_vocab_size = size;
        self
    }

    /// Enable subword embeddings.
    pub fn with_subwords(mut self, min_len: usize, max_len: usize) -> Self {
        self.config.use_subwords = true;
        self.config.subword_min_len = min_len;
        self.config.subword_max_len = max_len;
        self
    }

    /// Set a custom analyzer.
    pub fn analyzer(mut self, analyzer: Box<dyn Analyzer>) -> Self {
        self.analyzer = Some(analyzer);
        self
    }

    /// Build the text embedder.
    pub fn build(self) -> Result<TextEmbedder> {
        if let Some(analyzer) = self.analyzer {
            Ok(TextEmbedder::with_analyzer(self.config, analyzer))
        } else {
            TextEmbedder::new(self.config)
        }
    }
}

impl Default for TextEmbedderBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_config() {
        let config = EmbeddingConfig::default();
        assert_eq!(config.dimension, 128);
        assert_eq!(config.method, EmbeddingMethod::TfIdf);
        assert!(config.normalize);
    }

    #[test]
    fn test_embedding_method_parsing() {
        assert_eq!(
            EmbeddingMethod::parse_str("tf_idf").unwrap(),
            EmbeddingMethod::TfIdf
        );
        assert_eq!(
            EmbeddingMethod::parse_str("tfidf").unwrap(),
            EmbeddingMethod::TfIdf
        );
        assert_eq!(
            EmbeddingMethod::parse_str("bow").unwrap(),
            EmbeddingMethod::BagOfWords
        );
        assert!(EmbeddingMethod::parse_str("unknown").is_err());
    }

    #[test]
    fn test_vocabulary_creation() {
        let vocab = Vocabulary::new();
        assert_eq!(vocab.size(), 0);
        assert_eq!(vocab.total_docs, 0);
    }

    #[test]
    fn test_vocabulary_building() {
        let documents = vec![
            "the quick brown fox".to_string(),
            "the lazy dog".to_string(),
            "quick brown animals".to_string(),
        ];

        let analyzer = StandardAnalyzer::new().unwrap();
        let config = EmbeddingConfig {
            min_term_freq: 1, // Lower threshold to include all terms
            ..Default::default()
        };

        let vocab = Vocabulary::build_from_documents(&documents, &analyzer, &config).unwrap();

        assert!(vocab.size() > 0);
        assert_eq!(vocab.total_docs, 3);
        // StandardAnalyzer removes stop words like "the", so check for other terms
        assert!(vocab.get_term_stats("quick").is_some());
        assert!(vocab.get_term_stats("brown").is_some());
    }

    #[test]
    fn test_subword_generation() {
        let subwords = Vocabulary::generate_subwords("hello", 2, 4);

        assert!(subwords.contains(&"hello".to_string()));
        assert!(subwords.contains(&"<he>".to_string()));
        assert!(subwords.contains(&"<el>".to_string()));
        assert!(subwords.contains(&"<hell>".to_string()));
    }

    #[test]
    fn test_text_embedder_creation() {
        let config = EmbeddingConfig {
            dimension: 64,
            method: EmbeddingMethod::TfIdf,
            ..Default::default()
        };

        let embedder = TextEmbedder::new(config).unwrap();
        assert!(!embedder.is_trained());
        assert_eq!(embedder.get_config().dimension, 64);
    }

    #[test]
    fn test_text_embedder_training() {
        let documents = vec![
            "machine learning is fascinating".to_string(),
            "natural language processing".to_string(),
            "vector search and embeddings".to_string(),
        ];

        let config = EmbeddingConfig {
            min_term_freq: 1, // Lower threshold
            ..Default::default()
        };
        let mut embedder = TextEmbedder::new(config).unwrap();
        embedder.train(&documents).unwrap();

        assert!(embedder.is_trained());
        assert!(embedder.vocab_size() > 0);
    }

    #[test]
    fn test_text_embedding_generation() {
        let documents = vec![
            "hello world test".to_string(),
            "world machine learning".to_string(),
        ];

        let mut embedder = TextEmbedder::new(EmbeddingConfig {
            dimension: 10,
            method: EmbeddingMethod::BagOfWords,
            min_term_freq: 1,
            ..Default::default()
        })
        .unwrap();

        embedder.train(&documents).unwrap();

        let vector = embedder.embed_text("hello world").unwrap();
        assert_eq!(vector.dimension(), 10);
        assert!(vector.is_valid());
    }

    #[test]
    fn test_embedder_not_trained_error() {
        let embedder = TextEmbedder::new(EmbeddingConfig::default()).unwrap();

        let result = embedder.embed_text("test");
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_embedding() {
        let documents = vec!["text one".to_string(), "text two".to_string()];

        let mut embedder = TextEmbedder::new(EmbeddingConfig {
            dimension: 5,
            min_term_freq: 1,
            ..Default::default()
        })
        .unwrap();

        embedder.train(&documents).unwrap();

        let texts = vec!["text one".to_string(), "text two".to_string()];
        let vectors = embedder.embed_batch(&texts).unwrap();

        assert_eq!(vectors.len(), 2);
        assert!(vectors.iter().all(|v| v.dimension() == 5));
    }

    #[test]
    fn test_embedder_builder() {
        let embedder = TextEmbedderBuilder::new()
            .dimension(256)
            .method(EmbeddingMethod::Word2Vec)
            .normalize(false)
            .min_term_freq(1)
            .max_vocab_size(10000)
            .build()
            .unwrap();

        let config = embedder.get_config();
        assert_eq!(config.dimension, 256);
        assert_eq!(config.method, EmbeddingMethod::Word2Vec);
        assert!(!config.normalize);
        assert_eq!(config.min_term_freq, 1);
        assert_eq!(config.max_vocab_size, 10000);
    }

    #[test]
    fn test_different_embedding_methods() {
        let documents = vec!["test document".to_string()];

        for method in [
            EmbeddingMethod::TfIdf,
            EmbeddingMethod::BagOfWords,
            EmbeddingMethod::Random,
            EmbeddingMethod::Word2Vec,
        ] {
            let mut embedder = TextEmbedder::new(EmbeddingConfig {
                dimension: 8,
                method,
                min_term_freq: 1,
                ..Default::default()
            })
            .unwrap();

            if method != EmbeddingMethod::Random {
                embedder.train(&documents).unwrap();
            } else {
                // Random embeddings don't need training
                embedder.is_trained = true;
            }

            let vector = embedder.embed_text("test").unwrap();
            assert_eq!(vector.dimension(), 8);
            assert!(vector.is_valid());
        }
    }

    #[test]
    fn test_idf_calculation() {
        let documents = vec![
            "the cat sat on the mat".to_string(),
            "the dog ran in the park".to_string(),
            "cats and dogs are pets".to_string(),
        ];

        let analyzer = StandardAnalyzer::new().unwrap();
        let config = EmbeddingConfig {
            min_term_freq: 1, // Include all terms
            ..Default::default()
        };
        let vocab = Vocabulary::build_from_documents(&documents, &analyzer, &config).unwrap();

        let idf_dog = vocab.calculate_idf("dog"); // "dog" appears once
        let idf_cats = vocab.calculate_idf("cats"); // "cats" appears once

        // Both appear in 1/3 documents, so IDF should be similar
        assert!(idf_cats > 0.0);
        assert!(idf_dog > 0.0);
        assert!((idf_cats - idf_dog).abs() < 0.01); // Should be very similar
    }
}
