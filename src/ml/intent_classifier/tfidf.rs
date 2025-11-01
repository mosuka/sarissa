//! TF-IDF vectorizer for text feature extraction.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;

use crate::analysis::analyzer::analyzer::Analyzer;

/// TF-IDF vectorizer for text feature extraction.
///
/// Implements Term Frequency-Inverse Document Frequency vectorization,
/// a common text representation method in information retrieval and NLP.
///
/// # Algorithm
/// - **TF (Term Frequency)**: Normalized by document length
/// - **IDF (Inverse Document Frequency)**: log((N + 1) / (df + 1)) + 1
/// - **TF-IDF**: TF * IDF for each term
///
/// # Usage
/// 1. Create vectorizer with analyzer: `TfIdfVectorizer::new(analyzer)`
/// 2. Fit on training documents: `vectorizer.fit(&documents)`
/// 3. Transform new documents: `vectorizer.transform(document)`
pub struct TfIdfVectorizer {
    /// Vocabulary: word -> index mapping.
    vocabulary: HashMap<String, usize>,
    /// Inverse document frequency for each word.
    idf: Vec<f64>,
    /// Total number of documents seen during training.
    n_documents: usize,
    /// Analyzer for tokenization.
    analyzer: Arc<dyn Analyzer>,
}

impl std::fmt::Debug for TfIdfVectorizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TfIdfVectorizer")
            .field("vocabulary_size", &self.vocabulary.len())
            .field("n_documents", &self.n_documents)
            .field("analyzer", &self.analyzer.name())
            .finish()
    }
}

impl TfIdfVectorizer {
    /// Create a new TF-IDF vectorizer with the specified analyzer.
    ///
    /// The vectorizer starts empty and must be trained using `fit()` before
    /// it can transform documents.
    ///
    /// # Arguments
    /// * `analyzer` - Text analyzer for tokenizing documents
    ///
    /// # Returns
    /// New TF-IDF vectorizer instance
    pub fn new(analyzer: Arc<dyn Analyzer>) -> Self {
        Self {
            vocabulary: HashMap::new(),
            idf: Vec::new(),
            n_documents: 0,
            analyzer,
        }
    }

    /// Fit the vectorizer on training documents.
    ///
    /// Builds vocabulary from the documents and calculates IDF values.
    /// This must be called before using `transform()`.
    ///
    /// # Arguments
    /// * `documents` - Training documents to build vocabulary from
    ///
    /// # Returns
    /// Ok if successful
    ///
    /// # IDF Formula
    /// IDF(term) = log((N + 1) / (df + 1)) + 1
    /// where N is total documents and df is document frequency of the term
    pub fn fit(&mut self, documents: &[String]) -> Result<()> {
        self.n_documents = documents.len();
        let mut vocabulary = HashMap::new();
        let mut document_frequency: HashMap<String, usize> = HashMap::new();

        // Build vocabulary and count document frequencies
        for doc in documents {
            let tokens = Self::tokenize_with_analyzer(doc, &self.analyzer)?;
            let unique_tokens: std::collections::HashSet<_> = tokens.into_iter().collect();

            for token in unique_tokens {
                *document_frequency.entry(token.clone()).or_insert(0) += 1;
                if !vocabulary.contains_key(&token) {
                    let idx = vocabulary.len();
                    vocabulary.insert(token, idx);
                }
            }
        }

        // Calculate IDF for each term
        let mut idf = vec![0.0; vocabulary.len()];
        for (word, idx) in &vocabulary {
            let df = document_frequency.get(word).unwrap_or(&0);
            // IDF = log((N + 1) / (df + 1)) + 1
            idf[*idx] = ((self.n_documents as f64 + 1.0) / (*df as f64 + 1.0)).ln() + 1.0;
        }

        self.vocabulary = vocabulary;
        self.idf = idf;

        Ok(())
    }

    /// Transform a document into a TF-IDF feature vector.
    ///
    /// Converts a text document into a numerical vector based on the
    /// vocabulary learned during `fit()`. Terms not in the vocabulary
    /// are ignored.
    ///
    /// # Arguments
    /// * `document` - Document text to vectorize
    ///
    /// # Returns
    /// TF-IDF feature vector with length equal to vocabulary size
    ///
    /// # TF-IDF Calculation
    /// 1. Count term frequencies
    /// 2. Normalize by document length
    /// 3. Multiply by IDF for each term
    pub fn transform(&self, document: &str) -> Result<Vec<f64>> {
        let tokens = Self::tokenize_with_analyzer(document, &self.analyzer)?;
        let mut tf = vec![0.0; self.vocabulary.len()];

        // Count term frequencies
        for token in &tokens {
            if let Some(&idx) = self.vocabulary.get(token) {
                tf[idx] += 1.0;
            }
        }

        // Normalize by document length
        let doc_length = tokens.len() as f64;
        if doc_length > 0.0 {
            for count in &mut tf {
                *count /= doc_length;
            }
        }

        // Apply IDF
        for (idx, count) in tf.iter_mut().enumerate() {
            *count *= self.idf[idx];
        }

        Ok(tf)
    }

    /// Tokenize a document using the provided analyzer.
    ///
    /// Extracts text tokens from the document for vocabulary building
    /// and feature extraction.
    ///
    /// # Arguments
    /// * `text` - Text to tokenize
    /// * `analyzer` - Analyzer to use for tokenization
    ///
    /// # Returns
    /// Vector of token strings
    fn tokenize_with_analyzer(text: &str, analyzer: &Arc<dyn Analyzer>) -> Result<Vec<String>> {
        let tokens: Vec<String> = analyzer.analyze(text)?.map(|token| token.text).collect();
        Ok(tokens)
    }

    /// Get the size of the vocabulary.
    ///
    /// Returns the number of unique terms in the vocabulary,
    /// which is also the dimensionality of the feature vectors.
    pub fn vocabulary_size(&self) -> usize {
        self.vocabulary.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::analyzer::language::english::EnglishAnalyzer;

    #[test]
    fn test_tfidf_vectorizer() {
        let documents = vec![
            "what is machine learning".to_string(),
            "how to install python".to_string(),
            "buy laptop online".to_string(),
        ];

        let analyzer = Arc::new(EnglishAnalyzer::new().unwrap());
        let mut vectorizer = TfIdfVectorizer::new(analyzer);
        vectorizer.fit(&documents).unwrap();
        assert!(vectorizer.vocabulary_size() > 0);

        let features = vectorizer.transform("what is python").unwrap();
        assert_eq!(features.len(), vectorizer.vocabulary_size());
    }
}
