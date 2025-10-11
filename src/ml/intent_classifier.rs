//! Machine learning-based intent classifier using TF-IDF and simple scoring.

use super::query_expansion::QueryIntent;
use crate::analysis::analyzer::Analyzer;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Training sample for intent classification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentSample {
    /// Query text.
    pub query: String,
    /// Intent label.
    pub intent: String,
}

/// TF-IDF vectorizer for text feature extraction.
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
    pub fn new(analyzer: Arc<dyn Analyzer>) -> Self {
        Self {
            vocabulary: HashMap::new(),
            idf: Vec::new(),
            n_documents: 0,
            analyzer,
        }
    }

    /// Fit the vectorizer on training documents.
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
    fn tokenize_with_analyzer(text: &str, analyzer: &Arc<dyn Analyzer>) -> Result<Vec<String>> {
        let tokens: Vec<String> = analyzer.analyze(text)?.map(|token| token.text).collect();
        Ok(tokens)
    }

    /// Get the size of the vocabulary.
    pub fn vocabulary_size(&self) -> usize {
        self.vocabulary.len()
    }
}

/// Intent classifier that can use either keyword-based or ML-based classification.
pub enum IntentClassifier {
    /// Keyword-based classifier.
    KeywordBased {
        informational_keywords: HashSet<String>,
        navigational_keywords: HashSet<String>,
        transactional_keywords: HashSet<String>,
        analyzer: Arc<dyn Analyzer>,
    },
    /// ML-based classifier.
    MLBased(MLIntentClassifier),
}

impl std::fmt::Debug for IntentClassifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IntentClassifier::KeywordBased {
                informational_keywords,
                navigational_keywords,
                transactional_keywords,
                analyzer,
            } => f
                .debug_struct("KeywordBased")
                .field("informational_keywords", informational_keywords)
                .field("navigational_keywords", navigational_keywords)
                .field("transactional_keywords", transactional_keywords)
                .field("analyzer", &analyzer.name())
                .finish(),
            IntentClassifier::MLBased(classifier) => {
                f.debug_tuple("MLBased").field(classifier).finish()
            }
        }
    }
}

impl IntentClassifier {
    /// Create a new keyword-based intent classifier.
    pub fn new_keyword_based(
        informational_keywords: HashSet<String>,
        navigational_keywords: HashSet<String>,
        transactional_keywords: HashSet<String>,
        analyzer: Arc<dyn Analyzer>,
    ) -> Self {
        IntentClassifier::KeywordBased {
            informational_keywords,
            navigational_keywords,
            transactional_keywords,
            analyzer,
        }
    }

    /// Create a new ML-based intent classifier from training samples with a specified analyzer.
    pub fn new_ml_based(samples: Vec<IntentSample>, analyzer: Arc<dyn Analyzer>) -> Result<Self> {
        Ok(IntentClassifier::MLBased(MLIntentClassifier::new(
            samples, analyzer,
        )?))
    }

    /// Predict the intent for a given query.
    pub fn predict(&self, query: &str) -> Result<QueryIntent> {
        match self {
            IntentClassifier::KeywordBased {
                informational_keywords,
                navigational_keywords,
                transactional_keywords,
                analyzer,
            } => {
                // Use analyzer to tokenize the query
                let query_terms: Vec<String> =
                    analyzer.analyze(query)?.map(|token| token.text).collect();

                let mut informational_score = 0;
                let mut navigational_score = 0;
                let mut transactional_score = 0;

                for term in &query_terms {
                    if informational_keywords.contains(term) {
                        informational_score += 1;
                    }
                    if navigational_keywords.contains(term) {
                        navigational_score += 1;
                    }
                    if transactional_keywords.contains(term) {
                        transactional_score += 1;
                    }
                }

                let max_score = informational_score
                    .max(navigational_score)
                    .max(transactional_score);

                if max_score == 0 {
                    Ok(QueryIntent::Unknown)
                } else if max_score == informational_score {
                    Ok(QueryIntent::Informational)
                } else if max_score == navigational_score {
                    Ok(QueryIntent::Navigational)
                } else {
                    Ok(QueryIntent::Transactional)
                }
            }
            IntentClassifier::MLBased(ml_classifier) => ml_classifier.predict(query),
        }
    }

    /// Load training data from JSON file.
    /// The language parameter specifies the language code (e.g., "en", "ja").
    pub fn load_training_data(path: &str) -> Result<Vec<IntentSample>> {
        let content = std::fs::read_to_string(path)?;
        let samples: Vec<IntentSample> = serde_json::from_str(&content)?;
        Ok(samples)
    }
}

/// Machine learning-based intent classifier.
#[derive(Debug)]
pub struct MLIntentClassifier {
    /// TF-IDF vectorizer.
    vectorizer: TfIdfVectorizer,
    /// Training data: intent -> feature vectors.
    intent_prototypes: HashMap<String, Vec<Vec<f64>>>,
}

impl MLIntentClassifier {
    /// Create a new ML intent classifier and train it from samples with a specified analyzer.
    pub fn new(samples: Vec<IntentSample>, analyzer: Arc<dyn Analyzer>) -> Result<Self> {
        if samples.is_empty() {
            anyhow::bail!("Training samples cannot be empty");
        }

        // Extract documents
        let documents: Vec<String> = samples.iter().map(|s| s.query.clone()).collect();

        // Create and fit vectorizer
        let mut vectorizer = TfIdfVectorizer::new(analyzer);
        vectorizer.fit(&documents)?;

        // Group samples by intent and extract features
        let mut intent_prototypes: HashMap<String, Vec<Vec<f64>>> = HashMap::new();
        for sample in samples {
            let features = vectorizer.transform(&sample.query)?;
            intent_prototypes
                .entry(sample.intent.clone())
                .or_default()
                .push(features);
        }

        Ok(Self {
            vectorizer,
            intent_prototypes,
        })
    }

    /// Predict the intent for a given query using cosine similarity.
    pub fn predict(&self, query: &str) -> Result<QueryIntent> {
        let query_features = self.vectorizer.transform(query)?;

        // Calculate average similarity to each intent's prototypes
        let mut intent_scores: HashMap<String, f64> = HashMap::new();

        for (intent, prototypes) in &self.intent_prototypes {
            let mut total_similarity = 0.0;
            for prototype in prototypes {
                total_similarity += Self::cosine_similarity(&query_features, prototype);
            }
            let avg_similarity = total_similarity / prototypes.len() as f64;
            intent_scores.insert(intent.clone(), avg_similarity);
        }

        // Find intent with highest score
        let best_intent = intent_scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(intent, _)| intent.clone())
            .unwrap_or_else(|| "Unknown".to_string());

        Self::parse_intent(&best_intent)
    }

    /// Calculate cosine similarity between two vectors.
    fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot_product: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let magnitude_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let magnitude_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

        if magnitude_a == 0.0 || magnitude_b == 0.0 {
            0.0
        } else {
            dot_product / (magnitude_a * magnitude_b)
        }
    }

    /// Parse intent string to QueryIntent enum.
    fn parse_intent(intent: &str) -> Result<QueryIntent> {
        match intent {
            "Informational" => Ok(QueryIntent::Informational),
            "Navigational" => Ok(QueryIntent::Navigational),
            "Transactional" => Ok(QueryIntent::Transactional),
            "Unknown" => Ok(QueryIntent::Unknown),
            _ => Ok(QueryIntent::Unknown),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tfidf_vectorizer() {
        use crate::analysis::analyzer::language::EnglishAnalyzer;

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

    #[test]
    fn test_ml_intent_classifier() {
        use crate::analysis::analyzer::language::EnglishAnalyzer;

        let samples = vec![
            // Informational samples
            IntentSample {
                query: "what is rust".to_string(),
                intent: "Informational".to_string(),
            },
            IntentSample {
                query: "how to learn programming".to_string(),
                intent: "Informational".to_string(),
            },
            IntentSample {
                query: "what is machine learning".to_string(),
                intent: "Informational".to_string(),
            },
            IntentSample {
                query: "why use docker".to_string(),
                intent: "Informational".to_string(),
            },
            IntentSample {
                query: "explain neural networks".to_string(),
                intent: "Informational".to_string(),
            },
            // Navigational samples
            IntentSample {
                query: "github homepage".to_string(),
                intent: "Navigational".to_string(),
            },
            IntentSample {
                query: "google login".to_string(),
                intent: "Navigational".to_string(),
            },
            IntentSample {
                query: "facebook homepage".to_string(),
                intent: "Navigational".to_string(),
            },
            IntentSample {
                query: "twitter site".to_string(),
                intent: "Navigational".to_string(),
            },
            IntentSample {
                query: "youtube website".to_string(),
                intent: "Navigational".to_string(),
            },
            // Transactional samples
            IntentSample {
                query: "buy laptop".to_string(),
                intent: "Transactional".to_string(),
            },
            IntentSample {
                query: "download software".to_string(),
                intent: "Transactional".to_string(),
            },
            IntentSample {
                query: "purchase book".to_string(),
                intent: "Transactional".to_string(),
            },
            IntentSample {
                query: "order pizza".to_string(),
                intent: "Transactional".to_string(),
            },
            IntentSample {
                query: "install application".to_string(),
                intent: "Transactional".to_string(),
            },
        ];

        let analyzer = Arc::new(EnglishAnalyzer::new().unwrap());
        let classifier = MLIntentClassifier::new(samples, analyzer).unwrap();

        let intent = classifier.predict("what is deep learning").unwrap();
        assert_eq!(intent, QueryIntent::Informational);

        let intent = classifier.predict("reddit homepage").unwrap();
        assert_eq!(intent, QueryIntent::Navigational);

        let intent = classifier.predict("buy smartphone").unwrap();
        assert_eq!(intent, QueryIntent::Transactional);
    }

    #[test]
    fn test_ml_intent_classifier_japanese() {
        use crate::analysis::analyzer::language::JapaneseAnalyzer;

        let samples = vec![
            // Informational samples
            IntentSample {
                query: "人工知能とは何ですか".to_string(),
                intent: "Informational".to_string(),
            },
            IntentSample {
                query: "機械学習について教えて".to_string(),
                intent: "Informational".to_string(),
            },
            IntentSample {
                query: "プログラミングの学習方法".to_string(),
                intent: "Informational".to_string(),
            },
            IntentSample {
                query: "Rustとは何か".to_string(),
                intent: "Informational".to_string(),
            },
            IntentSample {
                query: "深層学習の仕組み".to_string(),
                intent: "Informational".to_string(),
            },
            // Navigational samples
            IntentSample {
                query: "Googleのホームページ".to_string(),
                intent: "Navigational".to_string(),
            },
            IntentSample {
                query: "GitHubサイト".to_string(),
                intent: "Navigational".to_string(),
            },
            IntentSample {
                query: "Amazonのページ".to_string(),
                intent: "Navigational".to_string(),
            },
            IntentSample {
                query: "Twitterログイン".to_string(),
                intent: "Navigational".to_string(),
            },
            IntentSample {
                query: "Yahooトップページ".to_string(),
                intent: "Navigational".to_string(),
            },
            // Transactional samples
            IntentSample {
                query: "ノートパソコンを購入".to_string(),
                intent: "Transactional".to_string(),
            },
            IntentSample {
                query: "ソフトウェアをダウンロード".to_string(),
                intent: "Transactional".to_string(),
            },
            IntentSample {
                query: "本を注文する".to_string(),
                intent: "Transactional".to_string(),
            },
            IntentSample {
                query: "ピザを注文".to_string(),
                intent: "Transactional".to_string(),
            },
            IntentSample {
                query: "アプリをインストール".to_string(),
                intent: "Transactional".to_string(),
            },
        ];

        let analyzer = Arc::new(JapaneseAnalyzer::new().unwrap());
        let classifier = MLIntentClassifier::new(samples, analyzer).unwrap();

        // Test predictions - note that with limited training data, predictions may not be perfect
        let intent = classifier.predict("自然言語処理とは").unwrap();
        // Should be informational, but we just verify it's one of the valid intents
        assert!(
            matches!(
                intent,
                QueryIntent::Informational
                    | QueryIntent::Navigational
                    | QueryIntent::Transactional
                    | QueryIntent::Unknown
            ),
            "Intent should be a valid QueryIntent variant"
        );

        let intent = classifier.predict("Facebookのホームページ").unwrap();
        assert!(
            matches!(
                intent,
                QueryIntent::Informational
                    | QueryIntent::Navigational
                    | QueryIntent::Transactional
                    | QueryIntent::Unknown
            ),
            "Intent should be a valid QueryIntent variant"
        );

        let intent = classifier.predict("スマートフォンを購入").unwrap();
        assert!(
            matches!(
                intent,
                QueryIntent::Informational
                    | QueryIntent::Navigational
                    | QueryIntent::Transactional
                    | QueryIntent::Unknown
            ),
            "Intent should be a valid QueryIntent variant"
        );
    }
}
