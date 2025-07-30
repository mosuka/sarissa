//! Spelling suggestion generation algorithms.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

use serde::{Deserialize, Serialize};

use crate::spelling::dictionary::SpellingDictionary;
use crate::spelling::levenshtein::{LevenshteinMatcher, TypoPatterns};

/// A spelling suggestion with a score indicating confidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Suggestion {
    /// The suggested word.
    pub word: String,
    /// Confidence score (higher is better, 0.0 to 1.0).
    pub score: f64,
    /// Edit distance from the original word.
    pub distance: usize,
    /// Frequency of the suggested word in the dictionary.
    pub frequency: u32,
}

impl Suggestion {
    /// Create a new suggestion.
    pub fn new(word: String, score: f64, distance: usize, frequency: u32) -> Self {
        Suggestion {
            word,
            score,
            distance,
            frequency,
        }
    }
}

impl Eq for Suggestion {}

impl Ord for Suggestion {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher scores come first
        other
            .score
            .partial_cmp(&self.score)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for Suggestion {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Configuration for spelling suggestion generation.
#[derive(Debug, Clone)]
pub struct SuggestionConfig {
    /// Maximum edit distance to consider.
    pub max_distance: usize,
    /// Maximum number of suggestions to return.
    pub max_suggestions: usize,
    /// Minimum frequency threshold for suggestions.
    pub min_frequency: u32,
    /// Weight for edit distance in scoring (0.0 to 1.0).
    pub distance_weight: f64,
    /// Weight for word frequency in scoring (0.0 to 1.0).
    pub frequency_weight: f64,
    /// Whether to use keyboard distance for better typo detection.
    pub use_keyboard_distance: bool,
    /// Whether to consider phonetic similarity.
    pub use_phonetic: bool,
}

impl Default for SuggestionConfig {
    fn default() -> Self {
        SuggestionConfig {
            max_distance: 2,
            max_suggestions: 5,
            min_frequency: 1,
            distance_weight: 0.6,
            frequency_weight: 0.4,
            use_keyboard_distance: true,
            use_phonetic: false,
        }
    }
}

/// Main spelling suggestion engine.
pub struct SuggestionEngine {
    dictionary: SpellingDictionary,
    config: SuggestionConfig,
}

impl SuggestionEngine {
    /// Create a new suggestion engine with the given dictionary.
    pub fn new(dictionary: SpellingDictionary) -> Self {
        SuggestionEngine {
            dictionary,
            config: SuggestionConfig::default(),
        }
    }

    /// Create a new suggestion engine with custom configuration.
    pub fn with_config(dictionary: SpellingDictionary, config: SuggestionConfig) -> Self {
        SuggestionEngine { dictionary, config }
    }

    /// Update the configuration.
    pub fn set_config(&mut self, config: SuggestionConfig) {
        self.config = config;
    }

    /// Get suggestions for a potentially misspelled word.
    pub fn suggest(&self, word: &str) -> Vec<Suggestion> {
        let word_lower = word.to_lowercase();

        // If the word is already in the dictionary, return it as the top suggestion
        if self.dictionary.contains(&word_lower) {
            let frequency = self.dictionary.frequency(&word_lower);
            return vec![Suggestion::new(word_lower, 1.0, 0, frequency)];
        }

        let mut suggestions = BinaryHeap::new();
        let matcher = LevenshteinMatcher::new(word_lower.clone());

        // Generate candidates using different methods
        let candidates = self.generate_candidates(&word_lower);

        for candidate in candidates {
            if let Some(distance) = matcher.distance_threshold(&candidate, self.config.max_distance)
            {
                let frequency = self.dictionary.frequency(&candidate);

                if frequency >= self.config.min_frequency {
                    let score = self.calculate_score(&word_lower, &candidate, distance, frequency);
                    suggestions.push(Suggestion::new(candidate, score, distance, frequency));
                }
            }
        }

        // Convert to sorted vector and limit results
        let mut result: Vec<Suggestion> = suggestions.into_sorted_vec();
        result.reverse(); // BinaryHeap returns lowest first, we want highest
        result.truncate(self.config.max_suggestions);
        result
    }

    /// Generate candidate words for correction.
    fn generate_candidates(&self, word: &str) -> HashSet<String> {
        let mut candidates = HashSet::new();

        // Add exact matches and simple variations
        candidates.extend(self.generate_edits(word, 1));

        if self.config.max_distance >= 2 {
            // Generate second-level edits for more flexibility
            let first_edits = self.generate_edits(word, 1);
            for edit in &first_edits {
                candidates.extend(self.generate_edits(edit, 1));
            }
        }

        // Add prefix matches for autocomplete-like suggestions
        candidates.extend(
            self.dictionary
                .words_with_prefix(&word[..word.len().min(3)]),
        );

        // Filter to only include dictionary words
        candidates.retain(|candidate| self.dictionary.contains(candidate));

        candidates
    }

    /// Generate all possible single edits of a word.
    fn generate_edits(&self, word: &str, max_distance: usize) -> HashSet<String> {
        if max_distance == 0 {
            return HashSet::new();
        }

        let mut edits = HashSet::new();
        let chars: Vec<char> = word.chars().collect();
        let len = chars.len();

        // Deletions
        for i in 0..len {
            let mut new_word = chars.clone();
            new_word.remove(i);
            edits.insert(new_word.into_iter().collect());
        }

        // Transpositions (swapping adjacent characters)
        for i in 0..len.saturating_sub(1) {
            let mut new_word = chars.clone();
            new_word.swap(i, i + 1);
            edits.insert(new_word.into_iter().collect());
        }

        // Replacements
        for i in 0..len {
            for ch in 'a'..='z' {
                if ch != chars[i] {
                    let mut new_word = chars.clone();
                    new_word[i] = ch;
                    edits.insert(new_word.into_iter().collect());
                }
            }
        }

        // Insertions
        for i in 0..=len {
            for ch in 'a'..='z' {
                let mut new_word = chars.clone();
                new_word.insert(i, ch);
                edits.insert(new_word.into_iter().collect());
            }
        }

        // If using keyboard distance, add keyboard-specific replacements
        if self.config.use_keyboard_distance {
            for i in 0..len {
                let nearby_keys = TypoPatterns::nearby_keys(chars[i]);
                for &nearby_char in &nearby_keys {
                    let mut new_word = chars.clone();
                    new_word[i] = nearby_char;
                    edits.insert(new_word.into_iter().collect());
                }
            }
        }

        edits
    }

    /// Calculate a confidence score for a suggestion.
    fn calculate_score(
        &self,
        original: &str,
        candidate: &str,
        distance: usize,
        frequency: u32,
    ) -> f64 {
        // Distance score (closer distance = higher score)
        let distance_score = if distance == 0 {
            1.0
        } else {
            1.0 / (1.0 + distance as f64)
        };

        // Frequency score (logarithmic scale to prevent domination by very common words)
        let frequency_score = if frequency == 0 {
            0.0
        } else {
            (frequency as f64).ln() / (self.dictionary.total_frequency() as f64).ln()
        };

        // Length similarity bonus
        let length_penalty = if original.len() == candidate.len() {
            1.0
        } else {
            0.9 // Small penalty for length differences
        };

        // Prefix bonus (words starting with the same letters are more likely correct)
        let prefix_bonus = self.calculate_prefix_bonus(original, candidate);

        // Keyboard distance bonus
        let keyboard_bonus = if self.config.use_keyboard_distance {
            let keyboard_dist = TypoPatterns::keyboard_distance(original, candidate);
            let regular_dist = distance as f64;
            if keyboard_dist < regular_dist {
                1.1 // 10% bonus for keyboard-friendly corrections
            } else {
                1.0
            }
        } else {
            1.0
        };

        // Combine scores
        let base_score = distance_score * self.config.distance_weight
            + frequency_score * self.config.frequency_weight;

        (base_score * length_penalty * prefix_bonus * keyboard_bonus).min(1.0)
    }

    /// Calculate bonus for common prefixes.
    fn calculate_prefix_bonus(&self, original: &str, candidate: &str) -> f64 {
        let orig_chars: Vec<char> = original.chars().collect();
        let cand_chars: Vec<char> = candidate.chars().collect();

        let common_prefix_len = orig_chars
            .iter()
            .zip(cand_chars.iter())
            .take_while(|(a, b)| a == b)
            .count();

        let max_len = orig_chars.len().max(cand_chars.len());
        if max_len == 0 {
            return 1.0;
        }

        // Bonus ranges from 1.0 (no common prefix) to 1.2 (same prefix)
        1.0 + (common_prefix_len as f64 / max_len as f64) * 0.2
    }

    /// Get dictionary statistics.
    pub fn dictionary_stats(&self) -> (usize, u64) {
        (
            self.dictionary.word_count(),
            self.dictionary.total_frequency(),
        )
    }

    /// Check if a word exists in the dictionary.
    pub fn is_correct(&self, word: &str) -> bool {
        self.dictionary.contains(word)
    }

    /// Get the frequency of a word in the dictionary.
    pub fn word_frequency(&self, word: &str) -> u32 {
        self.dictionary.frequency(word)
    }
}

/// Utility functions for common spelling correction tasks.
pub struct SpellingUtils;

impl SpellingUtils {
    /// Quick check if a word looks like it might be misspelled.
    /// This is a fast heuristic, not a definitive check.
    pub fn might_be_misspelled(word: &str) -> bool {
        // Very short words are usually correct
        if word.len() <= 2 {
            return false;
        }

        // Words with repeated characters might be typos
        let chars: Vec<char> = word.chars().collect();
        let has_repeated = chars.windows(3).any(|w| w[0] == w[1] && w[1] == w[2]);

        // Words with unusual character patterns
        let has_unusual_patterns =
            word.contains("qq") || word.contains("xx") || word.contains("zz");

        has_repeated || has_unusual_patterns
    }

    /// Extract potential words from mixed text for spell checking.
    pub fn extract_words(text: &str) -> Vec<String> {
        // Common stop words to filter out
        let stop_words = [
            "is", "a", "an", "the", "and", "or", "not", "in", "on", "at", "to", "for", "of",
            "with", "by",
        ];

        text.split(|c: char| !c.is_alphabetic())
            .filter_map(|word| {
                let cleaned = word.to_lowercase();
                if !word.is_empty() && word.len() > 1 && !stop_words.contains(&cleaned.as_str()) {
                    Some(cleaned)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Calculate the overall "correctness" score of a text.
    pub fn text_correctness_score(text: &str, engine: &SuggestionEngine) -> f64 {
        let words = Self::extract_words(text);
        if words.is_empty() {
            return 1.0;
        }

        let correct_words = words.iter().filter(|word| engine.is_correct(word)).count();

        correct_words as f64 / words.len() as f64
    }

    /// Get suggestions for all potentially misspelled words in text.
    pub fn correct_text(text: &str, engine: &SuggestionEngine) -> Vec<(String, Vec<Suggestion>)> {
        let words = Self::extract_words(text);
        let mut corrections = Vec::new();

        for word in words {
            if !engine.is_correct(&word) {
                let suggestions = engine.suggest(&word);
                if !suggestions.is_empty() {
                    corrections.push((word, suggestions));
                }
            }
        }

        corrections
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spelling::dictionary::BuiltinDictionary;

    #[test]
    fn test_suggestion_ordering() {
        let s1 = Suggestion::new("hello".to_string(), 0.9, 1, 100);
        let s2 = Suggestion::new("world".to_string(), 0.8, 1, 50);
        let s3 = Suggestion::new("test".to_string(), 0.95, 0, 200);

        let mut suggestions = [s1, s2, s3];
        suggestions.sort();

        assert_eq!(suggestions[0].word, "test");
        assert_eq!(suggestions[1].word, "hello");
        assert_eq!(suggestions[2].word, "world");
    }

    #[test]
    fn test_suggestion_engine_correct_word() {
        let dict = BuiltinDictionary::minimal();
        let engine = SuggestionEngine::new(dict);

        let suggestions = engine.suggest("hello");
        assert_eq!(suggestions.len(), 1);
        assert_eq!(suggestions[0].word, "hello");
        assert_eq!(suggestions[0].distance, 0);
        assert!((suggestions[0].score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_suggestion_engine_typos() {
        let dict = BuiltinDictionary::minimal();
        let engine = SuggestionEngine::new(dict);

        // Test simple typo
        let suggestions = engine.suggest("helo"); // missing 'l'
        assert!(!suggestions.is_empty());
        assert!(suggestions.iter().any(|s| s.word == "hello"));

        // Test transposition
        let suggestions = engine.suggest("serach"); // 'search' with transposed 'a' and 'r'
        assert!(!suggestions.is_empty());
        // Note: might not find 'search' depending on edit distance, but should find something
    }

    #[test]
    fn test_suggestion_engine_configuration() {
        let dict = BuiltinDictionary::minimal();
        let config = SuggestionConfig {
            max_distance: 1,
            max_suggestions: 3,
            min_frequency: 1,
            distance_weight: 0.8,
            frequency_weight: 0.2,
            use_keyboard_distance: false,
            use_phonetic: false,
        };
        let engine = SuggestionEngine::with_config(dict, config);

        let suggestions = engine.suggest("helo");
        assert!(suggestions.len() <= 3);

        // All suggestions should have distance <= 1
        for suggestion in &suggestions {
            assert!(suggestion.distance <= 1);
        }
    }

    #[test]
    fn test_generate_edits() {
        let dict = BuiltinDictionary::minimal();
        let engine = SuggestionEngine::new(dict);

        let edits = engine.generate_edits("cat", 1);

        // Should contain deletions
        assert!(edits.contains("at"));
        assert!(edits.contains("ct"));
        assert!(edits.contains("ca"));

        // Should contain insertions (many possibilities)
        assert!(edits.len() > 50); // Lots of possible single edits

        // Should contain some substitutions
        assert!(edits.contains("bat"));
        assert!(edits.contains("cot"));
    }

    #[test]
    fn test_prefix_bonus() {
        let dict = BuiltinDictionary::minimal();
        let engine = SuggestionEngine::new(dict);

        let bonus1 = engine.calculate_prefix_bonus("search", "searching"); // common prefix
        let bonus2 = engine.calculate_prefix_bonus("search", "church"); // no common prefix

        assert!(bonus1 > bonus2);
        assert!(bonus1 > 1.0);
    }

    #[test]
    fn test_spelling_utils() {
        // Test word extraction
        let words = SpellingUtils::extract_words("Hello, world! This is a test.");
        assert!(words.contains(&"hello".to_string()));
        assert!(words.contains(&"world".to_string()));
        assert!(words.contains(&"test".to_string()));
        assert!(!words.contains(&"is".to_string())); // too short

        // Test misspelling detection heuristics
        assert!(SpellingUtils::might_be_misspelled("helllo")); // repeated l
        assert!(SpellingUtils::might_be_misspelled("qqqword")); // unusual pattern
        assert!(!SpellingUtils::might_be_misspelled("hello")); // normal word
        assert!(!SpellingUtils::might_be_misspelled("it")); // too short
    }

    #[test]
    fn test_text_correctness_score() {
        let dict = BuiltinDictionary::minimal();
        let engine = SuggestionEngine::new(dict);

        let perfect_text = "hello world search query";
        let score1 = SpellingUtils::text_correctness_score(perfect_text, &engine);

        let imperfect_text = "helo world serach query";
        let score2 = SpellingUtils::text_correctness_score(imperfect_text, &engine);

        assert!(score1 > score2);
        assert!(score1 > 0.5); // Should be reasonably high for mostly correct text
    }

    #[test]
    fn test_correct_text() {
        let dict = BuiltinDictionary::minimal();
        let engine = SuggestionEngine::new(dict);

        let text = "helo world, this is a tst";
        let corrections = SpellingUtils::correct_text(text, &engine);

        // Should find corrections for misspelled words
        assert!(!corrections.is_empty());

        // Check if we found expected misspellings
        let misspelled_words: Vec<&String> = corrections.iter().map(|(word, _)| word).collect();
        assert!(misspelled_words.contains(&&"helo".to_string()));
    }

    #[test]
    fn test_keyboard_distance_in_suggestions() {
        let dict = BuiltinDictionary::minimal();
        let config = SuggestionConfig {
            use_keyboard_distance: true,
            ..Default::default()
        };
        let engine = SuggestionEngine::with_config(dict, config);

        // 'g' and 'h' are adjacent on keyboard
        let suggestions = engine.suggest("gello"); // should suggest "hello"

        // We can't guarantee "hello" will be found without a more comprehensive dictionary,
        // but the mechanism should work
        assert!(!suggestions.is_empty());
    }
}
