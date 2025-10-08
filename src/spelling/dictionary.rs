//! Dictionary management for spelling correction.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::error::Result;

/// A dictionary that stores words and their frequencies for spelling correction.
#[derive(Debug, Clone)]
pub struct SpellingDictionary {
    /// Words and their frequencies
    words: HashMap<String, u32>,
    /// Set of all words for fast lookup
    word_set: HashSet<String>,
    /// Total word count for probability calculations
    total_count: u64,
}

impl SpellingDictionary {
    /// Create a new empty dictionary.
    pub fn new() -> Self {
        SpellingDictionary {
            words: HashMap::new(),
            word_set: HashSet::new(),
            total_count: 0,
        }
    }

    /// Add a word to the dictionary with the given frequency.
    pub fn add_word(&mut self, word: String, frequency: u32) {
        let normalized = word.to_lowercase();

        // Update or insert the word
        let old_freq = self.words.get(&normalized).copied().unwrap_or(0);
        self.words.insert(normalized.clone(), frequency);
        self.word_set.insert(normalized);

        // Update total count
        self.total_count = self.total_count - old_freq as u64 + frequency as u64;
    }

    /// Increment the frequency of a word by 1.
    pub fn increment_word(&mut self, word: &str) {
        let normalized = word.to_lowercase();
        let current = self.words.get(&normalized).copied().unwrap_or(0);
        self.add_word(normalized, current + 1);
    }

    /// Check if a word exists in the dictionary.
    pub fn contains(&self, word: &str) -> bool {
        self.word_set.contains(&word.to_lowercase())
    }

    /// Get the frequency of a word.
    pub fn frequency(&self, word: &str) -> u32 {
        self.words.get(&word.to_lowercase()).copied().unwrap_or(0)
    }

    /// Get the probability of a word (frequency / total_count).
    pub fn probability(&self, word: &str) -> f64 {
        if self.total_count == 0 {
            return 0.0;
        }
        self.frequency(word) as f64 / self.total_count as f64
    }

    /// Get all words in the dictionary.
    pub fn words(&self) -> &HashMap<String, u32> {
        &self.words
    }

    /// Get the total number of unique words.
    pub fn word_count(&self) -> usize {
        self.words.len()
    }

    /// Get the total frequency count.
    pub fn total_frequency(&self) -> u64 {
        self.total_count
    }

    /// Load dictionary from a text file with one word per line.
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut dictionary = SpellingDictionary::new();
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        for line in reader.lines() {
            let line = line?;
            let word = line.trim();
            if !word.is_empty() && word.chars().all(|c| c.is_alphabetic()) {
                dictionary.increment_word(word);
            }
        }

        Ok(dictionary)
    }

    /// Load dictionary from a frequency file with format "word frequency" per line.
    pub fn load_from_frequency_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut dictionary = SpellingDictionary::new();
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        for line in reader.lines() {
            let line = line?;
            let parts: Vec<&str> = line.split_whitespace().collect();

            if parts.len() >= 2 {
                let word = parts[0];
                if let Ok(frequency) = parts[1].parse::<u32>()
                    && word.chars().all(|c| c.is_alphabetic())
                {
                    dictionary.add_word(word.to_string(), frequency);
                }
            }
        }

        Ok(dictionary)
    }

    /// Create a dictionary from a corpus of text.
    pub fn from_corpus(text: &str) -> Self {
        let mut dictionary = SpellingDictionary::new();

        // Simple tokenization - split on non-alphabetic characters
        let words = text
            .split(|c: char| !c.is_alphabetic())
            .filter(|word| !word.is_empty() && word.len() > 1)
            .map(|word| word.to_lowercase());

        for word in words {
            dictionary.increment_word(&word);
        }

        dictionary
    }

    /// Get words that start with the given prefix.
    pub fn words_with_prefix(&self, prefix: &str) -> Vec<String> {
        let prefix_lower = prefix.to_lowercase();
        self.word_set
            .iter()
            .filter(|word| word.starts_with(&prefix_lower))
            .cloned()
            .collect()
    }

    /// Get the most frequent words in the dictionary.
    pub fn most_frequent_words(&self, limit: usize) -> Vec<(String, u32)> {
        let mut word_freq: Vec<(String, u32)> = self
            .words
            .iter()
            .map(|(word, freq)| (word.clone(), *freq))
            .collect();

        word_freq.sort_by(|a, b| b.1.cmp(&a.1));
        word_freq.truncate(limit);
        word_freq
    }

    /// Merge another dictionary into this one.
    pub fn merge(&mut self, other: &SpellingDictionary) {
        for (word, frequency) in &other.words {
            let current = self.frequency(word);
            self.add_word(word.clone(), current + frequency);
        }
    }

    /// Remove words with frequency below the threshold.
    pub fn prune_low_frequency(&mut self, min_frequency: u32) {
        let words_to_remove: Vec<String> = self
            .words
            .iter()
            .filter(|&(_, &freq)| freq < min_frequency)
            .map(|(word, _)| word.clone())
            .collect();

        for word in words_to_remove {
            if let Some(freq) = self.words.remove(&word) {
                self.word_set.remove(&word);
                self.total_count -= freq as u64;
            }
        }
    }

    /// Save dictionary to a frequency file.
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        use std::io::Write;

        let mut file = File::create(path)?;
        let mut word_freq: Vec<(&String, &u32)> = self.words.iter().collect();
        word_freq.sort_by(|a, b| b.1.cmp(a.1));

        for (word, frequency) in word_freq {
            writeln!(file, "{word} {frequency}")?;
        }

        Ok(())
    }
}

impl Default for SpellingDictionary {
    fn default() -> Self {
        Self::new()
    }
}

/// A simple built-in dictionary with common English words.
pub struct BuiltinDictionary;

impl BuiltinDictionary {
    /// Create a dictionary with common English words.
    pub fn english() -> SpellingDictionary {
        let mut dict = SpellingDictionary::new();

        // Add common English words with estimated frequencies
        let common_words = vec![
            ("the", 1000000),
            ("be", 500000),
            ("to", 450000),
            ("of", 400000),
            ("and", 380000),
            ("a", 350000),
            ("in", 300000),
            ("that", 250000),
            ("have", 200000),
            ("i", 180000),
            ("it", 170000),
            ("for", 160000),
            ("not", 150000),
            ("on", 140000),
            ("with", 130000),
            ("he", 120000),
            ("as", 110000),
            ("you", 100000),
            ("do", 95000),
            ("at", 90000),
            ("this", 85000),
            ("but", 80000),
            ("his", 75000),
            ("by", 70000),
            ("from", 65000),
            ("they", 60000),
            ("we", 55000),
            ("say", 50000),
            ("her", 48000),
            ("she", 46000),
            ("or", 44000),
            ("an", 42000),
            ("will", 40000),
            ("my", 38000),
            ("one", 36000),
            ("all", 34000),
            ("would", 32000),
            ("there", 30000),
            ("their", 28000),
            ("what", 26000),
            ("so", 24000),
            ("up", 22000),
            ("out", 20000),
            ("if", 19000),
            ("about", 18000),
            ("who", 17000),
            ("get", 16000),
            ("which", 15000),
            ("go", 14000),
            ("me", 13000),
            ("when", 12000),
            ("make", 11000),
            ("can", 10000),
            ("like", 9500),
            ("time", 9000),
            ("no", 8500),
            ("just", 8000),
            ("him", 7500),
            ("know", 7000),
            ("take", 6500),
            ("hello", 7200),
            ("world", 6800),
            ("is", 8200),
            ("people", 6000),
            ("into", 5500),
            ("year", 5000),
            ("your", 4800),
            ("good", 4600),
            ("some", 4400),
            ("could", 4200),
            ("them", 4000),
            ("see", 3800),
            ("other", 3600),
            ("than", 3400),
            ("then", 3200),
            ("now", 3000),
            ("look", 2800),
            ("only", 2600),
            ("come", 2400),
            ("its", 2200),
            ("over", 2000),
            ("think", 1900),
            ("also", 1800),
            ("back", 1700),
            ("after", 1600),
            ("use", 1500),
            ("two", 1400),
            ("how", 1300),
            ("our", 1200),
            ("work", 1100),
            ("first", 1000),
            ("well", 950),
            ("way", 900),
            ("even", 850),
            ("new", 800),
            ("want", 750),
            ("because", 700),
            ("any", 650),
            ("these", 600),
            ("give", 550),
            ("day", 500),
            ("most", 480),
            ("us", 460),
            ("is", 440),
            ("was", 420),
            ("are", 400),
            ("been", 380),
            ("has", 360),
            ("had", 340),
            ("were", 320),
            ("said", 300),
            ("each", 280),
            ("which", 260),
            ("during", 240),
            ("where", 220),
            ("did", 200),
            ("does", 190),
            ("doing", 180),
            ("made", 170),
            ("find", 160),
            ("home", 150),
            ("help", 140),
            ("hand", 130),
            ("right", 120),
            ("world", 110),
            ("life", 100),
            ("love", 95),
            ("house", 90),
            ("water", 85),
            ("place", 80),
            ("word", 75),
            ("before", 70),
            ("through", 65),
            ("still", 60),
            ("here", 55),
            ("should", 50),
            ("never", 48),
            ("each", 46),
            ("those", 44),
            ("came", 42),
            ("may", 40),
            ("part", 38),
            ("against", 36),
            ("such", 34),
            ("turn", 32),
            ("every", 30),
            ("don", 28),
            ("point", 26),
            ("small", 24),
            ("end", 22),
            ("why", 20),
        ];

        for (word, freq) in common_words {
            dict.add_word(word.to_string(), freq);
        }

        dict
    }

    /// Create a minimal dictionary for testing.
    pub fn minimal() -> SpellingDictionary {
        let mut dict = SpellingDictionary::new();

        let words = vec![
            "hello",
            "world",
            "search",
            "query",
            "text",
            "word",
            "spell",
            "correct",
            "suggestion",
            "dictionary",
            "language",
            "english",
            "computer",
            "program",
            "software",
            "system",
            "data",
            "information",
            "process",
            "result",
            "value",
            "number",
            "string",
            "character",
        ];

        for word in words {
            dict.add_word(word.to_string(), 100);
        }

        dict
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_dictionary_basic_operations() {
        let mut dict = SpellingDictionary::new();

        assert!(!dict.contains("hello"));
        assert_eq!(dict.frequency("hello"), 0);
        assert_eq!(dict.word_count(), 0);

        dict.add_word("hello".to_string(), 5);
        assert!(dict.contains("hello"));
        assert_eq!(dict.frequency("hello"), 5);
        assert_eq!(dict.word_count(), 1);
        assert_eq!(dict.total_frequency(), 5);

        dict.increment_word("hello");
        assert_eq!(dict.frequency("hello"), 6);
        assert_eq!(dict.total_frequency(), 6);

        dict.add_word("world".to_string(), 3);
        assert_eq!(dict.word_count(), 2);
        assert_eq!(dict.total_frequency(), 9);
    }

    #[test]
    fn test_dictionary_case_insensitive() {
        let mut dict = SpellingDictionary::new();

        dict.add_word("Hello".to_string(), 5);
        assert!(dict.contains("hello"));
        assert!(dict.contains("HELLO"));
        assert!(dict.contains("Hello"));

        dict.increment_word("HELLO");
        assert_eq!(dict.frequency("hello"), 6);
    }

    #[test]
    fn test_dictionary_probability() {
        let mut dict = SpellingDictionary::new();

        dict.add_word("hello".to_string(), 6);
        dict.add_word("world".to_string(), 4);

        assert!((dict.probability("hello") - 0.6).abs() < 1e-6);
        assert!((dict.probability("world") - 0.4).abs() < 1e-6);
        assert_eq!(dict.probability("nonexistent"), 0.0);
    }

    #[test]
    fn test_from_corpus() {
        let corpus = "The quick brown fox jumps over the lazy dog. The dog was lazy.";
        let dict = SpellingDictionary::from_corpus(corpus);

        assert!(dict.contains("the"));
        assert!(dict.contains("quick"));
        assert!(dict.contains("dog"));
        assert_eq!(dict.frequency("the"), 3);
        assert_eq!(dict.frequency("dog"), 2);
        assert_eq!(dict.frequency("lazy"), 2);
        assert_eq!(dict.frequency("quick"), 1);
    }

    #[test]
    fn test_words_with_prefix() {
        let mut dict = SpellingDictionary::new();
        dict.add_word("search".to_string(), 1);
        dict.add_word("searching".to_string(), 1);
        dict.add_word("server".to_string(), 1);
        dict.add_word("query".to_string(), 1);

        let search_words = dict.words_with_prefix("sear");
        assert_eq!(search_words.len(), 2);
        assert!(search_words.contains(&"search".to_string()));
        assert!(search_words.contains(&"searching".to_string()));

        let se_words = dict.words_with_prefix("se");
        assert_eq!(se_words.len(), 3);
    }

    #[test]
    fn test_most_frequent_words() {
        let mut dict = SpellingDictionary::new();
        dict.add_word("common".to_string(), 100);
        dict.add_word("rare".to_string(), 1);
        dict.add_word("medium".to_string(), 50);

        let top_words = dict.most_frequent_words(2);
        assert_eq!(top_words.len(), 2);
        assert_eq!(top_words[0], ("common".to_string(), 100));
        assert_eq!(top_words[1], ("medium".to_string(), 50));
    }

    #[test]
    fn test_merge_dictionaries() {
        let mut dict1 = SpellingDictionary::new();
        dict1.add_word("hello".to_string(), 5);
        dict1.add_word("world".to_string(), 3);

        let mut dict2 = SpellingDictionary::new();
        dict2.add_word("hello".to_string(), 2);
        dict2.add_word("test".to_string(), 4);

        dict1.merge(&dict2);

        assert_eq!(dict1.frequency("hello"), 7);
        assert_eq!(dict1.frequency("world"), 3);
        assert_eq!(dict1.frequency("test"), 4);
        assert_eq!(dict1.word_count(), 3);
    }

    #[test]
    fn test_prune_low_frequency() {
        let mut dict = SpellingDictionary::new();
        dict.add_word("common".to_string(), 100);
        dict.add_word("rare".to_string(), 1);
        dict.add_word("medium".to_string(), 5);

        dict.prune_low_frequency(5);

        assert!(dict.contains("common"));
        assert!(dict.contains("medium"));
        assert!(!dict.contains("rare"));
        assert_eq!(dict.word_count(), 2);
    }

    #[test]
    fn test_file_operations() {
        let mut dict = SpellingDictionary::new();
        dict.add_word("hello".to_string(), 5);
        dict.add_word("world".to_string(), 3);

        // Test saving to file
        let temp_file = NamedTempFile::new().unwrap();
        dict.save_to_file(temp_file.path()).unwrap();

        // Test loading frequency file
        let loaded_dict = SpellingDictionary::load_from_frequency_file(temp_file.path()).unwrap();
        assert_eq!(loaded_dict.frequency("hello"), 5);
        assert_eq!(loaded_dict.frequency("world"), 3);
        assert_eq!(loaded_dict.word_count(), 2);
    }

    #[test]
    fn test_load_from_simple_file() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "hello").unwrap();
        writeln!(temp_file, "world").unwrap();
        writeln!(temp_file, "hello").unwrap();
        temp_file.flush().unwrap();

        let dict = SpellingDictionary::load_from_file(temp_file.path()).unwrap();
        assert_eq!(dict.frequency("hello"), 2);
        assert_eq!(dict.frequency("world"), 1);
        assert_eq!(dict.word_count(), 2);
    }

    #[test]
    fn test_builtin_dictionaries() {
        let english_dict = BuiltinDictionary::english();
        assert!(english_dict.contains("the"));
        assert!(english_dict.contains("hello"));
        assert!(english_dict.word_count() > 50);

        let minimal_dict = BuiltinDictionary::minimal();
        assert!(minimal_dict.contains("hello"));
        assert!(minimal_dict.contains("search"));
        assert!(minimal_dict.word_count() > 10);
    }
}
