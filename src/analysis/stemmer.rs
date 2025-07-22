//! Stemming algorithms for reducing words to their root forms.

use crate::analysis::filter::Filter;
use crate::analysis::token::TokenStream;
use crate::error::Result;
use std::collections::HashMap;

/// Trait for stemming algorithms.
pub trait Stemmer: Send + Sync {
    /// Stem a word to its root form.
    fn stem(&self, word: &str) -> String;

    /// Get the name of this stemmer.
    fn name(&self) -> &'static str;
}

/// Porter stemming algorithm implementation.
///
/// This is a simplified version of the Porter stemming algorithm
/// for reducing English words to their stems.
#[derive(Debug, Clone, Default)]
pub struct PorterStemmer {
    /// Cache for stemmed words to improve performance.
    #[allow(dead_code)]
    cache: HashMap<String, String>,
}

impl PorterStemmer {
    /// Create a new Porter stemmer.
    pub fn new() -> Self {
        PorterStemmer {
            cache: HashMap::new(),
        }
    }

    /// Check if a character is a vowel.
    #[allow(clippy::only_used_in_recursion)]
    fn is_vowel(&self, word: &str, pos: usize) -> bool {
        if pos >= word.len() {
            return false;
        }

        let chars: Vec<char> = word.chars().collect();
        let c = chars[pos].to_ascii_lowercase();

        match c {
            'a' | 'e' | 'i' | 'o' | 'u' => true,
            'y' if pos > 0 => !self.is_vowel(word, pos - 1),
            _ => false,
        }
    }

    /// Calculate the measure of a word (number of VC patterns).
    fn measure(&self, word: &str) -> usize {
        let mut m = 0;
        let n = word.len();
        let mut i = 0;

        // Skip initial consonants
        while i < n && !self.is_vowel(word, i) {
            i += 1;
        }

        // Count VC patterns
        while i < n {
            // Skip vowels
            while i < n && self.is_vowel(word, i) {
                i += 1;
            }

            if i >= n {
                break;
            }

            m += 1;

            // Skip consonants
            while i < n && !self.is_vowel(word, i) {
                i += 1;
            }
        }

        m
    }

    /// Check if word ends with a specific suffix.
    fn ends_with(&self, word: &str, suffix: &str) -> bool {
        word.len() >= suffix.len() && word[word.len() - suffix.len()..].eq_ignore_ascii_case(suffix)
    }

    /// Replace suffix if conditions are met.
    fn replace_suffix(
        &self,
        word: &str,
        old_suffix: &str,
        new_suffix: &str,
        min_measure: usize,
    ) -> String {
        if self.ends_with(word, old_suffix) {
            let stem = &word[..word.len() - old_suffix.len()];
            if self.measure(stem) >= min_measure {
                return format!("{stem}{new_suffix}");
            }
        }
        word.to_string()
    }

    /// Step 1a of Porter algorithm.
    fn step1a(&self, word: &str) -> String {
        if self.ends_with(word, "sses") {
            format!("{}ss", &word[..word.len() - 4])
        } else if self.ends_with(word, "ies") {
            format!("{}i", &word[..word.len() - 3])
        } else if self.ends_with(word, "ss") {
            word.to_string()
        } else if self.ends_with(word, "s") && word.len() > 1 {
            word[..word.len() - 1].to_string()
        } else {
            word.to_string()
        }
    }

    /// Step 1b of Porter algorithm.
    fn step1b(&self, word: &str) -> String {
        let original_word = word;
        let word = if self.ends_with(word, "eed") {
            self.replace_suffix(word, "eed", "ee", 1)
        } else if self.ends_with(word, "ed") {
            let stem = &word[..word.len() - 2];
            if self.contains_vowel(stem) {
                stem.to_string()
            } else {
                word.to_string()
            }
        } else if self.ends_with(word, "ing") {
            let stem = &word[..word.len() - 3];
            if self.contains_vowel(stem) {
                stem.to_string()
            } else {
                word.to_string()
            }
        } else {
            word.to_string()
        };

        // Post-processing for step 1b
        if word != original_word {
            if self.ends_with(&word, "at")
                || self.ends_with(&word, "bl")
                || self.ends_with(&word, "iz")
            {
                format!("{word}e")
            } else if self.ends_with_double_consonant(&word)
                && !self.ends_with(&word, "l")
                && !self.ends_with(&word, "s")
                && !self.ends_with(&word, "z")
            {
                word[..word.len() - 1].to_string()
            } else if self.measure(&word) == 1 && self.ends_cvc(&word) {
                format!("{word}e")
            } else {
                word
            }
        } else {
            word
        }
    }

    /// Check if word contains a vowel.
    fn contains_vowel(&self, word: &str) -> bool {
        for i in 0..word.len() {
            if self.is_vowel(word, i) {
                return true;
            }
        }
        false
    }

    /// Check if word ends with double consonant.
    fn ends_with_double_consonant(&self, word: &str) -> bool {
        let len = word.len();
        if len < 2 {
            return false;
        }

        let chars: Vec<char> = word.chars().collect();
        chars[len - 1] == chars[len - 2] && !self.is_vowel(word, len - 1)
    }

    /// Check if word ends with consonant-vowel-consonant pattern.
    fn ends_cvc(&self, word: &str) -> bool {
        let len = word.len();
        if len < 3 {
            return false;
        }

        !self.is_vowel(word, len - 3)
            && self.is_vowel(word, len - 2)
            && !self.is_vowel(word, len - 1)
            && !matches!(word.chars().last(), Some('w') | Some('x') | Some('y'))
    }

    /// Step 2 of Porter algorithm.
    fn step2(&self, word: &str) -> String {
        let suffixes = [
            ("ational", "ate"),
            ("tional", "tion"),
            ("enci", "ence"),
            ("anci", "ance"),
            ("izer", "ize"),
            ("abli", "able"),
            ("alli", "al"),
            ("entli", "ent"),
            ("eli", "e"),
            ("ousli", "ous"),
            ("ization", "ize"),
            ("ation", "ate"),
            ("ator", "ate"),
            ("alism", "al"),
            ("iveness", "ive"),
            ("fulness", "ful"),
            ("ousness", "ous"),
            ("aliti", "al"),
            ("iviti", "ive"),
            ("biliti", "ble"),
        ];

        for (old_suffix, new_suffix) in &suffixes {
            if self.ends_with(word, old_suffix) {
                return self.replace_suffix(word, old_suffix, new_suffix, 1);
            }
        }

        word.to_string()
    }

    /// Step 3 of Porter algorithm.
    fn step3(&self, word: &str) -> String {
        let suffixes = [
            ("icate", "ic"),
            ("ative", ""),
            ("alize", "al"),
            ("iciti", "ic"),
            ("ical", "ic"),
            ("ful", ""),
            ("ness", ""),
        ];

        for (old_suffix, new_suffix) in &suffixes {
            if self.ends_with(word, old_suffix) {
                return self.replace_suffix(word, old_suffix, new_suffix, 1);
            }
        }

        word.to_string()
    }

    /// Step 4 of Porter algorithm.
    fn step4(&self, word: &str) -> String {
        let suffixes = [
            "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment", "ent", "ion",
            "ou", "ism", "ate", "iti", "ous", "ive", "ize",
        ];

        for suffix in &suffixes {
            if self.ends_with(word, suffix) {
                let stem = &word[..word.len() - suffix.len()];
                if self.measure(stem) > 1 {
                    // Special case for ion
                    if *suffix != "ion" || self.ends_with(stem, "s") || self.ends_with(stem, "t")
                    {
                        return stem.to_string();
                    }
                }
            }
        }

        word.to_string()
    }

    /// Step 5 of Porter algorithm.
    fn step5(&self, word: &str) -> String {
        let word = if self.ends_with(word, "e") {
            let stem = &word[..word.len() - 1];
            let m = self.measure(stem);
            if m > 1 || (m == 1 && !self.ends_cvc(stem)) {
                stem.to_string()
            } else {
                word.to_string()
            }
        } else {
            word.to_string()
        };

        if self.ends_with(&word, "ll") && self.measure(&word) > 1 {
            word[..word.len() - 1].to_string()
        } else {
            word
        }
    }
}

impl Stemmer for PorterStemmer {
    fn stem(&self, word: &str) -> String {
        if word.len() <= 2 {
            return word.to_lowercase();
        }

        let word = word.to_lowercase();

        // Apply Porter algorithm steps
        let word = self.step1a(&word);
        let word = self.step1b(&word);
        let word = self.step2(&word);
        let word = self.step3(&word);
        let word = self.step4(&word);
        self.step5(&word)
    }

    fn name(&self) -> &'static str {
        "porter"
    }
}

/// Simple stemmer that just removes common suffixes.
#[derive(Debug, Clone, Default)]
pub struct SimpleStemmer {
    /// Common English suffixes to remove.
    suffixes: Vec<String>,
}

impl SimpleStemmer {
    /// Create a new simple stemmer.
    pub fn new() -> Self {
        let suffixes = vec![
            "ing".to_string(),
            "ed".to_string(),
            "er".to_string(),
            "est".to_string(),
            "ly".to_string(),
            "s".to_string(),
            "es".to_string(),
            "ies".to_string(),
            "ied".to_string(),
            "tion".to_string(),
            "sion".to_string(),
            "able".to_string(),
            "ible".to_string(),
            "ment".to_string(),
            "ness".to_string(),
            "ful".to_string(),
        ];

        SimpleStemmer { suffixes }
    }

    /// Create a simple stemmer with custom suffixes.
    pub fn with_suffixes(suffixes: Vec<String>) -> Self {
        SimpleStemmer { suffixes }
    }
}

impl Stemmer for SimpleStemmer {
    fn stem(&self, word: &str) -> String {
        let word = word.to_lowercase();

        if word.len() <= 3 {
            return word;
        }

        // Try to remove suffixes, longest first
        let mut sorted_suffixes = self.suffixes.clone();
        sorted_suffixes.sort_by_key(|b| std::cmp::Reverse(b.len()));

        for suffix in &sorted_suffixes {
            if word.len() > suffix.len() + 2 && word.ends_with(suffix) {
                return word[..word.len() - suffix.len()].to_string();
            }
        }

        word
    }

    fn name(&self) -> &'static str {
        "simple"
    }
}

/// Identity stemmer that returns words unchanged.
#[derive(Debug, Clone, Default)]
pub struct IdentityStemmer;

impl IdentityStemmer {
    pub fn new() -> Self {
        IdentityStemmer
    }
}

impl Stemmer for IdentityStemmer {
    fn stem(&self, word: &str) -> String {
        word.to_string()
    }

    fn name(&self) -> &'static str {
        "identity"
    }
}

/// Filter that applies stemming to tokens.
pub struct StemFilter {
    /// The stemmer to use.
    stemmer: Box<dyn Stemmer>,
}

impl std::fmt::Debug for StemFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StemFilter")
            .field("stemmer", &"<stemmer>")
            .finish()
    }
}

impl StemFilter {
    /// Create a new stem filter with the Porter stemmer.
    pub fn new() -> Self {
        StemFilter {
            stemmer: Box::new(PorterStemmer::new()),
        }
    }

    /// Create a stem filter with a custom stemmer.
    pub fn with_stemmer(stemmer: Box<dyn Stemmer>) -> Self {
        StemFilter { stemmer }
    }

    /// Create a stem filter with the simple stemmer.
    pub fn simple() -> Self {
        StemFilter {
            stemmer: Box::new(SimpleStemmer::new()),
        }
    }
}

impl Default for StemFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl Filter for StemFilter {
    fn filter(&self, tokens: TokenStream) -> Result<TokenStream> {
        let filtered_tokens = tokens
            .map(|token| {
                if token.is_stopped() {
                    token
                } else {
                    let stemmed = self.stemmer.stem(&token.text);
                    token.with_text(stemmed)
                }
            })
            .collect::<Vec<_>>();

        Ok(Box::new(filtered_tokens.into_iter()))
    }

    fn name(&self) -> &'static str {
        "stem"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::token::Token;

    #[test]
    fn test_porter_stemmer() {
        let stemmer = PorterStemmer::new();

        assert_eq!(stemmer.stem("running"), "run");
        assert_eq!(stemmer.stem("flies"), "fli");
        assert_eq!(stemmer.stem("died"), "di");
        assert_eq!(stemmer.stem("agreed"), "agre");
        assert_eq!(stemmer.stem("disabled"), "disabl");
        assert_eq!(stemmer.stem("measuring"), "measur");
        assert_eq!(stemmer.stem("itemization"), "item");
        assert_eq!(stemmer.stem("sensational"), "sensat");
        assert_eq!(stemmer.stem("traditional"), "tradit");
    }

    #[test]
    fn test_simple_stemmer() {
        let stemmer = SimpleStemmer::new();

        assert_eq!(stemmer.stem("running"), "runn");
        assert_eq!(stemmer.stem("flies"), "fli");
        assert_eq!(stemmer.stem("beautiful"), "beauti");
        assert_eq!(stemmer.stem("agreement"), "agree");
        assert_eq!(stemmer.stem("happiness"), "happi");
    }

    #[test]
    fn test_identity_stemmer() {
        let stemmer = IdentityStemmer::new();

        assert_eq!(stemmer.stem("running"), "running");
        assert_eq!(stemmer.stem("flies"), "flies");
        assert_eq!(stemmer.stem("test"), "test");
    }

    #[test]
    fn test_stem_filter() {
        let filter = StemFilter::new();
        let tokens = vec![
            Token::new("running", 0),
            Token::new("flies", 1),
            Token::new("test", 2).stop(),
        ];
        let token_stream = Box::new(tokens.into_iter());

        let result: Vec<Token> = filter.filter(token_stream).unwrap().collect();

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].text, "run");
        assert_eq!(result[1].text, "fli");
        assert_eq!(result[2].text, "test"); // Stopped tokens are not processed
        assert!(result[2].is_stopped());
    }

    #[test]
    fn test_simple_stem_filter() {
        let filter = StemFilter::simple();
        let tokens = vec![Token::new("running", 0), Token::new("beautiful", 1)];
        let token_stream = Box::new(tokens.into_iter());

        let result: Vec<Token> = filter.filter(token_stream).unwrap().collect();

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].text, "runn");
        assert_eq!(result[1].text, "beauti");
    }

    #[test]
    fn test_porter_measure() {
        let stemmer = PorterStemmer::new();

        assert_eq!(stemmer.measure("tree"), 0);
        assert_eq!(stemmer.measure("trees"), 1);
        assert_eq!(stemmer.measure("trouble"), 1);
        assert_eq!(stemmer.measure("troubles"), 2);
    }

    #[test]
    fn test_porter_vowel_detection() {
        let stemmer = PorterStemmer::new();
        let word = "trouble";

        assert!(!stemmer.is_vowel(word, 0)); // t
        assert!(!stemmer.is_vowel(word, 1)); // r
        assert!(stemmer.is_vowel(word, 2)); // o
        assert!(stemmer.is_vowel(word, 3)); // u is vowel
        assert!(!stemmer.is_vowel(word, 4)); // b
        assert!(!stemmer.is_vowel(word, 5)); // l
        assert!(stemmer.is_vowel(word, 6)); // e
    }
}
