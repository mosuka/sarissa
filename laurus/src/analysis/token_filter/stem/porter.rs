//! Porter stemming algorithm implementation.
//!
//! This module provides an implementation of the Porter stemming algorithm,
//! a widely-used algorithm for reducing English words to their stems.
//!
//! # Algorithm
//!
//! The Porter stemmer applies a series of rewrite rules in five steps:
//! 1. Plurals and -ed/-ing suffixes
//! 2. -ational → -ate, -tional → -tion, etc.
//! 3. -icate → -ic, -ative → "", etc.
//! 4. Remove -al, -ance, -ence, etc.
//! 5. Remove final -e and -ll
//!
//! # Examples
//!
//! ```
//! use laurus::analysis::token_filter::stem::Stemmer;
//! use laurus::analysis::token_filter::stem::porter::PorterStemmer;
//!
//! let stemmer = PorterStemmer::new();
//!
//! assert_eq!(stemmer.stem("running"), "run");
//! assert_eq!(stemmer.stem("flies"), "fli");
//! assert_eq!(stemmer.stem("traditional"), "tradit");
//! ```

use std::collections::HashMap;

use crate::analysis::token_filter::stem::Stemmer;

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

    /// Check whether the byte at `pos` is an ASCII vowel.
    ///
    /// `pos` is a **byte** offset: every caller derives it from
    /// `word.len()`, which is a byte length. The previous implementation
    /// mixed the two — it guarded on `pos >= word.len()` (bytes) then
    /// indexed a `Vec<char>` with `pos` (chars), so any word holding a
    /// multi-byte character panicked with an out-of-bounds index
    /// (e.g. `stem("naïve")`). The Porter algorithm is English-only, so
    /// treating every non-ASCII byte as a consonant is both faithful to
    /// the algorithm and panic-free.
    #[allow(clippy::only_used_in_recursion)]
    fn is_vowel(&self, word: &str, pos: usize) -> bool {
        let bytes = word.as_bytes();
        if pos >= bytes.len() {
            return false;
        }

        match bytes[pos].to_ascii_lowercase() {
            b'a' | b'e' | b'i' | b'o' | b'u' => true,
            b'y' if pos > 0 => !self.is_vowel(word, pos - 1),
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

    /// Check if word ends with `suffix` (ASCII, case-insensitive).
    ///
    /// Compares raw bytes: `&word[word.len() - suffix.len()..]` panics
    /// when the split point lands inside a multi-byte character. Every
    /// Porter suffix is ASCII, so a byte-tail comparison is exactly
    /// equivalent for ASCII input and simply yields `false` for a
    /// multi-byte tail — every suffix passed to `replace_suffix` below is
    /// ASCII, so once this returns true, `word.len() - suffix.len()` is
    /// guaranteed to land on a char boundary and the slice there is safe.
    fn ends_with(&self, word: &str, suffix: &str) -> bool {
        let w = word.as_bytes();
        let s = suffix.as_bytes();
        w.len() >= s.len() && w[w.len() - s.len()..].eq_ignore_ascii_case(s)
    }

    /// Replace suffix if conditions are met.
    ///
    /// Every suffix this module strips is ASCII (see `ends_with`'s doc),
    /// so `&word[..word.len() - old_suffix.len()]` below always lands on
    /// a char boundary once `ends_with` has matched.
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

    /// Check if word ends with a doubled ASCII consonant.
    ///
    /// The ASCII check is load-bearing, not cosmetic: callers strip the
    /// final **byte** when this returns true (see `step1b`), which is
    /// only a valid char boundary for a single-byte character. Without
    /// it, two identical UTF-8 continuation bytes (e.g. the tail of a
    /// 4-byte character) would satisfy a naive byte-equality test and
    /// make the caller slice mid-character.
    fn ends_with_double_consonant(&self, word: &str) -> bool {
        let bytes = word.as_bytes();
        let len = bytes.len();
        if len < 2 {
            return false;
        }

        bytes[len - 1].is_ascii_alphabetic()
            && bytes[len - 1] == bytes[len - 2]
            && !self.is_vowel(word, len - 1)
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
                    if *suffix != "ion" || self.ends_with(stem, "s") || self.ends_with(stem, "t") {
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

#[cfg(test)]
mod tests {
    use super::*;

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

    /// `is_vowel` used to guard on a byte length but index a `Vec<char>`
    /// with the same value, so any word holding a multi-byte character
    /// panicked with an out-of-bounds index. `naïve` (`ï` is 2 bytes)
    /// reaches `is_vowel(word, 4)` via `step5`'s "e" trim — this used to
    /// panic.
    #[test]
    fn stem_does_not_panic_on_latin_accented_words() {
        let stemmer = PorterStemmer::new();
        // No panic is the assertion; the exact stem isn't the point since
        // Porter is English-only.
        let _ = stemmer.stem("naïve");
        let _ = stemmer.stem("café");
        let _ = stemmer.stem("Zürich");
    }

    /// Japanese input holds no ASCII vowels at all, so every byte is
    /// treated as a consonant. The word is returned unchanged (measure
    /// stays 0 / no known suffix matches) rather than panicking.
    #[test]
    fn stem_does_not_panic_on_japanese_input() {
        let stemmer = PorterStemmer::new();
        let _ = stemmer.stem("日本語");
        let _ = stemmer.stem("形態素解析");
    }

    /// `ends_with_double_consonant` used to compare `chars()` for
    /// equality without an ASCII guard: two identical UTF-8 continuation
    /// bytes from a 4-byte character (e.g. an astral-plane codepoint or
    /// emoji) could satisfy that check and make `step1b` slice off a
    /// single byte, landing mid-character.
    #[test]
    fn stem_does_not_panic_on_astral_plane_input() {
        let stemmer = PorterStemmer::new();
        let _ = stemmer.stem("a𠀀ing");
        let _ = stemmer.stem("🍎🍎🍎");
    }

    /// `ends_with` now compares raw ASCII bytes; a multi-byte-suffixed
    /// word should simply fail to match rather than panic, and true
    /// ASCII matches must still work.
    #[test]
    fn ends_with_compares_ascii_suffixes_by_bytes() {
        let stemmer = PorterStemmer::new();
        assert!(stemmer.ends_with("日本語s", "s"));
        assert!(!stemmer.ends_with("日本語", "s"));
    }
}
