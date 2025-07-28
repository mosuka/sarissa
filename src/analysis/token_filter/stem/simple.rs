//! Simple stemmer implementation.

use super::Stemmer;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_stemmer() {
        let stemmer = SimpleStemmer::new();

        assert_eq!(stemmer.stem("running"), "runn");
        assert_eq!(stemmer.stem("flies"), "fli");
        assert_eq!(stemmer.stem("beautiful"), "beauti");
        assert_eq!(stemmer.stem("agreement"), "agree");
        assert_eq!(stemmer.stem("happiness"), "happi");
    }
}
