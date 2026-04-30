//! Lindera-based morphological tokenizer.
//!
//! This module provides a tokenizer using the Lindera library for
//! morphological analysis of CJK (Chinese, Japanese, Korean) languages.
//! Lindera performs dictionary-based word segmentation, which is essential
//! for languages that don't use spaces to separate words.
//!
//! # Dictionary loading
//!
//! Pass a filesystem path to a Lindera dictionary directory as the
//! `dict_uri` argument:
//!
//! - Japanese: an IPADIC build (e.g. `/var/lib/lindera/ipadic`)
//! - Korean: a ko-dic build (e.g. `/var/lib/lindera/ko-dic`)
//! - Chinese: a cc-cedict build (e.g. `/var/lib/lindera/cc-cedict`)
//!
//! `laurus` no longer enables Lindera's `embed-*` features by default,
//! so `embedded://*` URIs are not resolvable at runtime. The
//! laurus test suite continues to use `embedded://*` URIs because the
//! features are activated for the test build via `[dev-dependencies]`.
//!
//! # Examples
//!
//! ```
//! use laurus::analysis::tokenizer::lindera::LinderaTokenizer;
//! use laurus::analysis::tokenizer::Tokenizer;
//!
//! // In tests, the embedded ipadic is available; in production, supply a path.
//! let tokenizer = LinderaTokenizer::new("normal", "embedded://ipadic", None).unwrap();
//! let tokens: Vec<_> = tokenizer.tokenize("日本語の解析").unwrap().collect();
//!
//! // Tokens: ["日本", "語", "の", "解析"]
//! assert!(tokens.len() > 0);
//! ```

use std::borrow::Cow;
use std::str::FromStr;

use lindera::dictionary::{load_dictionary, load_user_dictionary};
use lindera::mode::Mode;
use lindera::segmenter::Segmenter;

use crate::analysis::token::{Token, TokenStream, TokenType};
use crate::analysis::tokenizer::Tokenizer;
use crate::error::{LaurusError, Result};

/// A tokenizer that uses Lindera for morphological analysis.
///
/// This tokenizer performs dictionary-based word segmentation for CJK languages,
/// breaking text into meaningful morphemes (words, particles, suffixes, etc.).
/// Pass a filesystem path to a Lindera dictionary directory at construction
/// time.
///
/// # Segmentation Modes
///
/// - `"normal"`: Standard segmentation
/// - `"search"`: Optimized for search (generates more tokens)
/// - `"decompose"`: Decomposes compound words
///
/// # Examples
///
/// ```
/// use laurus::analysis::tokenizer::lindera::LinderaTokenizer;
/// use laurus::analysis::tokenizer::Tokenizer;
///
/// // Japanese with IPADIC at runtime: pass a path like
/// // "/var/lib/lindera/ipadic". In tests we use the embedded form.
/// let tokenizer = LinderaTokenizer::new("normal", "embedded://ipadic", None).unwrap();
/// let tokens: Vec<_> = tokenizer.tokenize("形態素解析").unwrap().collect();
///
/// // Korean with ko-dic
/// let tokenizer = LinderaTokenizer::new("normal", "embedded://ko-dic", None).unwrap();
/// let tokens: Vec<_> = tokenizer.tokenize("한국어").unwrap().collect();
/// ```
pub struct LinderaTokenizer {
    // Add any necessary fields for the tokenizer
    inner: Segmenter,
}

impl LinderaTokenizer {
    /// Create a new Lindera tokenizer.
    ///
    /// # Arguments
    ///
    /// * `mode_str` - Segmentation mode: "normal", "search", or "decompose"
    /// * `dict_uri` - Lindera dictionary URI. In production, supply a
    ///   filesystem path to a dictionary directory (e.g.,
    ///   `"/var/lib/lindera/ipadic"`). `embedded://*` URIs only resolve
    ///   when the matching `embed-*` Lindera feature is enabled, which
    ///   `laurus` does not enable by default.
    /// * `user_dict_uri` - Optional user dictionary path for custom words
    ///
    /// # Returns
    ///
    /// A new `LinderaTokenizer` instance
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The mode string is invalid
    /// - The dictionary cannot be loaded
    /// - The user dictionary cannot be loaded
    ///
    /// # Examples
    ///
    /// ```
    /// use laurus::analysis::tokenizer::lindera::LinderaTokenizer;
    ///
    /// // Japanese tokenizer (test-only embedded URI; in production use a path).
    /// let tokenizer = LinderaTokenizer::new(
    ///     "normal",
    ///     "embedded://ipadic",
    ///     None
    /// ).unwrap();
    ///
    /// // With user dictionary
    /// // let tokenizer = LinderaTokenizer::new(
    /// //     "normal",
    /// //     "/var/lib/lindera/ipadic",
    /// //     Some("/etc/laurus/user_dict.csv")
    /// // ).unwrap();
    /// ```
    pub fn new(mode_str: &str, dict_uri: &str, user_dict_uri: Option<&str>) -> Result<Self> {
        let mode = Mode::from_str(mode_str)
            .map_err(|e| LaurusError::analysis(format!("Invalid mode '{}': {}", mode_str, e)))?;
        let dict = load_dictionary(dict_uri)
            .map_err(|e| LaurusError::analysis(format!("Failed to load dictionary: {}", e)))?;
        let metadata = &dict.metadata;
        let user_dict = match user_dict_uri {
            Some(uri) => Some(load_user_dictionary(uri, metadata).map_err(|e| {
                LaurusError::analysis(format!("Failed to load user dictionary: {}", e))
            })?),
            None => None,
        };
        let inner = Segmenter::new(mode, dict, user_dict);

        Ok(Self { inner })
    }

    /// Detect token type based on character content.
    ///
    /// Analyzes the token text to determine its type:
    /// - All numeric → Num
    /// - All Hiragana → Hiragana
    /// - All Katakana → Katakana
    /// - Contains Hangul → Hangul
    /// - Contains CJK → Cjk
    /// - ASCII alphanumeric → Alphanum
    /// - All punctuation → Punctuation
    /// - Otherwise → Other
    fn detect_token_type(text: &str) -> TokenType {
        if text.is_empty() {
            return TokenType::Other;
        }

        // Check if all characters are numeric
        if text.chars().all(|c| c.is_numeric()) {
            return TokenType::Num;
        }

        // Check if it's Hiragana
        if text.chars().all(|c| matches!(c, '\u{3040}'..='\u{309F}')) {
            return TokenType::Hiragana;
        }

        // Check if it's Katakana
        if text.chars().all(|c| matches!(c, '\u{30A0}'..='\u{30FF}')) {
            return TokenType::Katakana;
        }

        // Check if it's Hangul
        if text
            .chars()
            .any(|c| matches!(c, '\u{AC00}'..='\u{D7AF}' | '\u{1100}'..='\u{11FF}'))
        {
            return TokenType::Hangul;
        }

        // Check if it contains CJK characters
        if text.chars().any(|c| {
            matches!(c,
                '\u{4E00}'..='\u{9FFF}' |  // CJK Unified Ideographs
                '\u{3400}'..='\u{4DBF}' |  // CJK Extension A
                '\u{20000}'..='\u{2A6DF}' | // CJK Extension B
                '\u{2A700}'..='\u{2B73F}' | // CJK Extension C
                '\u{2B740}'..='\u{2B81F}' | // CJK Extension D
                '\u{2B820}'..='\u{2CEAF}'   // CJK Extension E
            )
        }) {
            return TokenType::Cjk;
        }

        // Check if it's alphanumeric (ASCII)
        if text
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        {
            return TokenType::Alphanum;
        }

        // Check if it's punctuation
        if text.chars().all(|c| c.is_ascii_punctuation()) {
            return TokenType::Punctuation;
        }

        TokenType::Other
    }
}

impl Tokenizer for LinderaTokenizer {
    fn tokenize(&self, text: &str) -> Result<TokenStream> {
        let mut tokens = Vec::new();

        for token in self
            .inner
            .segment(Cow::Borrowed(text))
            .map_err(|e| LaurusError::analysis(format!("Failed to segment text: {}", e)))?
        {
            let token_type = Self::detect_token_type(&token.surface);
            tokens.push(
                Token::with_offsets(
                    token.surface,
                    token.position,
                    token.byte_start,
                    token.byte_end,
                )
                .with_token_type(token_type),
            );
        }

        Ok(Box::new(tokens.into_iter()))
    }

    fn name(&self) -> &'static str {
        "lindera"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_japanese() {
        let tokenizer = LinderaTokenizer::new("normal", "embedded://ipadic", None).unwrap();

        let text = "日本語の形態素解析を行うことができます。";

        let tokens: Vec<Token> = tokenizer.tokenize(text).unwrap().collect();

        assert_eq!(tokens.len(), 11);
        assert_eq!(tokens[0].text, "日本語");
        assert_eq!(tokens[1].text, "の");
        assert_eq!(tokens[2].text, "形態素");
        assert_eq!(tokens[3].text, "解析");
        assert_eq!(tokens[4].text, "を");
        assert_eq!(tokens[5].text, "行う");
        assert_eq!(tokens[6].text, "こと");
        assert_eq!(tokens[7].text, "が");
        assert_eq!(tokens[8].text, "でき");
        assert_eq!(tokens[9].text, "ます");
        assert_eq!(tokens[10].text, "。");
    }

    #[test]
    fn test_tokenize_korean() {
        let tokenizer = LinderaTokenizer::new("normal", "embedded://ko-dic", None).unwrap();

        let text = "한국어의형태해석을실시할수있습니다.";

        let tokens: Vec<Token> = tokenizer.tokenize(text).unwrap().collect();

        assert_eq!(tokens.len(), 11);
        assert_eq!(tokens[0].text, "한국어");
        assert_eq!(tokens[1].text, "의");
        assert_eq!(tokens[2].text, "형태");
        assert_eq!(tokens[3].text, "해석");
        assert_eq!(tokens[4].text, "을");
        assert_eq!(tokens[5].text, "실시");
        assert_eq!(tokens[6].text, "할");
        assert_eq!(tokens[7].text, "수");
        assert_eq!(tokens[8].text, "있");
        assert_eq!(tokens[9].text, "습니다");
        assert_eq!(tokens[10].text, ".");
    }

    #[test]
    fn test_tokenize_chinese() {
        let tokenizer = LinderaTokenizer::new("normal", "embedded://cc-cedict", None).unwrap();

        let text = "能够进行汉语的形态素解析。";

        let tokens: Vec<Token> = tokenizer.tokenize(text).unwrap().collect();

        // Jieba tokenizes Chinese text differently from CC-CEDICT.
        assert!(!tokens.is_empty());
        // Verify that key Chinese words are present in the output.
        let texts: Vec<&str> = tokens.iter().map(|t| t.text.as_str()).collect();
        assert!(texts.contains(&"能够"));
        assert!(texts.contains(&"进行"));
        assert!(texts.contains(&"汉语"));
        assert!(texts.contains(&"解析"));
    }

    #[test]
    fn test_tokenizer_name() {
        let tokenizer = LinderaTokenizer::new("normal", "embedded://ipadic", None).unwrap();

        assert_eq!(tokenizer.name(), "lindera");
    }
}
