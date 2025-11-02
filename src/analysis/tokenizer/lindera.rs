//! Lindera-based morphological tokenizer.
//!
//! This module provides a tokenizer using the Lindera library for
//! morphological analysis of CJK (Chinese, Japanese, Korean) languages.
//! Lindera performs dictionary-based word segmentation, which is essential
//! for languages that don't use spaces to separate words.
//!
//! # Supported Languages
//!
//! - **Japanese**: Using UniDic dictionary (`embedded://unidic`)
//! - **Korean**: Using KO-DIC dictionary (`embedded://ko-dic`)
//! - **Chinese**: Using CC-CEDICT dictionary (`embedded://cc-cedict`)
//!
//! # Examples
//!
//! ```
//! use yatagarasu::analysis::tokenizer::lindera::LinderaTokenizer;
//! use yatagarasu::analysis::tokenizer::Tokenizer;
//!
//! // Japanese tokenization
//! let tokenizer = LinderaTokenizer::new("normal", "embedded://unidic", None).unwrap();
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
use crate::error::{Result, YatagarasuError};

/// A tokenizer that uses Lindera for morphological analysis.
///
/// This tokenizer performs dictionary-based word segmentation for CJK languages,
/// breaking text into meaningful morphemes (words, particles, suffixes, etc.).
/// It supports multiple dictionaries and segmentation modes.
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
/// use yatagarasu::analysis::tokenizer::lindera::LinderaTokenizer;
/// use yatagarasu::analysis::tokenizer::Tokenizer;
///
/// // Japanese with UniDic
/// let tokenizer = LinderaTokenizer::new("normal", "embedded://unidic", None).unwrap();
/// let tokens: Vec<_> = tokenizer.tokenize("形態素解析").unwrap().collect();
///
/// // Korean with KO-DIC
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
    /// * `dict_uri` - Dictionary URI (e.g., "embedded://unidic", "embedded://ko-dic")
    /// * `user_dict_uri` - Optional user dictionary URI for custom words
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
    /// use yatagarasu::analysis::tokenizer::lindera::LinderaTokenizer;
    ///
    /// // Japanese tokenizer
    /// let tokenizer = LinderaTokenizer::new(
    ///     "normal",
    ///     "embedded://unidic",
    ///     None
    /// ).unwrap();
    ///
    /// // With user dictionary
    /// // let tokenizer = LinderaTokenizer::new(
    /// //     "normal",
    /// //     "embedded://unidic",
    /// //     Some("/path/to/user_dict.csv")
    /// // ).unwrap();
    /// ```
    pub fn new(mode_str: &str, dict_uri: &str, user_dict_uri: Option<&str>) -> Result<Self> {
        let mode = Mode::from_str(mode_str).map_err(|e| {
            YatagarasuError::analysis(format!("Invalid mode '{}': {}", mode_str, e))
        })?;
        let dict = load_dictionary(dict_uri)
            .map_err(|e| YatagarasuError::analysis(format!("Failed to load dictionary: {}", e)))?;
        let metadata = &dict.metadata;
        let user_dict = match user_dict_uri {
            Some(uri) => Some(load_user_dictionary(uri, metadata).map_err(|e| {
                YatagarasuError::analysis(format!("Failed to load user dictionary: {}", e))
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
            .map_err(|e| YatagarasuError::analysis(format!("Failed to segment text: {}", e)))?
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
        let tokenizer = LinderaTokenizer::new("normal", "embedded://unidic", None).unwrap();

        let text = "日本語の形態素解析を行うことができます。";

        let tokens: Vec<Token> = tokenizer.tokenize(text).unwrap().collect();

        assert_eq!(tokens.len(), 13);
        assert_eq!(tokens[0].text, "日本");
        assert_eq!(tokens[1].text, "語");
        assert_eq!(tokens[2].text, "の");
        assert_eq!(tokens[3].text, "形態");
        assert_eq!(tokens[4].text, "素");
        assert_eq!(tokens[5].text, "解析");
        assert_eq!(tokens[6].text, "を");
        assert_eq!(tokens[7].text, "行う");
        assert_eq!(tokens[8].text, "こと");
        assert_eq!(tokens[9].text, "が");
        assert_eq!(tokens[10].text, "でき");
        assert_eq!(tokens[11].text, "ます");
        assert_eq!(tokens[12].text, "。");
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

        assert_eq!(tokens.len(), 8);
        assert_eq!(tokens[0].text, "能够");
        assert_eq!(tokens[1].text, "进行");
        assert_eq!(tokens[2].text, "汉语");
        assert_eq!(tokens[3].text, "的");
        assert_eq!(tokens[4].text, "形态");
        assert_eq!(tokens[5].text, "素");
        assert_eq!(tokens[6].text, "解析");
        assert_eq!(tokens[7].text, "。");
    }

    #[test]
    fn test_tokenizer_name() {
        let tokenizer = LinderaTokenizer::new("normal", "embedded://unidic", None).unwrap();

        assert_eq!(tokenizer.name(), "lindera");
    }
}
