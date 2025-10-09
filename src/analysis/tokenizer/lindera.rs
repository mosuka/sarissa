use std::borrow::Cow;
use std::str::FromStr;

use lindera::dictionary::{load_dictionary, load_user_dictionary};
use lindera::mode::Mode;
use lindera::segmenter::Segmenter;

use crate::analysis::token::{Token, TokenStream};
use crate::error::{Result, SarissaError};

use super::Tokenizer;

pub struct LinderaTokenizer {
    // Add any necessary fields for the tokenizer
    inner: Segmenter,
}

impl LinderaTokenizer {
    /// Create a new Lindera tokenizer.
    pub fn new(mode_str: &str, dict_uri: &str, user_dict_uri: Option<&str>) -> Result<Self> {
        let mode = Mode::from_str(mode_str)
            .map_err(|e| SarissaError::analysis(format!("Invalid mode '{}': {}", mode_str, e)))?;
        let dict = load_dictionary(dict_uri)
            .map_err(|e| SarissaError::analysis(format!("Failed to load dictionary: {}", e)))?;
        let metadata = &dict.metadata;
        let user_dict = match user_dict_uri {
            Some(uri) => Some(load_user_dictionary(&uri, metadata).map_err(|e| {
                SarissaError::analysis(format!("Failed to load user dictionary: {}", e))
            })?),
            None => None,
        };
        let inner = Segmenter::new(mode, dict, user_dict);

        Ok(Self { inner })
    }
}

impl Tokenizer for LinderaTokenizer {
    fn tokenize(&self, text: &str) -> Result<TokenStream> {
        let mut tokens = Vec::new();

        for token in self
            .inner
            .segment(Cow::Borrowed(text))
            .map_err(|e| SarissaError::analysis(format!("Failed to segment text: {}", e)))?
        {
            tokens.push(Token::with_offsets(
                token.surface,
                token.position,
                token.byte_start,
                token.byte_end,
            ));
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
