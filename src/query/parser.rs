//! Query parser for converting string queries to structured query objects.
//!
//! Similar to Lucene's QueryParser, this module parses query strings and
//! uses analyzers to tokenize and normalize terms.

use std::iter::Peekable;
use std::str::Chars;
use std::sync::Arc;

use crate::analysis::{Analyzer, StandardAnalyzer};
use crate::error::{Result, SarissaError};
use crate::query::{BooleanQuery, BooleanQueryBuilder, Occur, Query, TermQuery};

/// A query parser that supports basic query syntax with analyzer support.
///
/// Like Lucene's QueryParser, terms are analyzed using the configured analyzer
/// before creating TermQuery objects. This ensures proper tokenization, lowercasing,
/// and stop word removal.
///
/// Schema-less: no field validation, accepts any field name.
pub struct QueryParser {
    /// Default field to search in when no field is specified.
    default_field: Option<String>,

    /// Analyzer to use for parsing query terms.
    analyzer: Option<Arc<dyn Analyzer>>,
}

impl std::fmt::Debug for QueryParser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueryParser")
            .field("default_field", &self.default_field)
            .field("analyzer", &self.analyzer.as_ref().map(|_| "<Analyzer>"))
            .finish()
    }
}

impl Default for QueryParser {
    fn default() -> Self {
        Self::new()
    }
}

impl QueryParser {
    /// Create a new query parser without an analyzer (schema-less).
    ///
    /// Without an analyzer, terms are used as-is (case-sensitive, no tokenization).
    pub fn new() -> Self {
        QueryParser {
            default_field: None,
            analyzer: None,
        }
    }

    /// Create a query parser with the standard analyzer.
    ///
    /// This is the Lucene-style approach: terms are analyzed before matching.
    pub fn with_standard_analyzer() -> Result<Self> {
        Ok(QueryParser {
            default_field: None,
            analyzer: Some(Arc::new(StandardAnalyzer::new()?)),
        })
    }

    /// Create a query parser with a custom analyzer.
    pub fn with_analyzer(analyzer: Arc<dyn Analyzer>) -> Self {
        QueryParser {
            default_field: None,
            analyzer: Some(analyzer),
        }
    }

    /// Set the default field to search in when no field is specified.
    pub fn with_default_field<S: Into<String>>(mut self, field: S) -> Self {
        self.default_field = Some(field.into());
        self
    }

    /// Parse a query string into a Query object.
    ///
    /// Supported syntax:
    /// - Simple terms: `hello`
    /// - Field-specific terms: `title:hello`
    /// - Phrases: `"hello world"`
    /// - Boolean operators: `+required -forbidden optional`
    /// - Parentheses: `(title:hello OR body:world)`
    /// - AND/OR operators: `title:hello AND body:world`
    pub fn parse(&self, query_str: &str) -> Result<Box<dyn Query>> {
        let trimmed = query_str.trim();
        if trimmed.is_empty() {
            return Ok(Box::new(BooleanQuery::new()));
        }

        let mut parser = QueryStringParser::new(trimmed, &self.default_field, &self.analyzer);
        parser.parse()
    }

    /// Parse a query string for a specific field.
    pub fn parse_field(&self, field: &str, query_str: &str) -> Result<Box<dyn Query>> {
        let trimmed = query_str.trim();
        if trimmed.is_empty() {
            return Ok(Box::new(BooleanQuery::new()));
        }

        // Schema-less: no field validation
        let field_name = Some(field.to_string());
        let mut parser = QueryStringParser::new(trimmed, &field_name, &self.analyzer);
        parser.parse()
    }

    /// Get the default field.
    pub fn default_field(&self) -> Option<&str> {
        self.default_field.as_deref()
    }
}

/// Internal parser for parsing query strings.
struct QueryStringParser<'a> {
    chars: Peekable<Chars<'a>>,
    default_field: &'a Option<String>,
    analyzer: &'a Option<Arc<dyn Analyzer>>,
}

impl<'a> QueryStringParser<'a> {
    fn new(query_str: &'a str, default_field: &'a Option<String>, analyzer: &'a Option<Arc<dyn Analyzer>>) -> Self {
        QueryStringParser {
            chars: query_str.chars().peekable(),
            default_field,
            analyzer,
        }
    }

    /// Analyze a term using the configured analyzer.
    /// Returns the analyzed tokens or the original term if no analyzer is set.
    fn analyze_term(&self, term: &str) -> Result<Vec<String>> {
        if let Some(analyzer) = self.analyzer {
            let token_stream = analyzer.analyze(term)?;
            let tokens: Vec<String> = token_stream.map(|t| t.text).collect();
            Ok(tokens)
        } else {
            // No analyzer - use term as-is
            Ok(vec![term.to_string()])
        }
    }

    fn parse(&mut self) -> Result<Box<dyn Query>> {
        self.parse_or_expression()
    }

    fn parse_or_expression(&mut self) -> Result<Box<dyn Query>> {
        let mut left = self.parse_and_expression()?;

        while self.peek_word() == Some("OR") {
            self.consume_word("OR");
            let right = self.parse_and_expression()?;

            // Convert to boolean query
            let boolean_query = BooleanQueryBuilder::new()
                .should(left)
                .should(right)
                .build();

            left = Box::new(boolean_query);
        }

        Ok(left)
    }

    fn parse_and_expression(&mut self) -> Result<Box<dyn Query>> {
        let mut left = self.parse_term()?;

        while self.peek_word() == Some("AND") || self.should_continue_and() {
            if self.peek_word() == Some("AND") {
                self.consume_word("AND");
            }

            let right = self.parse_term()?;

            // Convert to boolean query
            let boolean_query = BooleanQueryBuilder::new().must(left).must(right).build();

            left = Box::new(boolean_query);
        }

        Ok(left)
    }

    fn parse_term(&mut self) -> Result<Box<dyn Query>> {
        self.skip_whitespace();

        if self.chars.peek().is_none() {
            return Ok(Box::new(BooleanQuery::new()));
        }

        // Check for prefix operators
        let occur = match self.chars.peek() {
            Some('+') => {
                self.chars.next();
                Occur::Must
            }
            Some('-') => {
                self.chars.next();
                Occur::MustNot
            }
            _ => Occur::Should,
        };

        self.skip_whitespace();

        // Check for parentheses
        if self.chars.peek() == Some(&'(') {
            self.chars.next();
            let inner = self.parse_or_expression()?;
            self.skip_whitespace();
            if self.chars.peek() == Some(&')') {
                self.chars.next();
            }
            return Ok(inner);
        }

        // Check for quoted phrase
        if self.chars.peek() == Some(&'"') {
            return self.parse_phrase();
        }

        // Parse field:term or just term
        let word = self.consume_word_or_quoted()?;

        // Check if it's a field:term pattern
        if word.contains(':') {
            let parts: Vec<&str> = word.splitn(2, ':').collect();
            if parts.len() == 2 {
                let field = parts[0];
                let term = parts[1];

                // Analyze the term
                return self.create_query_for_term(field, term, occur);
            }
        }

        // Use default field if available
        if let Some(default_field) = self.default_field {
            self.create_query_for_term(default_field, &word, occur)
        } else {
            Err(SarissaError::schema(
                "No default field specified and no field prefix found",
            ))
        }
    }

    /// Create a query for a term, applying analysis if configured.
    fn create_query_for_term(&self, field: &str, term: &str, occur: Occur) -> Result<Box<dyn Query>> {
        let analyzed_terms = self.analyze_term(term)?;

        if analyzed_terms.is_empty() {
            // Term was completely filtered out (e.g., stop word)
            return Ok(Box::new(BooleanQuery::new()));
        }

        if analyzed_terms.len() == 1 {
            // Single term - create a TermQuery
            let query = Box::new(TermQuery::new(field, &analyzed_terms[0]));
            Ok(self.wrap_with_occur(query, occur))
        } else {
            // Multiple tokens (e.g., "don't" -> ["don", "t"])
            // Create a BooleanQuery with SHOULD clauses
            let mut builder = BooleanQueryBuilder::new();
            for analyzed_term in analyzed_terms {
                let term_query = Box::new(TermQuery::new(field, &analyzed_term));
                builder = builder.should(term_query);
            }
            Ok(self.wrap_with_occur(Box::new(builder.build()), occur))
        }
    }

    fn parse_phrase(&mut self) -> Result<Box<dyn Query>> {
        // Consume opening quote
        self.chars.next();

        let mut phrase = String::new();
        while let Some(ch) = self.chars.peek() {
            if *ch == '"' {
                self.chars.next();
                break;
            }
            phrase.push(self.chars.next().unwrap());
        }

        // For now, treat phrases as simple terms
        // In a full implementation, we would create a PhraseQuery
        if let Some(default_field) = self.default_field {
            Ok(Box::new(TermQuery::new(default_field, phrase)))
        } else {
            Err(SarissaError::schema(
                "No default field specified for phrase query",
            ))
        }
    }

    fn wrap_with_occur(&self, query: Box<dyn Query>, occur: Occur) -> Box<dyn Query> {
        match occur {
            Occur::Should => query,
            Occur::Must | Occur::MustNot => {
                let boolean_query = match occur {
                    Occur::Must => BooleanQueryBuilder::new().must(query).build(),
                    Occur::MustNot => BooleanQueryBuilder::new().must_not(query).build(),
                    _ => unreachable!(),
                };
                Box::new(boolean_query)
            }
        }
    }

    fn consume_word_or_quoted(&mut self) -> Result<String> {
        let mut word = String::new();

        while let Some(ch) = self.chars.peek() {
            if ch.is_whitespace() || *ch == ')' {
                break;
            }
            word.push(self.chars.next().unwrap());
        }

        if word.is_empty() {
            Err(SarissaError::schema("Expected word but found end of input"))
        } else {
            Ok(word)
        }
    }

    fn consume_word(&mut self, expected: &str) {
        for _ in 0..expected.len() {
            self.chars.next();
        }
        self.skip_whitespace();
    }

    fn peek_word(&mut self) -> Option<&'static str> {
        self.skip_whitespace();

        // Save current position
        let remaining: String = self.chars.clone().collect();

        if remaining.starts_with("AND")
            && (remaining.len() == 3 || remaining.chars().nth(3).unwrap().is_whitespace())
        {
            Some("AND")
        } else if remaining.starts_with("OR")
            && (remaining.len() == 2 || remaining.chars().nth(2).unwrap().is_whitespace())
        {
            Some("OR")
        } else {
            None
        }
    }

    fn should_continue_and(&mut self) -> bool {
        self.skip_whitespace();

        // Check if there's more content that could be part of an AND expression
        if let Some(ch) = self.chars.peek() {
            *ch != ')' && self.peek_word().is_none()
        } else {
            false
        }
    }

    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.chars.peek() {
            if ch.is_whitespace() {
                self.chars.next();
            } else {
                break;
            }
        }
    }
}

/// Builder for creating query parsers.
pub struct QueryParserBuilder {
    default_field: Option<String>,
    analyzer: Option<Arc<dyn Analyzer>>,
}

impl std::fmt::Debug for QueryParserBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueryParserBuilder")
            .field("default_field", &self.default_field)
            .field("analyzer", &self.analyzer.as_ref().map(|_| "<Analyzer>"))
            .finish()
    }
}

impl QueryParserBuilder {
    /// Create a new query parser builder (schema-less).
    pub fn new() -> Self {
        QueryParserBuilder {
            default_field: None,
            analyzer: None,
        }
    }

    /// Set the default field.
    pub fn default_field<S: Into<String>>(mut self, field: S) -> Self {
        self.default_field = Some(field.into());
        self
    }

    /// Set the analyzer.
    pub fn analyzer(mut self, analyzer: Arc<dyn Analyzer>) -> Self {
        self.analyzer = Some(analyzer);
        self
    }

    /// Use the standard analyzer.
    pub fn with_standard_analyzer(mut self) -> Result<Self> {
        self.analyzer = Some(Arc::new(StandardAnalyzer::new()?));
        Ok(self)
    }

    /// Build the query parser.
    pub fn build(self) -> QueryParser {
        QueryParser {
            default_field: self.default_field,
            analyzer: self.analyzer,
        }
    }
}

impl Default for QueryParserBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(dead_code)]
    #[test]
    fn test_query_parser_creation() {
        let parser = QueryParser::new();

        assert!(parser.default_field().is_none());
    }

    #[test]
    fn test_query_parser_with_default_field() {
        let parser = QueryParser::new().with_default_field("title");

        assert_eq!(parser.default_field(), Some("title"));
    }

    #[test]
    fn test_parse_simple_term() {
        let parser = QueryParser::new().with_default_field("title");

        let query = parser.parse("hello").unwrap();
        assert_eq!(query.description(), "title:hello");
    }

    #[test]
    fn test_parse_field_term() {
        let parser = QueryParser::new();

        let query = parser.parse("title:hello").unwrap();
        assert_eq!(query.description(), "title:hello");
    }

    #[test]
    fn test_parse_boolean_and() {
        let parser = QueryParser::new();

        let query = parser.parse("title:hello AND body:world").unwrap();
        let desc = query.description();
        assert!(desc.contains("title:hello"));
        assert!(desc.contains("body:world"));
    }

    #[test]
    fn test_parse_boolean_or() {
        let parser = QueryParser::new();

        let query = parser.parse("title:hello OR title:world").unwrap();
        let desc = query.description();
        assert!(desc.contains("title:hello"));
        assert!(desc.contains("title:world"));
    }

    #[test]
    fn test_parse_required_term() {
        let parser = QueryParser::new();

        let query = parser.parse("+title:hello").unwrap();
        let desc = query.description();
        assert!(desc.contains("+title:hello"));
    }

    #[test]
    fn test_parse_forbidden_term() {
        let parser = QueryParser::new();

        let query = parser.parse("-title:spam").unwrap();
        let desc = query.description();
        assert!(desc.contains("-title:spam"));
    }

    #[test]
    fn test_parse_phrase() {
        let parser = QueryParser::new().with_default_field("title");

        let query = parser.parse("\"hello world\"").unwrap();
        assert_eq!(query.description(), "title:hello world");
    }

    #[test]
    fn test_parse_empty_query() {
        let parser = QueryParser::new();

        let query = parser.parse("").unwrap();
        assert!(query.description().contains("()"));
    }

    #[test]
    fn test_parse_whitespace_query() {
        let parser = QueryParser::new();

        let query = parser.parse("   ").unwrap();
        assert!(query.description().contains("()"));
    }

    #[test]
    fn test_parse_invalid_field() {
        let parser = QueryParser::new();

        // Schema-less mode: no field validation, any field is accepted
        let result = parser.parse("invalid_field:hello");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_no_default_field() {
        let parser = QueryParser::new();

        let result = parser.parse("hello");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_field_specific() {
        let parser = QueryParser::new();

        let query = parser.parse_field("title", "hello world").unwrap();
        // The parser treats "hello world" as two separate terms with implicit AND
        let desc = query.description();
        assert!(desc.contains("title:hello"));
        assert!(desc.contains("title:world"));
    }

    #[test]
    fn test_parse_field_invalid() {
        let parser = QueryParser::new();

        // Schema-less mode: no field validation, any field is accepted
        let result = parser.parse_field("invalid_field", "hello");
        assert!(result.is_ok());
    }

    #[test]
    fn test_complex_query() {
        let parser = QueryParser::new();

        let query = parser
            .parse("title:hello AND (body:world OR author:john)")
            .unwrap();
        let desc = query.description();
        // Complex boolean query should be parsed
        assert!(desc.contains("title:hello"));
    }

    #[test]
    fn test_query_parser_builder() {
        let parser = QueryParserBuilder::new().default_field("title").build();

        assert_eq!(parser.default_field(), Some("title"));
    }

    #[test]
    fn test_query_parser_with_analyzer() {
        let parser = QueryParser::with_standard_analyzer()
            .unwrap()
            .with_default_field("title");

        // Should lowercase "HELLO"
        let query = parser.parse("HELLO").unwrap();
        let desc = query.description();
        assert!(desc.contains("hello") && !desc.contains("HELLO"));
    }

    #[test]
    fn test_query_parser_analyzer_stop_words() {
        let parser = QueryParser::with_standard_analyzer()
            .unwrap()
            .with_default_field("title");

        // "the" should be removed
        let query = parser.parse("the quick fox").unwrap();
        let desc = query.description();
        assert!(!desc.contains("the"));
        assert!(desc.contains("quick"));
        assert!(desc.contains("fox"));
    }

    #[test]
    fn test_query_parser_analyzer_multiple_tokens() {
        let parser = QueryParser::with_standard_analyzer()
            .unwrap()
            .with_default_field("title");

        // Should create OR query for multiple tokens
        let query = parser.parse("Hello World").unwrap();
        let desc = query.description();
        assert!(desc.contains("hello"));
        assert!(desc.contains("world"));
    }

    #[test]
    fn test_implicit_and() {
        let parser = QueryParser::new();

        let query = parser.parse("title:hello body:world").unwrap();
        let desc = query.description();
        // Should create implicit AND
        assert!(desc.contains("title:hello"));
        assert!(desc.contains("body:world"));
    }

    #[test]
    fn test_mixed_operators() {
        let parser = QueryParser::new();

        let query = parser
            .parse("+title:required -title:forbidden title:optional")
            .unwrap();
        let desc = query.description();
        assert!(desc.contains("title:required"));
        assert!(desc.contains("title:forbidden"));
        assert!(desc.contains("title:optional"));
    }
}
