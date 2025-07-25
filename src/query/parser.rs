//! Query parser for converting string queries to structured query objects.

use crate::error::{Result, SarissaError};
use crate::query::{BooleanQuery, BooleanQueryBuilder, Occur, Query, TermQuery};
use crate::schema::Schema;
use std::iter::Peekable;
use std::str::Chars;

/// A simple query parser that supports basic query syntax.
#[derive(Debug)]
pub struct QueryParser {
    /// The schema to validate field names against.
    schema: Schema,
    /// Default field to search in when no field is specified.
    default_field: Option<String>,
}

impl QueryParser {
    /// Create a new query parser with the given schema.
    pub fn new(schema: Schema) -> Self {
        QueryParser {
            schema,
            default_field: None,
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

        let mut parser = QueryStringParser::new(trimmed, &self.schema, &self.default_field);
        parser.parse()
    }

    /// Parse a query string for a specific field.
    pub fn parse_field(&self, field: &str, query_str: &str) -> Result<Box<dyn Query>> {
        let trimmed = query_str.trim();
        if trimmed.is_empty() {
            return Ok(Box::new(BooleanQuery::new()));
        }

        // Validate field exists in schema
        if !self.schema.has_field(field) {
            return Err(SarissaError::schema(format!(
                "Field '{field}' does not exist"
            )));
        }

        let field_name = Some(field.to_string());
        let mut parser = QueryStringParser::new(trimmed, &self.schema, &field_name);
        parser.parse()
    }

    /// Get the schema for this parser.
    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// Get the default field.
    pub fn default_field(&self) -> Option<&str> {
        self.default_field.as_deref()
    }
}

/// Internal parser for parsing query strings.
struct QueryStringParser<'a> {
    chars: Peekable<Chars<'a>>,
    schema: &'a Schema,
    default_field: &'a Option<String>,
}

impl<'a> QueryStringParser<'a> {
    fn new(query_str: &'a str, schema: &'a Schema, default_field: &'a Option<String>) -> Self {
        QueryStringParser {
            chars: query_str.chars().peekable(),
            schema,
            default_field,
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

                // Validate field exists in schema
                if !self.schema.has_field(field) {
                    return Err(SarissaError::schema(format!(
                        "Field '{field}' does not exist"
                    )));
                }

                let query = Box::new(TermQuery::new(field, term));
                return Ok(self.wrap_with_occur(query, occur));
            }
        }

        // Use default field if available
        if let Some(default_field) = self.default_field {
            let query = Box::new(TermQuery::new(default_field, word));
            Ok(self.wrap_with_occur(query, occur))
        } else {
            Err(SarissaError::schema(
                "No default field specified and no field prefix found",
            ))
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
#[derive(Debug)]
pub struct QueryParserBuilder {
    schema: Schema,
    default_field: Option<String>,
}

impl QueryParserBuilder {
    /// Create a new query parser builder.
    pub fn new(schema: Schema) -> Self {
        QueryParserBuilder {
            schema,
            default_field: None,
        }
    }

    /// Set the default field.
    pub fn default_field<S: Into<String>>(mut self, field: S) -> Self {
        self.default_field = Some(field.into());
        self
    }

    /// Build the query parser.
    pub fn build(self) -> QueryParser {
        QueryParser {
            schema: self.schema,
            default_field: self.default_field,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{Schema, TextField};

    #[allow(dead_code)]
    fn create_test_schema() -> Schema {
        let mut schema = Schema::new();
        schema
            .add_field("title", Box::new(TextField::new().stored(true)))
            .unwrap();
        schema
            .add_field("body", Box::new(TextField::new()))
            .unwrap();
        schema
            .add_field("author", Box::new(TextField::new().stored(true)))
            .unwrap();
        schema
    }

    #[test]
    fn test_query_parser_creation() {
        let schema = create_test_schema();
        let parser = QueryParser::new(schema);

        assert!(parser.default_field().is_none());
        assert_eq!(parser.schema().len(), 3);
    }

    #[test]
    fn test_query_parser_with_default_field() {
        let schema = create_test_schema();
        let parser = QueryParser::new(schema).with_default_field("title");

        assert_eq!(parser.default_field(), Some("title"));
    }

    #[test]
    fn test_parse_simple_term() {
        let schema = create_test_schema();
        let parser = QueryParser::new(schema).with_default_field("title");

        let query = parser.parse("hello").unwrap();
        assert_eq!(query.description(), "title:hello");
    }

    #[test]
    fn test_parse_field_term() {
        let schema = create_test_schema();
        let parser = QueryParser::new(schema);

        let query = parser.parse("title:hello").unwrap();
        assert_eq!(query.description(), "title:hello");
    }

    #[test]
    fn test_parse_boolean_and() {
        let schema = create_test_schema();
        let parser = QueryParser::new(schema);

        let query = parser.parse("title:hello AND body:world").unwrap();
        let desc = query.description();
        assert!(desc.contains("title:hello"));
        assert!(desc.contains("body:world"));
    }

    #[test]
    fn test_parse_boolean_or() {
        let schema = create_test_schema();
        let parser = QueryParser::new(schema);

        let query = parser.parse("title:hello OR title:world").unwrap();
        let desc = query.description();
        assert!(desc.contains("title:hello"));
        assert!(desc.contains("title:world"));
    }

    #[test]
    fn test_parse_required_term() {
        let schema = create_test_schema();
        let parser = QueryParser::new(schema);

        let query = parser.parse("+title:hello").unwrap();
        let desc = query.description();
        assert!(desc.contains("+title:hello"));
    }

    #[test]
    fn test_parse_forbidden_term() {
        let schema = create_test_schema();
        let parser = QueryParser::new(schema);

        let query = parser.parse("-title:spam").unwrap();
        let desc = query.description();
        assert!(desc.contains("-title:spam"));
    }

    #[test]
    fn test_parse_phrase() {
        let schema = create_test_schema();
        let parser = QueryParser::new(schema).with_default_field("title");

        let query = parser.parse("\"hello world\"").unwrap();
        assert_eq!(query.description(), "title:hello world");
    }

    #[test]
    fn test_parse_empty_query() {
        let schema = create_test_schema();
        let parser = QueryParser::new(schema);

        let query = parser.parse("").unwrap();
        assert!(query.description().contains("()"));
    }

    #[test]
    fn test_parse_whitespace_query() {
        let schema = create_test_schema();
        let parser = QueryParser::new(schema);

        let query = parser.parse("   ").unwrap();
        assert!(query.description().contains("()"));
    }

    #[test]
    fn test_parse_invalid_field() {
        let schema = create_test_schema();
        let parser = QueryParser::new(schema);

        let result = parser.parse("invalid_field:hello");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_no_default_field() {
        let schema = create_test_schema();
        let parser = QueryParser::new(schema);

        let result = parser.parse("hello");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_field_specific() {
        let schema = create_test_schema();
        let parser = QueryParser::new(schema);

        let query = parser.parse_field("title", "hello world").unwrap();
        // The parser treats "hello world" as two separate terms with implicit AND
        let desc = query.description();
        assert!(desc.contains("title:hello"));
        assert!(desc.contains("title:world"));
    }

    #[test]
    fn test_parse_field_invalid() {
        let schema = create_test_schema();
        let parser = QueryParser::new(schema);

        let result = parser.parse_field("invalid_field", "hello");
        assert!(result.is_err());
    }

    #[test]
    fn test_complex_query() {
        let schema = create_test_schema();
        let parser = QueryParser::new(schema);

        let query = parser
            .parse("title:hello AND (body:world OR author:john)")
            .unwrap();
        let desc = query.description();
        // Complex boolean query should be parsed
        assert!(desc.contains("title:hello"));
    }

    #[test]
    fn test_query_parser_builder() {
        let schema = create_test_schema();

        let parser = QueryParserBuilder::new(schema)
            .default_field("title")
            .build();

        assert_eq!(parser.default_field(), Some("title"));
        assert_eq!(parser.schema().len(), 3);
    }

    #[test]
    fn test_implicit_and() {
        let schema = create_test_schema();
        let parser = QueryParser::new(schema);

        let query = parser.parse("title:hello body:world").unwrap();
        let desc = query.description();
        // Should create implicit AND
        assert!(desc.contains("title:hello"));
        assert!(desc.contains("body:world"));
    }

    #[test]
    fn test_mixed_operators() {
        let schema = create_test_schema();
        let parser = QueryParser::new(schema);

        let query = parser
            .parse("+title:required -title:forbidden title:optional")
            .unwrap();
        let desc = query.description();
        assert!(desc.contains("title:required"));
        assert!(desc.contains("title:forbidden"));
        assert!(desc.contains("title:optional"));
    }
}
