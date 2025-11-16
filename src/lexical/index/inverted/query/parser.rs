//! Query parser using pest.
//!
//! This parser supports the full query syntax including:
//! - Field-specific queries: `title:hello`
//! - Boolean operators: `AND`, `OR`
//! - Required/prohibited: `+required`, `-forbidden`
//! - Phrases: `"hello world"`
//! - Proximity search: `"hello world"~10`
//! - Fuzzy search: `roam~2`
//! - Range queries: `[100 TO 500]`, `{A TO Z}`
//! - Wildcards: `te?t`, `test*`
//! - Boosting: `jakarta^4`
//! - Grouping: `(title:hello OR body:world)`

use std::sync::Arc;

use pest::Parser;
use pest_derive::Parser;

use crate::analysis::analyzer::analyzer::Analyzer;
use crate::analysis::analyzer::per_field::PerFieldAnalyzer;
use crate::analysis::analyzer::standard::StandardAnalyzer;
use crate::document::field::NumericType;
use crate::error::{Result, YatagarasuError};
use crate::lexical::index::inverted::query::Query;
use crate::lexical::index::inverted::query::boolean::{BooleanClause, BooleanQuery, Occur};
use crate::lexical::index::inverted::query::fuzzy::FuzzyQuery;
use crate::lexical::index::inverted::query::phrase::PhraseQuery;
use crate::lexical::index::inverted::query::range::NumericRangeQuery;
use crate::lexical::index::inverted::query::term::TermQuery;
use crate::lexical::index::inverted::query::wildcard::WildcardQuery;

#[derive(Parser)]
#[grammar = "lexical/index/inverted/query/parser.pest"]
struct QueryStringParser;

/// Query parser.
///
/// Similar to Lucene's QueryParser, this requires an Analyzer to properly
/// normalize query terms before matching against the index.
pub struct QueryParser {
    /// Analyzer for tokenizing and normalizing query terms.
    /// Required - following Lucene's design where Analyzer is mandatory.
    analyzer: Arc<dyn Analyzer>,
    default_field: Option<String>,
    default_occur: Occur,
}

impl std::fmt::Debug for QueryParser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueryParser")
            .field("analyzer", &self.analyzer.name())
            .field("default_field", &self.default_field)
            .field("default_occur", &self.default_occur)
            .finish()
    }
}

impl QueryParser {
    /// Creates a new query parser with the given analyzer.
    ///
    /// Following Lucene's design, an Analyzer is required.
    ///
    /// # Arguments
    /// * `analyzer` - The analyzer to use for tokenizing and normalizing query terms
    ///
    /// # Example
    /// ```
    /// use yatagarasu::analysis::analyzer::standard::StandardAnalyzer;
    /// use yatagarasu::lexical::index::inverted::query::parser::QueryParser;
    /// use std::sync::Arc;
    ///
    /// let analyzer = Arc::new(StandardAnalyzer::new().unwrap());
    /// let parser = QueryParser::new(analyzer);
    /// ```
    pub fn new(analyzer: Arc<dyn Analyzer>) -> Self {
        Self {
            analyzer,
            default_field: None,
            default_occur: Occur::Should,
        }
    }

    /// Create a query parser with the standard analyzer.
    ///
    /// This is a convenience method for the common case.
    pub fn with_standard_analyzer() -> Result<Self> {
        Ok(QueryParser::new(Arc::new(StandardAnalyzer::new()?)))
    }

    /// Sets the default field.
    pub fn with_default_field(mut self, field: impl Into<String>) -> Self {
        self.default_field = Some(field.into());
        self
    }

    /// Sets the default occur.
    pub fn with_default_occur(mut self, occur: Occur) -> Self {
        self.default_occur = occur;
        self
    }

    /// Get the default field.
    pub fn default_field(&self) -> Option<&str> {
        self.default_field.as_deref()
    }

    /// Parse a field-specific query.
    pub fn parse_field(&self, field: &str, query_str: &str) -> Result<Box<dyn Query>> {
        // Handle phrase queries specially (preserve quotes)
        let full_query = if query_str.contains(' ') && !query_str.starts_with('"') {
            format!("{field}:\"{query_str}\"")
        } else {
            format!("{field}:{query_str}")
        };
        self.parse(&full_query)
    }

    /// Parses a query string into a Query object.
    pub fn parse(&self, query_str: &str) -> Result<Box<dyn Query>> {
        let pairs = QueryStringParser::parse(Rule::query, query_str)
            .map_err(|e| YatagarasuError::parse(format!("Parse error: {e}")))?;

        for pair in pairs {
            if pair.as_rule() == Rule::query {
                for inner_pair in pair.into_inner() {
                    if inner_pair.as_rule() == Rule::boolean_query {
                        return self.parse_boolean_query(inner_pair);
                    }
                }
            }
        }

        Err(YatagarasuError::parse("No valid query found".to_string()))
    }

    fn parse_boolean_query(&self, pair: pest::iterators::Pair<Rule>) -> Result<Box<dyn Query>> {
        let mut current_occur = self.default_occur;
        let mut terms: Vec<(Occur, Box<dyn Query>)> = Vec::new();

        for inner_pair in pair.into_inner() {
            match inner_pair.as_rule() {
                Rule::boolean_op => {
                    let op = inner_pair.as_str();
                    current_occur = match op.to_uppercase().as_str() {
                        "AND" => Occur::Must,
                        "OR" => Occur::Should,
                        _ => Occur::Should,
                    };
                }
                Rule::clause => {
                    let (occur, query) = self.parse_clause(inner_pair, current_occur)?;
                    terms.push((occur, query));
                }
                _ => {}
            }
        }

        // If only one term, return it directly
        if terms.len() == 1 {
            return Ok(terms.into_iter().next().unwrap().1);
        }

        // Build boolean query
        let mut bool_query = BooleanQuery::new();
        for (occur, query) in terms {
            bool_query.add_clause(BooleanClause::new(query, occur));
        }

        Ok(Box::new(bool_query))
    }

    fn parse_clause(
        &self,
        pair: pest::iterators::Pair<Rule>,
        default_occur: Occur,
    ) -> Result<(Occur, Box<dyn Query>)> {
        for inner_pair in pair.into_inner() {
            match inner_pair.as_rule() {
                Rule::required_clause => {
                    for sub_pair in inner_pair.into_inner() {
                        if sub_pair.as_rule() == Rule::sub_clause {
                            let query = self.parse_sub_clause(sub_pair)?;
                            return Ok((Occur::Must, query));
                        }
                    }
                }
                Rule::prohibited_clause => {
                    for sub_pair in inner_pair.into_inner() {
                        if sub_pair.as_rule() == Rule::sub_clause {
                            let query = self.parse_sub_clause(sub_pair)?;
                            return Ok((Occur::MustNot, query));
                        }
                    }
                }
                Rule::sub_clause => {
                    let query = self.parse_sub_clause(inner_pair)?;
                    return Ok((default_occur, query));
                }
                _ => {}
            }
        }

        Err(YatagarasuError::parse("Invalid clause".to_string()))
    }

    fn parse_sub_clause(&self, pair: pest::iterators::Pair<Rule>) -> Result<Box<dyn Query>> {
        for inner_pair in pair.into_inner() {
            match inner_pair.as_rule() {
                Rule::grouped_query => return self.parse_grouped_query(inner_pair),
                Rule::field_query => return self.parse_field_query(inner_pair),
                Rule::term_query => return self.parse_term_query(inner_pair),
                _ => {}
            }
        }

        Err(YatagarasuError::parse("Invalid sub-clause".to_string()))
    }

    fn parse_grouped_query(&self, pair: pest::iterators::Pair<Rule>) -> Result<Box<dyn Query>> {
        let mut boost = 1.0;
        let mut query: Option<Box<dyn Query>> = None;

        for inner_pair in pair.into_inner() {
            match inner_pair.as_rule() {
                Rule::boolean_query => {
                    query = Some(self.parse_boolean_query(inner_pair)?);
                }
                Rule::boost => {
                    boost = self.parse_boost(inner_pair)?;
                }
                _ => {}
            }
        }

        if let Some(mut q) = query {
            if boost != 1.0 {
                q.set_boost(boost);
            }
            Ok(q)
        } else {
            Err(YatagarasuError::parse("Invalid grouped query".to_string()))
        }
    }

    fn parse_field_query(&self, pair: pest::iterators::Pair<Rule>) -> Result<Box<dyn Query>> {
        let mut field: Option<String> = None;

        for inner_pair in pair.into_inner() {
            match inner_pair.as_rule() {
                Rule::field => {
                    field = Some(inner_pair.as_str().to_string());
                }
                Rule::field_value => {
                    let field_name = field
                        .ok_or_else(|| YatagarasuError::parse("Missing field name".to_string()))?;
                    return self.parse_field_value(inner_pair, Some(&field_name));
                }
                _ => {}
            }
        }

        Err(YatagarasuError::parse("Invalid field query".to_string()))
    }

    fn parse_term_query(&self, pair: pest::iterators::Pair<Rule>) -> Result<Box<dyn Query>> {
        for inner_pair in pair.into_inner() {
            if inner_pair.as_rule() == Rule::field_value {
                return self.parse_field_value(inner_pair, None);
            }
        }

        Err(YatagarasuError::parse("Invalid term query".to_string()))
    }

    fn parse_field_value(
        &self,
        pair: pest::iterators::Pair<Rule>,
        field: Option<&str>,
    ) -> Result<Box<dyn Query>> {
        for inner_pair in pair.into_inner() {
            match inner_pair.as_rule() {
                Rule::range_query => return self.parse_range_query(inner_pair, field),
                Rule::phrase_query => return self.parse_phrase_query(inner_pair, field),
                Rule::fuzzy_term => return self.parse_fuzzy_term(inner_pair, field),
                Rule::wildcard_term => return self.parse_wildcard_term(inner_pair, field),
                Rule::simple_term => return self.parse_simple_term(inner_pair, field),
                _ => {}
            }
        }

        Err(YatagarasuError::parse("Invalid field value".to_string()))
    }

    fn parse_range_query(
        &self,
        pair: pest::iterators::Pair<Rule>,
        field: Option<&str>,
    ) -> Result<Box<dyn Query>> {
        let field_name = field
            .or(self.default_field.as_deref())
            .ok_or_else(|| YatagarasuError::parse("No field specified".to_string()))?;

        let mut lower_inclusive = true;
        let mut upper_inclusive = true;
        let mut lower: Option<String> = None;
        let mut upper: Option<String> = None;

        for inner_pair in pair.into_inner() {
            match inner_pair.as_rule() {
                Rule::range_inclusive => {
                    lower_inclusive = true;
                    upper_inclusive = true;
                    for range_part in inner_pair.into_inner() {
                        if range_part.as_rule() == Rule::range_value {
                            if lower.is_none() {
                                lower = Some(self.parse_range_value(range_part)?);
                            } else {
                                upper = Some(self.parse_range_value(range_part)?);
                            }
                        }
                    }
                }
                Rule::range_exclusive => {
                    lower_inclusive = false;
                    upper_inclusive = false;
                    for range_part in inner_pair.into_inner() {
                        if range_part.as_rule() == Rule::range_value {
                            if lower.is_none() {
                                lower = Some(self.parse_range_value(range_part)?);
                            } else {
                                upper = Some(self.parse_range_value(range_part)?);
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        // Try to parse as numeric range, fallback to term query for text ranges
        let lower_num = lower.as_ref().and_then(|s| s.parse::<f64>().ok());
        let upper_num = upper.as_ref().and_then(|s| s.parse::<f64>().ok());

        if lower_num.is_some() || upper_num.is_some() {
            // Numeric range query
            let query = NumericRangeQuery::new(
                field_name,
                NumericType::Float,
                lower_num,
                upper_num,
                lower_inclusive,
                upper_inclusive,
            );
            Ok(Box::new(query))
        } else {
            // Text range - use a term query as fallback
            let term = format!(
                "{}{} TO {}{}",
                if lower_inclusive { "[" } else { "{" },
                lower.as_deref().unwrap_or("*"),
                upper.as_deref().unwrap_or("*"),
                if upper_inclusive { "]" } else { "}" }
            );
            Ok(Box::new(TermQuery::new(field_name, &term)))
        }
    }

    fn parse_range_value(&self, pair: pest::iterators::Pair<Rule>) -> Result<String> {
        let value = pair.as_str();
        if value == "*" {
            Ok("*".to_string())
        } else {
            Ok(value.trim_matches('"').to_string())
        }
    }

    fn parse_phrase_query(
        &self,
        pair: pest::iterators::Pair<Rule>,
        field: Option<&str>,
    ) -> Result<Box<dyn Query>> {
        let field_name = field
            .or(self.default_field.as_deref())
            .ok_or_else(|| YatagarasuError::parse("No field specified".to_string()))?;

        let mut phrase_content = String::new();
        let mut slop: Option<u32> = None;
        let mut boost = 1.0;

        for inner_pair in pair.into_inner() {
            match inner_pair.as_rule() {
                Rule::phrase_content => {
                    phrase_content = inner_pair.as_str().to_string();
                }
                Rule::proximity => {
                    for prox_pair in inner_pair.into_inner() {
                        if prox_pair.as_rule() == Rule::number {
                            slop = Some(prox_pair.as_str().parse().unwrap_or(0));
                        }
                    }
                }
                Rule::boost => {
                    boost = self.parse_boost(inner_pair)?;
                }
                _ => {}
            }
        }

        let terms = self.analyze_term(Some(field_name), &phrase_content)?;
        let mut phrase_query = PhraseQuery::new(field_name, terms);

        if let Some(slop_value) = slop {
            phrase_query = phrase_query.with_slop(slop_value);
        }

        if boost != 1.0 {
            phrase_query = phrase_query.with_boost(boost);
        }

        Ok(Box::new(phrase_query))
    }

    fn parse_fuzzy_term(
        &self,
        pair: pest::iterators::Pair<Rule>,
        field: Option<&str>,
    ) -> Result<Box<dyn Query>> {
        let field_name = field
            .or(self.default_field.as_deref())
            .ok_or_else(|| YatagarasuError::parse("No field specified".to_string()))?;

        let mut term = String::new();
        let mut fuzziness: u8 = 2; // Default fuzziness

        for inner_pair in pair.into_inner() {
            match inner_pair.as_rule() {
                Rule::term => {
                    term = inner_pair.as_str().to_string();
                }
                Rule::fuzziness => {
                    for fuzz_pair in inner_pair.into_inner() {
                        if fuzz_pair.as_rule() == Rule::number {
                            fuzziness = fuzz_pair.as_str().parse().unwrap_or(2);
                        }
                    }
                }
                _ => {}
            }
        }

        // ✅ Normalize the term using the analyzer (like Lucene does)
        // This ensures the query term is in the same form as indexed terms
        let terms = self.analyze_term(Some(field_name), &term)?;
        let normalized_term = if terms.is_empty() {
            // Fallback to original term if analyzer produces no tokens
            &term
        } else {
            // Use the first token (following Lucene's behavior)
            &terms[0]
        };

        Ok(Box::new(
            FuzzyQuery::new(field_name, normalized_term).max_edits(fuzziness as u32),
        ))
    }

    fn parse_wildcard_term(
        &self,
        pair: pest::iterators::Pair<Rule>,
        field: Option<&str>,
    ) -> Result<Box<dyn Query>> {
        let field_name = field
            .or(self.default_field.as_deref())
            .ok_or_else(|| YatagarasuError::parse("No field specified".to_string()))?;

        let mut pattern = String::new();

        for inner_pair in pair.into_inner() {
            if inner_pair.as_rule() == Rule::wildcard_pattern {
                pattern = inner_pair.as_str().to_string();
            }
        }

        Ok(Box::new(WildcardQuery::new(field_name, &pattern)?))
    }

    fn parse_simple_term(
        &self,
        pair: pest::iterators::Pair<Rule>,
        field: Option<&str>,
    ) -> Result<Box<dyn Query>> {
        let field_name = field
            .or(self.default_field.as_deref())
            .ok_or_else(|| YatagarasuError::parse("No field specified".to_string()))?;

        let mut term = String::new();
        let mut boost = 1.0;

        for inner_pair in pair.into_inner() {
            match inner_pair.as_rule() {
                Rule::term => {
                    term = inner_pair.as_str().to_string();
                }
                Rule::boost => {
                    boost = self.parse_boost(inner_pair)?;
                }
                _ => {}
            }
        }

        let terms = self.analyze_term(Some(field_name), &term)?;

        if terms.is_empty() {
            return Err(YatagarasuError::parse(
                "No terms after analysis".to_string(),
            ));
        }

        if terms.len() == 1 {
            let query = TermQuery::new(field_name, &terms[0]);
            if boost != 1.0 {
                Ok(Box::new(query.with_boost(boost)))
            } else {
                Ok(Box::new(query))
            }
        } else {
            // Multiple terms - create a phrase query
            let query = PhraseQuery::new(field_name, terms);
            if boost != 1.0 {
                Ok(Box::new(query.with_boost(boost)))
            } else {
                Ok(Box::new(query))
            }
        }
    }

    fn parse_boost(&self, pair: pest::iterators::Pair<Rule>) -> Result<f32> {
        for inner_pair in pair.into_inner() {
            if inner_pair.as_rule() == Rule::boost_value {
                return Ok(inner_pair.as_str().parse().unwrap_or(1.0));
            }
        }
        Ok(1.0)
    }

    fn analyze_term(&self, field: Option<&str>, term: &str) -> Result<Vec<String>> {
        let token_stream = if let Some(field_name) = field {
            // Use field-specific analyzer if available (PerFieldAnalyzer)
            if let Some(per_field) = self.analyzer.as_any().downcast_ref::<PerFieldAnalyzer>() {
                per_field.analyze_field(field_name, term)?
            } else {
                self.analyzer.analyze(term)?
            }
        } else {
            self.analyzer.analyze(term)?
        };

        let tokens: Vec<String> = token_stream.into_iter().map(|t| t.text).collect();
        Ok(tokens)
    }
}

/// Builder for QueryParser.
pub struct QueryParserBuilder {
    analyzer: Arc<dyn Analyzer>,
    default_field: Option<String>,
    default_occur: Occur,
}

impl QueryParserBuilder {
    /// Creates a new builder with the given analyzer.
    pub fn new(analyzer: Arc<dyn Analyzer>) -> Self {
        Self {
            analyzer,
            default_field: None,
            default_occur: Occur::Should,
        }
    }

    /// Sets the default field.
    pub fn default_field(mut self, field: impl Into<String>) -> Self {
        self.default_field = Some(field.into());
        self
    }

    /// Sets the default occur.
    pub fn default_occur(mut self, occur: Occur) -> Self {
        self.default_occur = occur;
        self
    }

    /// Builds the parser.
    pub fn build(self) -> Result<QueryParser> {
        Ok(QueryParser {
            analyzer: self.analyzer,
            default_field: self.default_field,
            default_occur: self.default_occur,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::analyzer::standard::StandardAnalyzer;

    /// Helper function to create a test parser with StandardAnalyzer
    fn create_test_parser() -> QueryParser {
        let analyzer = Arc::new(StandardAnalyzer::new().unwrap());
        QueryParser::new(analyzer)
    }

    #[test]
    fn test_simple_term() {
        let parser = create_test_parser().with_default_field("content");
        let query = parser.parse("hello").unwrap();
        assert!(format!("{query:?}").contains("TermQuery"));
    }

    #[test]
    fn test_field_query() {
        let parser = create_test_parser().with_default_field("content");
        let query = parser.parse("title:hello").unwrap();
        assert!(format!("{query:?}").contains("TermQuery"));
    }

    #[test]
    fn test_boolean_query() {
        let parser = create_test_parser().with_default_field("content");
        let query = parser.parse("hello AND world").unwrap();
        assert!(format!("{query:?}").contains("BooleanQuery"));
    }

    #[test]
    fn test_phrase_query() {
        let parser = create_test_parser().with_default_field("content");
        let query = parser.parse("\"hello world\"").unwrap();
        assert!(format!("{query:?}").contains("PhraseQuery"));
    }

    #[test]
    fn test_fuzzy_query() {
        let parser = create_test_parser().with_default_field("content");
        let query = parser.parse("hello~2").unwrap();
        assert!(format!("{query:?}").contains("FuzzyQuery"));
    }

    #[test]
    fn test_wildcard_query() {
        let parser = create_test_parser().with_default_field("content");
        let query = parser.parse("hel*").unwrap();
        assert!(format!("{query:?}").contains("WildcardQuery"));
    }

    #[test]
    fn test_required_clause() {
        let parser = create_test_parser().with_default_field("content");
        let query = parser.parse("+hello world").unwrap();
        assert!(format!("{query:?}").contains("BooleanQuery"));
    }

    #[test]
    fn test_prohibited_clause() {
        let parser = create_test_parser().with_default_field("content");
        let query = parser.parse("hello -world").unwrap();
        assert!(format!("{query:?}").contains("BooleanQuery"));
    }

    #[test]
    fn test_grouped_query() {
        let parser = create_test_parser().with_default_field("content");
        let query = parser.parse("(hello OR world) AND test").unwrap();
        assert!(format!("{query:?}").contains("BooleanQuery"));
    }

    #[test]
    fn test_proximity_search() {
        let parser = create_test_parser().with_default_field("content");
        let query = parser.parse("\"hello world\"~10").unwrap();
        assert!(format!("{query:?}").contains("PhraseQuery"));
    }
}
