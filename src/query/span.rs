//! Span queries for positional and proximity-based searching.
//!
//! Span queries provide advanced search capabilities based on term positions
//! within documents, enabling complex proximity and phrase searches.

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::full_text::reader::IndexReader;
use crate::query::{Matcher, Query, Scorer};

/// A span represents a term occurrence with position information.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Span {
    /// Start position (inclusive)
    pub start: u32,
    /// End position (exclusive)
    pub end: u32,
    /// Term that generated this span
    pub term: String,
}

impl Span {
    /// Create a new span.
    pub fn new(start: u32, end: u32, term: String) -> Self {
        Span { start, end, term }
    }

    /// Get the length of this span.
    pub fn length(&self) -> u32 {
        self.end.saturating_sub(self.start)
    }

    /// Check if this span overlaps with another span.
    pub fn overlaps(&self, other: &Span) -> bool {
        self.start < other.end && other.start < self.end
    }

    /// Check if this span contains another span.
    pub fn contains(&self, other: &Span) -> bool {
        self.start <= other.start && other.end <= self.end
    }

    /// Get the distance between this span and another span.
    pub fn distance_to(&self, other: &Span) -> u32 {
        if self.overlaps(other) {
            0
        } else if self.end <= other.start {
            other.start - self.end
        } else {
            self.start - other.end
        }
    }
}

/// Base trait for span queries.
pub trait SpanQuery: Send + Sync + std::fmt::Debug {
    /// Get spans for a document.
    fn get_spans(&self, doc_id: u32, reader: &dyn IndexReader) -> Result<Vec<Span>>;

    /// Get the field name this span query operates on.
    fn field_name(&self) -> &str;

    /// Clone this span query.
    fn clone_box(&self) -> Box<dyn SpanQuery>;
}

/// A span query that matches a single term.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanTermQuery {
    /// Field to search in
    field: String,
    /// Term to search for
    term: String,
    /// Boost factor
    boost: f32,
}

impl SpanTermQuery {
    /// Create a new span term query.
    pub fn new<F: Into<String>, T: Into<String>>(field: F, term: T) -> Self {
        SpanTermQuery {
            field: field.into(),
            term: term.into(),
            boost: 1.0,
        }
    }

    /// Set the boost factor.
    pub fn boost(mut self, boost: f32) -> Self {
        self.boost = boost;
        self
    }

    /// Get the term.
    pub fn term(&self) -> &str {
        &self.term
    }
}

impl SpanQuery for SpanTermQuery {
    fn get_spans(&self, _doc_id: u32, _reader: &dyn IndexReader) -> Result<Vec<Span>> {
        // In a real implementation, this would fetch term positions from the index
        // For now, we'll return a placeholder
        Ok(vec![Span::new(0, 1, self.term.clone())])
    }

    fn field_name(&self) -> &str {
        &self.field
    }

    fn clone_box(&self) -> Box<dyn SpanQuery> {
        Box::new(self.clone())
    }
}

/// A span query that matches terms near each other.
#[derive(Debug)]
pub struct SpanNearQuery {
    /// Field to search in
    field: String,
    /// Clauses that must appear near each other
    clauses: Vec<Box<dyn SpanQuery>>,
    /// Maximum distance between terms
    slop: u32,
    /// Whether terms must appear in order
    in_order: bool,
    /// Boost factor
    boost: f32,
}

impl SpanNearQuery {
    /// Create a new span near query.
    pub fn new<F: Into<String>>(
        field: F,
        clauses: Vec<Box<dyn SpanQuery>>,
        slop: u32,
        in_order: bool,
    ) -> Self {
        SpanNearQuery {
            field: field.into(),
            clauses,
            slop,
            in_order,
            boost: 1.0,
        }
    }

    /// Set the boost factor.
    pub fn boost(mut self, boost: f32) -> Self {
        self.boost = boost;
        self
    }

    /// Get the slop (maximum distance).
    pub fn slop(&self) -> u32 {
        self.slop
    }

    /// Check if terms must be in order.
    pub fn is_in_order(&self) -> bool {
        self.in_order
    }

    /// Get the clauses.
    pub fn clauses(&self) -> &[Box<dyn SpanQuery>] {
        &self.clauses
    }
}

impl SpanQuery for SpanNearQuery {
    fn get_spans(&self, doc_id: u32, reader: &dyn IndexReader) -> Result<Vec<Span>> {
        let mut all_clause_spans = Vec::new();

        // Get spans for each clause
        for clause in &self.clauses {
            let clause_spans = clause.get_spans(doc_id, reader)?;
            all_clause_spans.push(clause_spans);
        }

        // Find combinations of spans that satisfy the proximity requirements
        let mut result_spans = Vec::new();
        self.find_near_spans(&all_clause_spans, 0, Vec::new(), &mut result_spans);

        Ok(result_spans)
    }

    fn field_name(&self) -> &str {
        &self.field
    }

    fn clone_box(&self) -> Box<dyn SpanQuery> {
        let cloned_clauses: Vec<Box<dyn SpanQuery>> = self
            .clauses
            .iter()
            .map(|clause| clause.clone_box())
            .collect();

        Box::new(SpanNearQuery {
            field: self.field.clone(),
            clauses: cloned_clauses,
            slop: self.slop,
            in_order: self.in_order,
            boost: self.boost,
        })
    }
}

impl SpanNearQuery {
    /// Recursively find combinations of spans that satisfy proximity requirements.
    fn find_near_spans(
        &self,
        all_clause_spans: &[Vec<Span>],
        clause_index: usize,
        current_spans: Vec<Span>,
        result_spans: &mut Vec<Span>,
    ) {
        if clause_index >= all_clause_spans.len() {
            // We have spans for all clauses, check if they satisfy proximity
            if self.spans_satisfy_proximity(&current_spans)
                && let Some(combined_span) = self.combine_spans(&current_spans)
            {
                result_spans.push(combined_span);
            }
            return;
        }

        // Try each span from the current clause
        for span in &all_clause_spans[clause_index] {
            let mut new_current = current_spans.clone();
            new_current.push(span.clone());
            self.find_near_spans(
                all_clause_spans,
                clause_index + 1,
                new_current,
                result_spans,
            );
        }
    }

    /// Check if a set of spans satisfies the proximity requirements.
    fn spans_satisfy_proximity(&self, spans: &[Span]) -> bool {
        if spans.len() < 2 {
            return true;
        }

        let mut sorted_spans = spans.to_vec();
        sorted_spans.sort_by_key(|s| s.start);

        // Check order requirement
        if self.in_order {
            for (i, span) in sorted_spans.iter().enumerate().take(spans.len()) {
                if span.term != self.clauses[i].field_name() {
                    return false;
                }
            }
        }

        // Check slop requirement
        let total_span = Span::new(
            sorted_spans[0].start,
            sorted_spans.last().unwrap().end,
            "combined".to_string(),
        );

        let term_length: u32 = sorted_spans.iter().map(|s| s.length()).sum();
        let gaps = total_span.length().saturating_sub(term_length);

        gaps <= self.slop
    }

    /// Combine multiple spans into a single span covering all of them.
    fn combine_spans(&self, spans: &[Span]) -> Option<Span> {
        if spans.is_empty() {
            return None;
        }

        let start = spans.iter().map(|s| s.start).min().unwrap();
        let end = spans.iter().map(|s| s.end).max().unwrap();
        let terms: Vec<String> = spans.iter().map(|s| s.term.clone()).collect();
        let combined_term = format!("near({})", terms.join(","));

        Some(Span::new(start, end, combined_term))
    }
}

/// A span query that matches the first span that contains the second.
#[derive(Debug)]
pub struct SpanContainingQuery {
    /// Field to search in
    field: String,
    /// The containing span query
    big: Box<dyn SpanQuery>,
    /// The contained span query
    little: Box<dyn SpanQuery>,
    /// Boost factor
    boost: f32,
}

impl SpanContainingQuery {
    /// Create a new span containing query.
    pub fn new<F: Into<String>>(
        field: F,
        big: Box<dyn SpanQuery>,
        little: Box<dyn SpanQuery>,
    ) -> Self {
        SpanContainingQuery {
            field: field.into(),
            big,
            little,
            boost: 1.0,
        }
    }

    /// Set the boost factor.
    pub fn boost(mut self, boost: f32) -> Self {
        self.boost = boost;
        self
    }
}

impl SpanQuery for SpanContainingQuery {
    fn get_spans(&self, doc_id: u32, reader: &dyn IndexReader) -> Result<Vec<Span>> {
        let big_spans = self.big.get_spans(doc_id, reader)?;
        let little_spans = self.little.get_spans(doc_id, reader)?;

        let mut result = Vec::new();

        for big_span in &big_spans {
            for little_span in &little_spans {
                if big_span.contains(little_span) {
                    result.push(big_span.clone());
                    break; // Only add each big span once
                }
            }
        }

        Ok(result)
    }

    fn field_name(&self) -> &str {
        &self.field
    }

    fn clone_box(&self) -> Box<dyn SpanQuery> {
        Box::new(SpanContainingQuery {
            field: self.field.clone(),
            big: self.big.clone_box(),
            little: self.little.clone_box(),
            boost: self.boost,
        })
    }
}

/// A span query that matches spans within a certain distance of another span.
#[derive(Debug)]
pub struct SpanWithinQuery {
    /// Field to search in
    field: String,
    /// The span query to match
    include: Box<dyn SpanQuery>,
    /// The span query that defines the boundaries
    exclude: Box<dyn SpanQuery>,
    /// Maximum distance from exclude spans
    distance: u32,
    /// Boost factor
    boost: f32,
}

impl SpanWithinQuery {
    /// Create a new span within query.
    pub fn new<F: Into<String>>(
        field: F,
        include: Box<dyn SpanQuery>,
        exclude: Box<dyn SpanQuery>,
        distance: u32,
    ) -> Self {
        SpanWithinQuery {
            field: field.into(),
            include,
            exclude,
            distance,
            boost: 1.0,
        }
    }

    /// Set the boost factor.
    pub fn boost(mut self, boost: f32) -> Self {
        self.boost = boost;
        self
    }
}

impl SpanQuery for SpanWithinQuery {
    fn get_spans(&self, doc_id: u32, reader: &dyn IndexReader) -> Result<Vec<Span>> {
        let include_spans = self.include.get_spans(doc_id, reader)?;
        let exclude_spans = self.exclude.get_spans(doc_id, reader)?;

        let mut result = Vec::new();

        for include_span in &include_spans {
            let mut within_distance = false;

            for exclude_span in &exclude_spans {
                if include_span.distance_to(exclude_span) <= self.distance {
                    within_distance = true;
                    break;
                }
            }

            if within_distance {
                result.push(include_span.clone());
            }
        }

        Ok(result)
    }

    fn field_name(&self) -> &str {
        &self.field
    }

    fn clone_box(&self) -> Box<dyn SpanQuery> {
        Box::new(SpanWithinQuery {
            field: self.field.clone(),
            include: self.include.clone_box(),
            exclude: self.exclude.clone_box(),
            distance: self.distance,
            boost: self.boost,
        })
    }
}

/// A wrapper that adapts a SpanQuery to the regular Query interface.
#[derive(Debug)]
pub struct SpanQueryWrapper {
    /// The underlying span query
    span_query: Box<dyn SpanQuery>,
    /// Boost factor
    boost: f32,
}

impl SpanQueryWrapper {
    /// Create a new span query wrapper.
    pub fn new(span_query: Box<dyn SpanQuery>) -> Self {
        SpanQueryWrapper {
            span_query,
            boost: 1.0,
        }
    }

    /// Set the boost factor.
    pub fn boost(mut self, boost: f32) -> Self {
        self.boost = boost;
        self
    }

    /// Get the underlying span query.
    pub fn span_query(&self) -> &dyn SpanQuery {
        self.span_query.as_ref()
    }
}

impl Query for SpanQueryWrapper {
    fn matcher(&self, _reader: &dyn IndexReader) -> Result<Box<dyn Matcher>> {
        // In a real implementation, this would create a SpanMatcher
        Err(crate::error::SarissaError::query(
            "SpanQueryWrapper matcher not implemented",
        ))
    }

    fn scorer(&self, _reader: &dyn IndexReader) -> Result<Box<dyn Scorer>> {
        // In a real implementation, this would create a SpanScorer
        Err(crate::error::SarissaError::query(
            "SpanQueryWrapper scorer not implemented",
        ))
    }

    fn boost(&self) -> f32 {
        self.boost
    }

    fn set_boost(&mut self, boost: f32) {
        self.boost = boost;
    }

    fn clone_box(&self) -> Box<dyn Query> {
        Box::new(SpanQueryWrapper {
            span_query: self.span_query.clone_box(),
            boost: self.boost,
        })
    }

    fn description(&self) -> String {
        format!("SpanQueryWrapper({})", self.span_query.field_name())
    }

    fn is_empty(&self, _reader: &dyn IndexReader) -> Result<bool> {
        // For now, assume span queries are never empty
        Ok(false)
    }

    fn cost(&self, _reader: &dyn IndexReader) -> Result<u64> {
        // Return a default cost for now
        Ok(1)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Builder for creating complex span queries.
#[derive(Debug)]
pub struct SpanQueryBuilder {
    field: String,
}

impl SpanQueryBuilder {
    /// Create a new span query builder.
    pub fn new<F: Into<String>>(field: F) -> Self {
        SpanQueryBuilder {
            field: field.into(),
        }
    }

    /// Create a span term query.
    pub fn term<T: Into<String>>(&self, term: T) -> SpanTermQuery {
        SpanTermQuery::new(&self.field, term)
    }

    /// Create a span near query.
    pub fn near(
        &self,
        clauses: Vec<Box<dyn SpanQuery>>,
        slop: u32,
        in_order: bool,
    ) -> SpanNearQuery {
        SpanNearQuery::new(&self.field, clauses, slop, in_order)
    }

    /// Create a span containing query.
    pub fn containing(
        &self,
        big: Box<dyn SpanQuery>,
        little: Box<dyn SpanQuery>,
    ) -> SpanContainingQuery {
        SpanContainingQuery::new(&self.field, big, little)
    }

    /// Create a span within query.
    pub fn within(
        &self,
        include: Box<dyn SpanQuery>,
        exclude: Box<dyn SpanQuery>,
        distance: u32,
    ) -> SpanWithinQuery {
        SpanWithinQuery::new(&self.field, include, exclude, distance)
    }

    /// Create a phrase query using span near with zero slop.
    pub fn phrase(&self, terms: Vec<String>) -> SpanNearQuery {
        let clauses: Vec<Box<dyn SpanQuery>> = terms
            .into_iter()
            .map(|term| Box::new(self.term(term)) as Box<dyn SpanQuery>)
            .collect();

        self.near(clauses, 0, true)
    }

    /// Create a proximity query using span near.
    pub fn proximity(&self, terms: Vec<String>, slop: u32) -> SpanNearQuery {
        let clauses: Vec<Box<dyn SpanQuery>> = terms
            .into_iter()
            .map(|term| Box::new(self.term(term)) as Box<dyn SpanQuery>)
            .collect();

        self.near(clauses, slop, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_span_operations() {
        let span1 = Span::new(5, 10, "hello".to_string());
        let span2 = Span::new(8, 12, "world".to_string());
        let span3 = Span::new(15, 20, "test".to_string());

        assert_eq!(span1.length(), 5);
        assert!(span1.overlaps(&span2));
        assert!(!span1.overlaps(&span3));
        assert_eq!(span1.distance_to(&span3), 5);
        assert_eq!(span1.distance_to(&span2), 0); // overlapping
    }

    #[test]
    fn test_span_containment() {
        let big_span = Span::new(0, 20, "sentence".to_string());
        let small_span = Span::new(5, 10, "word".to_string());
        let outside_span = Span::new(25, 30, "other".to_string());

        assert!(big_span.contains(&small_span));
        assert!(!big_span.contains(&outside_span));
        assert!(!small_span.contains(&big_span));
    }

    #[test]
    fn test_span_term_query() {
        let query = SpanTermQuery::new("content", "hello").boost(2.0);

        assert_eq!(query.field_name(), "content");
        assert_eq!(query.term(), "hello");
        assert_eq!(query.boost, 2.0);
    }

    #[test]
    fn test_span_near_query() {
        let term1 = Box::new(SpanTermQuery::new("content", "hello")) as Box<dyn SpanQuery>;
        let term2 = Box::new(SpanTermQuery::new("content", "world")) as Box<dyn SpanQuery>;

        let near_query = SpanNearQuery::new("content", vec![term1, term2], 5, true);

        assert_eq!(near_query.field_name(), "content");
        assert_eq!(near_query.slop(), 5);
        assert!(near_query.is_in_order());
        assert_eq!(near_query.clauses().len(), 2);
    }

    #[test]
    fn test_span_query_builder() {
        let builder = SpanQueryBuilder::new("content");

        // Test term query
        let term_query = builder.term("hello");
        assert_eq!(term_query.field_name(), "content");
        assert_eq!(term_query.term(), "hello");

        // Test phrase query
        let phrase_query = builder.phrase(vec!["hello".to_string(), "world".to_string()]);
        assert_eq!(phrase_query.field_name(), "content");
        assert_eq!(phrase_query.slop(), 0);
        assert!(phrase_query.is_in_order());

        // Test proximity query
        let proximity_query = builder.proximity(vec!["hello".to_string(), "world".to_string()], 10);
        assert_eq!(proximity_query.field_name(), "content");
        assert_eq!(proximity_query.slop(), 10);
        assert!(!proximity_query.is_in_order());
    }

    #[test]
    fn test_span_query_wrapper() {
        let span_query = Box::new(SpanTermQuery::new("content", "hello")) as Box<dyn SpanQuery>;
        let wrapper = SpanQueryWrapper::new(span_query).boost(1.5);

        assert_eq!(wrapper.boost, 1.5);
        assert_eq!(wrapper.span_query().field_name(), "content");
    }

    #[test]
    fn test_span_proximity_checking() {
        let query = SpanNearQuery::new(
            "content",
            vec![], // Empty for this test
            3,      // slop = 3
            false,  // not in order
        );

        // Test spans that should satisfy proximity (distance = 2, which is <= slop of 3)
        let spans = vec![
            Span::new(0, 1, "hello".to_string()),
            Span::new(3, 4, "world".to_string()),
        ];
        assert!(query.spans_satisfy_proximity(&spans));

        // Test spans that should not satisfy proximity (distance = 5, which is > slop of 3)
        let spans = vec![
            Span::new(0, 1, "hello".to_string()),
            Span::new(6, 7, "world".to_string()),
        ];
        assert!(!query.spans_satisfy_proximity(&spans));
    }
}
