//! Range query implementation for querying within value ranges.

use std::fmt::Debug;

use chrono::{DateTime, Utc};

use crate::lexical::document::field::NumericType;
use crate::error::Result;
use crate::lexical::index::inverted::query::Query;
use crate::lexical::index::inverted::query::matcher::{EmptyMatcher, Matcher, PreComputedMatcher};
use crate::lexical::index::inverted::query::scorer::{BM25Scorer, Scorer};
use crate::lexical::reader::LexicalIndexReader;

/// Bound type for range queries.
#[derive(Debug, Clone, PartialEq)]
pub enum Bound<T> {
    /// Inclusive bound.
    Included(T),
    /// Exclusive bound.
    Excluded(T),
    /// Unbounded (no limit).
    Unbounded,
}

impl<T: PartialOrd> Bound<T> {
    /// Check if a value satisfies this bound as a lower bound.
    pub fn contains_lower(&self, value: &T) -> bool {
        match self {
            Bound::Included(bound) => value >= bound,
            Bound::Excluded(bound) => value > bound,
            Bound::Unbounded => true,
        }
    }

    /// Check if a value satisfies this bound as an upper bound.
    pub fn contains_upper(&self, value: &T) -> bool {
        match self {
            Bound::Included(bound) => value <= bound,
            Bound::Excluded(bound) => value < bound,
            Bound::Unbounded => true,
        }
    }
}

/// A query that matches documents with field values within a specified range.
#[derive(Debug, Clone)]
pub struct RangeQuery {
    /// The field to search in.
    field: String,
    /// Lower bound of the range.
    lower_bound: Bound<String>,
    /// Upper bound of the range.
    upper_bound: Bound<String>,
    /// The boost factor for this query.
    boost: f32,
}

impl RangeQuery {
    /// Create a new range query with both bounds inclusive.
    pub fn new<S: Into<String>>(field: S, lower: Option<String>, upper: Option<String>) -> Self {
        let lower_bound = match lower {
            Some(val) => Bound::Included(val),
            None => Bound::Unbounded,
        };
        let upper_bound = match upper {
            Some(val) => Bound::Included(val),
            None => Bound::Unbounded,
        };

        RangeQuery {
            field: field.into(),
            lower_bound,
            upper_bound,
            boost: 1.0,
        }
    }

    /// Create a range query with custom bound types.
    pub fn with_bounds<S: Into<String>>(
        field: S,
        lower_bound: Bound<String>,
        upper_bound: Bound<String>,
    ) -> Self {
        RangeQuery {
            field: field.into(),
            lower_bound,
            upper_bound,
            boost: 1.0,
        }
    }

    /// Create a range query for values greater than or equal to the given value.
    pub fn greater_than_or_equal<S: Into<String>>(field: S, value: String) -> Self {
        Self::with_bounds(field, Bound::Included(value), Bound::Unbounded)
    }

    /// Create a range query for values greater than the given value.
    pub fn greater_than<S: Into<String>>(field: S, value: String) -> Self {
        Self::with_bounds(field, Bound::Excluded(value), Bound::Unbounded)
    }

    /// Create a range query for values less than or equal to the given value.
    pub fn less_than_or_equal<S: Into<String>>(field: S, value: String) -> Self {
        Self::with_bounds(field, Bound::Unbounded, Bound::Included(value))
    }

    /// Create a range query for values less than the given value.
    pub fn less_than<S: Into<String>>(field: S, value: String) -> Self {
        Self::with_bounds(field, Bound::Unbounded, Bound::Excluded(value))
    }

    /// Set the boost factor for this query.
    pub fn with_boost(mut self, boost: f32) -> Self {
        self.boost = boost;
        self
    }

    /// Get the field name.
    pub fn field(&self) -> &str {
        &self.field
    }

    /// Get the lower bound.
    pub fn lower_bound(&self) -> &Bound<String> {
        &self.lower_bound
    }

    /// Get the upper bound.
    pub fn upper_bound(&self) -> &Bound<String> {
        &self.upper_bound
    }

    /// Check if a term falls within the range.
    pub fn contains(&self, term: &str) -> bool {
        self.lower_bound.contains_lower(&term.to_string())
            && self.upper_bound.contains_upper(&term.to_string())
    }
}

impl Query for RangeQuery {
    fn matcher(&self, _reader: &dyn LexicalIndexReader) -> Result<Box<dyn Matcher>> {
        Ok(Box::new(EmptyMatcher::new()))
    }

    fn scorer(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Scorer>> {
        // Range queries typically match fewer documents than term queries
        let total_docs = reader.doc_count();
        let estimated_doc_freq = total_docs / 10; // Estimate: 10% of docs might match a range
        let estimated_term_freq = estimated_doc_freq * 2; // Average 2 occurrences per doc

        Ok(Box::new(BM25Scorer::new(
            estimated_doc_freq.max(1),
            estimated_term_freq.max(1),
            total_docs,
            10.0, // Estimated average field length
            total_docs,
            self.boost,
        )))
    }

    fn boost(&self) -> f32 {
        self.boost
    }

    fn set_boost(&mut self, boost: f32) {
        self.boost = boost;
    }

    fn description(&self) -> String {
        format!(
            "RangeQuery(field:{}, lower:{:?}, upper:{:?})",
            self.field, self.lower_bound, self.upper_bound
        )
    }

    fn clone_box(&self) -> Box<dyn Query> {
        Box::new(self.clone())
    }

    fn is_empty(&self, _reader: &dyn LexicalIndexReader) -> Result<bool> {
        Ok(false)
    }

    fn cost(&self, _reader: &dyn LexicalIndexReader) -> Result<u64> {
        Ok(500) // Range queries are moderately expensive
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Matcher for range queries.
#[derive(Debug)]
pub struct RangeMatcher {
    /// Current document ID.
    current_doc: u64,
    /// Boost factor for scoring.
    #[allow(dead_code)]
    boost: f32,
    /// Whether we've reached the end.
    exhausted: bool,
}

impl RangeMatcher {
    /// Create a new range matcher.
    pub fn new(boost: f32) -> Self {
        RangeMatcher {
            current_doc: 0,
            boost,
            exhausted: true,
        }
    }
}

impl Matcher for RangeMatcher {
    fn doc_id(&self) -> u64 {
        if self.exhausted {
            u64::MAX
        } else {
            self.current_doc
        }
    }

    fn next(&mut self) -> Result<bool> {
        Ok(false)
    }

    fn skip_to(&mut self, _target: u64) -> Result<bool> {
        Ok(false)
    }

    fn cost(&self) -> u64 {
        0
    }

    fn is_exhausted(&self) -> bool {
        self.exhausted
    }
}

/// Specialized numeric range query for optimal performance.
#[derive(Debug, Clone)]
pub struct NumericRangeQuery {
    /// The field to search in.
    field: String,
    /// The numeric type of the field.
    numeric_type: NumericType,
    /// Lower bound as bytes (for efficient comparison).
    lower_bound: Option<Vec<u8>>,
    /// Upper bound as bytes (for efficient comparison).
    upper_bound: Option<Vec<u8>>,
    /// Whether lower bound is inclusive.
    lower_inclusive: bool,
    /// Whether upper bound is inclusive.
    upper_inclusive: bool,
    /// The boost factor for this query.
    boost: f32,
}

impl NumericRangeQuery {
    /// Create a new numeric range query.
    pub fn new<S: Into<String>>(
        field: S,
        numeric_type: NumericType,
        lower: Option<f64>,
        upper: Option<f64>,
        lower_inclusive: bool,
        upper_inclusive: bool,
    ) -> Self {
        let lower_bound = lower.map(|v| Self::encode_numeric(v, numeric_type));
        let upper_bound = upper.map(|v| Self::encode_numeric(v, numeric_type));

        NumericRangeQuery {
            field: field.into(),
            numeric_type,
            lower_bound,
            upper_bound,
            lower_inclusive,
            upper_inclusive,
            boost: 1.0,
        }
    }

    /// Create a range query for integers.
    pub fn i64_range<S: Into<String>>(field: S, lower: Option<i64>, upper: Option<i64>) -> Self {
        Self::new(
            field,
            NumericType::Integer,
            lower.map(|v| v as f64),
            upper.map(|v| v as f64),
            true,
            true,
        )
    }

    /// Create a range query for floats with inclusive bounds.
    pub fn f64_range<S: Into<String>>(field: S, lower: Option<f64>, upper: Option<f64>) -> Self {
        Self::new(field, NumericType::Float, lower, upper, true, true)
    }

    /// Create a range query for floats with exclusive upper bound.
    pub fn f64_range_exclusive_upper<S: Into<String>>(
        field: S,
        lower: Option<f64>,
        upper: Option<f64>,
    ) -> Self {
        Self::new(field, NumericType::Float, lower, upper, true, false)
    }

    /// Create a range query for floats with exclusive lower bound.
    pub fn f64_range_exclusive_lower<S: Into<String>>(
        field: S,
        lower: Option<f64>,
        upper: Option<f64>,
    ) -> Self {
        Self::new(field, NumericType::Float, lower, upper, false, true)
    }

    /// Create a range query for floats with both bounds exclusive.
    pub fn f64_range_exclusive<S: Into<String>>(
        field: S,
        lower: Option<f64>,
        upper: Option<f64>,
    ) -> Self {
        Self::new(field, NumericType::Float, lower, upper, false, false)
    }

    /// Create a greater than query.
    pub fn greater_than<S: Into<String>>(field: S, numeric_type: NumericType, value: f64) -> Self {
        Self::new(field, numeric_type, Some(value), None, false, true)
    }

    /// Create a greater than or equal query.
    pub fn greater_than_or_equal<S: Into<String>>(
        field: S,
        numeric_type: NumericType,
        value: f64,
    ) -> Self {
        Self::new(field, numeric_type, Some(value), None, true, true)
    }

    /// Create a less than query.
    pub fn less_than<S: Into<String>>(field: S, numeric_type: NumericType, value: f64) -> Self {
        Self::new(field, numeric_type, None, Some(value), true, false)
    }

    /// Create a less than or equal query.
    pub fn less_than_or_equal<S: Into<String>>(
        field: S,
        numeric_type: NumericType,
        value: f64,
    ) -> Self {
        Self::new(field, numeric_type, None, Some(value), true, true)
    }

    /// Set the boost factor for this query.
    pub fn with_boost(mut self, boost: f32) -> Self {
        self.boost = boost;
        self
    }

    /// Get the field name.
    pub fn field(&self) -> &str {
        &self.field
    }

    /// Get the numeric type.
    pub fn numeric_type(&self) -> NumericType {
        self.numeric_type
    }

    /// Encode a numeric value for efficient storage and comparison.
    pub fn encode_numeric(value: f64, numeric_type: NumericType) -> Vec<u8> {
        match numeric_type {
            NumericType::Integer => (value as i64).to_be_bytes().to_vec(),
            NumericType::Float => value.to_be_bytes().to_vec(),
        }
    }

    /// Check if an encoded value falls within the range.
    pub fn contains_encoded(&self, encoded_value: &[u8]) -> bool {
        // Check lower bound
        if let Some(ref lower) = self.lower_bound {
            let cmp = encoded_value.cmp(lower);
            if self.lower_inclusive {
                if cmp == std::cmp::Ordering::Less {
                    return false;
                }
            } else if cmp != std::cmp::Ordering::Greater {
                return false;
            }
        }

        // Check upper bound
        if let Some(ref upper) = self.upper_bound {
            let cmp = encoded_value.cmp(upper);
            if self.upper_inclusive {
                if cmp == std::cmp::Ordering::Greater {
                    return false;
                }
            } else if cmp != std::cmp::Ordering::Less {
                return false;
            }
        }

        true
    }

    /// Check if a numeric value falls within the range.
    pub fn contains_numeric(&self, value: f64) -> bool {
        let encoded = Self::encode_numeric(value, self.numeric_type);
        self.contains_encoded(&encoded)
    }

    /// Get the minimum f64 value if the range has a lower bound.
    pub fn min_f64(&self) -> Option<f64> {
        self.lower_bound
            .as_ref()
            .map(|bytes| match self.numeric_type {
                NumericType::Float => {
                    f64::from_be_bytes(bytes.as_slice().try_into().unwrap_or([0; 8]))
                }
                NumericType::Integer => {
                    i64::from_be_bytes(bytes.as_slice().try_into().unwrap_or([0; 8])) as f64
                }
            })
    }

    /// Get the maximum f64 value if the range has an upper bound.
    pub fn max_f64(&self) -> Option<f64> {
        self.upper_bound
            .as_ref()
            .map(|bytes| match self.numeric_type {
                NumericType::Float => {
                    f64::from_be_bytes(bytes.as_slice().try_into().unwrap_or([0; 8]))
                }
                NumericType::Integer => {
                    i64::from_be_bytes(bytes.as_slice().try_into().unwrap_or([0; 8])) as f64
                }
            })
    }

    /// Get the minimum i64 value if the range has a lower bound.
    pub fn min_i64(&self) -> Option<i64> {
        self.min_f64().map(|v| v as i64)
    }

    /// Get the maximum i64 value if the range has an upper bound.
    pub fn max_i64(&self) -> Option<i64> {
        self.max_f64().map(|v| v as i64)
    }

    /// Count the number of documents that match this range query.
    pub fn count_matching_documents(&self, reader: &dyn LexicalIndexReader) -> Result<u64> {
        let mut count = 0u64;
        let total_docs = reader.max_doc();

        for doc_id in 0..total_docs {
            if let Ok(Some(doc)) = reader.document(doc_id)
                && let Some(field_value) = doc.get_field(&self.field)
            {
                let numeric_value = match &field_value.value {
                    crate::lexical::document::field::FieldValue::Float(f) => *f,
                    crate::lexical::document::field::FieldValue::Integer(i) => *i as f64,
                    _ => continue, // Not a numeric field
                };

                if self.contains_numeric(numeric_value) {
                    count += 1;
                }
            }
        }

        Ok(count)
    }
}

/// A scorer specialized for numeric range queries.
#[derive(Debug, Clone)]
pub struct RangeScorer {
    /// Lower bound value for proximity calculation.
    lower_bound: Option<f64>,
    /// Upper bound value for proximity calculation.
    upper_bound: Option<f64>,
    /// Range width for selectivity calculation.
    range_width: f64,
    /// Total number of documents.
    total_docs: u64,
    /// Number of documents matching the range.
    matching_docs: u64,
    /// Boost factor.
    boost: f32,
}

impl RangeScorer {
    /// Create a new range scorer.
    pub fn new(
        lower_bound: Option<f64>,
        upper_bound: Option<f64>,
        matching_docs: u64,
        total_docs: u64,
        boost: f32,
    ) -> Self {
        // Calculate range width for selectivity scoring
        let range_width = match (lower_bound, upper_bound) {
            (Some(lower), Some(upper)) => (upper - lower).abs(),
            (Some(_), None) | (None, Some(_)) => 1000.0, // Large default for unbounded ranges
            (None, None) => f64::INFINITY,               // Unbounded range
        };

        RangeScorer {
            lower_bound,
            upper_bound,
            range_width,
            total_docs,
            matching_docs,
            boost,
        }
    }

    /// Calculate selectivity-based IDF for the range.
    fn range_idf(&self) -> f32 {
        if self.matching_docs == 0 || self.total_docs == 0 {
            // Return a default IDF for ranges with no matches
            return 1.0;
        }

        let n = self.total_docs as f32;
        let df = self.matching_docs as f32;

        // Base IDF calculation using BM25 formula
        let base_idf = ((n - df + 0.5) / (df + 0.5)).ln();

        // Check for invalid values and ensure non-negative
        let base_idf = if base_idf.is_nan() || base_idf.is_infinite() || base_idf < 0.0 {
            // When most documents match (negative IDF), use a small positive value
            // This maintains relative ranking while avoiding negative scores
            0.5
        } else {
            base_idf
        };

        // Apply range selectivity bonus - narrower ranges get higher scores
        let selectivity_multiplier = if self.range_width.is_finite() && self.range_width > 0.0 {
            // Smaller range width = higher selectivity = higher score
            1.0_f32 + (100.0_f32 / (self.range_width as f32 + 1.0_f32)).min(2.0_f32)
        } else {
            1.0_f32 // No bonus for unbounded ranges
        };

        // Ensure minimum meaningful score
        let min_score = 0.5_f32;
        (base_idf * selectivity_multiplier).max(min_score)
    }

    /// Calculate proximity-based score for a numeric value.
    fn proximity_score(&self, value: f64) -> f32 {
        match (self.lower_bound, self.upper_bound) {
            (Some(lower), Some(upper)) => {
                // For bounded ranges, calculate distance from range center
                let center = (lower + upper) / 2.0;
                let distance_from_center = (value - center).abs();
                let max_distance = (upper - lower) / 2.0;

                if max_distance > 0.0 {
                    // Values closer to center get higher scores
                    (1.0 + (1.0 - (distance_from_center / max_distance)).max(0.0)) as f32
                } else {
                    1.0_f32
                }
            }
            (Some(bound), None) | (None, Some(bound)) => {
                // For single-bounded ranges, closer to bound is better
                let distance = (value - bound).abs();
                (1.0 + (1.0 / (distance + 1.0)).min(1.0)) as f32
            }
            (None, None) => {
                // Unbounded range - constant score
                1.0_f32
            }
        }
    }
}

impl Scorer for RangeScorer {
    fn score(&self, _doc_id: u64, value: f32, _field_length: Option<f32>) -> f32 {
        let idf = self.range_idf();

        // If value is provided (non-zero), use it for proximity calculation
        // Otherwise use a default score
        let proximity = if value != 0.0 {
            self.proximity_score(value as f64)
        } else {
            // Default proximity score when no value is provided
            1.5
        };

        // Final score combines IDF with proximity bonus
        let score = self.boost * idf * proximity;

        // Ensure score is never NaN or infinite
        if score.is_nan() || score.is_infinite() {
            return self.boost * idf * 1.5; // Return reasonable default
        }

        score
    }

    fn boost(&self) -> f32 {
        self.boost
    }

    fn set_boost(&mut self, boost: f32) {
        self.boost = boost;
    }

    fn max_score(&self) -> f32 {
        if self.matching_docs == 0 {
            return 0.0;
        }

        let idf = self.range_idf();
        let max_proximity = 2.0; // Maximum proximity bonus
        self.boost * idf * max_proximity
    }

    fn name(&self) -> &'static str {
        "RangeScorer"
    }
}

impl Query for NumericRangeQuery {
    fn matcher(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Matcher>> {
        // Try to use BKD Tree if available
        if let Some(bkd_tree) = reader.get_bkd_tree(&self.field)? {
            let min_value = self.min_f64();
            let max_value = self.max_f64();

            let doc_ids = bkd_tree.range_search(min_value, max_value);
            return Ok(Box::new(PreComputedMatcher::new(doc_ids)));
        }

        // Fallback: scan all documents and filter by numeric range
        let mut matching_docs = Vec::new();
        let max_doc = reader.max_doc();

        for doc_id in 0..max_doc {
            if let Ok(Some(doc)) = reader.document(doc_id)
                && let Some(field_value) = doc.get_field(&self.field)
            {
                let numeric_value = match &field_value.value {
                    crate::lexical::document::field::FieldValue::Float(f) => Some(*f),
                    crate::lexical::document::field::FieldValue::Integer(i) => Some(*i as f64),
                    // WORKAROUND: Parse text values as numbers (needed because stored docs lose type info)
                    crate::lexical::document::field::FieldValue::Text(s) => s.parse::<f64>().ok(),
                    _ => None,
                };

                if let Some(numeric_value) = numeric_value
                    && self.contains_numeric(numeric_value)
                {
                    matching_docs.push(doc_id);
                }
            }
        }

        Ok(Box::new(PreComputedMatcher::new(matching_docs)))
    }

    fn scorer(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Scorer>> {
        let total_docs = reader.doc_count();
        if total_docs == 0 {
            return Ok(Box::new(BM25Scorer::new(0, 0, 0, 1.0, 1, self.boost)));
        }

        // Count actual matching documents for accurate scoring
        let matching_docs = self.count_matching_documents(reader)?;

        // Create specialized range scorer with actual statistics
        Ok(Box::new(RangeScorer::new(
            self.min_f64(),
            self.max_f64(),
            matching_docs,
            total_docs,
            self.boost,
        )))
    }

    fn boost(&self) -> f32 {
        self.boost
    }

    fn set_boost(&mut self, boost: f32) {
        self.boost = boost;
    }

    fn description(&self) -> String {
        format!(
            "NumericRangeQuery(field:{}, type:{:?}, lower:{:?}, upper:{:?})",
            self.field, self.numeric_type, self.lower_bound, self.upper_bound
        )
    }

    fn clone_box(&self) -> Box<dyn Query> {
        Box::new(self.clone())
    }

    fn is_empty(&self, _reader: &dyn LexicalIndexReader) -> Result<bool> {
        Ok(false)
    }

    fn cost(&self, _reader: &dyn LexicalIndexReader) -> Result<u64> {
        Ok(100) // Numeric range queries are efficient
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Optimized matcher for numeric range queries.
#[derive(Debug)]
pub struct NumericRangeMatcher {
    /// Lower bound as bytes.
    lower_bound: Option<Vec<u8>>,
    /// Upper bound as bytes.
    upper_bound: Option<Vec<u8>>,
    /// Whether lower bound is inclusive.
    lower_inclusive: bool,
    /// Whether upper bound is inclusive.
    upper_inclusive: bool,
    /// Current document ID.
    current_doc: u64,
    /// Boost factor for scoring.
    #[allow(dead_code)]
    boost: f32,
    /// Whether we've reached the end.
    exhausted: bool,
}

impl NumericRangeMatcher {
    /// Create a new numeric range matcher.
    pub fn new(
        lower_bound: Option<Vec<u8>>,
        upper_bound: Option<Vec<u8>>,
        lower_inclusive: bool,
        upper_inclusive: bool,
        boost: f32,
    ) -> Self {
        NumericRangeMatcher {
            lower_bound,
            upper_bound,
            lower_inclusive,
            upper_inclusive,
            current_doc: 0,
            boost,
            exhausted: true, // Will be set to false when actual matching is implemented
        }
    }

    /// Check if an encoded value falls within the range.
    pub fn contains_encoded(&self, encoded_value: &[u8]) -> bool {
        // Check lower bound
        if let Some(ref lower) = self.lower_bound {
            let cmp = encoded_value.cmp(lower);
            if self.lower_inclusive {
                if cmp == std::cmp::Ordering::Less {
                    return false;
                }
            } else if cmp != std::cmp::Ordering::Greater {
                return false;
            }
        }

        // Check upper bound
        if let Some(ref upper) = self.upper_bound {
            let cmp = encoded_value.cmp(upper);
            if self.upper_inclusive {
                if cmp == std::cmp::Ordering::Greater {
                    return false;
                }
            } else if cmp != std::cmp::Ordering::Less {
                return false;
            }
        }

        true
    }
}

impl Matcher for NumericRangeMatcher {
    fn doc_id(&self) -> u64 {
        if self.exhausted {
            u64::MAX
        } else {
            self.current_doc
        }
    }

    fn next(&mut self) -> Result<bool> {
        // Implementation would iterate through document postings
        // and check if numeric values fall within range
        Ok(false)
    }

    fn skip_to(&mut self, _target: u64) -> Result<bool> {
        // Implementation would skip to target document
        // and check subsequent documents
        Ok(false)
    }

    fn cost(&self) -> u64 {
        // Numeric range queries are typically efficient
        100
    }

    fn is_exhausted(&self) -> bool {
        self.exhausted
    }
}

/// A specialized matcher for numeric range queries that filters documents.
#[derive(Debug)]
pub struct NumericRangeFilterMatcher {
    /// The numeric range query for reference.
    query: NumericRangeQuery,
    /// The underlying AllMatcher.
    all_matcher: crate::lexical::index::inverted::query::matcher::AllMatcher,
    /// Reader reference for document access.
    reader: Option<std::sync::Arc<dyn LexicalIndexReader>>,
}

impl NumericRangeFilterMatcher {
    /// Create a new numeric range filter matcher.
    pub fn new(query: NumericRangeQuery, reader: std::sync::Arc<dyn LexicalIndexReader>) -> Self {
        let max_doc = reader.max_doc();
        NumericRangeFilterMatcher {
            query,
            all_matcher: crate::lexical::index::inverted::query::matcher::AllMatcher::new(max_doc),
            reader: Some(reader),
        }
    }

    /// Check if the current document matches the numeric range.
    fn check_current_document(&self) -> bool {
        if let Some(ref reader) = self.reader {
            let doc_id = self.all_matcher.doc_id();
            if doc_id == u64::MAX {
                return false;
            }

            // Get the document
            if let Ok(Some(doc)) = reader.document(doc_id) {
                // Get the field value
                if let Some(field_value) = doc.get_field(&self.query.field) {
                    // Check if it's a numeric field and extract the value
                    let numeric_value = match &field_value.value {
                        crate::lexical::document::field::FieldValue::Float(f) => *f,
                        crate::lexical::document::field::FieldValue::Integer(i) => *i as f64,
                        _ => return false, // Not a numeric field
                    };

                    // Check if the value is within range
                    return self.query.contains_numeric(numeric_value);
                }
            }
        }

        false
    }

    /// Advance to the next document that matches the range.
    fn advance_to_next_matching(&mut self) -> Result<bool> {
        loop {
            // If exhausted, return false
            if self.all_matcher.is_exhausted() {
                return Ok(false);
            }

            // Check if current document matches
            if self.check_current_document() {
                return Ok(true);
            }

            // Move to next document
            if !self.all_matcher.next()? {
                return Ok(false);
            }
        }
    }
}

impl Matcher for NumericRangeFilterMatcher {
    fn doc_id(&self) -> u64 {
        self.all_matcher.doc_id()
    }

    fn next(&mut self) -> Result<bool> {
        // Move to next document in AllMatcher
        if !self.all_matcher.next()? {
            return Ok(false);
        }

        // Find the next matching document
        self.advance_to_next_matching()
    }

    fn skip_to(&mut self, target: u64) -> Result<bool> {
        // Skip AllMatcher to target
        if !self.all_matcher.skip_to(target)? {
            return Ok(false);
        }

        // Find the next matching document from target
        self.advance_to_next_matching()
    }

    fn cost(&self) -> u64 {
        self.all_matcher.cost()
    }

    fn is_exhausted(&self) -> bool {
        self.all_matcher.is_exhausted()
    }
}

/// Specialized datetime range query for optimal performance.
#[derive(Debug, Clone)]
pub struct DateTimeRangeQuery {
    /// The field to search in.
    field: String,
    /// Lower bound as timestamp.
    lower_bound: Option<i64>,
    /// Upper bound as timestamp.
    upper_bound: Option<i64>,
    /// Whether lower bound is inclusive.
    lower_inclusive: bool,
    /// Whether upper bound is inclusive.
    upper_inclusive: bool,
    /// The boost factor for this query.
    boost: f32,
}

impl DateTimeRangeQuery {
    /// Create a new datetime range query.
    pub fn new<S: Into<String>>(
        field: S,
        lower: Option<DateTime<Utc>>,
        upper: Option<DateTime<Utc>>,
        lower_inclusive: bool,
        upper_inclusive: bool,
    ) -> Self {
        DateTimeRangeQuery {
            field: field.into(),
            lower_bound: lower.map(|dt| dt.timestamp()),
            upper_bound: upper.map(|dt| dt.timestamp()),
            lower_inclusive,
            upper_inclusive,
            boost: 1.0,
        }
    }

    /// Create a datetime range with both bounds inclusive.
    pub fn between<S: Into<String>>(field: S, start: DateTime<Utc>, end: DateTime<Utc>) -> Self {
        Self::new(field, Some(start), Some(end), true, true)
    }

    /// Create a query for dates after the given datetime.
    pub fn after<S: Into<String>>(field: S, datetime: DateTime<Utc>) -> Self {
        Self::new(field, Some(datetime), None, false, true)
    }

    /// Create a query for dates on or after the given datetime.
    pub fn on_or_after<S: Into<String>>(field: S, datetime: DateTime<Utc>) -> Self {
        Self::new(field, Some(datetime), None, true, true)
    }

    /// Create a query for dates before the given datetime.
    pub fn before<S: Into<String>>(field: S, datetime: DateTime<Utc>) -> Self {
        Self::new(field, None, Some(datetime), true, false)
    }

    /// Create a query for dates on or before the given datetime.
    pub fn on_or_before<S: Into<String>>(field: S, datetime: DateTime<Utc>) -> Self {
        Self::new(field, None, Some(datetime), true, true)
    }

    /// Set the boost factor for this query.
    pub fn with_boost(mut self, boost: f32) -> Self {
        self.boost = boost;
        self
    }

    /// Get the field name.
    pub fn field(&self) -> &str {
        &self.field
    }

    /// Check if a timestamp falls within the range.
    pub fn contains_timestamp(&self, timestamp: i64) -> bool {
        // Check lower bound
        if let Some(lower) = self.lower_bound {
            if self.lower_inclusive {
                if timestamp < lower {
                    return false;
                }
            } else if timestamp <= lower {
                return false;
            }
        }

        // Check upper bound
        if let Some(upper) = self.upper_bound {
            if self.upper_inclusive {
                if timestamp > upper {
                    return false;
                }
            } else if timestamp >= upper {
                return false;
            }
        }

        true
    }

    /// Check if a datetime falls within the range.
    pub fn contains_datetime(&self, datetime: &DateTime<Utc>) -> bool {
        self.contains_timestamp(datetime.timestamp())
    }
}

impl Query for DateTimeRangeQuery {
    fn matcher(&self, _reader: &dyn LexicalIndexReader) -> Result<Box<dyn Matcher>> {
        Ok(Box::new(DateTimeRangeMatcher::new(
            self.lower_bound,
            self.upper_bound,
            self.lower_inclusive,
            self.upper_inclusive,
            self.boost,
        )))
    }

    fn scorer(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Scorer>> {
        // Range queries typically match fewer documents than term queries
        let total_docs = reader.doc_count();
        let estimated_doc_freq = total_docs / 10; // Estimate: 10% of docs might match a range
        let estimated_term_freq = estimated_doc_freq * 2; // Average 2 occurrences per doc

        Ok(Box::new(BM25Scorer::new(
            estimated_doc_freq.max(1),
            estimated_term_freq.max(1),
            total_docs,
            10.0, // Estimated average field length
            total_docs,
            self.boost,
        )))
    }

    fn boost(&self) -> f32 {
        self.boost
    }

    fn set_boost(&mut self, boost: f32) {
        self.boost = boost;
    }

    fn description(&self) -> String {
        format!(
            "DateTimeRangeQuery(field:{}, lower:{:?}, upper:{:?})",
            self.field, self.lower_bound, self.upper_bound
        )
    }

    fn clone_box(&self) -> Box<dyn Query> {
        Box::new(self.clone())
    }

    fn is_empty(&self, _reader: &dyn LexicalIndexReader) -> Result<bool> {
        Ok(false)
    }

    fn cost(&self, _reader: &dyn LexicalIndexReader) -> Result<u64> {
        Ok(100) // DateTime range queries are efficient
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Optimized matcher for datetime range queries.
#[derive(Debug)]
pub struct DateTimeRangeMatcher {
    /// Lower bound as timestamp.
    lower_bound: Option<i64>,
    /// Upper bound as timestamp.
    upper_bound: Option<i64>,
    /// Whether lower bound is inclusive.
    lower_inclusive: bool,
    /// Whether upper bound is inclusive.
    upper_inclusive: bool,
    /// Current document ID.
    current_doc: u64,
    /// Boost factor for scoring.
    #[allow(dead_code)]
    boost: f32,
    /// Whether we've reached the end.
    exhausted: bool,
}

impl DateTimeRangeMatcher {
    /// Create a new datetime range matcher.
    pub fn new(
        lower_bound: Option<i64>,
        upper_bound: Option<i64>,
        lower_inclusive: bool,
        upper_inclusive: bool,
        boost: f32,
    ) -> Self {
        DateTimeRangeMatcher {
            lower_bound,
            upper_bound,
            lower_inclusive,
            upper_inclusive,
            current_doc: 0,
            boost,
            exhausted: true, // Will be set to false when actual matching is implemented
        }
    }

    /// Check if a timestamp falls within the range.
    pub fn contains_timestamp(&self, timestamp: i64) -> bool {
        // Check lower bound
        if let Some(lower) = self.lower_bound {
            if self.lower_inclusive {
                if timestamp < lower {
                    return false;
                }
            } else if timestamp <= lower {
                return false;
            }
        }

        // Check upper bound
        if let Some(upper) = self.upper_bound {
            if self.upper_inclusive {
                if timestamp > upper {
                    return false;
                }
            } else if timestamp >= upper {
                return false;
            }
        }

        true
    }
}

impl Matcher for DateTimeRangeMatcher {
    fn doc_id(&self) -> u64 {
        if self.exhausted {
            u64::MAX
        } else {
            self.current_doc
        }
    }

    fn next(&mut self) -> Result<bool> {
        // Implementation would iterate through document postings
        // and check if datetime values fall within range
        Ok(false)
    }

    fn skip_to(&mut self, _target: u64) -> Result<bool> {
        // Implementation would skip to target document
        // and check subsequent documents
        Ok(false)
    }

    fn cost(&self) -> u64 {
        // DateTime range queries are typically efficient
        100
    }

    fn is_exhausted(&self) -> bool {
        self.exhausted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bound_contains_lower() {
        let bound = Bound::Included("hello".to_string());
        assert!(bound.contains_lower(&"hello".to_string()));
        assert!(bound.contains_lower(&"world".to_string()));
        assert!(!bound.contains_lower(&"apple".to_string()));

        let bound = Bound::Excluded("hello".to_string());
        assert!(!bound.contains_lower(&"hello".to_string()));
        assert!(bound.contains_lower(&"world".to_string()));
        assert!(!bound.contains_lower(&"apple".to_string()));

        let bound: Bound<String> = Bound::Unbounded;
        assert!(bound.contains_lower(&"anything".to_string()));
    }

    #[test]
    fn test_bound_contains_upper() {
        let bound = Bound::Included("hello".to_string());
        assert!(bound.contains_upper(&"hello".to_string()));
        assert!(!bound.contains_upper(&"world".to_string()));
        assert!(bound.contains_upper(&"apple".to_string()));

        let bound = Bound::Excluded("hello".to_string());
        assert!(!bound.contains_upper(&"hello".to_string()));
        assert!(!bound.contains_upper(&"world".to_string()));
        assert!(bound.contains_upper(&"apple".to_string()));

        let bound: Bound<String> = Bound::Unbounded;
        assert!(bound.contains_upper(&"anything".to_string()));
    }

    #[test]
    fn test_range_query_creation() {
        let query = RangeQuery::new(
            "field",
            Some("apple".to_string()),
            Some("zebra".to_string()),
        );

        assert_eq!(query.field(), "field");
        assert_eq!(*query.lower_bound(), Bound::Included("apple".to_string()));
        assert_eq!(*query.upper_bound(), Bound::Included("zebra".to_string()));
        assert_eq!(query.boost(), 1.0);
    }

    #[test]
    fn test_range_query_contains() {
        let query = RangeQuery::new(
            "field",
            Some("apple".to_string()),
            Some("zebra".to_string()),
        );

        assert!(query.contains("apple"));
        assert!(query.contains("hello"));
        assert!(query.contains("zebra"));
        assert!(!query.contains("aardvark"));
        assert!(!query.contains("zoo"));
    }

    #[test]
    fn test_range_query_greater_than() {
        let query = RangeQuery::greater_than("field", "hello".to_string());

        assert!(!query.contains("hello"));
        assert!(query.contains("world"));
        assert!(!query.contains("apple"));
    }

    #[test]
    fn test_range_query_less_than_or_equal() {
        let query = RangeQuery::less_than_or_equal("field", "hello".to_string());

        assert!(query.contains("hello"));
        assert!(!query.contains("world"));
        assert!(query.contains("apple"));
    }

    #[test]
    fn test_range_query_with_boost() {
        let query = RangeQuery::new("field", None, None).with_boost(2.5);

        assert_eq!(query.boost(), 2.5);
    }

    #[test]
    fn test_numeric_range_query_creation() {
        let query = NumericRangeQuery::i64_range("price", Some(100), Some(200));

        assert_eq!(query.field(), "price");
        assert_eq!(query.numeric_type(), NumericType::Integer);
        assert!(query.contains_numeric(150.0));
        assert!(!query.contains_numeric(50.0));
        assert!(!query.contains_numeric(250.0));
    }

    #[test]
    fn test_numeric_range_query_f64() {
        let query = NumericRangeQuery::f64_range("price", Some(99.99), Some(199.99));

        assert!(query.contains_numeric(150.0));
        assert!(query.contains_numeric(99.99));
        assert!(query.contains_numeric(199.99));
        assert!(!query.contains_numeric(99.98));
        assert!(!query.contains_numeric(200.0));
    }

    #[test]
    fn test_numeric_range_query_greater_than() {
        let query = NumericRangeQuery::greater_than("rating", NumericType::Integer, 3.0);

        assert!(query.contains_numeric(4.0));
        assert!(query.contains_numeric(5.0));
        assert!(!query.contains_numeric(3.0));
        assert!(!query.contains_numeric(2.0));
    }

    #[test]
    fn test_numeric_range_query_less_than_or_equal() {
        let query = NumericRangeQuery::less_than_or_equal("rating", NumericType::Integer, 3.0);

        assert!(query.contains_numeric(3.0));
        assert!(query.contains_numeric(2.0));
        assert!(query.contains_numeric(1.0));
        assert!(!query.contains_numeric(4.0));
    }

    #[test]
    fn test_datetime_range_query_creation() {
        use chrono::{TimeZone, Utc};

        let start = Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap();
        let end = Utc.with_ymd_and_hms(2023, 12, 31, 23, 59, 59).unwrap();

        let query = DateTimeRangeQuery::between("created_at", start, end);

        assert_eq!(query.field(), "created_at");
        assert!(query.contains_datetime(&Utc.with_ymd_and_hms(2023, 6, 15, 12, 0, 0).unwrap()));
        assert!(query.contains_datetime(&start));
        assert!(query.contains_datetime(&end));
        assert!(!query.contains_datetime(&Utc.with_ymd_and_hms(2022, 12, 31, 23, 59, 59).unwrap()));
        assert!(!query.contains_datetime(&Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap()));
    }

    #[test]
    fn test_datetime_range_query_after() {
        use chrono::{TimeZone, Utc};

        let threshold = Utc.with_ymd_and_hms(2023, 6, 15, 12, 0, 0).unwrap();
        let query = DateTimeRangeQuery::after("updated_at", threshold);

        assert!(query.contains_datetime(&Utc.with_ymd_and_hms(2023, 6, 15, 12, 0, 1).unwrap()));
        assert!(query.contains_datetime(&Utc.with_ymd_and_hms(2023, 12, 31, 23, 59, 59).unwrap()));
        assert!(!query.contains_datetime(&threshold));
        assert!(!query.contains_datetime(&Utc.with_ymd_and_hms(2023, 6, 15, 11, 59, 59).unwrap()));
    }

    #[test]
    fn test_datetime_range_query_on_or_before() {
        use chrono::{TimeZone, Utc};

        let threshold = Utc.with_ymd_and_hms(2023, 6, 15, 12, 0, 0).unwrap();
        let query = DateTimeRangeQuery::on_or_before("updated_at", threshold);

        assert!(query.contains_datetime(&threshold));
        assert!(query.contains_datetime(&Utc.with_ymd_and_hms(2023, 6, 15, 11, 59, 59).unwrap()));
        assert!(query.contains_datetime(&Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap()));
        assert!(!query.contains_datetime(&Utc.with_ymd_and_hms(2023, 6, 15, 12, 0, 1).unwrap()));
    }

    #[test]
    fn test_numeric_encoding() {
        // Test that encoding preserves ordering for different numeric types
        let values = [1.0, 2.0, 3.0, 100.0, 1000.0];

        for &numeric_type in &[
            NumericType::Integer,
            NumericType::Integer,
            NumericType::Float,
            NumericType::Float,
        ] {
            let encoded_values: Vec<_> = values
                .iter()
                .map(|&v| NumericRangeQuery::encode_numeric(v, numeric_type))
                .collect();

            // Check that encoded values maintain ordering
            for i in 1..encoded_values.len() {
                assert!(
                    encoded_values[i - 1] < encoded_values[i],
                    "Ordering not preserved for {numeric_type:?}"
                );
            }
        }
    }

    #[test]
    fn test_query_costs() {
        let string_range = RangeQuery::new("field", Some("a".to_string()), Some("z".to_string()));
        let numeric_range = NumericRangeQuery::i64_range("price", Some(0), Some(100));
        let datetime_range =
            DateTimeRangeQuery::between("created_at", chrono::Utc::now(), chrono::Utc::now());

        // These are placeholder implementations, but we can test the cost method exists
        assert!(string_range.cost(&EmptyReader).unwrap() > 0);
        assert!(numeric_range.cost(&EmptyReader).unwrap() > 0);
        assert!(datetime_range.cost(&EmptyReader).unwrap() > 0);

        // Numeric and datetime queries should be more efficient
        assert!(
            numeric_range.cost(&EmptyReader).unwrap() < string_range.cost(&EmptyReader).unwrap()
        );
        assert!(
            datetime_range.cost(&EmptyReader).unwrap() < string_range.cost(&EmptyReader).unwrap()
        );
    }

    // Placeholder reader for testing
    #[derive(Debug)]
    struct EmptyReader;

    impl LexicalIndexReader for EmptyReader {
        fn doc_count(&self) -> u64 {
            0
        }
        fn max_doc(&self) -> u64 {
            0
        }
        fn is_deleted(&self, _doc_id: u64) -> bool {
            false
        }
        fn document(
            &self,
            _doc_id: u64,
        ) -> crate::error::Result<Option<crate::lexical::document::document::Document>> {
            Ok(None)
        }
        fn term_info(
            &self,
            _field: &str,
            _term: &str,
        ) -> crate::error::Result<Option<crate::lexical::reader::ReaderTermInfo>> {
            Ok(None)
        }
        fn postings(
            &self,
            _field: &str,
            _term: &str,
        ) -> crate::error::Result<Option<Box<dyn crate::lexical::reader::PostingIterator>>>
        {
            Ok(None)
        }
        fn field_stats(
            &self,
            _field: &str,
        ) -> crate::error::Result<Option<crate::lexical::reader::FieldStats>> {
            Ok(None)
        }
        fn close(&mut self) -> crate::error::Result<()> {
            Ok(())
        }
        fn is_closed(&self) -> bool {
            false
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }
}
