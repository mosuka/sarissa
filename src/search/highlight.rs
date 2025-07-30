//! Text highlighting functionality for search results.

use std::collections::HashSet;
use std::ops::Range;

use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::analysis::{Analyzer, StandardAnalyzer, Token};
use crate::error::Result;
use crate::query::Query;

/// Configuration for text highlighting.
#[derive(Debug, Clone)]
pub struct HighlightConfig {
    /// HTML tag to wrap highlighted terms (e.g., "mark", "em", "strong").
    pub tag: String,
    /// CSS class to add to highlight tags.
    pub css_class: Option<String>,
    /// Maximum number of fragments to return.
    pub max_fragments: usize,
    /// Length of each fragment in characters.
    pub fragment_size: usize,
    /// Number of characters to overlap between fragments.
    pub fragment_overlap: usize,
    /// Separator between fragments.
    pub fragment_separator: String,
    /// Whether to return the entire field if no highlights are found.
    pub return_entire_field_if_no_highlight: bool,
    /// Maximum length of returned text.
    pub max_analyzed_chars: usize,
}

impl Default for HighlightConfig {
    fn default() -> Self {
        HighlightConfig {
            tag: "mark".to_string(),
            css_class: None,
            max_fragments: 5,
            fragment_size: 150,
            fragment_overlap: 20,
            fragment_separator: " ... ".to_string(),
            return_entire_field_if_no_highlight: false,
            max_analyzed_chars: 1_000_000,
        }
    }
}

impl HighlightConfig {
    /// Create a new highlight configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the HTML tag for highlighting.
    pub fn tag(mut self, tag: String) -> Self {
        self.tag = tag;
        self
    }

    /// Set the CSS class for highlight tags.
    pub fn css_class(mut self, css_class: String) -> Self {
        self.css_class = Some(css_class);
        self
    }

    /// Set the maximum number of fragments.
    pub fn max_fragments(mut self, max_fragments: usize) -> Self {
        self.max_fragments = max_fragments;
        self
    }

    /// Set the fragment size.
    pub fn fragment_size(mut self, fragment_size: usize) -> Self {
        self.fragment_size = fragment_size;
        self
    }

    /// Build the opening HTML tag.
    pub fn opening_tag(&self) -> String {
        if let Some(ref css_class) = self.css_class {
            format!("<{} class=\"{}\">", self.tag, css_class)
        } else {
            format!("<{}>", self.tag)
        }
    }

    /// Build the closing HTML tag.
    pub fn closing_tag(&self) -> String {
        format!("</{}>", self.tag)
    }
}

/// Represents a highlighted fragment of text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HighlightFragment {
    /// The highlighted text fragment.
    pub text: String,
    /// Starting position in the original text.
    pub start_offset: usize,
    /// Ending position in the original text.
    pub end_offset: usize,
    /// Score indicating relevance of this fragment.
    pub score: f32,
}

impl HighlightFragment {
    /// Create a new highlight fragment.
    pub fn new(text: String, start_offset: usize, end_offset: usize, score: f32) -> Self {
        HighlightFragment {
            text,
            start_offset,
            end_offset,
            score,
        }
    }
}

/// Represents highlight information for a field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldHighlight {
    /// Field name.
    pub field_name: String,
    /// Highlighted fragments.
    pub fragments: Vec<HighlightFragment>,
    /// Whether the entire field content was returned.
    pub is_entire_field: bool,
}

impl FieldHighlight {
    /// Create a new field highlight.
    pub fn new(field_name: String) -> Self {
        FieldHighlight {
            field_name,
            fragments: Vec::new(),
            is_entire_field: false,
        }
    }

    /// Add a fragment to this field highlight.
    pub fn add_fragment(&mut self, fragment: HighlightFragment) {
        self.fragments.push(fragment);
    }

    /// Get the best fragment (highest score).
    pub fn best_fragment(&self) -> Option<&HighlightFragment> {
        self.fragments.iter().max_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Combine all fragments into a single string.
    pub fn combined_text(&self, separator: &str) -> String {
        self.fragments
            .iter()
            .map(|f| &f.text)
            .cloned()
            .collect::<Vec<_>>()
            .join(separator)
    }
}

/// Text range with highlighting information.
#[derive(Debug, Clone)]
struct HighlightSpan {
    /// Range in the original text.
    range: Range<usize>,
    /// Whether this span should be highlighted.
    highlight: bool,
    /// Score for this span (higher = more important).
    score: f32,
}

impl HighlightSpan {
    fn new(range: Range<usize>, highlight: bool, score: f32) -> Self {
        HighlightSpan {
            range,
            highlight,
            score,
        }
    }
}

/// Main highlighter that can highlight text based on search queries.
pub struct Highlighter {
    /// Configuration for highlighting.
    config: HighlightConfig,
    /// Text analyzer for tokenization.
    analyzer: Box<dyn Analyzer>,
}

impl std::fmt::Debug for Highlighter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Highlighter")
            .field("config", &self.config)
            .field("analyzer", &"<dyn Analyzer>")
            .finish()
    }
}

impl Highlighter {
    /// Create a new highlighter.
    pub fn new(config: HighlightConfig) -> Self {
        Highlighter {
            config,
            analyzer: Box::new(StandardAnalyzer::new().unwrap()),
        }
    }

    /// Create a highlighter with a custom analyzer.
    pub fn with_analyzer(config: HighlightConfig, analyzer: Box<dyn Analyzer>) -> Self {
        Highlighter { config, analyzer }
    }

    /// Highlight text based on a query.
    pub fn highlight<Q: Query>(
        &self,
        query: &Q,
        field_name: &str,
        text: &str,
    ) -> Result<FieldHighlight> {
        // Limit text length
        let text = if text.len() > self.config.max_analyzed_chars {
            &text[..self.config.max_analyzed_chars]
        } else {
            text
        };

        // Extract terms from query
        let highlight_terms = self.extract_query_terms(query)?;

        if highlight_terms.is_empty() {
            return self.create_no_highlight_result(field_name, text);
        }

        // Find highlight spans
        let highlight_spans = self.find_highlight_spans(text, &highlight_terms)?;

        if highlight_spans.is_empty() {
            return self.create_no_highlight_result(field_name, text);
        }

        // Create fragments
        let fragments = self.create_fragments(text, &highlight_spans)?;

        let mut field_highlight = FieldHighlight::new(field_name.to_string());
        for fragment in fragments {
            field_highlight.add_fragment(fragment);
        }

        Ok(field_highlight)
    }

    /// Extract terms to highlight from a query.
    fn extract_query_terms<Q: Query>(&self, query: &Q) -> Result<HashSet<String>> {
        // This is a simplified implementation
        // In a real implementation, we would:
        // 1. Traverse the query tree
        // 2. Extract all terms, phrases, and patterns
        // 3. Handle different query types appropriately

        let mut terms = HashSet::new();

        // For now, we'll add some basic term extraction
        let description = query.description();

        // Simple heuristic: extract words from the description
        let words: Vec<&str> = description.split_whitespace().collect();
        for word in words {
            // Clean up the word (remove quotes, parentheses, etc.)
            let cleaned = word.trim_matches(|c: char| !c.is_alphanumeric());
            if !cleaned.is_empty() && cleaned.len() > 1 {
                terms.insert(cleaned.to_lowercase());
            }
        }

        Ok(terms)
    }

    /// Find highlight spans in text.
    fn find_highlight_spans(
        &self,
        text: &str,
        terms: &HashSet<String>,
    ) -> Result<Vec<HighlightSpan>> {
        let mut spans = Vec::new();

        // Tokenize the text
        let tokens = self.analyzer.analyze(text)?;
        let tokens: Vec<Token> = tokens.collect();

        // Find matching tokens
        for token in &tokens {
            if terms.contains(&token.text.to_lowercase()) {
                let score = self.calculate_term_score(&token.text, terms);
                spans.push(HighlightSpan::new(
                    token.start_offset..token.start_offset + token.text.len(),
                    true,
                    score,
                ));
            }
        }

        // Also find phrase matches (simple implementation)
        for term in terms {
            if term.contains(' ') {
                // This is a phrase
                if let Ok(regex) = Regex::new(&format!(r"(?i)\b{}\b", regex::escape(term))) {
                    for mat in regex.find_iter(text) {
                        spans.push(HighlightSpan::new(
                            mat.range(),
                            true,
                            2.0, // Phrases get higher score
                        ));
                    }
                }
            }
        }

        // Sort spans by position
        spans.sort_by_key(|span| span.range.start);

        // Merge overlapping spans
        let merged_spans = self.merge_overlapping_spans(spans);

        Ok(merged_spans)
    }

    /// Calculate score for a term match.
    fn calculate_term_score(&self, term: &str, all_terms: &HashSet<String>) -> f32 {
        // Simple scoring based on term length and rarity
        let base_score = 1.0;
        let length_bonus = (term.len() as f32).log2() * 0.1;
        let rarity_bonus = 1.0 / (all_terms.len() as f32).sqrt();

        base_score + length_bonus + rarity_bonus
    }

    /// Merge overlapping highlight spans.
    fn merge_overlapping_spans(&self, mut spans: Vec<HighlightSpan>) -> Vec<HighlightSpan> {
        if spans.is_empty() {
            return spans;
        }

        let mut merged = Vec::new();
        let mut current = spans.remove(0);

        for span in spans {
            if span.range.start <= current.range.end {
                // Overlapping spans - merge them
                current.range.end = current.range.end.max(span.range.end);
                current.score = current.score.max(span.score);
            } else {
                // Non-overlapping - push current and start new one
                merged.push(current);
                current = span;
            }
        }

        merged.push(current);
        merged
    }

    /// Create text fragments with highlighting.
    fn create_fragments(
        &self,
        text: &str,
        spans: &[HighlightSpan],
    ) -> Result<Vec<HighlightFragment>> {
        let mut fragments = Vec::new();

        // Group spans into fragments
        let fragment_groups = self.group_spans_into_fragments(text, spans);

        for (group_spans, fragment_range) in fragment_groups {
            let fragment_text = self.apply_highlighting(
                &text[fragment_range.clone()],
                &group_spans,
                fragment_range.start,
            )?;
            let score = group_spans.iter().map(|s| s.score).sum::<f32>() / group_spans.len() as f32;

            fragments.push(HighlightFragment::new(
                fragment_text,
                fragment_range.start,
                fragment_range.end,
                score,
            ));
        }

        // Sort fragments by score (highest first)
        fragments.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Limit number of fragments
        fragments.truncate(self.config.max_fragments);

        Ok(fragments)
    }

    /// Group highlight spans into fragments.
    fn group_spans_into_fragments(
        &self,
        text: &str,
        spans: &[HighlightSpan],
    ) -> Vec<(Vec<HighlightSpan>, Range<usize>)> {
        let mut groups = Vec::new();
        let text_len = text.len();

        for span in spans {
            // Calculate fragment boundaries around this span
            let fragment_start = span
                .range
                .start
                .saturating_sub(self.config.fragment_size / 2);
            let fragment_end = (span.range.end + self.config.fragment_size / 2).min(text_len);

            // Adjust to word boundaries
            let fragment_start = self.find_word_boundary(text, fragment_start, false);
            let fragment_end = self.find_word_boundary(text, fragment_end, true);

            let fragment_range = fragment_start..fragment_end;

            // Find all spans that overlap with this fragment
            let mut group_spans = Vec::new();
            for candidate_span in spans {
                if candidate_span.range.start < fragment_range.end
                    && candidate_span.range.end > fragment_range.start
                {
                    // Adjust span coordinates relative to fragment
                    let relative_start = candidate_span
                        .range
                        .start
                        .saturating_sub(fragment_range.start);
                    let relative_end =
                        (candidate_span.range.end - fragment_range.start).min(fragment_range.len());

                    group_spans.push(HighlightSpan::new(
                        relative_start..relative_end,
                        candidate_span.highlight,
                        candidate_span.score,
                    ));
                }
            }

            if !group_spans.is_empty() {
                groups.push((group_spans, fragment_range));
            }
        }

        // Remove duplicate fragments (simple deduplication)
        groups.dedup_by(|(_, range1), (_, range2)| {
            (range1.start as i32 - range2.start as i32).abs() < 50
        });

        groups
    }

    /// Find word boundary near a position.
    fn find_word_boundary(&self, text: &str, pos: usize, forward: bool) -> usize {
        let chars: Vec<char> = text.chars().collect();
        let mut current_pos = pos.min(chars.len());

        if forward {
            // Find next word boundary
            while current_pos < chars.len() && chars[current_pos].is_alphanumeric() {
                current_pos += 1;
            }
        } else {
            // Find previous word boundary
            while current_pos > 0 && chars[current_pos - 1].is_alphanumeric() {
                current_pos -= 1;
            }
        }

        current_pos
    }

    /// Apply highlighting markup to text.
    fn apply_highlighting(
        &self,
        text: &str,
        spans: &[HighlightSpan],
        _offset: usize,
    ) -> Result<String> {
        if spans.is_empty() {
            return Ok(text.to_string());
        }

        let mut result = String::new();
        let mut last_pos = 0;

        for span in spans {
            if span.highlight {
                // Add text before the highlight
                result.push_str(&text[last_pos..span.range.start]);

                // Add highlighted text
                result.push_str(&self.config.opening_tag());
                result.push_str(&text[span.range.clone()]);
                result.push_str(&self.config.closing_tag());

                last_pos = span.range.end;
            }
        }

        // Add remaining text
        if last_pos < text.len() {
            result.push_str(&text[last_pos..]);
        }

        Ok(result)
    }

    /// Create result when no highlights are found.
    fn create_no_highlight_result(&self, field_name: &str, text: &str) -> Result<FieldHighlight> {
        let mut field_highlight = FieldHighlight::new(field_name.to_string());

        if self.config.return_entire_field_if_no_highlight {
            field_highlight.is_entire_field = true;
            field_highlight.add_fragment(HighlightFragment::new(
                text.to_string(),
                0,
                text.len(),
                0.0,
            ));
        }

        Ok(field_highlight)
    }
}

/// Utility for creating highlighted snippets without full query analysis.
#[derive(Debug)]
pub struct SimpleHighlighter {
    config: HighlightConfig,
}

impl SimpleHighlighter {
    /// Create a new simple highlighter.
    pub fn new(config: HighlightConfig) -> Self {
        SimpleHighlighter { config }
    }

    /// Highlight specific terms in text.
    pub fn highlight_terms(&self, text: &str, terms: &[&str]) -> String {
        let mut result = text.to_string();

        // Sort terms by length (longest first) to avoid partial replacements
        let mut sorted_terms: Vec<&str> = terms.to_vec();
        sorted_terms.sort_by_key(|term| std::cmp::Reverse(term.len()));

        for term in sorted_terms {
            if !term.is_empty() {
                let pattern = format!(r"(?i)\b{}\b", regex::escape(term));
                if let Ok(regex) = Regex::new(&pattern) {
                    result = regex
                        .replace_all(&result, |caps: &regex::Captures| {
                            format!(
                                "{}{}{}",
                                self.config.opening_tag(),
                                &caps[0],
                                self.config.closing_tag()
                            )
                        })
                        .to_string();
                }
            }
        }

        result
    }

    /// Create a snippet of text around the first occurrence of any term.
    pub fn create_snippet(&self, text: &str, terms: &[&str], max_length: usize) -> String {
        if terms.is_empty() || text.is_empty() {
            return if text.len() <= max_length {
                text.to_string()
            } else {
                format!("{}...", &text[..max_length])
            };
        }

        // Find the first occurrence of any term
        let mut first_match_pos = None;
        for term in terms {
            if let Some(pos) = text.to_lowercase().find(&term.to_lowercase()) {
                if first_match_pos.is_none() || pos < first_match_pos.unwrap() {
                    first_match_pos = Some(pos);
                }
            }
        }

        if let Some(match_pos) = first_match_pos {
            // Create snippet around the match
            let start = match_pos.saturating_sub(max_length / 3);
            let end = (match_pos + max_length * 2 / 3).min(text.len());

            let mut snippet = text[start..end].to_string();

            // Add ellipsis if we truncated
            if start > 0 {
                snippet = format!("...{snippet}");
            }
            if end < text.len() {
                snippet = format!("{snippet}...");
            }

            snippet
        } else {
            // No matches found, return beginning of text
            if text.len() <= max_length {
                text.to_string()
            } else {
                format!("{}...", &text[..max_length])
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::TermQuery;

    #[test]
    fn test_highlight_config() {
        let config = HighlightConfig::new()
            .tag("em".to_string())
            .css_class("highlight".to_string())
            .max_fragments(3)
            .fragment_size(100);

        assert_eq!(config.tag, "em");
        assert_eq!(config.css_class, Some("highlight".to_string()));
        assert_eq!(config.max_fragments, 3);
        assert_eq!(config.fragment_size, 100);

        assert_eq!(config.opening_tag(), "<em class=\"highlight\">");
        assert_eq!(config.closing_tag(), "</em>");
    }

    #[test]
    fn test_highlight_fragment() {
        let fragment = HighlightFragment::new(
            "This is a <mark>test</mark> fragment".to_string(),
            10,
            50,
            1.5,
        );

        assert_eq!(fragment.text, "This is a <mark>test</mark> fragment");
        assert_eq!(fragment.start_offset, 10);
        assert_eq!(fragment.end_offset, 50);
        assert_eq!(fragment.score, 1.5);
    }

    #[test]
    fn test_field_highlight() {
        let mut field_highlight = FieldHighlight::new("content".to_string());

        field_highlight.add_fragment(HighlightFragment::new("fragment 1".to_string(), 0, 10, 1.0));
        field_highlight.add_fragment(HighlightFragment::new(
            "fragment 2".to_string(),
            20,
            30,
            2.0,
        ));

        assert_eq!(field_highlight.fragments.len(), 2);
        assert_eq!(field_highlight.best_fragment().unwrap().score, 2.0);
        assert_eq!(
            field_highlight.combined_text(" | "),
            "fragment 1 | fragment 2"
        );
    }

    #[test]
    fn test_simple_highlighter() {
        let config = HighlightConfig::default();
        let highlighter = SimpleHighlighter::new(config);

        let text = "This is a test document with some test content.";
        let terms = vec!["test", "content"];

        let highlighted = highlighter.highlight_terms(text, &terms);
        assert!(highlighted.contains("<mark>test</mark>"));
        assert!(highlighted.contains("<mark>content</mark>"));

        let snippet = highlighter.create_snippet(text, &terms, 30);
        assert!(snippet.len() <= 35); // Account for ellipsis
        assert!(snippet.contains("test"));
    }

    #[test]
    fn test_highlighter_extract_terms() {
        let config = HighlightConfig::default();
        let highlighter = Highlighter::new(config);

        let query = TermQuery::new("field", "search");
        let terms = highlighter.extract_query_terms(&query).unwrap();

        // Note: This is a simplified test since term extraction is basic
        assert!(!terms.is_empty());
    }

    #[test]
    fn test_merge_overlapping_spans() {
        let config = HighlightConfig::default();
        let highlighter = Highlighter::new(config);

        let spans = vec![
            HighlightSpan::new(0..5, true, 1.0),
            HighlightSpan::new(3..8, true, 1.5),
            HighlightSpan::new(10..15, true, 1.2),
        ];

        let merged = highlighter.merge_overlapping_spans(spans);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].range, 0..8);
        assert_eq!(merged[1].range, 10..15);
    }

    #[test]
    fn test_word_boundary_finding() {
        let config = HighlightConfig::default();
        let highlighter = Highlighter::new(config);

        let text = "The quick brown fox jumps";

        // Find word boundary before position 7 (middle of "quick")
        let boundary = highlighter.find_word_boundary(text, 7, false);
        assert_eq!(boundary, 4); // Start of "quick"

        // Find word boundary after position 7
        let boundary = highlighter.find_word_boundary(text, 7, true);
        assert_eq!(boundary, 9); // End of "quick"
    }
}
