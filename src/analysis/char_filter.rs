//! Char filter implementations for text normalization.
//!
//! This module provides filters that pre-process the text string before it is
//! passed to the tokenizer. This allows for normalization operations like
//! Unicode normalization or regex replacement.
//!
//! # Available Filters
//!
//! - [`unicode_normalize::UnicodeNormalizationCharFilter`] - Unicode normalization (NFC, NFD, etc.)
//! - [`pattern_replace::PatternReplaceCharFilter`] - Regex-based replacement
//! - [`japanese_iteration_mark::JapaneseIterationMarkCharFilter`] - Japanese iteration mark normalization
//! - [`mapping::MappingCharFilter`] - Character mapping replacement
//!
//! # Examples
//!
//! ```
//! use sarissa::analysis::char_filter::CharFilter;
//! // use sarissa::analysis::char_filter::unicode_normalize::UnicodeNormalizationCharFilter;
//! // (Example usage to be added after implementation)
//! ```

/// Represents a change in the text, mapping a range in the original text
/// to a range in the new text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Transformation {
    pub original_start: usize,
    pub original_end: usize,
    pub new_start: usize,
    pub new_end: usize,
}

impl Transformation {
    pub fn new(
        original_start: usize,
        original_end: usize,
        new_start: usize,
        new_end: usize,
    ) -> Self {
        Self {
            original_start,
            original_end,
            new_start,
            new_end,
        }
    }
}

/// Trait for character filters that transform text before tokenization.
///
/// Implementations can modify the text content and returns the modified text
/// along with a list of transformations that occurred.
pub trait CharFilter: Send + Sync {
    /// Apply this filter to the input text.
    ///
    /// # Arguments
    ///
    /// * `input` - The input text to filter
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// - The filtered text.
    /// - A vector of `Transformation`s describing changes made.
    fn filter(&self, input: &str) -> (String, Vec<Transformation>);

    /// Get the name of this char filter.
    fn name(&self) -> &'static str;
}

pub mod japanese_iteration_mark;
pub mod mapping;
pub mod pattern_replace;
pub mod unicode_normalize;
