//! Term dictionary enumeration API.
//!
//! This module provides traits and types for efficiently enumerating terms in the index,
//! similar to Lucene's Terms and TermsEnum.

use crate::error::Result;

/// Statistics about a term in the index.
#[derive(Debug, Clone)]
pub struct TermStats {
    /// The term text
    pub term: String,
    /// Number of documents containing this term
    pub doc_freq: u64,
    /// Total number of occurrences across all documents
    pub total_term_freq: u64,
}

/// Iterator over terms in a field's term dictionary.
///
/// This is similar to Lucene's TermsEnum, providing sequential access to
/// indexed terms in sorted order.
///
/// # Example (conceptual - not yet implemented)
///
/// ```ignore
/// let terms = reader.terms("content")?;
/// let mut terms_enum = terms.iterator()?;
///
/// while let Some(term) = terms_enum.next()? {
///     println!("Term: {}, DocFreq: {}", term.term, term.doc_freq);
/// }
/// ```ignore
pub trait TermsEnum: Send + Sync {
    /// Advance to the next term in the enumeration.
    ///
    /// Returns `None` when there are no more terms.
    fn next(&mut self) -> Result<Option<TermStats>>;

    /// Seek to the first term greater than or equal to the target.
    ///
    /// Returns `true` if an exact match was found, `false` if positioned
    /// at the next term greater than target, or error if no such term exists.
    ///
    /// This is useful for implementing prefix queries efficiently.
    fn seek(&mut self, target: &str) -> Result<bool>;

    /// Seek to the exact term.
    ///
    /// Returns `true` if the term exists, `false` otherwise.
    fn seek_exact(&mut self, term: &str) -> Result<bool>;

    /// Get the current term without advancing the iterator.
    ///
    /// Returns `None` if the iterator hasn't been advanced or is exhausted.
    fn current(&self) -> Option<&TermStats>;

    /// Get statistics for the current term.
    ///
    /// This is equivalent to `current()` but returns a copy.
    fn term_stats(&self) -> Option<TermStats> {
        self.current().cloned()
    }
}

/// Access to the term dictionary for a specific field.
///
/// This is similar to Lucene's Terms, representing all terms indexed in a field.
///
/// # Example (conceptual - not yet implemented)
///
/// ```ignore
/// let terms = reader.terms("content")?;
/// println!("Total terms: {}", terms.size());
///
/// // Iterate over all terms
/// let mut iter = terms.iterator()?;
/// while let Some(term) = iter.next()? {
///     println!("{}: {} docs", term.term, term.doc_freq);
/// }
///
/// // Or seek to specific position
/// let mut iter = terms.iterator()?;
/// if iter.seek("prefix")? {
///     println!("Found exact match");
/// }
/// ```ignore
pub trait Terms: Send + Sync {
    /// Get an iterator over all terms in this field.
    fn iterator(&self) -> Result<Box<dyn TermsEnum>>;

    /// Get the number of unique terms in this field.
    ///
    /// Returns `None` if the count is not available or too expensive to compute.
    fn size(&self) -> Option<u64>;

    /// Get the sum of document frequencies across all terms.
    ///
    /// This is the total number of term-document pairs.
    fn sum_doc_freq(&self) -> Option<u64>;

    /// Get the sum of total term frequencies across all terms.
    ///
    /// This is the total number of term occurrences in the index.
    fn sum_total_term_freq(&self) -> Option<u64>;

    /// Check if this field has term frequencies stored.
    fn has_freqs(&self) -> bool {
        true
    }

    /// Check if this field has positions stored.
    fn has_positions(&self) -> bool {
        false
    }

    /// Check if this field has offsets stored.
    fn has_offsets(&self) -> bool {
        false
    }

    /// Check if this field has payloads stored.
    fn has_payloads(&self) -> bool {
        false
    }
}

/// Extension trait for LexicalIndexReader to provide term dictionary access.
///
/// This will eventually be added to the LexicalIndexReader trait, but is defined
/// separately here to avoid breaking changes during development.
///
/// # Example (conceptual - not yet implemented)
///
/// ```ignore
/// use sarissa::lexical::terms::TermDictionaryAccess;
///
/// let reader = index.reader()?;
/// let terms = reader.terms("content")?;
/// let mut iter = terms.iterator()?;
///
/// while let Some(term_stats) = iter.next()? {
///     println!("{}: {} docs", term_stats.term, term_stats.doc_freq);
/// }
/// ```ignore
pub trait TermDictionaryAccess {
    /// Get access to the term dictionary for the specified field.
    ///
    /// Returns `None` if the field doesn't exist in the index.
    fn terms(&self, field: &str) -> Result<Option<Box<dyn Terms>>>;

    /// Check if a specific term exists in a field.
    ///
    /// This is a convenience method equivalent to:
    /// ```ignore
    /// reader.terms(field)?.and_then(|terms| {
    ///     let mut iter = terms.iterator()?;
    ///     iter.seek_exact(term)
    /// })
    /// ```ignore
    fn term_exists(&self, field: &str, term: &str) -> Result<bool> {
        if let Some(terms) = self.terms(field)? {
            let mut iter = terms.iterator()?;
            iter.seek_exact(term)
        } else {
            Ok(false)
        }
    }
}

// Implement TermsEnum for Box<dyn TermsEnum> to allow composition
impl TermsEnum for Box<dyn TermsEnum> {
    fn next(&mut self) -> Result<Option<TermStats>> {
        (**self).next()
    }

    fn seek(&mut self, target: &str) -> Result<bool> {
        (**self).seek(target)
    }

    fn seek_exact(&mut self, term: &str) -> Result<bool> {
        (**self).seek_exact(term)
    }

    fn current(&self) -> Option<&TermStats> {
        (**self).current()
    }
}

// TODO: Add automaton intersection support: terms.intersect(automaton)
// TODO: Add range query support: terms.range(min, max)

// ============================================================================
// Concrete Implementations for InvertedIndex
// ============================================================================

use crate::lexical::index::structures::dictionary::{HybridTermDictionary, TermInfo};
use std::sync::Arc;

/// Iterator over terms in a term dictionary.
pub struct InvertedIndexTermsEnum {
    /// Iterator over the dictionary entries
    terms: Vec<(String, TermInfo)>,
    /// Current position in the iterator
    position: usize,
    /// Current term stats (cached)
    current: Option<TermStats>,
}

impl InvertedIndexTermsEnum {
    /// Create a new terms enum for a field.
    ///
    /// This extracts all terms for the specified field from the term dictionary.
    /// Terms are stored in the format "field:term", so we filter by prefix.
    pub fn new(field: &str, dict: &Arc<HybridTermDictionary>) -> Self {
        let field_prefix = format!("{}:", field);
        let mut terms = Vec::new();

        // Iterate over all terms in the dictionary
        for (term, info) in dict.iter() {
            // Check if this term belongs to our field
            if let Some(term_text) = term.strip_prefix(&field_prefix) {
                terms.push((term_text.to_string(), info.clone()));
            }
        }

        // Terms are already sorted since HybridTermDictionary.iter() uses SortedTermDictionary
        InvertedIndexTermsEnum {
            terms,
            position: 0,
            current: None,
        }
    }
}

impl TermsEnum for InvertedIndexTermsEnum {
    fn next(&mut self) -> Result<Option<TermStats>> {
        if self.position >= self.terms.len() {
            self.current = None;
            return Ok(None);
        }

        let (term, info) = &self.terms[self.position];
        let stats = TermStats {
            term: term.clone(),
            doc_freq: info.doc_frequency,
            total_term_freq: info.total_frequency,
        };

        self.current = Some(stats.clone());
        self.position += 1;

        Ok(Some(stats))
    }

    fn seek(&mut self, target: &str) -> Result<bool> {
        // Binary search for the target term or the next term greater than it
        let result = self
            .terms
            .binary_search_by(|(term, _)| term.as_str().cmp(target));

        match result {
            Ok(index) => {
                // Exact match found
                self.position = index;
                let (term, info) = &self.terms[index];
                self.current = Some(TermStats {
                    term: term.clone(),
                    doc_freq: info.doc_frequency,
                    total_term_freq: info.total_frequency,
                });
                Ok(true)
            }
            Err(index) => {
                // No exact match; position at the next term
                self.position = index;
                if index < self.terms.len() {
                    let (term, info) = &self.terms[index];
                    self.current = Some(TermStats {
                        term: term.clone(),
                        doc_freq: info.doc_frequency,
                        total_term_freq: info.total_frequency,
                    });
                    Ok(false)
                } else {
                    self.current = None;
                    Ok(false)
                }
            }
        }
    }

    fn seek_exact(&mut self, term: &str) -> Result<bool> {
        // Binary search for exact match
        let result = self.terms.binary_search_by(|(t, _)| t.as_str().cmp(term));

        match result {
            Ok(index) => {
                self.position = index;
                let (term, info) = &self.terms[index];
                self.current = Some(TermStats {
                    term: term.clone(),
                    doc_freq: info.doc_frequency,
                    total_term_freq: info.total_frequency,
                });
                Ok(true)
            }
            Err(_) => {
                self.current = None;
                Ok(false)
            }
        }
    }

    fn current(&self) -> Option<&TermStats> {
        self.current.as_ref()
    }
}

/// Implementation of Terms trait for a specific field.
pub struct InvertedIndexTerms {
    field: String,
    dict: Arc<HybridTermDictionary>,
    // Cached statistics
    size: Option<u64>,
    sum_doc_freq: Option<u64>,
    sum_total_term_freq: Option<u64>,
}

impl InvertedIndexTerms {
    /// Create a new Terms instance for a field.
    pub fn new(field: &str, dict: Arc<HybridTermDictionary>) -> Self {
        let field_prefix = format!("{}:", field);

        // Calculate statistics by iterating through matching terms
        let mut size = 0u64;
        let mut sum_doc_freq = 0u64;
        let mut sum_total_term_freq = 0u64;

        for (term, info) in dict.iter() {
            if term.starts_with(&field_prefix) {
                size += 1;
                sum_doc_freq += info.doc_frequency;
                sum_total_term_freq += info.total_frequency;
            }
        }

        InvertedIndexTerms {
            field: field.to_string(),
            dict,
            size: Some(size),
            sum_doc_freq: Some(sum_doc_freq),
            sum_total_term_freq: Some(sum_total_term_freq),
        }
    }
}

impl Terms for InvertedIndexTerms {
    fn iterator(&self) -> Result<Box<dyn TermsEnum>> {
        Ok(Box::new(InvertedIndexTermsEnum::new(
            &self.field,
            &self.dict,
        )))
    }

    fn size(&self) -> Option<u64> {
        self.size
    }

    fn sum_doc_freq(&self) -> Option<u64> {
        self.sum_doc_freq
    }

    fn sum_total_term_freq(&self) -> Option<u64> {
        self.sum_total_term_freq
    }
}
