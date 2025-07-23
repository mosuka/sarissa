//! Matcher implementations for query execution.

use crate::error::Result;
use crate::index::reader::PostingIterator;
use std::collections::BinaryHeap;
use std::cmp::Ordering;
use std::fmt::Debug;

/// Trait for document matchers.
pub trait Matcher: Send + Debug {
    /// Get the current document ID.
    fn doc_id(&self) -> u64;

    /// Move to the next matching document.
    fn next(&mut self) -> Result<bool>;

    /// Skip to the first document >= target.
    fn skip_to(&mut self, target: u64) -> Result<bool>;

    /// Get the cost of iterating through this matcher.
    fn cost(&self) -> u64;

    /// Check if this matcher is exhausted.
    fn is_exhausted(&self) -> bool;
}

/// A matcher that matches no documents.
#[derive(Debug)]
pub struct EmptyMatcher {
    exhausted: bool,
}

impl EmptyMatcher {
    /// Create a new empty matcher.
    pub fn new() -> Self {
        EmptyMatcher { exhausted: true }
    }
}

impl Default for EmptyMatcher {
    fn default() -> Self {
        Self::new()
    }
}

impl Matcher for EmptyMatcher {
    fn doc_id(&self) -> u64 {
        u64::MAX
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

/// A matcher that matches all documents.
#[derive(Debug)]
pub struct AllMatcher {
    current_doc: u64,
    max_doc: u64,
}

impl AllMatcher {
    /// Create a new all matcher.
    pub fn new(max_doc: u64) -> Self {
        AllMatcher {
            current_doc: 0,
            max_doc,
        }
    }
}

impl Matcher for AllMatcher {
    fn doc_id(&self) -> u64 {
        if self.current_doc >= self.max_doc {
            u64::MAX
        } else {
            self.current_doc
        }
    }

    fn next(&mut self) -> Result<bool> {
        if self.current_doc >= self.max_doc {
            Ok(false)
        } else {
            self.current_doc += 1;
            Ok(self.current_doc < self.max_doc)
        }
    }

    fn skip_to(&mut self, target: u64) -> Result<bool> {
        if target >= self.max_doc {
            self.current_doc = self.max_doc;
            Ok(false)
        } else {
            self.current_doc = target;
            Ok(true)
        }
    }

    fn cost(&self) -> u64 {
        self.max_doc
    }

    fn is_exhausted(&self) -> bool {
        self.current_doc >= self.max_doc
    }
}

/// A matcher based on a posting iterator.
#[derive(Debug)]
pub struct PostingMatcher {
    posting_iter: Box<dyn PostingIterator>,
    exhausted: bool,
    cost: u64,
}

impl PostingMatcher {
    /// Create a new posting matcher.
    pub fn new(posting_iter: Box<dyn PostingIterator>) -> Self {
        let cost = posting_iter.cost();
        PostingMatcher {
            posting_iter,
            exhausted: false,
            cost,
        }
    }

    /// Create an exhausted posting matcher.
    pub fn exhausted() -> Self {
        // Create a dummy iterator that's already exhausted
        let doc_ids = vec![];
        let term_freqs = vec![];
        let posting_iter = Box::new(crate::index::reader::BasicPostingIterator::new(doc_ids, term_freqs).unwrap());
        PostingMatcher {
            posting_iter,
            exhausted: true,
            cost: 0,
        }
    }
}

impl Matcher for PostingMatcher {
    fn doc_id(&self) -> u64 {
        if self.exhausted {
            u64::MAX
        } else {
            self.posting_iter.doc_id()
        }
    }

    fn next(&mut self) -> Result<bool> {
        if self.exhausted {
            Ok(false)
        } else {
            let has_next = self.posting_iter.next()?;
            if !has_next {
                self.exhausted = true;
            }
            Ok(has_next)
        }
    }

    fn skip_to(&mut self, target: u64) -> Result<bool> {
        if self.exhausted {
            Ok(false)
        } else {
            let result = self.posting_iter.skip_to(target)?;
            if !result {
                self.exhausted = true;
            }
            Ok(result)
        }
    }

    fn cost(&self) -> u64 {
        self.cost
    }

    fn is_exhausted(&self) -> bool {
        self.exhausted
    }
}

/// A helper struct for tracking matchers in the disjunction heap.
#[derive(Debug)]
struct MatcherEntry {
    matcher: Box<dyn Matcher>,
}

impl PartialEq for MatcherEntry {
    fn eq(&self, other: &Self) -> bool {
        self.matcher.doc_id() == other.matcher.doc_id()
    }
}

impl Eq for MatcherEntry {}

impl PartialOrd for MatcherEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MatcherEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: lower doc IDs come first
        other.matcher.doc_id().cmp(&self.matcher.doc_id())
    }
}

/// A matcher that implements disjunction (OR) of multiple matchers.
#[derive(Debug)]
pub struct DisjunctionMatcher {
    /// Min-heap of active matchers, ordered by current doc_id.
    heap: BinaryHeap<MatcherEntry>,
    /// Current document ID.
    current_doc: u64,
    /// Whether this matcher is exhausted.
    exhausted: bool,
    /// Total cost estimate.
    cost: u64,
}

impl DisjunctionMatcher {
    /// Create a new disjunction matcher from multiple matchers.
    pub fn new(mut matchers: Vec<Box<dyn Matcher>>) -> Self {
        let mut heap = BinaryHeap::new();
        let mut cost = 0;

        // Initialize heap with all non-empty matchers
        for matcher in matchers.drain(..) {
            if !matcher.is_exhausted() {
                cost += matcher.cost();
                heap.push(MatcherEntry { matcher });
            }
        }

        let current_doc = heap.peek().map(|entry| entry.matcher.doc_id()).unwrap_or(u64::MAX);
        let exhausted = heap.is_empty();

        DisjunctionMatcher {
            heap,
            current_doc,
            exhausted,
            cost,
        }
    }

    /// Create an empty disjunction matcher.
    pub fn empty() -> Self {
        DisjunctionMatcher {
            heap: BinaryHeap::new(),
            current_doc: u64::MAX,
            exhausted: true,
            cost: 0,
        }
    }

    /// Advance to the next document, skipping duplicates.
    fn advance_to_next_doc(&mut self) -> Result<()> {
        if self.exhausted {
            return Ok(());
        }

        let current_doc = self.current_doc;

        // Advance all matchers that are at the current document
        let mut matchers_to_reinsert = Vec::new();
        while let Some(entry) = self.heap.peek() {
            if entry.matcher.doc_id() == current_doc {
                let mut entry = self.heap.pop().unwrap();
                if entry.matcher.next()? && !entry.matcher.is_exhausted() {
                    matchers_to_reinsert.push(entry);
                }
            } else {
                break;
            }
        }

        // Reinsert advanced matchers
        for entry in matchers_to_reinsert {
            self.heap.push(entry);
        }

        // Update current document
        if let Some(entry) = self.heap.peek() {
            self.current_doc = entry.matcher.doc_id();
        } else {
            self.current_doc = u64::MAX;
            self.exhausted = true;
        }

        Ok(())
    }
}

impl Matcher for DisjunctionMatcher {
    fn doc_id(&self) -> u64 {
        self.current_doc
    }

    fn next(&mut self) -> Result<bool> {
        if self.exhausted {
            return Ok(false);
        }

        self.advance_to_next_doc()?;
        Ok(!self.exhausted)
    }

    fn skip_to(&mut self, target: u64) -> Result<bool> {
        if self.exhausted || target <= self.current_doc {
            return Ok(!self.exhausted);
        }

        // Skip all matchers to target or beyond
        let mut matchers_to_reinsert = Vec::new();
        while let Some(mut entry) = self.heap.pop() {
            if entry.matcher.skip_to(target)? && !entry.matcher.is_exhausted() {
                matchers_to_reinsert.push(entry);
            }
        }

        // Reinsert matchers that successfully skipped
        for entry in matchers_to_reinsert {
            self.heap.push(entry);
        }

        // Update current document
        if let Some(entry) = self.heap.peek() {
            self.current_doc = entry.matcher.doc_id();
            self.exhausted = false;
        } else {
            self.current_doc = u64::MAX;
            self.exhausted = true;
        }

        Ok(!self.exhausted)
    }

    fn cost(&self) -> u64 {
        self.cost
    }

    fn is_exhausted(&self) -> bool {
        self.exhausted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_matcher() {
        let mut matcher = EmptyMatcher::new();

        assert_eq!(matcher.doc_id(), u64::MAX);
        assert!(matcher.is_exhausted());
        assert_eq!(matcher.cost(), 0);
        assert!(!matcher.next().unwrap());
        assert!(!matcher.skip_to(5).unwrap());
    }

    #[test]
    fn test_all_matcher() {
        let mut matcher = AllMatcher::new(5);

        assert_eq!(matcher.doc_id(), 0);
        assert!(!matcher.is_exhausted());
        assert_eq!(matcher.cost(), 5);

        // Test iteration
        assert!(matcher.next().unwrap());
        assert_eq!(matcher.doc_id(), 1);

        assert!(matcher.next().unwrap());
        assert_eq!(matcher.doc_id(), 2);

        // Test skip_to
        assert!(matcher.skip_to(4).unwrap());
        assert_eq!(matcher.doc_id(), 4);

        assert!(!matcher.skip_to(10).unwrap());
        assert_eq!(matcher.doc_id(), u64::MAX);
        assert!(matcher.is_exhausted());
    }

    #[test]
    fn test_posting_matcher() {
        let doc_ids = vec![0, 1, 2, 3, 4];
        let term_freqs = vec![1, 1, 1, 1, 1];
        let posting_iter = Box::new(crate::index::reader::BasicPostingIterator::new(doc_ids, term_freqs).unwrap());
        let mut matcher = PostingMatcher::new(posting_iter);

        assert_eq!(matcher.doc_id(), 0);
        assert!(!matcher.is_exhausted());
        assert_eq!(matcher.cost(), 5);

        // Test iteration
        for i in 1..5 {
            assert!(matcher.next().unwrap());
            assert_eq!(matcher.doc_id(), i);
        }

        assert!(!matcher.next().unwrap());
        assert!(matcher.is_exhausted());
    }

    #[test]
    fn test_exhausted_posting_matcher() {
        let matcher = PostingMatcher::exhausted();

        assert_eq!(matcher.doc_id(), u64::MAX);
        assert!(matcher.is_exhausted());
        assert_eq!(matcher.cost(), 0);
    }
}
