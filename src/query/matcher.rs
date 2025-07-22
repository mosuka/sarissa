//! Matcher implementations for query execution.

use crate::error::Result;
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
    current_doc: u64,
    exhausted: bool,
    cost: u64,
}

impl PostingMatcher {
    /// Create a new posting matcher.
    pub fn new(cost: u64) -> Self {
        PostingMatcher {
            current_doc: 0,
            exhausted: false,
            cost,
        }
    }

    /// Create an exhausted posting matcher.
    pub fn exhausted() -> Self {
        PostingMatcher {
            current_doc: u64::MAX,
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
            self.current_doc
        }
    }

    fn next(&mut self) -> Result<bool> {
        if self.exhausted {
            Ok(false)
        } else {
            // TODO: Implement actual posting list iteration
            // For now, simulate some documents
            self.current_doc += 1;
            if self.current_doc >= 10 {
                self.exhausted = true;
                Ok(false)
            } else {
                Ok(true)
            }
        }
    }

    fn skip_to(&mut self, target: u64) -> Result<bool> {
        if self.exhausted {
            Ok(false)
        } else if target >= 10 {
            self.current_doc = u64::MAX;
            self.exhausted = true;
            Ok(false)
        } else {
            self.current_doc = target;
            Ok(true)
        }
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
        let mut matcher = PostingMatcher::new(100);

        assert_eq!(matcher.doc_id(), 0);
        assert!(!matcher.is_exhausted());
        assert_eq!(matcher.cost(), 100);

        // Test iteration
        for i in 1..10 {
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
