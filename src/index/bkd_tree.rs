//! Simple BKD Tree implementation for numeric range queries.
//! 
//! This is a simplified version of Apache Lucene's BKD Tree data structure,
//! optimized for 1-dimensional numeric range filtering.

use std::cmp::Ordering;

/// A simple BKD Tree for efficient numeric range queries.
/// 
/// This implementation stores sorted (value, doc_id) pairs and provides
/// efficient range search capabilities using binary search.
#[derive(Debug, Clone)]
pub struct SimpleBKDTree {
    /// Sorted array of (value, doc_id) pairs.
    /// Sorted by value first, then by doc_id for stable ordering.
    sorted_entries: Vec<(f64, u64)>,
    
    /// Block size for chunked processing (similar to Lucene's 512-1024).
    block_size: usize,
    
    /// Field name this tree is built for.
    field_name: String,
}

impl SimpleBKDTree {
    /// Create a new BKD Tree from unsorted (value, doc_id) pairs.
    pub fn new(field_name: String, mut entries: Vec<(f64, u64)>) -> Self {
        // Sort by value first, then by doc_id for stable ordering
        entries.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
        });
        
        SimpleBKDTree {
            sorted_entries: entries,
            block_size: 512, // Similar to Lucene's default
            field_name,
        }
    }
    
    /// Create an empty BKD Tree.
    pub fn empty(field_name: String) -> Self {
        SimpleBKDTree {
            sorted_entries: Vec::new(),
            block_size: 512,
            field_name,
        }
    }
    
    /// Get the field name this tree is built for.
    pub fn field_name(&self) -> &str {
        &self.field_name
    }
    
    /// Get the number of entries in this tree.
    pub fn size(&self) -> usize {
        self.sorted_entries.len()
    }
    
    /// Check if the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.sorted_entries.is_empty()
    }
    
    /// Perform a range search and return matching document IDs.
    /// 
    /// # Arguments
    /// * `min` - Minimum value (inclusive, or None for unbounded)
    /// * `max` - Maximum value (inclusive, or None for unbounded)
    /// 
    /// # Returns
    /// Vector of document IDs that match the range criteria.
    pub fn range_search(&self, min: Option<f64>, max: Option<f64>) -> Vec<u64> {
        if self.sorted_entries.is_empty() {
            return Vec::new();
        }
        
        // Find the range of indices that match our criteria
        let start_idx = match min {
            Some(min_val) => self.find_first_gte(min_val),
            None => 0,
        };
        
        let end_idx = match max {
            Some(max_val) => self.find_last_lte(max_val),
            None => self.sorted_entries.len().saturating_sub(1),
        };
        
        if start_idx > end_idx {
            return Vec::new();
        }
        
        // Extract document IDs from the matching range
        let mut doc_ids = Vec::new();
        for i in start_idx..=end_idx {
            doc_ids.push(self.sorted_entries[i].1);
        }
        
        // Sort document IDs for consistent ordering
        doc_ids.sort_unstable();
        doc_ids.dedup(); // Remove duplicates if any
        
        doc_ids
    }
    
    /// Find the first index where value >= target using binary search.
    fn find_first_gte(&self, target: f64) -> usize {
        let mut left = 0;
        let mut right = self.sorted_entries.len();
        
        while left < right {
            let mid = left + (right - left) / 2;
            if self.sorted_entries[mid].0 >= target {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        
        left
    }
    
    /// Find the last index where value <= target using binary search.
    fn find_last_lte(&self, target: f64) -> usize {
        if self.sorted_entries.is_empty() {
            return 0;
        }
        
        let mut left = 0;
        let mut right = self.sorted_entries.len();
        
        while left < right {
            let mid = left + (right - left) / 2;
            if self.sorted_entries[mid].0 <= target {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        
        if left > 0 { left - 1 } else { 0 }
    }
    
    /// Get statistics about this BKD Tree.
    pub fn stats(&self) -> BKDTreeStats {
        let num_blocks = self.sorted_entries.len().div_ceil(self.block_size);
        let min_value = self.sorted_entries.first().map(|(v, _)| *v);
        let max_value = self.sorted_entries.last().map(|(v, _)| *v);
        
        BKDTreeStats {
            field_name: self.field_name.clone(),
            total_entries: self.sorted_entries.len(),
            num_blocks,
            block_size: self.block_size,
            min_value,
            max_value,
        }
    }
}

/// Statistics about a BKD Tree.
#[derive(Debug, Clone)]
pub struct BKDTreeStats {
    /// Field name this tree is built for.
    pub field_name: String,
    /// Total number of entries.
    pub total_entries: usize,
    /// Number of blocks.
    pub num_blocks: usize,
    /// Block size.
    pub block_size: usize,
    /// Minimum value in the tree.
    pub min_value: Option<f64>,
    /// Maximum value in the tree.
    pub max_value: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn create_test_tree() -> SimpleBKDTree {
        let entries = vec![
            (1.0, 10),   // doc 10, value 1.0
            (3.0, 20),   // doc 20, value 3.0
            (2.0, 30),   // doc 30, value 2.0
            (5.0, 40),   // doc 40, value 5.0
            (4.0, 50),   // doc 50, value 4.0
            (1.5, 60),   // doc 60, value 1.5
        ];
        SimpleBKDTree::new("test_field".to_string(), entries)
    }
    
    #[test]
    fn test_bkd_tree_creation() {
        let tree = create_test_tree();
        
        assert_eq!(tree.size(), 6);
        assert_eq!(tree.field_name(), "test_field");
        assert!(!tree.is_empty());
        
        // Verify entries are sorted by value
        let expected_order = vec![
            (1.0, 10), (1.5, 60), (2.0, 30), (3.0, 20), (4.0, 50), (5.0, 40)
        ];
        assert_eq!(tree.sorted_entries, expected_order);
    }
    
    #[test]
    fn test_empty_tree() {
        let tree = SimpleBKDTree::empty("empty_field".to_string());
        
        assert_eq!(tree.size(), 0);
        assert!(tree.is_empty());
        assert_eq!(tree.range_search(Some(1.0), Some(5.0)), Vec::<u64>::new());
    }
    
    #[test]
    fn test_range_search_exact_bounds() {
        let tree = create_test_tree();
        
        // Range [2.0, 4.0] should match docs 30, 20, 50
        let results = tree.range_search(Some(2.0), Some(4.0));
        let mut expected = vec![30, 20, 50]; // docs with values 2.0, 3.0, 4.0
        expected.sort();
        
        assert_eq!(results, expected);
    }
    
    #[test]
    fn test_range_search_partial_bounds() {
        let tree = create_test_tree();
        
        // Range [3.0, None] should match docs 20, 50, 40
        let results = tree.range_search(Some(3.0), None);
        let mut expected = vec![20, 50, 40]; // docs with values 3.0, 4.0, 5.0
        expected.sort();
        
        assert_eq!(results, expected);
        
        // Range [None, 2.0] should match docs 10, 60, 30
        let results = tree.range_search(None, Some(2.0));
        let mut expected = vec![10, 60, 30]; // docs with values 1.0, 1.5, 2.0
        expected.sort();
        
        assert_eq!(results, expected);
    }
    
    #[test]
    fn test_range_search_no_bounds() {
        let tree = create_test_tree();
        
        // Range [None, None] should match all docs
        let results = tree.range_search(None, None);
        let mut expected = vec![10, 20, 30, 40, 50, 60];
        expected.sort();
        
        assert_eq!(results, expected);
    }
    
    #[test]
    fn test_range_search_no_matches() {
        let tree = create_test_tree();
        
        // Range [10.0, 20.0] should match no docs
        let results = tree.range_search(Some(10.0), Some(20.0));
        assert_eq!(results, Vec::<u64>::new());
        
        // Range [2.5, 2.5] should match no docs (no exact match)
        let results = tree.range_search(Some(2.5), Some(2.5));
        assert_eq!(results, Vec::<u64>::new());
    }
    
    #[test]
    fn test_range_search_single_value() {
        let tree = create_test_tree();
        
        // Range [3.0, 3.0] should match doc 20
        let results = tree.range_search(Some(3.0), Some(3.0));
        assert_eq!(results, vec![20]);
    }
    
    #[test]
    fn test_stats() {
        let tree = create_test_tree();
        let stats = tree.stats();
        
        assert_eq!(stats.field_name, "test_field");
        assert_eq!(stats.total_entries, 6);
        assert_eq!(stats.block_size, 512);
        assert_eq!(stats.min_value, Some(1.0));
        assert_eq!(stats.max_value, Some(5.0));
    }
    
    #[test]
    fn test_binary_search_functions() {
        let tree = create_test_tree();
        
        // Test find_first_gte
        assert_eq!(tree.find_first_gte(0.5), 0);  // Before all values
        assert_eq!(tree.find_first_gte(1.0), 0);  // Exact match first
        assert_eq!(tree.find_first_gte(1.2), 1);  // Between values
        assert_eq!(tree.find_first_gte(3.0), 3);  // Exact match middle
        assert_eq!(tree.find_first_gte(6.0), 6);  // After all values
        
        // Test find_last_lte
        assert_eq!(tree.find_last_lte(0.5), 0);   // Before all values
        assert_eq!(tree.find_last_lte(1.0), 0);   // Exact match first
        assert_eq!(tree.find_last_lte(1.2), 0);   // Between values
        assert_eq!(tree.find_last_lte(3.0), 3);   // Exact match middle
        assert_eq!(tree.find_last_lte(6.0), 5);   // After all values
    }
}