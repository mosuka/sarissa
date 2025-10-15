//! Term dictionary for efficient term-to-posting lookup.
//!
//! This module provides a high-performance term dictionary implementation
//! based on sorted arrays and hash tables for different use cases.

use std::collections::BTreeMap;

use ahash::AHashMap;

use crate::error::{Result, SageError};
use crate::storage::{StorageInput, StorageOutput, StructReader, StructWriter};

/// Information about a term in the dictionary.
#[derive(Debug, Clone, PartialEq)]
pub struct TermInfo {
    /// Offset to the posting list in the posting file.
    pub posting_offset: u64,
    /// Length of the posting list in bytes.
    pub posting_length: u64,
    /// Document frequency (number of documents containing this term).
    pub doc_frequency: u64,
    /// Total frequency across all documents.
    pub total_frequency: u64,
}

impl TermInfo {
    /// Create new term info.
    pub fn new(
        posting_offset: u64,
        posting_length: u64,
        doc_frequency: u64,
        total_frequency: u64,
    ) -> Self {
        TermInfo {
            posting_offset,
            posting_length,
            doc_frequency,
            total_frequency,
        }
    }
}

/// A sorted array-based term dictionary for prefix queries and ordered iteration.
#[derive(Debug, Clone)]
pub struct SortedTermDictionary {
    /// Sorted terms.
    terms: Vec<String>,
    /// Term info for each term (parallel array).
    term_infos: Vec<TermInfo>,
}

impl SortedTermDictionary {
    /// Create a new empty sorted term dictionary.
    pub fn new() -> Self {
        SortedTermDictionary {
            terms: Vec::new(),
            term_infos: Vec::new(),
        }
    }

    /// Create from a map of terms to term info.
    pub fn from_map(map: BTreeMap<String, TermInfo>) -> Self {
        let mut terms = Vec::with_capacity(map.len());
        let mut term_infos = Vec::with_capacity(map.len());

        for (term, info) in map.into_iter() {
            terms.push(term);
            term_infos.push(info);
        }

        SortedTermDictionary { terms, term_infos }
    }

    /// Look up a term and return its info.
    pub fn get(&self, term: &str) -> Option<&TermInfo> {
        match self
            .terms
            .binary_search_by(|probe| probe.as_str().cmp(term))
        {
            Ok(index) => Some(&self.term_infos[index]),
            Err(_) => None,
        }
    }

    /// Find terms with the given prefix.
    pub fn find_prefix(&self, prefix: &str) -> Vec<(&str, &TermInfo)> {
        let start_pos = match self
            .terms
            .binary_search_by(|probe| probe.as_str().cmp(prefix))
        {
            Ok(pos) => pos,
            Err(pos) => pos,
        };

        let mut results = Vec::new();
        for i in start_pos..self.terms.len() {
            if self.terms[i].starts_with(prefix) {
                results.push((self.terms[i].as_str(), &self.term_infos[i]));
            } else {
                break;
            }
        }

        results
    }

    /// Find terms in a range.
    pub fn find_range(&self, start: &str, end: &str) -> Vec<(&str, &TermInfo)> {
        let start_pos = match self
            .terms
            .binary_search_by(|probe| probe.as_str().cmp(start))
        {
            Ok(pos) => pos,
            Err(pos) => pos,
        };

        let end_pos = match self.terms.binary_search_by(|probe| probe.as_str().cmp(end)) {
            Ok(pos) => pos, // end is exclusive, so don't include it
            Err(pos) => pos,
        };

        let mut results = Vec::new();
        for i in start_pos..end_pos.min(self.terms.len()) {
            results.push((self.terms[i].as_str(), &self.term_infos[i]));
        }

        results
    }

    /// Get the number of terms.
    pub fn len(&self) -> usize {
        self.terms.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    /// Get an iterator over all terms.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &TermInfo)> {
        self.terms
            .iter()
            .zip(self.term_infos.iter())
            .map(|(term, info)| (term.as_str(), info))
    }

    /// Read the dictionary from storage.
    pub fn read_from_storage<R: StorageInput>(reader: &mut StructReader<R>) -> Result<Self> {
        // Read header
        let magic = reader.read_u32()?;
        if magic != 0x53544443 {
            // "STDC"
            return Err(SageError::index(
                "Invalid sorted dictionary magic number",
            ));
        }

        let version = reader.read_u32()?;
        if version != 1 {
            return Err(SageError::index(format!(
                "Unsupported sorted dictionary version: {version}"
            )));
        }

        let term_count = reader.read_varint()? as usize;
        let mut terms = Vec::with_capacity(term_count);
        let mut term_infos = Vec::with_capacity(term_count);

        // Read terms and term infos
        for _ in 0..term_count {
            let term = reader.read_string()?;
            let posting_offset = reader.read_u64()?;
            let posting_length = reader.read_u64()?;
            let doc_frequency = reader.read_u64()?;
            let total_frequency = reader.read_u64()?;

            terms.push(term);
            term_infos.push(TermInfo {
                posting_offset,
                posting_length,
                doc_frequency,
                total_frequency,
            });
        }

        Ok(SortedTermDictionary { terms, term_infos })
    }
}

impl Default for SortedTermDictionary {
    fn default() -> Self {
        Self::new()
    }
}

/// A hash-based term dictionary for fast random access.
#[derive(Debug, Clone)]
pub struct HashTermDictionary {
    /// Hash map from terms to term info.
    terms: AHashMap<String, TermInfo>,
}

impl HashTermDictionary {
    /// Create a new empty hash term dictionary.
    pub fn new() -> Self {
        HashTermDictionary {
            terms: AHashMap::new(),
        }
    }

    /// Create with initial capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        HashTermDictionary {
            terms: AHashMap::with_capacity(capacity),
        }
    }

    /// Insert a term with its info.
    pub fn insert(&mut self, term: String, info: TermInfo) {
        self.terms.insert(term, info);
    }

    /// Look up a term and return its info.
    pub fn get(&self, term: &str) -> Option<&TermInfo> {
        self.terms.get(term)
    }

    /// Check if a term exists.
    pub fn contains(&self, term: &str) -> bool {
        self.terms.contains_key(term)
    }

    /// Get the number of terms.
    pub fn len(&self) -> usize {
        self.terms.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    /// Get an iterator over all terms.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &TermInfo)> {
        self.terms.iter().map(|(term, info)| (term.as_str(), info))
    }

    /// Convert to a sorted dictionary.
    pub fn to_sorted(&self) -> SortedTermDictionary {
        let map: BTreeMap<String, TermInfo> = self
            .terms
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        SortedTermDictionary::from_map(map)
    }

    /// Write to storage.
    pub fn write_to_storage<W: StorageOutput>(&self, writer: &mut StructWriter<W>) -> Result<()> {
        // Write magic number for hash dictionary
        writer.write_u32(0x48544443)?; // "HTDC"

        // Write version
        writer.write_u32(1)?;

        // Write number of terms
        writer.write_varint(self.terms.len() as u64)?;

        // Write terms and their info
        for (term, info) in &self.terms {
            writer.write_string(term)?;

            // Write TermInfo
            writer.write_u64(info.posting_offset)?;
            writer.write_u64(info.posting_length)?;
            writer.write_u64(info.doc_frequency)?;
            writer.write_u64(info.total_frequency)?;
        }

        Ok(())
    }

    /// Read from storage.
    pub fn read_from_storage<R: StorageInput>(reader: &mut StructReader<R>) -> Result<Self> {
        // Read magic number
        let magic = reader.read_u32()?;
        if magic != 0x48544443 {
            // "HTDC"
            return Err(SageError::index("Invalid hash dictionary magic number"));
        }

        // Read version
        let version = reader.read_u32()?;
        if version != 1 {
            return Err(SageError::index(format!(
                "Unsupported hash dictionary version: {version}"
            )));
        }

        // Read number of terms
        let term_count = reader.read_varint()? as usize;

        // Read terms and their info
        let mut terms = AHashMap::with_capacity(term_count);

        for _ in 0..term_count {
            let term = reader.read_string()?;
            let info = TermInfo {
                posting_offset: reader.read_u64()?,
                posting_length: reader.read_u64()?,
                doc_frequency: reader.read_u64()?,
                total_frequency: reader.read_u64()?,
            };

            terms.insert(term, info);
        }

        Ok(HashTermDictionary { terms })
    }
}

impl Default for HashTermDictionary {
    fn default() -> Self {
        Self::new()
    }
}

/// A hybrid term dictionary that provides both fast access and prefix queries.
#[derive(Debug, Clone)]
pub struct HybridTermDictionary {
    /// Hash dictionary for fast random access.
    hash_dict: HashTermDictionary,
    /// Sorted dictionary for prefix and range queries.
    sorted_dict: SortedTermDictionary,
}

impl HybridTermDictionary {
    /// Create a new hybrid dictionary from a hash dictionary.
    pub fn from_hash(hash_dict: HashTermDictionary) -> Self {
        let sorted_dict = hash_dict.to_sorted();
        HybridTermDictionary {
            hash_dict,
            sorted_dict,
        }
    }

    /// Read hybrid term dictionary from storage.
    pub fn read_from_storage<R: StorageInput>(reader: &mut StructReader<R>) -> Result<Self> {
        let sorted_dict = SortedTermDictionary::read_from_storage(reader)?;
        let mut hash_dict = HashTermDictionary::with_capacity(sorted_dict.len());

        for (term, info) in sorted_dict.iter() {
            hash_dict.insert(term.to_string(), info.clone());
        }

        Ok(HybridTermDictionary {
            hash_dict,
            sorted_dict,
        })
    }

    /// Look up a term (uses hash dictionary for speed).
    pub fn get(&self, term: &str) -> Option<&TermInfo> {
        self.hash_dict.get(term)
    }

    /// Find terms with the given prefix (uses sorted dictionary).
    pub fn find_prefix(&self, prefix: &str) -> Vec<(&str, &TermInfo)> {
        self.sorted_dict.find_prefix(prefix)
    }

    /// Find terms in a range (uses sorted dictionary).
    pub fn find_range(&self, start: &str, end: &str) -> Vec<(&str, &TermInfo)> {
        self.sorted_dict.find_range(start, end)
    }

    /// Get the number of terms.
    pub fn len(&self) -> usize {
        self.hash_dict.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.hash_dict.is_empty()
    }

    /// Get an iterator over all terms (ordered).
    pub fn iter(&self) -> impl Iterator<Item = (&str, &TermInfo)> {
        self.sorted_dict.iter()
    }

    /// Write the dictionary to storage.
    pub fn write_to_storage<W: StorageOutput>(&self, writer: &mut StructWriter<W>) -> Result<()> {
        self.sorted_dict.write_to_storage(writer)
    }
}

/// Builder for creating term dictionaries.
pub struct TermDictionaryBuilder {
    terms: BTreeMap<String, TermInfo>,
}

impl TermDictionaryBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        TermDictionaryBuilder {
            terms: BTreeMap::new(),
        }
    }

    /// Add a term with its info.
    pub fn add_term(&mut self, term: String, info: TermInfo) {
        self.terms.insert(term, info);
    }

    /// Build a sorted term dictionary.
    pub fn build_sorted(self) -> SortedTermDictionary {
        SortedTermDictionary::from_map(self.terms)
    }

    /// Build a hash term dictionary.
    pub fn build_hash(self) -> HashTermDictionary {
        let mut hash_dict = HashTermDictionary::with_capacity(self.terms.len());
        for (term, info) in self.terms {
            hash_dict.insert(term, info);
        }
        hash_dict
    }

    /// Build a hybrid term dictionary.
    pub fn build_hybrid(self) -> HybridTermDictionary {
        let hash_dict = self.build_hash();
        HybridTermDictionary::from_hash(hash_dict)
    }

    /// Get the current number of terms.
    pub fn len(&self) -> usize {
        self.terms.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }
}

impl Default for TermDictionaryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Dictionary statistics.
#[derive(Debug, Clone)]
pub struct DictionaryStats {
    /// Number of terms.
    pub term_count: usize,
    /// Total size in memory (bytes).
    pub memory_size: usize,
    /// Average term length.
    pub avg_term_length: f64,
    /// Total document frequency.
    pub total_doc_frequency: u64,
    /// Total term frequency.
    pub total_term_frequency: u64,
}

impl SortedTermDictionary {
    /// Write to storage.
    pub fn write_to_storage<W: StorageOutput>(&self, writer: &mut StructWriter<W>) -> Result<()> {
        // Write magic number for sorted dictionary
        writer.write_u32(0x53544443)?; // "STDC"

        // Write version
        writer.write_u32(1)?;

        // Write number of terms
        writer.write_varint(self.terms.len() as u64)?;

        // Write terms and their info
        for (term, info) in self.terms.iter().zip(self.term_infos.iter()) {
            writer.write_string(term)?;

            // Write TermInfo
            writer.write_u64(info.posting_offset)?;
            writer.write_u64(info.posting_length)?;
            writer.write_u64(info.doc_frequency)?;
            writer.write_u64(info.total_frequency)?;
        }

        Ok(())
    }

    /// Get statistics about the dictionary.
    pub fn stats(&self) -> DictionaryStats {
        let term_count = self.terms.len();
        let total_term_length: usize = self.terms.iter().map(|t| t.len()).sum();
        let avg_term_length = if term_count > 0 {
            total_term_length as f64 / term_count as f64
        } else {
            0.0
        };

        let total_doc_frequency = self.term_infos.iter().map(|info| info.doc_frequency).sum();
        let total_term_frequency = self
            .term_infos
            .iter()
            .map(|info| info.total_frequency)
            .sum();

        // Estimate memory size
        let memory_size =
            total_term_length + (self.term_infos.len() * std::mem::size_of::<TermInfo>());

        DictionaryStats {
            term_count,
            memory_size,
            avg_term_length,
            total_doc_frequency,
            total_term_frequency,
        }
    }
}

impl HashTermDictionary {
    /// Get statistics about the dictionary.
    pub fn stats(&self) -> DictionaryStats {
        let term_count = self.terms.len();
        let total_term_length: usize = self.terms.keys().map(|t| t.len()).sum();
        let avg_term_length = if term_count > 0 {
            total_term_length as f64 / term_count as f64
        } else {
            0.0
        };

        let total_doc_frequency = self.terms.values().map(|info| info.doc_frequency).sum();
        let total_term_frequency = self.terms.values().map(|info| info.total_frequency).sum();

        // Estimate memory size (includes hash map overhead)
        let memory_size =
            total_term_length + (self.terms.len() * (std::mem::size_of::<TermInfo>() + 64));

        DictionaryStats {
            term_count,
            memory_size,
            avg_term_length,
            total_doc_frequency,
            total_term_frequency,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{MemoryStorage, Storage, StorageConfig};
    use std::sync::Arc;

    fn create_test_term_info(offset: u64) -> TermInfo {
        TermInfo::new(offset, 100, 5, 20)
    }

    #[test]
    fn test_sorted_term_dictionary() {
        let mut builder = TermDictionaryBuilder::new();
        builder.add_term("apple".to_string(), create_test_term_info(0));
        builder.add_term("banana".to_string(), create_test_term_info(100));
        builder.add_term("cherry".to_string(), create_test_term_info(200));
        builder.add_term("apricot".to_string(), create_test_term_info(300));

        let dict = builder.build_sorted();

        // Test exact lookup
        assert!(dict.get("apple").is_some());
        assert!(dict.get("banana").is_some());
        assert!(dict.get("nonexistent").is_none());

        // Test prefix search
        let ap_results = dict.find_prefix("ap");
        assert_eq!(ap_results.len(), 2);
        assert!(ap_results.iter().any(|(term, _)| *term == "apple"));
        assert!(ap_results.iter().any(|(term, _)| *term == "apricot"));

        // Test range search
        let range_results = dict.find_range("apple", "cherry");
        assert_eq!(range_results.len(), 3); // apple, apricot, banana
    }

    #[test]
    fn test_hash_term_dictionary() {
        let mut dict = HashTermDictionary::new();
        dict.insert("apple".to_string(), create_test_term_info(0));
        dict.insert("banana".to_string(), create_test_term_info(100));
        dict.insert("cherry".to_string(), create_test_term_info(200));

        assert!(dict.contains("apple"));
        assert!(dict.contains("banana"));
        assert!(!dict.contains("nonexistent"));

        assert_eq!(dict.len(), 3);
        assert!(!dict.is_empty());

        let info = dict.get("apple").unwrap();
        assert_eq!(info.posting_offset, 0);
    }

    #[test]
    fn test_hybrid_term_dictionary() {
        let mut hash_dict = HashTermDictionary::new();
        hash_dict.insert("apple".to_string(), create_test_term_info(0));
        hash_dict.insert("banana".to_string(), create_test_term_info(100));
        hash_dict.insert("apricot".to_string(), create_test_term_info(200));

        let hybrid_dict = HybridTermDictionary::from_hash(hash_dict);

        // Test hash-based lookup
        assert!(hybrid_dict.get("apple").is_some());
        assert!(hybrid_dict.get("nonexistent").is_none());

        // Test prefix search
        let ap_results = hybrid_dict.find_prefix("ap");
        assert_eq!(ap_results.len(), 2);
    }

    #[test]
    fn test_dictionary_serialization() {
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));

        let mut builder = TermDictionaryBuilder::new();
        builder.add_term("apple".to_string(), create_test_term_info(0));
        builder.add_term("banana".to_string(), create_test_term_info(100));
        builder.add_term("cherry".to_string(), create_test_term_info(200));

        let original_dict = builder.build_sorted();

        // Write to storage
        {
            let output = storage.create_output("test_dict.bin").unwrap();
            let mut writer = StructWriter::new(output);
            original_dict.write_to_storage(&mut writer).unwrap();
            writer.close().unwrap();
        }

        // Read from storage
        {
            let input = storage.open_input("test_dict.bin").unwrap();
            let mut reader = StructReader::new(input).unwrap();
            let loaded_dict = SortedTermDictionary::read_from_storage(&mut reader).unwrap();

            assert_eq!(loaded_dict.len(), original_dict.len());

            for term in ["apple", "banana", "cherry"] {
                let orig_info = original_dict.get(term).unwrap();
                let loaded_info = loaded_dict.get(term).unwrap();
                assert_eq!(orig_info, loaded_info);
            }
        }
    }

    #[test]
    fn test_dictionary_stats() {
        let mut builder = TermDictionaryBuilder::new();
        builder.add_term("short".to_string(), TermInfo::new(0, 50, 1, 1));
        builder.add_term("longer_term".to_string(), TermInfo::new(50, 100, 5, 10));
        builder.add_term(
            "longest_term_here".to_string(),
            TermInfo::new(150, 200, 3, 8),
        );

        let dict = builder.build_sorted();
        let stats = dict.stats();

        assert_eq!(stats.term_count, 3);
        assert!(stats.avg_term_length > 0.0);
        assert_eq!(stats.total_doc_frequency, 9); // 1 + 5 + 3
        assert_eq!(stats.total_term_frequency, 19); // 1 + 10 + 8
        assert!(stats.memory_size > 0);
    }

    #[test]
    fn test_empty_dictionary() {
        let dict = SortedTermDictionary::new();
        assert!(dict.is_empty());
        assert_eq!(dict.len(), 0);
        assert!(dict.get("anything").is_none());
        assert!(dict.find_prefix("any").is_empty());
    }

    #[test]
    fn test_dictionary_builder() {
        let mut builder = TermDictionaryBuilder::new();
        assert!(builder.is_empty());

        builder.add_term("test".to_string(), create_test_term_info(0));
        assert_eq!(builder.len(), 1);

        let sorted = builder.build_sorted();
        assert_eq!(sorted.len(), 1);
        assert!(sorted.get("test").is_some());
    }
}
