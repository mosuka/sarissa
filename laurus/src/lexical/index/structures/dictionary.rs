//! Term dictionary data structures for mapping terms to posting list metadata.
//!
//! This module provides multiple term dictionary implementations—sorted, hash-based,
//! and hybrid—for efficiently mapping terms to their [`TermInfo`] (posting list
//! offset, length, document frequency, and total frequency). A [`TermDictionaryBuilder`]
//! is also provided for constructing any of the dictionary variants.

use std::collections::BTreeMap;

use ahash::AHashMap;

use crate::error::{LaurusError, Result};
use crate::storage::structured::{StructReader, StructWriter};
use crate::storage::{StorageInput, StorageOutput};

/// One block's max-impact metadata for Block-Max-WAND (#403 PR-C).
///
/// Each block covers up to [`BLOCK_SIZE`] consecutive postings of a
/// term. The pair records:
///
/// - `last_doc_id` — the doc id of the last posting in the block; used
///   to locate the block containing a target doc id by binary search,
///   and to seek to the next block during WAND skipping.
/// - `max_factor` — the tightest possible BM25 TF-component value over
///   the postings in this block, computed with default BM25 parameters
///   (`k1 = 1.2`, `b = 0.75`) and the segment's average field length.
///   `BM25Scorer::block_max_score_at` multiplies it by `boost · idf`
///   to expose a per-block upper bound to the searcher loop.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlockMax {
    /// Last doc id covered by this block.
    pub last_doc_id: u64,
    /// Block-level TF-component upper bound (default BM25 parameters).
    pub max_factor: f32,
}

/// Posting blocks are 128 entries wide, matching Lucene 9 / Tantivy.
pub const BLOCK_SIZE: usize = 128;

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
    /// Term-level tightened BM25 TF-component upper bound (#403 PR-B2).
    ///
    /// Maximum of `(tf · (k1 + 1)) / (tf + k1 · (1 - b + b · (L / avg_L)))`
    /// over every posting in the list. `0.0` means "unset"; scorers
    /// fall back to the loose `k1 + 1` synthetic ceiling.
    pub max_score_factor: f32,
    /// Per-block (`BLOCK_SIZE`-wide) max-impact metadata used by
    /// Block-Max-WAND (#403 PR-C). Empty if not available
    /// (legacy v2 dictionaries or aggregated cross-segment views) —
    /// scorers then fall back to [`Self::max_score_factor`].
    pub block_max: Vec<BlockMax>,
}

impl TermInfo {
    /// Create new term info with `max_score_factor = 0.0` (loose bound).
    ///
    /// Use [`TermInfo::with_max_score_factor`] to build a value with the
    /// tightened block-max bound from the index. Existing call sites
    /// keep working — the searcher silently falls back to the synthetic
    /// upper bound when the field is `0.0`.
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
            max_score_factor: 0.0,
            block_max: Vec::new(),
        }
    }

    /// Create new term info with a precomputed tightened max-score
    /// factor (#403 PR-B2). Pass `0.0` if the factor is not available
    /// — scorers will fall back to the loose `k1 + 1` bound.
    pub fn with_max_score_factor(
        posting_offset: u64,
        posting_length: u64,
        doc_frequency: u64,
        total_frequency: u64,
        max_score_factor: f32,
    ) -> Self {
        TermInfo {
            posting_offset,
            posting_length,
            doc_frequency,
            total_frequency,
            max_score_factor,
            block_max: Vec::new(),
        }
    }

    /// Create new term info with both the term-level factor and the
    /// per-block max-impact metadata used by Block-Max-WAND (#403 PR-C).
    pub fn with_block_max(
        posting_offset: u64,
        posting_length: u64,
        doc_frequency: u64,
        total_frequency: u64,
        max_score_factor: f32,
        block_max: Vec<BlockMax>,
    ) -> Self {
        TermInfo {
            posting_offset,
            posting_length,
            doc_frequency,
            total_frequency,
            max_score_factor,
            block_max,
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
    ///
    /// Supports both **v1** (legacy) and **v2** (#403 PR-B2) layouts:
    ///
    /// - v1 entries store only the four `u64` fields. `max_score_factor`
    ///   is filled with `0.0`, which the BM25 scorer treats as "fall
    ///   back to the loose `k1 + 1` upper bound" — segments produced
    ///   before this PR continue to load and search correctly.
    /// - v2 entries append a single `f32` per term holding the
    ///   precomputed tightened TF-component upper bound.
    pub fn read_from_storage<R: StorageInput>(reader: &mut StructReader<R>) -> Result<Self> {
        // Read header
        let magic = reader.read_u32()?;
        if magic != 0x53544443 {
            // "STDC"
            return Err(LaurusError::index("Invalid sorted dictionary magic number"));
        }

        let version = reader.read_u32()?;
        if version != 1 && version != 2 && version != 3 {
            return Err(LaurusError::index(format!(
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
            let max_score_factor = if version >= 2 {
                reader.read_f32()?
            } else {
                0.0
            };
            // v3: per-block (last_doc_id, max_factor) array. v1/v2
            // entries decode with an empty `block_max`, which scorers
            // treat as "fall back to the term-level
            // `max_score_factor`" (#403 PR-C).
            let block_max = if version >= 3 {
                let block_count = reader.read_varint()? as usize;
                let mut blocks = Vec::with_capacity(block_count);
                for _ in 0..block_count {
                    let last_doc_id = reader.read_u64()?;
                    let mf = reader.read_f32()?;
                    blocks.push(BlockMax {
                        last_doc_id,
                        max_factor: mf,
                    });
                }
                blocks
            } else {
                Vec::new()
            };

            terms.push(term);
            term_infos.push(TermInfo {
                posting_offset,
                posting_length,
                doc_frequency,
                total_frequency,
                max_score_factor,
                block_max,
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

    /// Write to storage in **v3** layout (#403 PR-C). Each entry now
    /// carries the v2 `max_score_factor: f32` plus a length-prefixed
    /// per-block `(last_doc_id: u64, max_factor: f32)` array used by
    /// Block-Max-WAND. See [`SortedTermDictionary::write_to_storage`]
    /// for the matching format on the sorted side.
    pub fn write_to_storage<W: StorageOutput>(&self, writer: &mut StructWriter<W>) -> Result<()> {
        // Write magic number for hash dictionary
        writer.write_u32(0x48544443)?; // "HTDC"

        // Write version
        writer.write_u32(3)?;

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
            writer.write_f32(info.max_score_factor)?;
            writer.write_varint(info.block_max.len() as u64)?;
            for block in &info.block_max {
                writer.write_u64(block.last_doc_id)?;
                writer.write_f32(block.max_factor)?;
            }
        }

        Ok(())
    }

    /// Read from storage. Accepts **v1** (legacy), **v2** (#403 PR-B2)
    /// and **v3** (#403 PR-C) layouts. Older entries fill the missing
    /// fields with their "unset" defaults — `max_score_factor = 0.0`
    /// and an empty `block_max` — which scorers treat as "fall back
    /// to the looser bound" so older segments continue to load.
    pub fn read_from_storage<R: StorageInput>(reader: &mut StructReader<R>) -> Result<Self> {
        // Read magic number
        let magic = reader.read_u32()?;
        if magic != 0x48544443 {
            // "HTDC"
            return Err(LaurusError::index("Invalid hash dictionary magic number"));
        }

        // Read version
        let version = reader.read_u32()?;
        if version != 1 && version != 2 && version != 3 {
            return Err(LaurusError::index(format!(
                "Unsupported hash dictionary version: {version}"
            )));
        }

        // Read number of terms
        let term_count = reader.read_varint()? as usize;

        // Read terms and their info
        let mut terms = AHashMap::with_capacity(term_count);

        for _ in 0..term_count {
            let term = reader.read_string()?;
            let posting_offset = reader.read_u64()?;
            let posting_length = reader.read_u64()?;
            let doc_frequency = reader.read_u64()?;
            let total_frequency = reader.read_u64()?;
            let max_score_factor = if version >= 2 {
                reader.read_f32()?
            } else {
                0.0
            };
            let block_max = if version >= 3 {
                let block_count = reader.read_varint()? as usize;
                let mut blocks = Vec::with_capacity(block_count);
                for _ in 0..block_count {
                    let last_doc_id = reader.read_u64()?;
                    let mf = reader.read_f32()?;
                    blocks.push(BlockMax {
                        last_doc_id,
                        max_factor: mf,
                    });
                }
                blocks
            } else {
                Vec::new()
            };
            let info = TermInfo {
                posting_offset,
                posting_length,
                doc_frequency,
                total_frequency,
                max_score_factor,
                block_max,
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
    /// Write to storage in **v3** layout (#403 PR-C).
    ///
    /// Each entry carries the v2 `max_score_factor: f32` plus a
    /// length-prefixed per-block `(last_doc_id: u64, max_factor: f32)`
    /// array used by Block-Max-WAND. v1/v2 readers are no longer able
    /// to load new segments; readers from this codebase accept v1, v2
    /// and v3 (see [`SortedTermDictionary::read_from_storage`]).
    pub fn write_to_storage<W: StorageOutput>(&self, writer: &mut StructWriter<W>) -> Result<()> {
        // Write magic number for sorted dictionary
        writer.write_u32(0x53544443)?; // "STDC"

        // Write version
        writer.write_u32(3)?;

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
            writer.write_f32(info.max_score_factor)?;
            writer.write_varint(info.block_max.len() as u64)?;
            for block in &info.block_max {
                writer.write_u64(block.last_doc_id)?;
                writer.write_f32(block.max_factor)?;
            }
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
    use crate::storage::Storage;

    use crate::storage::memory::MemoryStorage;
    use crate::storage::memory::MemoryStorageConfig;
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
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

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
