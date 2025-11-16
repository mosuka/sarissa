//! Posting lists and inverted index implementation.
//!
//! This module provides the core inverted index data structures for efficient
//! term-to-document mapping with frequency and position information.

use ahash::AHashMap;

use crate::error::{Result, PlatypusError};
use crate::storage::structured::{StructReader, StructWriter};
use crate::storage::{StorageInput, StorageOutput};

/// A single posting in a posting list.
#[derive(Debug, Clone, PartialEq)]
pub struct Posting {
    /// Document ID.
    pub doc_id: u64,
    /// Term frequency in the document.
    pub frequency: u32,
    /// Positions of the term in the document (for phrase queries).
    pub positions: Option<Vec<u32>>,
    /// Weight/score for this posting.
    pub weight: f32,
}

impl Posting {
    /// Create a new posting.
    pub fn new(doc_id: u64) -> Self {
        Posting {
            doc_id,
            frequency: 1,
            positions: None,
            weight: 1.0,
        }
    }

    /// Create a posting with frequency.
    pub fn with_frequency(doc_id: u64, frequency: u32) -> Self {
        Posting {
            doc_id,
            frequency,
            positions: None,
            weight: 1.0,
        }
    }

    /// Create a posting with positions.
    pub fn with_positions(doc_id: u64, positions: Vec<u32>) -> Self {
        let frequency = positions.len() as u32;
        Posting {
            doc_id,
            frequency,
            positions: Some(positions),
            weight: 1.0,
        }
    }

    /// Set the weight for this posting.
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }

    /// Add a position to this posting.
    pub fn add_position(&mut self, position: u32) {
        match &mut self.positions {
            Some(positions) => {
                positions.push(position);
                self.frequency = positions.len() as u32;
            }
            None => {
                self.positions = Some(vec![position]);
                self.frequency = 1;
            }
        }
    }

    /// Get the term frequency.
    pub fn frequency(&self) -> u32 {
        self.frequency
    }

    /// Get positions if available.
    pub fn positions(&self) -> Option<&[u32]> {
        self.positions.as_deref()
    }
}

/// A posting list for a specific term.
#[derive(Debug, Clone)]
pub struct PostingList {
    /// The term this posting list represents.
    pub term: String,
    /// The postings in this list.
    pub postings: Vec<Posting>,
    /// Total frequency across all documents.
    pub total_frequency: u64,
    /// Document frequency (number of documents containing this term).
    pub doc_frequency: u64,
}

impl PostingList {
    /// Create a new empty posting list.
    pub fn new(term: String) -> Self {
        PostingList {
            term,
            postings: Vec::new(),
            total_frequency: 0,
            doc_frequency: 0,
        }
    }

    /// Add a posting to this list.
    pub fn add_posting(&mut self, posting: Posting) {
        self.total_frequency += posting.frequency as u64;
        self.doc_frequency += 1;

        // Insert in sorted order by doc_id
        match self
            .postings
            .binary_search_by_key(&posting.doc_id, |p| p.doc_id)
        {
            Ok(pos) => {
                // Document already exists, merge the posting
                let existing = &mut self.postings[pos];
                existing.frequency += posting.frequency;
                self.total_frequency += posting.frequency as u64;

                if let Some(new_positions) = posting.positions {
                    match &mut existing.positions {
                        Some(positions) => positions.extend(new_positions),
                        None => existing.positions = Some(new_positions),
                    }
                }
            }
            Err(pos) => {
                // Insert new posting
                self.postings.insert(pos, posting);
            }
        }
    }

    /// Get the length of the posting list.
    pub fn len(&self) -> usize {
        self.postings.len()
    }

    /// Check if the posting list is empty.
    pub fn is_empty(&self) -> bool {
        self.postings.is_empty()
    }

    /// Get an iterator over the postings.
    pub fn iter(&'_ self) -> std::slice::Iter<'_, Posting> {
        self.postings.iter()
    }

    /// Optimize the posting list by removing duplicates and sorting.
    pub fn optimize(&mut self) {
        self.postings.sort_by_key(|p| p.doc_id);
        self.postings.dedup_by_key(|p| p.doc_id);
    }

    /// Encode the posting list to binary format.
    pub fn encode<W: StorageOutput>(&self, writer: &mut StructWriter<W>) -> Result<()> {
        // Write term
        writer.write_string(&self.term)?;

        // Write metadata
        writer.write_varint(self.total_frequency)?;
        writer.write_varint(self.doc_frequency)?;
        writer.write_varint(self.postings.len() as u64)?;

        // Write postings with delta compression for doc IDs
        let mut prev_doc_id = 0u64;
        for posting in &self.postings {
            // Delta-compressed doc ID
            let delta = posting.doc_id - prev_doc_id;
            writer.write_varint(delta)?;
            prev_doc_id = posting.doc_id;

            // Frequency
            writer.write_varint(posting.frequency as u64)?;

            // Weight
            writer.write_f32(posting.weight)?;

            // Positions (optional)
            if let Some(positions) = &posting.positions {
                writer.write_u8(1)?; // Has positions flag
                writer.write_varint(positions.len() as u64)?;

                // Delta-compress positions
                let mut prev_pos = 0u32;
                for &pos in positions {
                    let delta = pos - prev_pos;
                    writer.write_varint(delta as u64)?;
                    prev_pos = pos;
                }
            } else {
                writer.write_u8(0)?; // No positions flag
            }
        }

        Ok(())
    }

    /// Decode a posting list from binary format.
    pub fn decode<R: StorageInput>(reader: &mut StructReader<R>) -> Result<Self> {
        // Read term
        let term = reader.read_string()?;

        // Read metadata
        let total_frequency = reader.read_varint()?;
        let doc_frequency = reader.read_varint()?;
        let posting_count = reader.read_varint()? as usize;

        let mut postings = Vec::with_capacity(posting_count);
        let mut prev_doc_id = 0u64;

        for _ in 0..posting_count {
            // Read delta-compressed doc ID
            let delta = reader.read_varint()?;
            let doc_id = prev_doc_id + delta;
            prev_doc_id = doc_id;

            // Read frequency
            let frequency = reader.read_varint()? as u32;

            // Read weight
            let weight = reader.read_f32()?;

            // Read positions
            let has_positions = reader.read_u8()? != 0;
            let positions = if has_positions {
                let pos_count = reader.read_varint()? as usize;
                let mut positions = Vec::with_capacity(pos_count);
                let mut prev_pos = 0u32;

                for _ in 0..pos_count {
                    let delta = reader.read_varint()? as u32;
                    let pos = prev_pos + delta;
                    positions.push(pos);
                    prev_pos = pos;
                }

                Some(positions)
            } else {
                None
            };

            postings.push(Posting {
                doc_id,
                frequency,
                positions,
                weight,
            });
        }

        Ok(PostingList {
            term,
            postings,
            total_frequency,
            doc_frequency,
        })
    }
}

/// Simple in-memory posting list iterator.
///
/// # Purpose
/// Used for sequentially processing a `Vec<Posting>` in memory.
///
/// # Implemented Traits
/// - Standard Rust `Iterator` trait
/// - Does NOT implement `reader::PostingIterator` trait
///
/// # Features
/// - Basic iteration (`next()` only)
/// - No skip functionality
/// - No block caching
///
/// # Use Cases
/// - When you need to process an in-memory `Vec<Posting>` rather than reading from an index
/// - When advanced query features (like `skip_to()`) are not needed
pub struct PostingIterator {
    postings: Vec<Posting>,
    position: usize,
}

impl PostingIterator {
    /// Create a new posting iterator.
    pub fn new(postings: Vec<Posting>) -> Self {
        PostingIterator {
            postings,
            position: 0,
        }
    }

    /// Create an empty iterator.
    pub fn empty() -> Self {
        PostingIterator {
            postings: Vec::new(),
            position: 0,
        }
    }

    /// Get the current posting.
    pub fn current(&self) -> Option<&Posting> {
        self.postings.get(self.position)
    }

    /// Advance to the next posting.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Option<&Posting> {
        if self.position < self.postings.len() {
            let posting = &self.postings[self.position];
            self.position += 1;
            Some(posting)
        } else {
            None
        }
    }

    /// Skip to the first posting with doc_id >= target.
    pub fn skip_to(&mut self, target_doc_id: u64) -> bool {
        while self.position < self.postings.len() {
            if self.postings[self.position].doc_id >= target_doc_id {
                return true;
            }
            self.position += 1;
        }
        false
    }

    /// Check if the iterator is exhausted.
    pub fn is_exhausted(&self) -> bool {
        self.position >= self.postings.len()
    }

    /// Get the total number of postings.
    pub fn len(&self) -> usize {
        self.postings.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.postings.is_empty()
    }
}

impl Iterator for PostingIterator {
    type Item = Posting;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position < self.postings.len() {
            let posting = self.postings[self.position].clone();
            self.position += 1;
            Some(posting)
        } else {
            None
        }
    }
}

/// An in-memory index mapping terms to posting lists.
///
/// This is a lightweight data structure used for building segments.
/// It maintains a hash map from terms to their posting lists and provides
/// efficient methods for adding postings and serializing to storage.
#[derive(Debug)]
pub struct TermPostingIndex {
    /// Term dictionary mapping terms to posting lists.
    terms: AHashMap<String, PostingList>,
    /// Total number of documents indexed.
    doc_count: u64,
    /// Total number of terms indexed.
    term_count: u64,
}

impl TermPostingIndex {
    /// Create a new empty term posting index.
    pub fn new() -> Self {
        TermPostingIndex {
            terms: AHashMap::new(),
            doc_count: 0,
            term_count: 0,
        }
    }

    /// Add a posting to the index.
    pub fn add_posting(&mut self, term: String, posting: Posting) {
        let posting_list = self.terms.entry(term.clone()).or_insert_with(|| {
            self.term_count += 1;
            PostingList::new(term)
        });

        posting_list.add_posting(posting);
    }

    /// Add multiple postings for a document.
    pub fn add_document(&mut self, doc_id: u64, terms: Vec<(String, u32, Option<Vec<u32>>)>) {
        for (term, frequency, positions) in terms {
            let posting = if let Some(positions) = positions {
                Posting::with_positions(doc_id, positions)
            } else {
                Posting::with_frequency(doc_id, frequency)
            };

            self.add_posting(term, posting);
        }

        self.doc_count = self.doc_count.max(doc_id + 1);
    }

    /// Get a posting list for a term.
    pub fn get_posting_list(&self, term: &str) -> Option<&PostingList> {
        self.terms.get(term)
    }

    /// Get an iterator for a term.
    pub fn get_posting_iterator(&self, term: &str) -> PostingIterator {
        match self.terms.get(term) {
            Some(posting_list) => PostingIterator::new(posting_list.postings.clone()),
            None => PostingIterator::empty(),
        }
    }

    /// Get the number of documents in the index.
    pub fn doc_count(&self) -> u64 {
        self.doc_count
    }

    /// Get the number of unique terms in the index.
    pub fn term_count(&self) -> u64 {
        self.term_count
    }

    /// Get all terms in the index.
    pub fn terms(&self) -> impl Iterator<Item = &String> {
        self.terms.keys()
    }

    /// Optimize the index by optimizing all posting lists.
    pub fn optimize(&mut self) {
        for posting_list in self.terms.values_mut() {
            posting_list.optimize();
        }
    }

    /// Write the inverted index to storage.
    pub fn write_to_storage<W: StorageOutput>(&self, writer: &mut StructWriter<W>) -> Result<()> {
        // Write header
        writer.write_u32(0x494E5658)?; // Magic number "INVX"
        writer.write_u32(1)?; // Version
        writer.write_varint(self.doc_count)?;
        writer.write_varint(self.term_count)?;
        writer.write_varint(self.terms.len() as u64)?;

        // Sort terms for deterministic output
        let mut sorted_terms: Vec<_> = self.terms.iter().collect();
        sorted_terms.sort_by_key(|(term, _)| *term);

        // Write posting lists
        for (_, posting_list) in sorted_terms {
            posting_list.encode(writer)?;
        }

        Ok(())
    }

    /// Read an inverted index from storage.
    pub fn read_from_storage<R: StorageInput>(reader: &mut StructReader<R>) -> Result<Self> {
        // Read header
        let magic = reader.read_u32()?;
        if magic != 0x494E5658 {
            return Err(PlatypusError::index("Invalid inverted index file format"));
        }

        let version = reader.read_u32()?;
        if version != 1 {
            return Err(PlatypusError::index(format!(
                "Unsupported index version: {version}"
            )));
        }

        let doc_count = reader.read_varint()?;
        let term_count = reader.read_varint()?;
        let posting_list_count = reader.read_varint()? as usize;

        let mut terms = AHashMap::with_capacity(posting_list_count);

        // Read posting lists
        for _ in 0..posting_list_count {
            let posting_list = PostingList::decode(reader)?;
            terms.insert(posting_list.term.clone(), posting_list);
        }

        Ok(TermPostingIndex {
            terms,
            doc_count,
            term_count,
        })
    }
}

impl Default for TermPostingIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about posting lists and the inverted index.
#[derive(Debug, Clone)]
pub struct PostingStats {
    /// Total number of posting lists.
    pub posting_list_count: usize,
    /// Total number of postings.
    pub total_postings: usize,
    /// Average postings per list.
    pub avg_postings_per_list: f64,
    /// Largest posting list size.
    pub max_posting_list_size: usize,
    /// Total compressed size in bytes.
    pub compressed_size: usize,
}

impl TermPostingIndex {
    /// Get statistics about the inverted index.
    pub fn stats(&self) -> PostingStats {
        let posting_list_count = self.terms.len();
        let total_postings: usize = self.terms.values().map(|pl| pl.postings.len()).sum();
        let avg_postings_per_list = if posting_list_count > 0 {
            total_postings as f64 / posting_list_count as f64
        } else {
            0.0
        };
        let max_posting_list_size = self
            .terms
            .values()
            .map(|pl| pl.postings.len())
            .max()
            .unwrap_or(0);

        PostingStats {
            posting_list_count,
            total_postings,
            avg_postings_per_list,
            max_posting_list_size,
            compressed_size: 0, // TODO: Calculate actual compressed size
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

    #[test]
    fn test_posting_creation() {
        let posting = Posting::new(1);
        assert_eq!(posting.doc_id, 1);
        assert_eq!(posting.frequency, 1);
        assert_eq!(posting.positions, None);
        assert_eq!(posting.weight, 1.0);

        let posting = Posting::with_frequency(2, 5);
        assert_eq!(posting.doc_id, 2);
        assert_eq!(posting.frequency, 5);

        let posting = Posting::with_positions(3, vec![10, 20, 30]);
        assert_eq!(posting.doc_id, 3);
        assert_eq!(posting.frequency, 3);
        assert_eq!(posting.positions, Some(vec![10, 20, 30]));
    }

    #[test]
    fn test_posting_list() {
        let mut list = PostingList::new("test".to_string());
        assert!(list.is_empty());

        list.add_posting(Posting::new(1));
        list.add_posting(Posting::new(3));
        list.add_posting(Posting::new(2));

        assert_eq!(list.len(), 3);
        assert_eq!(list.doc_frequency, 3);

        // Should be sorted by doc_id
        let doc_ids: Vec<u64> = list.postings.iter().map(|p| p.doc_id).collect();
        assert_eq!(doc_ids, vec![1, 2, 3]);
    }

    #[test]
    fn test_posting_iterator() {
        let postings = vec![
            Posting::new(1),
            Posting::new(3),
            Posting::new(5),
            Posting::new(7),
        ];

        let mut iter = PostingIterator::new(postings);

        assert_eq!(iter.current().unwrap().doc_id, 1);
        assert_eq!(iter.next().unwrap().doc_id, 1);
        assert_eq!(iter.current().unwrap().doc_id, 3);

        // Test skip_to
        assert!(iter.skip_to(5));
        assert_eq!(iter.current().map(|p| p.doc_id), Some(5));
        assert_eq!(iter.current().unwrap().doc_id, 5);

        // Skip past end
        assert!(!iter.skip_to(10));
        assert!(iter.is_exhausted());
    }

    #[test]
    fn test_inverted_index() {
        let mut index = TermPostingIndex::new();

        // Add document 1: "hello world"
        index.add_document(
            1,
            vec![
                ("hello".to_string(), 1, Some(vec![0])),
                ("world".to_string(), 1, Some(vec![1])),
            ],
        );

        // Add document 2: "hello rust world"
        index.add_document(
            2,
            vec![
                ("hello".to_string(), 1, Some(vec![0])),
                ("rust".to_string(), 1, Some(vec![1])),
                ("world".to_string(), 1, Some(vec![2])),
            ],
        );

        assert_eq!(index.doc_count(), 3); // doc_id 2 + 1
        assert_eq!(index.term_count(), 3); // hello, world, rust

        // Test posting lists
        let hello_list = index.get_posting_list("hello").unwrap();
        assert_eq!(hello_list.postings.len(), 2);
        assert_eq!(hello_list.doc_frequency, 2);

        let rust_list = index.get_posting_list("rust").unwrap();
        assert_eq!(rust_list.postings.len(), 1);
        assert_eq!(rust_list.doc_frequency, 1);

        assert!(index.get_posting_list("nonexistent").is_none());
    }

    #[test]
    fn test_posting_list_encoding() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        let mut original_list = PostingList::new("test".to_string());
        original_list.add_posting(Posting::with_positions(1, vec![0, 5, 10]));
        original_list.add_posting(Posting::with_frequency(3, 2));
        original_list.add_posting(Posting::new(5));

        // Encode
        {
            let output = storage.create_output("test_posting.bin").unwrap();
            let mut writer = StructWriter::new(output);
            original_list.encode(&mut writer).unwrap();
            writer.close().unwrap();
        }

        // Decode
        {
            let input = storage.open_input("test_posting.bin").unwrap();
            let mut reader = StructReader::new(input).unwrap();
            let decoded_list = PostingList::decode(&mut reader).unwrap();

            assert_eq!(decoded_list.term, original_list.term);
            assert_eq!(decoded_list.postings.len(), original_list.postings.len());
            assert_eq!(decoded_list.doc_frequency, original_list.doc_frequency);
            assert_eq!(decoded_list.total_frequency, original_list.total_frequency);

            for (orig, decoded) in original_list
                .postings
                .iter()
                .zip(decoded_list.postings.iter())
            {
                assert_eq!(orig.doc_id, decoded.doc_id);
                assert_eq!(orig.frequency, decoded.frequency);
                assert_eq!(orig.positions, decoded.positions);
            }
        }
    }

    #[test]
    fn test_inverted_index_serialization() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        let mut original_index = TermPostingIndex::new();
        original_index.add_document(
            1,
            vec![
                ("hello".to_string(), 2, Some(vec![0, 5])),
                ("world".to_string(), 1, Some(vec![1])),
            ],
        );
        original_index.add_document(
            2,
            vec![
                ("hello".to_string(), 1, Some(vec![2])),
                ("rust".to_string(), 3, Some(vec![0, 3, 6])),
            ],
        );

        // Write to storage
        {
            let output = storage.create_output("test_index.bin").unwrap();
            let mut writer = StructWriter::new(output);
            original_index.write_to_storage(&mut writer).unwrap();
            writer.close().unwrap();
        }

        // Read from storage
        {
            let input = storage.open_input("test_index.bin").unwrap();
            let mut reader = StructReader::new(input).unwrap();
            let loaded_index = TermPostingIndex::read_from_storage(&mut reader).unwrap();

            assert_eq!(loaded_index.doc_count(), original_index.doc_count());
            assert_eq!(loaded_index.term_count(), original_index.term_count());

            // Test specific terms
            for term in ["hello", "world", "rust"] {
                let orig_list = original_index.get_posting_list(term);
                let loaded_list = loaded_index.get_posting_list(term);

                match (orig_list, loaded_list) {
                    (Some(orig), Some(loaded)) => {
                        assert_eq!(orig.postings.len(), loaded.postings.len());
                        assert_eq!(orig.doc_frequency, loaded.doc_frequency);
                    }
                    (None, None) => {}
                    _ => panic!("Mismatch in term existence: {term}"),
                }
            }
        }
    }

    #[test]
    fn test_posting_stats() {
        let mut index = TermPostingIndex::new();

        // Add several documents
        for doc_id in 0..100 {
            index.add_document(
                doc_id,
                vec![
                    ("common".to_string(), 1, None),
                    (format!("term_{}", doc_id % 10), 1, None),
                ],
            );
        }

        let stats = index.stats();
        assert!(stats.posting_list_count > 0);
        assert!(stats.total_postings > 0);
        assert!(stats.avg_postings_per_list > 0.0);
        assert!(stats.max_posting_list_size > 0);
    }
}
