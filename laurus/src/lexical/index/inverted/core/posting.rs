//! Posting lists and inverted index implementation.
//!
//! This module provides the core inverted index data structures for efficient
//! term-to-document mapping with frequency and position information.

use ahash::AHashMap;
use bitpacking::{BitPacker, BitPacker4x};

use crate::error::{LaurusError, Result};
use crate::storage::structured::{StructReader, StructWriter};
use crate::storage::{StorageInput, StorageOutput};

/// Block length for `BitPacker4x` (SSE3 / scalar fallback): 128 ints per block.
const POSTING_BLOCK_LEN: usize = BitPacker4x::BLOCK_LEN;

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

/// Structure-of-Arrays posting list for cache-efficient iteration.
///
/// Unlike the AoS [`PostingList`] which stores each posting as a struct with
/// `doc_id`, `frequency`, `positions`, and `weight`, this layout stores each
/// field in its own contiguous array.  Sequential access to a single field
/// (e.g. all `doc_ids`) hits a tight cache line sequence instead of striding
/// over position data.
///
/// Position data is **not** included; use the original [`PostingList`] for
/// phrase queries.
#[derive(Debug, Clone)]
pub struct SoAPostingList {
    /// The term this posting list represents.
    pub term: String,
    /// Document IDs, sorted ascending.
    pub doc_ids: Vec<u64>,
    /// Term frequencies, parallel to `doc_ids`.
    pub frequencies: Vec<u32>,
    /// Per-document weights, parallel to `doc_ids`.
    pub weights: Vec<f32>,
    /// Total term frequency across all documents.
    pub total_frequency: u64,
    /// Document frequency (number of documents containing this term).
    pub doc_frequency: u64,
}

impl SoAPostingList {
    /// Returns the number of postings.
    pub fn len(&self) -> usize {
        self.doc_ids.len()
    }

    /// Returns `true` if the posting list is empty.
    pub fn is_empty(&self) -> bool {
        self.doc_ids.is_empty()
    }

    /// Returns an iterator that yields `(doc_id, frequency, weight)` tuples.
    pub fn iter(&self) -> SoAPostingIterator<'_> {
        SoAPostingIterator {
            list: self,
            position: 0,
        }
    }
}

/// Iterator over a [`SoAPostingList`] that yields
/// `(doc_id, frequency, weight)` tuples.
#[derive(Debug)]
pub struct SoAPostingIterator<'a> {
    list: &'a SoAPostingList,
    position: usize,
}

impl<'a> SoAPostingIterator<'a> {
    /// Skip forward until the current doc_id is >= `target`.
    ///
    /// Returns `true` if a posting with `doc_id >= target` was found,
    /// or `false` if the iterator is exhausted.
    ///
    /// # Arguments
    ///
    /// * `target` - The minimum doc_id to seek to.
    pub fn skip_to(&mut self, target: u64) -> bool {
        while self.position < self.list.doc_ids.len() {
            if self.list.doc_ids[self.position] >= target {
                return true;
            }
            self.position += 1;
        }
        // Exhausted — position the cursor at end so next() returns None.
        true
    }
}

impl<'a> Iterator for SoAPostingIterator<'a> {
    /// `(doc_id, frequency, weight)`
    type Item = (u64, u32, f32);

    fn next(&mut self) -> Option<Self::Item> {
        if self.position < self.list.doc_ids.len() {
            let i = self.position;
            self.position += 1;
            Some((
                self.list.doc_ids[i],
                self.list.frequencies[i],
                self.list.weights[i],
            ))
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.list.doc_ids.len() - self.position;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for SoAPostingIterator<'a> {}

/// Compact posting for hot-path scanning where positions are not needed.
///
/// This reduces memory footprint per posting from ~40+ bytes to 16 bytes,
/// improving cache efficiency during posting list traversal.
#[derive(Debug, Clone, PartialEq)]
pub struct CompactPosting {
    /// Document ID.
    pub doc_id: u64,
    /// Term frequency in the document.
    pub frequency: u32,
    /// Document-level weight/boost.
    pub weight: f32,
}

impl CompactPosting {
    /// Create a new compact posting.
    ///
    /// # Arguments
    /// * `doc_id` - Document ID
    /// * `frequency` - Term frequency
    /// * `weight` - Document weight
    pub fn new(doc_id: u64, frequency: u32, weight: f32) -> Self {
        CompactPosting {
            doc_id,
            frequency,
            weight,
        }
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
        // Insert in sorted order by doc_id
        match self
            .postings
            .binary_search_by_key(&posting.doc_id, |p| p.doc_id)
        {
            Ok(pos) => {
                // Document already exists, merge the posting.
                // Only update total_frequency (not doc_frequency, since doc already counted).
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
                // Insert new posting. Update both counters.
                self.total_frequency += posting.frequency as u64;
                self.doc_frequency += 1;
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

    /// Convert postings to compact format, dropping position data.
    ///
    /// Useful for query types that don't need position information (e.g., TermQuery, BooleanQuery).
    ///
    /// # Returns
    /// Vector of compact postings without position data.
    pub fn to_compact(&self) -> Vec<CompactPosting> {
        self.postings
            .iter()
            .map(|p| CompactPosting {
                doc_id: p.doc_id,
                frequency: p.frequency,
                weight: p.weight,
            })
            .collect()
    }

    /// Convert to Structure-of-Arrays layout for cache-efficient iteration.
    ///
    /// The returned [`SoAPostingList`] stores `doc_ids`, `frequencies`, and
    /// `weights` in separate contiguous arrays.  This layout is faster for
    /// sequential BM25 scoring because each field is read in a tight loop
    /// without skipping over position data.
    ///
    /// # Returns
    ///
    /// A new `SoAPostingList` with the same data (positions are dropped).
    pub fn to_soa(&self) -> SoAPostingList {
        let len = self.postings.len();
        let mut doc_ids = Vec::with_capacity(len);
        let mut frequencies = Vec::with_capacity(len);
        let mut weights = Vec::with_capacity(len);

        for p in &self.postings {
            doc_ids.push(p.doc_id);
            frequencies.push(p.frequency);
            weights.push(p.weight);
        }

        SoAPostingList {
            term: self.term.clone(),
            doc_ids,
            frequencies,
            weights,
            total_frequency: self.total_frequency,
            doc_frequency: self.doc_frequency,
        }
    }

    /// Encode the posting list to binary format using block-based bit-packing.
    ///
    /// # Layout
    ///
    /// The on-disk format is structure-of-arrays (SoA), with each field stored
    /// in its own contiguous section:
    ///
    /// ```text
    /// [term: string]
    /// [total_frequency: varint]
    /// [doc_frequency: varint]
    /// [posting_count N: varint]
    /// [any_positions: u8]                     (1 if any posting has positions)
    ///
    /// // Section 1: doc_ids — sorted ascending, FOR + delta bit-packed
    /// repeat (N / 128) times: [num_bits: u8] [packed: num_bits * 16 bytes]
    /// repeat (N % 128) times: [delta: varint]
    ///
    /// // Section 2: frequencies — raw bit-packed
    /// repeat (N / 128) times: [num_bits: u8] [packed: num_bits * 16 bytes]
    /// repeat (N % 128) times: [freq: varint]
    ///
    /// // Section 3: weights — raw f32 array
    /// repeat N times: [weight: f32]
    ///
    /// // Section 4: positions (only when any_positions == 1)
    /// repeat N times:
    ///   [has_positions: u8]
    ///   if 1: [count: varint] [delta: varint] * count
    /// ```
    ///
    /// # Constraints
    ///
    /// - `doc_id` must fit in `u32` (i.e. < 2^32). Per-segment doc-id space is
    ///   bounded by Lucene/Tantivy convention; this is enforced at encode time.
    ///
    /// # Arguments
    ///
    /// * `writer` - The structured output writer.
    pub fn encode<W: StorageOutput>(&self, writer: &mut StructWriter<W>) -> Result<()> {
        writer.write_string(&self.term)?;
        writer.write_varint(self.total_frequency)?;
        writer.write_varint(self.doc_frequency)?;

        let n = self.postings.len();
        writer.write_varint(n as u64)?;

        let any_positions = self.postings.iter().any(|p| p.positions.is_some());
        writer.write_u8(u8::from(any_positions))?;

        if n == 0 {
            return Ok(());
        }

        let bitpacker = BitPacker4x::new();
        let full_blocks = n / POSTING_BLOCK_LEN;
        let tail = n % POSTING_BLOCK_LEN;

        // Section 1: doc_ids (FOR + sorted-delta bit-pack).
        let mut doc_buf = [0u32; POSTING_BLOCK_LEN];
        // Reusable scratch for the largest possible packed block (32 bits per
        // value * 128 / 8 = 512 bytes).
        let mut packed = vec![0u8; 32 * POSTING_BLOCK_LEN / 8];
        let mut initial: u32 = 0;
        for b in 0..full_blocks {
            for (i, slot) in doc_buf.iter_mut().enumerate() {
                let did = self.postings[b * POSTING_BLOCK_LEN + i].doc_id;
                *slot = u32::try_from(did).map_err(|_| {
                    LaurusError::index(format!(
                        "doc_id {did} exceeds u32::MAX; segment is too large for bit-packed posting format"
                    ))
                })?;
            }
            let num_bits = bitpacker.num_bits_sorted(initial, &doc_buf);
            let bytes = num_bits as usize * POSTING_BLOCK_LEN / 8;
            bitpacker.compress_sorted(initial, &doc_buf, &mut packed[..bytes], num_bits);
            writer.write_u8(num_bits)?;
            writer.write_raw(&packed[..bytes])?;
            initial = doc_buf[POSTING_BLOCK_LEN - 1];
        }
        // Tail doc_ids (varint deltas continuing from `initial`).
        let mut prev_did: u64 = initial as u64;
        for i in 0..tail {
            let did = self.postings[full_blocks * POSTING_BLOCK_LEN + i].doc_id;
            // Reject overflow even in the tail so reads stay symmetric.
            u32::try_from(did).map_err(|_| {
                LaurusError::index(format!(
                    "doc_id {did} exceeds u32::MAX; segment is too large for bit-packed posting format"
                ))
            })?;
            writer.write_varint(did - prev_did)?;
            prev_did = did;
        }

        // Section 2: frequencies (raw bit-pack).
        let mut freq_buf = [0u32; POSTING_BLOCK_LEN];
        for b in 0..full_blocks {
            for (i, slot) in freq_buf.iter_mut().enumerate() {
                *slot = self.postings[b * POSTING_BLOCK_LEN + i].frequency;
            }
            let num_bits = bitpacker.num_bits(&freq_buf);
            let bytes = num_bits as usize * POSTING_BLOCK_LEN / 8;
            bitpacker.compress(&freq_buf, &mut packed[..bytes], num_bits);
            writer.write_u8(num_bits)?;
            writer.write_raw(&packed[..bytes])?;
        }
        for i in 0..tail {
            let freq = self.postings[full_blocks * POSTING_BLOCK_LEN + i].frequency;
            writer.write_varint(freq as u64)?;
        }

        // Section 3: weights (raw f32).
        for posting in &self.postings {
            writer.write_f32(posting.weight)?;
        }

        // Section 4: positions (only when at least one posting carries them).
        if any_positions {
            for posting in &self.postings {
                if let Some(positions) = &posting.positions {
                    writer.write_u8(1)?;
                    writer.write_varint(positions.len() as u64)?;
                    let mut prev_pos = 0u32;
                    for &pos in positions {
                        let delta = pos.saturating_sub(prev_pos);
                        writer.write_varint(delta as u64)?;
                        prev_pos = pos;
                    }
                } else {
                    writer.write_u8(0)?;
                }
            }
        }

        Ok(())
    }

    /// Decode a posting list previously written by [`Self::encode`].
    ///
    /// Reads the SoA-laid sections (doc_ids, frequencies, weights, optional
    /// positions) and rebuilds the AoS [`Vec<Posting>`].
    ///
    /// # Arguments
    ///
    /// * `reader` - The structured input reader positioned at a posting-list
    ///   header.
    pub fn decode<R: StorageInput>(reader: &mut StructReader<R>) -> Result<Self> {
        let term = reader.read_string()?;
        let total_frequency = reader.read_varint()?;
        let doc_frequency = reader.read_varint()?;
        let n = reader.read_varint()? as usize;
        let any_positions = reader.read_u8()? != 0;

        if n == 0 {
            return Ok(PostingList {
                term,
                postings: Vec::new(),
                total_frequency,
                doc_frequency,
            });
        }

        let bitpacker = BitPacker4x::new();
        let full_blocks = n / POSTING_BLOCK_LEN;
        let tail = n % POSTING_BLOCK_LEN;

        // Section 1: doc_ids.
        let mut doc_ids: Vec<u32> = Vec::with_capacity(n);
        let mut buf = [0u32; POSTING_BLOCK_LEN];
        let mut initial: u32 = 0;
        for _ in 0..full_blocks {
            let num_bits = reader.read_u8()?;
            let bytes = num_bits as usize * POSTING_BLOCK_LEN / 8;
            let compressed = reader.read_raw(bytes)?;
            bitpacker.decompress_sorted(initial, &compressed, &mut buf, num_bits);
            doc_ids.extend_from_slice(&buf);
            initial = buf[POSTING_BLOCK_LEN - 1];
        }
        let mut prev_did: u64 = initial as u64;
        for _ in 0..tail {
            let delta = reader.read_varint()?;
            let did = prev_did + delta;
            doc_ids.push(u32::try_from(did).map_err(|_| {
                LaurusError::index(format!(
                    "decoded doc_id {did} exceeds u32::MAX; corrupted posting list"
                ))
            })?);
            prev_did = did;
        }

        // Section 2: frequencies.
        let mut frequencies: Vec<u32> = Vec::with_capacity(n);
        for _ in 0..full_blocks {
            let num_bits = reader.read_u8()?;
            let bytes = num_bits as usize * POSTING_BLOCK_LEN / 8;
            let compressed = reader.read_raw(bytes)?;
            bitpacker.decompress(&compressed, &mut buf, num_bits);
            frequencies.extend_from_slice(&buf);
        }
        for _ in 0..tail {
            frequencies.push(reader.read_varint()? as u32);
        }

        // Section 3: weights.
        let mut weights: Vec<f32> = Vec::with_capacity(n);
        for _ in 0..n {
            weights.push(reader.read_f32()?);
        }

        // Section 4: positions.
        let positions_per_posting: Vec<Option<Vec<u32>>> = if any_positions {
            let mut out = Vec::with_capacity(n);
            for _ in 0..n {
                let has = reader.read_u8()? != 0;
                if has {
                    let count = reader.read_varint()? as usize;
                    let mut positions = Vec::with_capacity(count);
                    let mut prev_pos = 0u32;
                    for _ in 0..count {
                        let delta = reader.read_varint()? as u32;
                        let pos = prev_pos + delta;
                        positions.push(pos);
                        prev_pos = pos;
                    }
                    out.push(Some(positions));
                } else {
                    out.push(None);
                }
            }
            out
        } else {
            (0..n).map(|_| None).collect()
        };

        let postings: Vec<Posting> = doc_ids
            .into_iter()
            .zip(frequencies)
            .zip(weights)
            .zip(positions_per_posting)
            .map(|(((did, freq), w), pos)| Posting {
                doc_id: did as u64,
                frequency: freq,
                positions: pos,
                weight: w,
            })
            .collect();

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
            return Err(LaurusError::index("Invalid inverted index file format"));
        }

        let version = reader.read_u32()?;
        if version != 1 {
            return Err(LaurusError::index(format!(
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

    #[test]
    fn test_soa_posting_list() {
        let mut list = PostingList::new("hello".to_string());
        list.add_posting(Posting::with_frequency(1, 3).with_weight(1.0));
        list.add_posting(Posting::with_frequency(5, 1).with_weight(2.0));
        list.add_posting(Posting::with_frequency(9, 2).with_weight(0.5));

        let soa = list.to_soa();
        assert_eq!(soa.len(), 3);
        assert_eq!(soa.doc_ids, &[1, 5, 9]);
        assert_eq!(soa.frequencies, &[3, 1, 2]);
        assert_eq!(soa.weights, &[1.0, 2.0, 0.5]);
        assert_eq!(soa.term, "hello");
        assert_eq!(soa.total_frequency, list.total_frequency);
        assert_eq!(soa.doc_frequency, list.doc_frequency);

        // Test iterator
        let mut iter = soa.iter();
        let first = iter.next().unwrap();
        assert_eq!(first, (1, 3, 1.0));
        let second = iter.next().unwrap();
        assert_eq!(second, (5, 1, 2.0));
        let third = iter.next().unwrap();
        assert_eq!(third, (9, 2, 0.5));
        assert!(iter.next().is_none());

        // Test skip_to
        let mut iter = soa.iter();
        assert!(iter.skip_to(5));
        assert_eq!(iter.next().unwrap(), (5, 1, 2.0));
        assert!(iter.skip_to(100));
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_compact_posting() {
        let posting = CompactPosting::new(42, 3, 1.5);
        assert_eq!(posting.doc_id, 42);
        assert_eq!(posting.frequency, 3);
        assert_eq!(posting.weight, 1.5);
    }

    #[test]
    fn test_posting_list_to_compact() {
        let mut list = PostingList::new("test".to_string());
        list.add_posting(Posting::with_positions(1, vec![0, 5, 10]).with_weight(1.0));
        list.add_posting(Posting::with_positions(2, vec![3, 7]).with_weight(2.0));

        let compact = list.to_compact();
        assert_eq!(compact.len(), 2);
        assert_eq!(compact[0].doc_id, 1);
        assert_eq!(compact[0].frequency, 3);
        assert_eq!(compact[0].weight, 1.0);
        assert_eq!(compact[1].doc_id, 2);
        assert_eq!(compact[1].frequency, 2);
        assert_eq!(compact[1].weight, 2.0);
    }

    #[test]
    fn test_compact_posting_size() {
        assert_eq!(std::mem::size_of::<CompactPosting>(), 16);
    }

    /// Round-trip a posting list of size `n` through encode/decode and assert
    /// that every field is preserved. `with_positions` controls whether each
    /// posting carries position data.
    fn round_trip_n(n: usize, with_positions: bool) {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        let mut original = PostingList::new(format!("term_n{n}"));
        // Build deterministic but non-trivial postings: doc_id step 3, frequency
        // varying 1..=7, weight derived from doc_id.
        for i in 0..n {
            let did = (i as u64) * 3 + 1;
            let freq = ((i % 7) + 1) as u32;
            let weight = 0.25 + (i % 4) as f32 * 0.5;
            let posting = if with_positions {
                let mut positions = Vec::with_capacity(freq as usize);
                let mut p = (i % 11) as u32;
                for _ in 0..freq {
                    positions.push(p);
                    p += 2;
                }
                Posting::with_positions(did, positions).with_weight(weight)
            } else {
                Posting::with_frequency(did, freq).with_weight(weight)
            };
            original.add_posting(posting);
        }

        let path = format!("rt_n{n}_pos{with_positions}.bin");
        {
            let output = storage.create_output(&path).unwrap();
            let mut writer = StructWriter::new(output);
            original.encode(&mut writer).unwrap();
            writer.close().unwrap();
        }
        let input = storage.open_input(&path).unwrap();
        let mut reader = StructReader::new(input).unwrap();
        let decoded = PostingList::decode(&mut reader).unwrap();

        assert_eq!(decoded.term, original.term, "term mismatch (n={n})");
        assert_eq!(
            decoded.total_frequency, original.total_frequency,
            "total_frequency mismatch (n={n})"
        );
        assert_eq!(
            decoded.doc_frequency, original.doc_frequency,
            "doc_frequency mismatch (n={n})"
        );
        assert_eq!(
            decoded.postings.len(),
            original.postings.len(),
            "len mismatch (n={n})"
        );
        for (i, (orig, dec)) in original
            .postings
            .iter()
            .zip(decoded.postings.iter())
            .enumerate()
        {
            assert_eq!(orig.doc_id, dec.doc_id, "doc_id mismatch at i={i} (n={n})");
            assert_eq!(
                orig.frequency, dec.frequency,
                "frequency mismatch at i={i} (n={n})"
            );
            assert_eq!(orig.weight, dec.weight, "weight mismatch at i={i} (n={n})");
            assert_eq!(
                orig.positions, dec.positions,
                "positions mismatch at i={i} (n={n})"
            );
        }
    }

    #[test]
    fn test_round_trip_empty() {
        round_trip_n(0, false);
        round_trip_n(0, true);
    }

    #[test]
    fn test_round_trip_single() {
        round_trip_n(1, false);
        round_trip_n(1, true);
    }

    /// Block size minus one: tail-only path, no full bit-packed block.
    #[test]
    fn test_round_trip_below_block() {
        round_trip_n(127, false);
        round_trip_n(127, true);
    }

    /// Exactly one full block, zero tail.
    #[test]
    fn test_round_trip_exact_block() {
        round_trip_n(128, false);
        round_trip_n(128, true);
    }

    /// One full block plus a single tail element.
    #[test]
    fn test_round_trip_block_plus_one() {
        round_trip_n(129, false);
        round_trip_n(129, true);
    }

    #[test]
    fn test_round_trip_two_blocks() {
        round_trip_n(256, false);
        round_trip_n(256, true);
    }

    #[test]
    fn test_round_trip_many_blocks() {
        round_trip_n(1000, false);
        round_trip_n(1000, true);
    }

    /// Mixed positions (some postings carry positions, some don't) verifies
    /// that the per-posting `has_positions` flag inside section 4 still works.
    #[test]
    fn test_round_trip_mixed_positions() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let mut original = PostingList::new("mixed".to_string());
        for i in 0..200u64 {
            let mut posting = Posting::with_frequency(i * 2, ((i % 5) + 1) as u32);
            if i % 3 == 0 {
                posting.add_position((i % 17) as u32);
                posting.add_position(((i % 17) + 5) as u32);
            }
            original.add_posting(posting);
        }

        let path = "rt_mixed.bin";
        {
            let output = storage.create_output(path).unwrap();
            let mut writer = StructWriter::new(output);
            original.encode(&mut writer).unwrap();
            writer.close().unwrap();
        }
        let input = storage.open_input(path).unwrap();
        let mut reader = StructReader::new(input).unwrap();
        let decoded = PostingList::decode(&mut reader).unwrap();

        assert_eq!(decoded.postings.len(), original.postings.len());
        for (orig, dec) in original.postings.iter().zip(decoded.postings.iter()) {
            assert_eq!(orig, dec);
        }
    }

    /// Encoding a doc_id beyond `u32::MAX` must fail with a clear error so we
    /// don't silently corrupt the bit-packed segment.
    #[test]
    fn test_encode_rejects_u64_doc_id_overflow() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let mut list = PostingList::new("overflow".to_string());
        // Fill a full block so the overflow check exercises the bit-packed
        // path, not just the tail.
        for i in 0..127u64 {
            list.add_posting(Posting::new(i));
        }
        list.add_posting(Posting::new(u64::from(u32::MAX) + 1));

        let output = storage.create_output("overflow.bin").unwrap();
        let mut writer = StructWriter::new(output);
        let err = list
            .encode(&mut writer)
            .expect_err("expected overflow error");
        let msg = format!("{err}");
        assert!(
            msg.contains("exceeds u32::MAX"),
            "unexpected error message: {msg}"
        );
    }
}
