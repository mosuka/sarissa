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

/// Branching factor for the multi-level skip table over a posting list
/// (#503). Matches Lucene 90's `Lucene90PostingsFormat` — every level
/// has 1/`SKIP_INTERVAL` of the lower level's entries, so seeking via
/// [`InvertedIndexPostingIterator::skip_to`][super::super::reader::InvertedIndexPostingIterator]
/// reaches O(log_8 N) instead of the linear-scan O(N) that
/// `block_cache` paid before.
///
/// The constant is co-located with the on-disk encoder
/// ([`PostingList::encode_v2`]) because the writer and reader must
/// agree on the same interval; bumping it requires a format bump.
pub const SKIP_INTERVAL: usize = 8;

/// Build the multi-level skip table over a sorted doc-id slice.
///
/// Each output level holds the "last doc id" of each `step`-wide window
/// of `doc_ids`, where `step = SKIP_INTERVAL^(level + 1)`. The function
/// keeps adding levels until the top one has at most `SKIP_INTERVAL`
/// entries — at that point a single `partition_point` on the top level
/// covers the whole posting list.
///
/// Returns an empty `Vec` when `doc_ids.len() < SKIP_INTERVAL`: the
/// linear-scan fallback inside `skip_to` is already O(N) ≤ O(SKIP_INTERVAL)
/// for these short lists, so paying the skip-table build cost is a net
/// loss.
///
/// # Arguments
///
/// * `doc_ids` - Ascending-sorted doc ids of the posting list.
///
/// # Returns
///
/// `Vec<Vec<u32>>` where index `0` is level 0 (step = `SKIP_INTERVAL`)
/// and the last index is the top level (≤ `SKIP_INTERVAL` entries).
pub fn build_skip_levels(doc_ids: &[u32]) -> Vec<Vec<u32>> {
    let n = doc_ids.len();
    if n < SKIP_INTERVAL {
        return Vec::new();
    }

    let mut levels: Vec<Vec<u32>> = Vec::new();
    let mut step = SKIP_INTERVAL;
    // Level 0: stride directly over `doc_ids`.
    loop {
        let len = n / step;
        if len == 0 {
            break;
        }
        let mut level = Vec::with_capacity(len);
        for i in 0..len {
            // Last doc id of the i-th window of `step` postings.
            level.push(doc_ids[(i + 1) * step - 1]);
        }
        levels.push(level);
        // Stop once the top level has collapsed to a single window —
        // a further level would have zero entries.
        if len <= 1 {
            break;
        }
        // Saturate to avoid overflow on absurdly large lists.
        step = match step.checked_mul(SKIP_INTERVAL) {
            Some(s) => s,
            None => break,
        };
    }
    levels
}

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

/// SoA-native decoded posting data, produced directly by
/// [`PostingList::decode_soa`] without an intermediate `Vec<Posting>`
/// reassembly.
///
/// Designed for the query hot path: the iterator can be backed by parallel
/// arrays and serve `doc_id` / `term_freq` / `weight` from sequential `u32` /
/// `f32` slices instead of striding over a 40-byte `Posting` struct.
///
/// Doc IDs are kept as `u32` because per-segment doc-id space is bounded to
/// `u32::MAX` (Lucene / Tantivy convention) — see [`PostingList::encode`] for
/// the matching invariant on the writer side.
#[derive(Debug, Clone)]
pub struct DecodedPostingList {
    /// The term this posting list represents.
    pub term: String,
    /// Doc IDs, sorted ascending. Parallel to `frequencies` / `weights`.
    pub doc_ids: Vec<u32>,
    /// Term frequencies. Parallel to `doc_ids` / `weights`.
    pub frequencies: Vec<u32>,
    /// Per-document weights. Parallel to `doc_ids` / `frequencies`.
    pub weights: Vec<f32>,
    /// Optional per-posting positions sidecar. `None` when **no** posting in
    /// this list carries position data (the common case for boolean / BM25
    /// queries that don't need phrases). When `Some(v)`, `v[i]` is the
    /// positions for posting `i`: `Some(Vec<u32>)` if positions are present,
    /// `None` otherwise.
    pub positions: Option<Vec<Option<Vec<u32>>>>,
    /// Multi-level skip table over `doc_ids` (#503). Index 0 is level 0
    /// (step = [`SKIP_INTERVAL`]); the last index is the top level
    /// (≤ `SKIP_INTERVAL` entries). Empty when the posting list is too
    /// short to benefit (`doc_ids.len() < SKIP_INTERVAL`) or when the
    /// decoder loaded a legacy v1 segment that did not carry skip
    /// metadata — in the latter case the reader path rebuilds the
    /// table on load via [`build_skip_levels`].
    pub skip_levels: Vec<Vec<u32>>,
    /// Total term frequency across all documents.
    pub total_frequency: u64,
    /// Document frequency (number of documents containing this term).
    pub doc_frequency: u64,
}

impl DecodedPostingList {
    /// Number of postings in this list.
    pub fn len(&self) -> usize {
        self.doc_ids.len()
    }

    /// Returns `true` if there are no postings.
    pub fn is_empty(&self) -> bool {
        self.doc_ids.is_empty()
    }

    /// Build a SoA decoded view from an AoS [`PostingList`]. Used by callers
    /// that already hold an in-memory `Vec<Posting>` and want the SoA-native
    /// iterator path.
    ///
    /// # Arguments
    ///
    /// * `list` - The AoS posting list to convert.
    pub fn from_posting_list(list: &PostingList) -> Self {
        let n = list.postings.len();
        let mut doc_ids = Vec::with_capacity(n);
        let mut frequencies = Vec::with_capacity(n);
        let mut weights = Vec::with_capacity(n);
        let any_positions = list.postings.iter().any(|p| p.positions.is_some());
        let mut positions: Option<Vec<Option<Vec<u32>>>> = if any_positions {
            Some(Vec::with_capacity(n))
        } else {
            None
        };

        for posting in &list.postings {
            doc_ids.push(posting.doc_id as u32);
            frequencies.push(posting.frequency);
            weights.push(posting.weight);
            if let Some(out) = positions.as_mut() {
                out.push(posting.positions.clone());
            }
        }

        let skip_levels = build_skip_levels(&doc_ids);
        DecodedPostingList {
            term: list.term.clone(),
            doc_ids,
            frequencies,
            weights,
            positions,
            skip_levels,
            total_frequency: list.total_frequency,
            doc_frequency: list.doc_frequency,
        }
    }

    /// Reassemble an AoS [`PostingList`] from this SoA view. Useful for tests
    /// and back-compat code paths that still expect `Vec<Posting>`.
    pub fn into_posting_list(self) -> PostingList {
        let n = self.doc_ids.len();
        let positions_iter: Box<dyn Iterator<Item = Option<Vec<u32>>>> = match self.positions {
            Some(v) => Box::new(v.into_iter()),
            None => Box::new(std::iter::repeat_with(|| None).take(n)),
        };

        let postings: Vec<Posting> = self
            .doc_ids
            .into_iter()
            .zip(self.frequencies)
            .zip(self.weights)
            .zip(positions_iter)
            .map(|(((did, freq), w), pos)| Posting {
                doc_id: did as u64,
                frequency: freq,
                positions: pos,
                weight: w,
            })
            .collect();

        PostingList {
            term: self.term,
            postings,
            total_frequency: self.total_frequency,
            doc_frequency: self.doc_frequency,
        }
    }
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
        self.encode_header_and_payload(writer, /* with_skip_levels */ false)
    }

    /// Encode the posting list in v2 format, which adds a multi-level
    /// skip table after the header (#503).
    ///
    /// The header and payload sections (doc_ids / frequencies / weights /
    /// positions) are byte-identical to [`Self::encode`]; the only
    /// difference is the new **Skip levels** section inserted between
    /// `any_positions` and Section 1. Concretely:
    ///
    /// ```text
    /// [term: string]
    /// [total_frequency: varint]
    /// [doc_frequency: varint]
    /// [posting_count N: varint]
    /// [any_positions: u8]
    ///
    /// // ───── New v2 section: multi-level skip table ─────
    /// [num_skip_levels: u8]
    /// repeat num_skip_levels times:
    ///   [level_len: varint]
    ///   repeat level_len times: [doc_id: u32]   // raw little-endian u32
    ///
    /// // Sections 1-4 identical to v1
    /// ```
    ///
    /// A v1 reader cannot decode a v2 payload — the format is gated by
    /// [`TermPostingIndex`]'s on-disk version field, which v2 readers
    /// inspect before dispatching to [`Self::decode_soa_v2`].
    ///
    /// # Arguments
    ///
    /// * `writer` - The structured output writer.
    pub fn encode_v2<W: StorageOutput>(&self, writer: &mut StructWriter<W>) -> Result<()> {
        self.encode_header_and_payload(writer, /* with_skip_levels */ true)
    }

    /// Shared encoder used by both [`Self::encode`] (v1) and
    /// [`Self::encode_v2`]. The two only differ in whether the
    /// **Skip levels** section is emitted between the header and
    /// Section 1.
    fn encode_header_and_payload<W: StorageOutput>(
        &self,
        writer: &mut StructWriter<W>,
        with_skip_levels: bool,
    ) -> Result<()> {
        writer.write_string(&self.term)?;
        writer.write_varint(self.total_frequency)?;
        writer.write_varint(self.doc_frequency)?;

        let n = self.postings.len();
        writer.write_varint(n as u64)?;

        let any_positions = self.postings.iter().any(|p| p.positions.is_some());
        writer.write_u8(u8::from(any_positions))?;

        if with_skip_levels {
            // Build the skip table from the in-memory postings. The
            // doc_ids we send through `build_skip_levels` must match
            // the ones the bit-packer is about to emit — we already
            // enforce `doc_id ≤ u32::MAX` in Section 1, so reuse the
            // same conversion + overflow check here.
            let mut doc_ids_u32 = Vec::with_capacity(n);
            for posting in &self.postings {
                let did = posting.doc_id;
                doc_ids_u32.push(u32::try_from(did).map_err(|_| {
                    LaurusError::index(format!(
                        "doc_id {did} exceeds u32::MAX; segment is too large for bit-packed posting format"
                    ))
                })?);
            }
            let levels = build_skip_levels(&doc_ids_u32);
            writer.write_u8(u8::try_from(levels.len()).map_err(|_| {
                LaurusError::index(format!(
                    "skip level count {} exceeds u8::MAX; refusing to encode",
                    levels.len()
                ))
            })?)?;
            for level in &levels {
                writer.write_varint(level.len() as u64)?;
                for &did in level {
                    writer.write_u32(did)?;
                }
            }
        }

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

    /// Decode a posting list previously written by [`Self::encode`] in
    /// SoA-native form, **without** an intermediate `Vec<Posting>`
    /// reassembly.
    ///
    /// This is the fast path for the query hot loop: callers can store the
    /// returned [`DecodedPostingList`] in their iterator directly and serve
    /// `doc_id` / `term_freq` / `weight` from the parallel `u32` / `f32`
    /// slices, avoiding both the malloc and the AoS struct stride that the
    /// `Vec<Posting>` form pays.
    ///
    /// # Arguments
    ///
    /// * `reader` - The structured input reader positioned at a posting-list
    ///   header.
    pub fn decode_soa<R: StorageInput>(reader: &mut StructReader<R>) -> Result<DecodedPostingList> {
        Self::decode_soa_inner(reader, /* with_skip_levels */ false)
    }

    /// Decode a posting list previously written by [`Self::encode_v2`]
    /// (#503). Reads the on-disk skip levels into
    /// [`DecodedPostingList::skip_levels`] verbatim; the iterator then
    /// uses them directly without rebuilding from `doc_ids`.
    ///
    /// # Arguments
    ///
    /// * `reader` - The structured input reader positioned at a v2
    ///   posting-list header.
    pub fn decode_soa_v2<R: StorageInput>(
        reader: &mut StructReader<R>,
    ) -> Result<DecodedPostingList> {
        Self::decode_soa_inner(reader, /* with_skip_levels */ true)
    }

    /// Shared decoder used by both [`Self::decode_soa`] (v1) and
    /// [`Self::decode_soa_v2`]. The two only differ in whether the
    /// **Skip levels** section is consumed from the input stream; v1
    /// rebuilds the skip table from `doc_ids` at the end instead.
    fn decode_soa_inner<R: StorageInput>(
        reader: &mut StructReader<R>,
        with_skip_levels: bool,
    ) -> Result<DecodedPostingList> {
        let term = reader.read_string()?;
        let total_frequency = reader.read_varint()?;
        let doc_frequency = reader.read_varint()?;
        let n = reader.read_varint()? as usize;
        let any_positions = reader.read_u8()? != 0;

        // v2-only section: multi-level skip table. Always present (even
        // for short posting lists where `num_skip_levels = 0`), so the
        // byte layout stays deterministic.
        let mut disk_skip_levels: Vec<Vec<u32>> = Vec::new();
        if with_skip_levels {
            let num_levels = reader.read_u8()? as usize;
            disk_skip_levels.reserve(num_levels);
            for _ in 0..num_levels {
                let level_len = reader.read_varint()? as usize;
                let mut level = Vec::with_capacity(level_len);
                for _ in 0..level_len {
                    level.push(reader.read_u32()?);
                }
                disk_skip_levels.push(level);
            }
        }

        if n == 0 {
            return Ok(DecodedPostingList {
                term,
                doc_ids: Vec::new(),
                frequencies: Vec::new(),
                weights: Vec::new(),
                positions: if any_positions {
                    Some(Vec::new())
                } else {
                    None
                },
                skip_levels: disk_skip_levels,
                total_frequency,
                doc_frequency,
            });
        }

        let bitpacker = BitPacker4x::new();
        let full_blocks = n / POSTING_BLOCK_LEN;
        let tail = n % POSTING_BLOCK_LEN;

        // Section 1: doc_ids.
        //
        // Issue #504: prefer the zero-copy `read_raw_with` path so
        // mmap-backed segments hand the compressed block straight to
        // `bitpacking` without the intermediate `Vec<u8>` allocation +
        // `copy_from_slice` that `read_raw` would otherwise perform.
        // The fallback (buffered file I/O) still works through the
        // same callback — it just routes through a heap buffer.
        let mut doc_ids: Vec<u32> = Vec::with_capacity(n);
        let mut buf = [0u32; POSTING_BLOCK_LEN];
        let mut initial: u32 = 0;
        for _ in 0..full_blocks {
            let num_bits = reader.read_u8()?;
            let bytes = num_bits as usize * POSTING_BLOCK_LEN / 8;
            reader.read_raw_with(bytes, |compressed| {
                bitpacker.decompress_sorted(initial, compressed, &mut buf, num_bits);
            })?;
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

        // Section 2: frequencies (same zero-copy block path).
        let mut frequencies: Vec<u32> = Vec::with_capacity(n);
        for _ in 0..full_blocks {
            let num_bits = reader.read_u8()?;
            let bytes = num_bits as usize * POSTING_BLOCK_LEN / 8;
            reader.read_raw_with(bytes, |compressed| {
                bitpacker.decompress(compressed, &mut buf, num_bits);
            })?;
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

        // Section 4: positions (only materialised when at least one posting
        // carries them; absent otherwise to keep the SoA path zero-allocation
        // for the common BM25 / boolean case).
        let positions = if any_positions {
            let mut out: Vec<Option<Vec<u32>>> = Vec::with_capacity(n);
            for _ in 0..n {
                let has = reader.read_u8()? != 0;
                if has {
                    let count = reader.read_varint()? as usize;
                    let mut p = Vec::with_capacity(count);
                    let mut prev_pos = 0u32;
                    for _ in 0..count {
                        let delta = reader.read_varint()? as u32;
                        let pos = prev_pos + delta;
                        p.push(pos);
                        prev_pos = pos;
                    }
                    out.push(Some(p));
                } else {
                    out.push(None);
                }
            }
            Some(out)
        } else {
            None
        };

        // Issue #503: v2 segments carry skip levels on disk; v1 segments
        // do not, so build the table from the decoded `doc_ids` at load
        // time. The build cost is paid once per segment open, not per
        // query, so the fallback path stays cheap.
        let skip_levels = if with_skip_levels {
            disk_skip_levels
        } else {
            build_skip_levels(&doc_ids)
        };

        Ok(DecodedPostingList {
            term,
            doc_ids,
            frequencies,
            weights,
            positions,
            skip_levels,
            total_frequency,
            doc_frequency,
        })
    }

    /// Decode a posting list previously written by [`Self::encode`] into the
    /// AoS [`Vec<Posting>`] form. Thin wrapper over [`Self::decode_soa`] kept
    /// for callers that need positions per `Posting` or want a back-compat
    /// view.
    ///
    /// # Arguments
    ///
    /// * `reader` - The structured input reader positioned at a posting-list
    ///   header.
    pub fn decode<R: StorageInput>(reader: &mut StructReader<R>) -> Result<Self> {
        Ok(Self::decode_soa(reader)?.into_posting_list())
    }

    /// Decode a posting list previously written by [`Self::encode_v2`]
    /// into the AoS [`Vec<Posting>`] form (#503). The on-disk skip
    /// levels are consumed but discarded — AoS callers (tests, legacy
    /// code paths) do not use them.
    ///
    /// # Arguments
    ///
    /// * `reader` - The structured input reader positioned at a v2
    ///   posting-list header.
    pub fn decode_v2<R: StorageInput>(reader: &mut StructReader<R>) -> Result<Self> {
        Ok(Self::decode_soa_v2(reader)?.into_posting_list())
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

    /// On-disk version of the [`TermPostingIndex`] format used by
    /// [`Self::write_to_storage`]. Version 2 introduces the
    /// multi-level skip table per posting list (#503); v1 segments
    /// remain readable via [`Self::read_from_storage`]'s back-compat
    /// branch.
    const ON_DISK_VERSION: u32 = 2;

    /// Write the inverted index to storage.
    pub fn write_to_storage<W: StorageOutput>(&self, writer: &mut StructWriter<W>) -> Result<()> {
        // Write header
        writer.write_u32(0x494E5658)?; // Magic number "INVX"
        writer.write_u32(Self::ON_DISK_VERSION)?;
        writer.write_varint(self.doc_count)?;
        writer.write_varint(self.term_count)?;
        writer.write_varint(self.terms.len() as u64)?;

        // Sort terms for deterministic output
        let mut sorted_terms: Vec<_> = self.terms.iter().collect();
        sorted_terms.sort_by_key(|(term, _)| *term);

        // v2: every posting list carries an on-disk skip table.
        for (_, posting_list) in sorted_terms {
            posting_list.encode_v2(writer)?;
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
        if version != 1 && version != 2 {
            return Err(LaurusError::index(format!(
                "Unsupported index version: {version}"
            )));
        }

        let doc_count = reader.read_varint()?;
        let term_count = reader.read_varint()?;
        let posting_list_count = reader.read_varint()? as usize;

        let mut terms = AHashMap::with_capacity(posting_list_count);

        // Dispatch posting-list decode by on-disk version. v1 segments
        // do not carry skip levels; the SoA decoder rebuilds them at
        // load time (#503 back-compat fallback).
        for _ in 0..posting_list_count {
            let posting_list = if version == 1 {
                PostingList::decode(reader)?
            } else {
                PostingList::decode_v2(reader)?
            };
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

    /// `decode_soa` must produce the same data as the AoS `decode` path
    /// across full-block / tail / mixed-positions sizes. This protects the
    /// fast SoA path against drift from the back-compat AoS path.
    #[test]
    fn test_decode_soa_matches_decode_aos() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        for &(n, with_pos) in &[
            (0usize, false),
            (1, false),
            (1, true),
            (127, false),
            (128, true),
            (129, false),
            (256, true),
            (1000, false),
        ] {
            let mut original = PostingList::new(format!("term_n{n}_p{with_pos}"));
            for i in 0..n {
                let did = (i as u64) * 3 + 1;
                let freq = ((i % 7) + 1) as u32;
                let weight = 0.25 + (i % 4) as f32 * 0.5;
                let p = if with_pos {
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
                original.add_posting(p);
            }

            let path = format!("soa_match_n{n}_p{with_pos}.bin");
            {
                let output = storage.create_output(&path).unwrap();
                let mut writer = StructWriter::new(output);
                original.encode(&mut writer).unwrap();
                writer.close().unwrap();
            }

            // Decode through both paths and compare.
            let aos_decoded = {
                let input = storage.open_input(&path).unwrap();
                let mut reader = StructReader::new(input).unwrap();
                PostingList::decode(&mut reader).unwrap()
            };
            let soa_decoded = {
                let input = storage.open_input(&path).unwrap();
                let mut reader = StructReader::new(input).unwrap();
                PostingList::decode_soa(&mut reader).unwrap()
            };

            assert_eq!(aos_decoded.term, soa_decoded.term);
            assert_eq!(aos_decoded.total_frequency, soa_decoded.total_frequency);
            assert_eq!(aos_decoded.doc_frequency, soa_decoded.doc_frequency);
            assert_eq!(aos_decoded.postings.len(), soa_decoded.len());
            for (i, p) in aos_decoded.postings.iter().enumerate() {
                assert_eq!(p.doc_id as u32, soa_decoded.doc_ids[i], "doc_id at {i}");
                assert_eq!(p.frequency, soa_decoded.frequencies[i], "freq at {i}");
                assert_eq!(p.weight, soa_decoded.weights[i], "weight at {i}");
                let soa_pos = soa_decoded.positions.as_ref().and_then(|v| v[i].clone());
                assert_eq!(p.positions, soa_pos, "positions at {i}");
            }
        }
    }

    /// `DecodedPostingList::from_posting_list` followed by
    /// `into_posting_list` must round-trip every field on both
    /// positions-present and positions-absent shapes.
    #[test]
    fn test_decoded_posting_list_aos_soa_roundtrip() {
        let mut list = PostingList::new("term".to_string());
        for i in 0..200u64 {
            let mut p = Posting::with_frequency(i * 2 + 7, ((i % 5) + 1) as u32)
                .with_weight(0.5 + (i % 3) as f32);
            if i % 4 == 0 {
                p.add_position((i % 13) as u32);
                p.add_position(((i % 13) + 4) as u32);
            }
            list.add_posting(p);
        }

        let soa = DecodedPostingList::from_posting_list(&list);
        assert_eq!(soa.len(), list.postings.len());
        let rebuilt = soa.into_posting_list();
        assert_eq!(rebuilt.term, list.term);
        assert_eq!(rebuilt.total_frequency, list.total_frequency);
        assert_eq!(rebuilt.doc_frequency, list.doc_frequency);
        for (orig, dec) in list.postings.iter().zip(rebuilt.postings.iter()) {
            assert_eq!(orig, dec);
        }
    }

    // ─────────────────────────────────────────────────────────────
    // #503: multi-level skip table — build + on-disk v2 round-trip.
    // ─────────────────────────────────────────────────────────────

    /// `build_skip_levels` must return an empty `Vec` for posting lists
    /// shorter than `SKIP_INTERVAL` — the tail linear scan in `skip_to`
    /// is already O(SKIP_INTERVAL) for these cases, so paying the
    /// skip-table cost is a net loss.
    #[test]
    fn test_build_skip_levels_below_interval() {
        for n in 0..SKIP_INTERVAL {
            let doc_ids: Vec<u32> = (0..n as u32).collect();
            let levels = build_skip_levels(&doc_ids);
            assert!(
                levels.is_empty(),
                "expected empty skip levels at n={n}, got {levels:?}"
            );
        }
    }

    /// Exactly one bottom-level entry: `n = SKIP_INTERVAL`. The top
    /// level should have a single entry equal to the last doc id.
    #[test]
    fn test_build_skip_levels_single_block() {
        let doc_ids: Vec<u32> = (0..SKIP_INTERVAL as u32).collect();
        let levels = build_skip_levels(&doc_ids);
        assert_eq!(levels.len(), 1, "{levels:?}");
        assert_eq!(levels[0], vec![SKIP_INTERVAL as u32 - 1]);
    }

    /// At `n = SKIP_INTERVAL * SKIP_INTERVAL` the table should top out
    /// at exactly two levels: L0 with `SKIP_INTERVAL` entries, L1 with
    /// 1 entry pointing at the last doc id.
    #[test]
    fn test_build_skip_levels_two_levels() {
        let n = SKIP_INTERVAL * SKIP_INTERVAL;
        let doc_ids: Vec<u32> = (0..n as u32).collect();
        let levels = build_skip_levels(&doc_ids);
        assert_eq!(levels.len(), 2);
        assert_eq!(levels[0].len(), SKIP_INTERVAL);
        for (i, &v) in levels[0].iter().enumerate() {
            assert_eq!(v, ((i + 1) * SKIP_INTERVAL - 1) as u32);
        }
        assert_eq!(levels[1], vec![(n - 1) as u32]);
    }

    /// 5000 sequential doc ids exercises 4 levels (8, 64, 512, 4096
    /// strides). Verify each entry equals `doc_ids[(i + 1) * step - 1]`.
    #[test]
    fn test_build_skip_levels_5k_dense() {
        let n: usize = 5_000;
        let doc_ids: Vec<u32> = (0..n as u32).collect();
        let levels = build_skip_levels(&doc_ids);
        let mut step = SKIP_INTERVAL;
        for level in &levels {
            let expected_len = n / step;
            assert_eq!(level.len(), expected_len, "step={step}");
            for (i, &v) in level.iter().enumerate() {
                assert_eq!(v, ((i + 1) * step - 1) as u32, "step={step} i={i}");
            }
            step *= SKIP_INTERVAL;
        }
        assert!(
            levels.last().unwrap().len() <= SKIP_INTERVAL,
            "top level should fit in a single skip window: {levels:?}"
        );
    }

    /// Encoding via `encode_v2` and decoding via `decode_soa_v2` must
    /// round-trip every field including the multi-level skip table —
    /// the on-disk levels must equal the in-memory `build_skip_levels`
    /// result.
    #[test]
    fn test_round_trip_v2_preserves_skip_levels() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        // Pick a size that triggers ≥ 2 skip levels so the encoder's
        // count varint actually runs.
        let n: usize = 2_000;
        let mut original = PostingList::new("v2_round_trip".to_string());
        for i in 0..n {
            let did = (i as u64) * 3 + 1;
            let freq = ((i % 7) + 1) as u32;
            let weight = 0.25 + (i % 4) as f32 * 0.5;
            original.add_posting(Posting::with_frequency(did, freq).with_weight(weight));
        }

        let path = "v2_round_trip.bin";
        {
            let output = storage.create_output(path).unwrap();
            let mut writer = StructWriter::new(output);
            original.encode_v2(&mut writer).unwrap();
            writer.close().unwrap();
        }
        let input = storage.open_input(path).unwrap();
        let mut reader = StructReader::new(input).unwrap();
        let decoded = PostingList::decode_soa_v2(&mut reader).unwrap();

        // Postings preserved.
        assert_eq!(decoded.len(), n);
        for (i, posting) in original.postings.iter().enumerate() {
            assert_eq!(decoded.doc_ids[i], posting.doc_id as u32, "doc at {i}");
            assert_eq!(decoded.frequencies[i], posting.frequency, "freq at {i}");
        }

        // Skip levels match what `build_skip_levels` produces.
        let expected_levels = build_skip_levels(&decoded.doc_ids);
        assert_eq!(decoded.skip_levels.len(), expected_levels.len());
        for (i, (got, want)) in decoded
            .skip_levels
            .iter()
            .zip(expected_levels.iter())
            .enumerate()
        {
            assert_eq!(got, want, "level {i} mismatch");
        }
    }

    /// Decoding a v1-encoded posting list must still populate
    /// `skip_levels` (backward-compat fallback rebuilds it at load).
    #[test]
    fn test_v1_decode_populates_skip_levels_fallback() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let n: usize = 1_000;
        let mut original = PostingList::new("v1_compat".to_string());
        for i in 0..n {
            let did = (i as u64) * 2;
            original.add_posting(Posting::with_frequency(did, 1));
        }

        let path = "v1_compat.bin";
        {
            let output = storage.create_output(path).unwrap();
            let mut writer = StructWriter::new(output);
            original.encode(&mut writer).unwrap();
            writer.close().unwrap();
        }
        let input = storage.open_input(path).unwrap();
        let mut reader = StructReader::new(input).unwrap();
        let decoded = PostingList::decode_soa(&mut reader).unwrap();

        assert_eq!(decoded.len(), n);
        let expected_levels = build_skip_levels(&decoded.doc_ids);
        assert_eq!(decoded.skip_levels, expected_levels);
    }

    /// `TermPostingIndex` v1 segments (no skip levels on disk) must
    /// still load correctly through `read_from_storage`'s back-compat
    /// branch. We synthesize a v1 byte stream by writing the magic +
    /// version=1 header and then `encode` (v1) for each posting list.
    #[test]
    fn test_term_posting_index_v1_back_compat() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        let mut original = TermPostingIndex::new();
        original.add_document(
            1,
            vec![
                ("hello".to_string(), 2, Some(vec![0, 5])),
                ("world".to_string(), 1, Some(vec![1])),
            ],
        );
        original.add_document(
            2,
            vec![
                ("hello".to_string(), 1, Some(vec![2])),
                ("rust".to_string(), 3, Some(vec![0, 3, 6])),
            ],
        );

        // Synthesize a v1 file by hand (write_to_storage emits v2).
        let path = "tpi_v1.bin";
        {
            let output = storage.create_output(path).unwrap();
            let mut writer = StructWriter::new(output);
            writer.write_u32(0x494E5658).unwrap(); // "INVX"
            writer.write_u32(1).unwrap(); // version = 1 (legacy)
            writer.write_varint(original.doc_count()).unwrap();
            writer.write_varint(original.term_count()).unwrap();
            writer.write_varint(3).unwrap(); // 3 distinct terms
            let mut terms: Vec<_> = original.terms.iter().collect();
            terms.sort_by_key(|(t, _)| *t);
            for (_, posting_list) in terms {
                posting_list.encode(&mut writer).unwrap();
            }
            writer.close().unwrap();
        }

        let input = storage.open_input(path).unwrap();
        let mut reader = StructReader::new(input).unwrap();
        let loaded = TermPostingIndex::read_from_storage(&mut reader).unwrap();

        assert_eq!(loaded.doc_count(), original.doc_count());
        assert_eq!(loaded.term_count(), original.term_count());
        for term in ["hello", "world", "rust"] {
            let want = original.get_posting_list(term).expect("term exists");
            let got = loaded.get_posting_list(term).expect("term loaded");
            assert_eq!(got.postings.len(), want.postings.len(), "term={term}");
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
