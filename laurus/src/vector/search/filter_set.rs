//! Typed allow-set for filter-aware vector search (Issue
//! [#739](https://github.com/mosuka/laurus/issues/739)).
//!
//! Filter-aware traversal (#645) and the inline Flat / IVF filters (#740) need
//! `O(1)`/`O(log)` membership tests against the set of documents a filter
//! matches. [`FilterSet`] is that set, in one of two representations chosen by
//! shape:
//!
//! - [`FilterSet::Bitmap`] — a [`RoaringTreemap`] behind an [`Arc`]. Used for
//!   dense sets and, crucially, to **share** the bitmap the lexical filter
//!   cache already built (`InvertedIndexReader::matching_doc_ids`, #578/#764)
//!   so the engine's filtered hybrid search materialises the set once instead
//!   of `RoaringTreemap → Vec<u64> → AHashSet`.
//! - [`FilterSet::Hash`] — an [`AHashSet`], preserving the sparse-filter path
//!   #645 already handles (no regression).
//!
//! Doc ids are the global `u64` space shared by the lexical and vector sides,
//! so a lexical `RoaringTreemap` is usable here directly with no translation.

use std::sync::Arc;

use ahash::AHashSet;
use roaring::RoaringTreemap;

/// Entry count at or above which [`FilterSet::from_doc_ids`] stores a raw id
/// list as a [`RoaringTreemap`] rather than an [`AHashSet`].
///
/// Below this, small/sparse filters stay on `AHashSet` (the #645 path), so
/// there is no regression for typical selective filters. Dense filters cross
/// the threshold and gain the bitmap's compact, branch-light membership tests.
const DENSE_THRESHOLD: u64 = 4096;

/// The set of document ids a filter matches, used for membership tests during
/// vector search. See the module docs for the representation trade-offs.
#[derive(Debug, Clone)]
pub enum FilterSet {
    /// Dense / shared representation. The [`Arc`] lets the lexical filter
    /// cache's bitmap be reused without copying.
    Bitmap(Arc<RoaringTreemap>),
    /// Sparse representation (the #645 default).
    Hash(AHashSet<u64>),
}

impl FilterSet {
    /// Returns `true` if `id` is in the allow-set.
    pub fn contains(&self, id: u64) -> bool {
        match self {
            FilterSet::Bitmap(b) => b.contains(id),
            FilterSet::Hash(h) => h.contains(&id),
        }
    }

    /// Number of documents in the allow-set.
    pub fn len(&self) -> u64 {
        match self {
            FilterSet::Bitmap(b) => b.len(),
            FilterSet::Hash(h) => h.len() as u64,
        }
    }

    /// Returns `true` if the allow-set is empty.
    pub fn is_empty(&self) -> bool {
        match self {
            FilterSet::Bitmap(b) => b.is_empty(),
            FilterSet::Hash(h) => h.is_empty(),
        }
    }

    /// Iterate the allowed document ids.
    ///
    /// Used by the HNSW tiny-allow-set brute-force path (when the set is
    /// smaller than `ef_search`, scoring every allowed doc directly is cheaper
    /// and exact). Iteration order is unspecified.
    pub fn iter(&self) -> Box<dyn Iterator<Item = u64> + '_> {
        match self {
            FilterSet::Bitmap(b) => Box::new(b.iter()),
            FilterSet::Hash(h) => Box::new(h.iter().copied()),
        }
    }

    /// Wrap an already-built [`RoaringTreemap`] (e.g. from the lexical filter
    /// cache) as a [`FilterSet::Bitmap`] without copying the set.
    pub fn from_bitmap(bitmap: Arc<RoaringTreemap>) -> Self {
        FilterSet::Bitmap(bitmap)
    }

    /// Build a [`FilterSet`] from a raw id list, picking the representation by
    /// size: a [`RoaringTreemap`] when `ids.len() >= DENSE_THRESHOLD`, else an
    /// [`AHashSet`]. Used for the external `allowed_ids: Vec<u64>` path; the
    /// engine prefers [`from_bitmap`](Self::from_bitmap) to share the cached set.
    ///
    /// # Arguments
    ///
    /// * `ids` - The allowed document ids (order/duplicates are irrelevant).
    pub fn from_doc_ids(ids: &[u64]) -> Self {
        if ids.len() as u64 >= DENSE_THRESHOLD {
            FilterSet::Bitmap(Arc::new(ids.iter().copied().collect()))
        } else {
            FilterSet::Hash(ids.iter().copied().collect())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_variant_membership() {
        let fs = FilterSet::Hash([1u64, 5, 9].into_iter().collect());
        assert!(fs.contains(5));
        assert!(!fs.contains(6));
        assert_eq!(fs.len(), 3);
        assert!(!fs.is_empty());
    }

    #[test]
    fn bitmap_variant_membership() {
        let fs = FilterSet::from_bitmap(Arc::new([1u64, 5, 9].into_iter().collect()));
        assert!(fs.contains(5));
        assert!(!fs.contains(6));
        assert_eq!(fs.len(), 3);
        assert!(!fs.is_empty());
    }

    #[test]
    fn from_doc_ids_picks_hash_when_sparse() {
        let fs = FilterSet::from_doc_ids(&[3, 7, 15]);
        assert!(matches!(fs, FilterSet::Hash(_)));
        assert!(fs.contains(7));
        assert!(!fs.contains(8));
    }

    #[test]
    fn from_doc_ids_picks_bitmap_when_dense() {
        let ids: Vec<u64> = (0..DENSE_THRESHOLD).collect();
        let fs = FilterSet::from_doc_ids(&ids);
        assert!(matches!(fs, FilterSet::Bitmap(_)));
        assert_eq!(fs.len(), DENSE_THRESHOLD);
        assert!(fs.contains(0));
        assert!(fs.contains(DENSE_THRESHOLD - 1));
        assert!(!fs.contains(DENSE_THRESHOLD));
    }

    #[test]
    fn both_variants_agree_on_membership() {
        let ids: Vec<u64> = vec![2, 4, 8, 16, 32, 64];
        let hash = FilterSet::Hash(ids.iter().copied().collect());
        let bitmap = FilterSet::from_bitmap(Arc::new(ids.iter().copied().collect()));
        for id in 0..70u64 {
            assert_eq!(hash.contains(id), bitmap.contains(id), "disagree on {id}");
        }
    }
}
