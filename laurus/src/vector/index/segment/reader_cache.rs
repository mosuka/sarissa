//! Per-segment reader cache shared across segment-per-commit vector index
//! types (Issue #889; originated in HNSW's segment-per-commit design,
//! Issue #634).
//!
//! Before this cache existed, every search against a segmented index
//! reloaded every managed segment from disk on each call, re-parsing the
//! segment header and rebuilding its in-memory search state. For a
//! multi-segment index the per-query I/O often dominated the actual search
//! cost.
//!
//! This cache holds an `Arc<R>` per `segment_id` (`R` is the index type's
//! concrete reader — `HnswIndexReader`, `FlatVectorIndexReader`,
//! `IvfIndexReader`) so that the second and subsequent searches against the
//! same segment hit memory only. Entries are invalidated by the owning
//! index's merge path immediately after the segment manager removes a
//! source segment as part of a merge.
//!
//! Issue [#660](https://github.com/mosuka/laurus/issues/660). Reuses the
//! storage-level per-segment input cache landed in
//! [#522](https://github.com/mosuka/laurus/issues/522); the two layers are
//! complementary — #522 caches the open file handle, #660 caches the fully
//! parsed reader state on top.

use std::sync::Arc;

use ahash::AHashMap;
use parking_lot::RwLock;

use crate::error::Result;

/// Caches `Arc<R>` per `segment_id`.
///
/// The cache is intentionally minimal: it stores every requested reader for
/// the lifetime of the owning index (i.e. no LRU eviction or memory
/// budget). A future change can swap the inner map for an LRU-backed
/// implementation without touching call sites.
#[derive(Debug)]
pub struct SegmentedReaderCache<R> {
    inner: RwLock<AHashMap<String, Arc<R>>>,
}

impl<R> Default for SegmentedReaderCache<R> {
    // Not derived: `#[derive(Default)]` on a generic struct adds an `R:
    // Default` bound, but `R` here is only ever stored behind an `Arc` and
    // never itself defaulted.
    fn default() -> Self {
        Self {
            inner: RwLock::new(AHashMap::default()),
        }
    }
}

impl<R> SegmentedReaderCache<R> {
    /// Create an empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the cached reader for `segment_id`, invoking `loader` on miss.
    ///
    /// Concurrent callers requesting the same `segment_id` may race; the
    /// double-checked lock pattern below ensures only one of them runs
    /// `loader`, the rest pick up the already-inserted entry.
    ///
    /// # Arguments
    ///
    /// * `segment_id` - The segment identifier to look up.
    /// * `loader` - Invoked only on a cache miss; returns the freshly-built
    ///   reader.
    ///
    /// # Returns
    ///
    /// `Ok(Arc::clone(&reader))` if the reader is found or successfully
    /// loaded. Propagates `loader`'s error otherwise.
    pub fn get_or_load<F>(&self, segment_id: &str, loader: F) -> Result<Arc<R>>
    where
        F: FnOnce() -> Result<R>,
    {
        if let Some(reader) = self.inner.read().get(segment_id) {
            return Ok(Arc::clone(reader));
        }

        // Slow path: take the write lock and re-check before loading so that
        // concurrent misses on the same `segment_id` don't load twice.
        let mut guard = self.inner.write();
        if let Some(reader) = guard.get(segment_id) {
            return Ok(Arc::clone(reader));
        }

        let reader = Arc::new(loader()?);
        guard.insert(segment_id.to_string(), Arc::clone(&reader));
        Ok(reader)
    }

    /// Remove the cached reader for `segment_id`.
    ///
    /// Safe to call for unknown segment ids — a missing entry is a no-op.
    pub fn invalidate(&self, segment_id: &str) {
        self.inner.write().remove(segment_id);
    }

    /// Remove all cached readers.
    ///
    /// Intended for tests and for future LRU integration; not invoked from
    /// the live search path.
    pub fn clear(&self) {
        self.inner.write().clear();
    }

    /// Return the number of cached entries.
    ///
    /// Primarily exposed for tests. The value may be stale by the time the
    /// caller reads it because other threads may concurrently mutate the
    /// cache, so do not rely on it for control flow.
    pub fn len(&self) -> usize {
        self.inner.read().len()
    }

    /// Return `true` if the cache currently holds no entries.
    pub fn is_empty(&self) -> bool {
        self.inner.read().is_empty()
    }

    /// Return `true` if the cache currently holds an entry for `segment_id`.
    ///
    /// Primarily exposed for tests.
    pub fn contains(&self, segment_id: &str) -> bool {
        self.inner.read().contains_key(segment_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::LaurusError;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// The cache only stores `Arc<R>` and these tests don't need the reader
    /// to be functional — they only need to assert that the loader is
    /// invoked the right number of times for a given access pattern.
    ///
    /// Instead, return an `Err` from the loader and assert the call count.
    /// `get_or_load` propagates the error without inserting anything, so we
    /// can use call counters to distinguish hits from misses. `()` stands in
    /// for a concrete reader type here.

    #[test]
    fn get_or_load_invokes_loader_on_miss_only() {
        let cache: SegmentedReaderCache<()> = SegmentedReaderCache::new();
        let calls = AtomicUsize::new(0);

        // First call: loader is invoked, returns Err so nothing is cached.
        let r1 = cache.get_or_load("seg-a", || {
            calls.fetch_add(1, Ordering::SeqCst);
            Err(LaurusError::other("intentional miss"))
        });
        assert!(r1.is_err());
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(cache.len(), 0, "no entry inserted on loader error");

        // Second call: same segment, loader still invoked (no entry cached).
        let r2 = cache.get_or_load("seg-a", || {
            calls.fetch_add(1, Ordering::SeqCst);
            Err(LaurusError::other("still missing"))
        });
        assert!(r2.is_err());
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn invalidate_is_noop_for_unknown_segment() {
        let cache: SegmentedReaderCache<()> = SegmentedReaderCache::new();
        cache.invalidate("never-cached");
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn clear_empties_the_cache() {
        let cache: SegmentedReaderCache<()> = SegmentedReaderCache::new();
        assert!(cache.is_empty());
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn contains_reports_membership() {
        let cache: SegmentedReaderCache<()> = SegmentedReaderCache::new();
        assert!(!cache.contains("seg-a"));
        // No way to insert via the public API without a real reader; that
        // case is covered by the integration test in `segmented_field`.
    }

    #[test]
    fn get_or_load_caches_successful_loads() {
        let cache: SegmentedReaderCache<u32> = SegmentedReaderCache::new();
        let calls = AtomicUsize::new(0);
        let load = || {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(42)
        };

        let r1 = cache.get_or_load("seg-a", load).unwrap();
        assert_eq!(*r1, 42);
        assert_eq!(cache.len(), 1);

        let r2 = cache.get_or_load("seg-a", load).unwrap();
        assert_eq!(*r2, 42);
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "second call must hit the cache, not reinvoke the loader"
        );
    }
}
