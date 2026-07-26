//! Shared multi-segment fan-out search layer for segment-per-commit vector
//! indexes (Issue #889; originated in HNSW's segment-per-commit design,
//! Issues #634/#880/#883).
//!
//! [`SegmentedReaderFacade`] and [`SegmentFanoutSearcher`] operate purely
//! through the [`VectorIndexReader`]/[`VectorIndexSearcher`] trait objects —
//! every per-reader call they make (`contains_vector`, `get_vector`,
//! `vector_ids`, ...) is already a trait method, so the fan-out logic itself
//! needs no index-type-specific code. Only the per-segment concrete searcher
//! construction differs between index types, which is why
//! [`SegmentFanoutSearcher`] takes it as a closure rather than hard-coding a
//! concrete searcher type.

use std::sync::Arc;

use crate::error::Result;
use crate::maintenance::deletion::DeletionBitmap;
use crate::vector::core::distance::DistanceMetric;
use crate::vector::core::vector::Vector;
use crate::vector::reader::{
    ValidationReport, VectorIndexMetadata, VectorIndexReader, VectorIterator, VectorStats,
};
use crate::vector::search::searcher::{
    VectorIndexQuery, VectorIndexQueryResults, VectorIndexSearcher,
};

/// Read facade over a segment-per-commit index's sealed segments
/// (newest-wins, deletion-filtered).
///
/// Materializes the distinct live `(doc_id, field) -> segment` mapping once
/// at construction; per-vector data is fetched from the owning segment
/// reader on demand.
#[derive(Debug)]
pub struct SegmentedReaderFacade {
    /// Sealed readers, newest generation first.
    readers: Vec<Arc<dyn VectorIndexReader>>,
    /// Distinct live keys with the owning (newest) reader's index, in
    /// first-seen (newest-segment) order.
    entries: Vec<(u64, String, usize)>,
    dimension: usize,
    metric: DistanceMetric,
}

impl SegmentedReaderFacade {
    /// Build the facade over `readers` (already ordered newest-generation
    /// first by the caller), filtering out anything `bitmap` marks deleted.
    pub fn new(
        readers: Vec<Arc<dyn VectorIndexReader>>,
        bitmap: Option<Arc<DeletionBitmap>>,
        dimension: usize,
        metric: DistanceMetric,
    ) -> Self {
        let mut seen: std::collections::HashSet<(u64, String)> = std::collections::HashSet::new();
        let mut entries = Vec::new();
        for (idx, reader) in readers.iter().enumerate() {
            if let Ok(ids) = reader.vector_ids() {
                for (doc_id, field) in ids {
                    if let Some(b) = &bitmap
                        && b.is_deleted(doc_id)
                    {
                        continue;
                    }
                    if seen.insert((doc_id, field.clone())) {
                        entries.push((doc_id, field, idx));
                    }
                }
            }
        }
        Self {
            readers,
            entries,
            dimension,
            metric,
        }
    }

    fn owner_of(&self, doc_id: u64, field_name: &str) -> Option<&Arc<dyn VectorIndexReader>> {
        self.entries
            .iter()
            .find(|(d, f, _)| *d == doc_id && f == field_name)
            .map(|(_, _, idx)| &self.readers[*idx])
    }
}

impl VectorIndexReader for SegmentedReaderFacade {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn get_vector(&self, doc_id: u64, field_name: &str) -> Result<Option<Vector>> {
        match self.owner_of(doc_id, field_name) {
            Some(reader) => reader.get_vector(doc_id, field_name),
            None => Ok(None),
        }
    }

    fn get_vectors_for_doc(&self, doc_id: u64) -> Result<Vec<(String, Vector)>> {
        let mut out = Vec::new();
        for (d, field, idx) in &self.entries {
            if *d == doc_id
                && let Some(v) = self.readers[*idx].get_vector(doc_id, field)?
            {
                out.push((field.clone(), v));
            }
        }
        Ok(out)
    }

    fn get_vectors(&self, doc_ids: &[(u64, String)]) -> Result<Vec<Option<Vector>>> {
        doc_ids
            .iter()
            .map(|(d, f)| self.get_vector(*d, f))
            .collect()
    }

    fn vector_ids(&self) -> Result<Vec<(u64, String)>> {
        Ok(self
            .entries
            .iter()
            .map(|(d, f, _)| (*d, f.clone()))
            .collect())
    }

    fn vector_count(&self) -> usize {
        self.entries.len()
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn distance_metric(&self) -> DistanceMetric {
        self.metric
    }

    fn stats(&self) -> VectorStats {
        VectorStats {
            vector_count: self.entries.len(),
            dimension: self.dimension,
            memory_usage: 0,
            build_time_ms: 0,
        }
    }

    fn contains_vector(&self, doc_id: u64, field_name: &str) -> bool {
        self.entries
            .iter()
            .any(|(d, f, _)| *d == doc_id && f == field_name)
    }

    fn get_vector_range(
        &self,
        start_doc_id: u64,
        end_doc_id: u64,
    ) -> Result<Vec<(u64, String, Vector)>> {
        let mut out = Vec::new();
        for (doc_id, field, idx) in &self.entries {
            if *doc_id >= start_doc_id
                && *doc_id < end_doc_id
                && let Some(v) = self.readers[*idx].get_vector(*doc_id, field)?
            {
                out.push((*doc_id, field.clone(), v));
            }
        }
        Ok(out)
    }

    fn get_vectors_by_field(&self, field_name: &str) -> Result<Vec<(u64, Vector)>> {
        let mut out = Vec::new();
        for (doc_id, field, idx) in &self.entries {
            if field == field_name
                && let Some(v) = self.readers[*idx].get_vector(*doc_id, field)?
            {
                out.push((*doc_id, v));
            }
        }
        Ok(out)
    }

    fn field_names(&self) -> Result<Vec<String>> {
        let mut names: Vec<String> = Vec::new();
        for (_, field, _) in &self.entries {
            if !names.iter().any(|n| n == field) {
                names.push(field.clone());
            }
        }
        Ok(names)
    }

    fn vector_iterator(&self) -> Result<Box<dyn VectorIterator>> {
        // Materialize through the newest-wins entries; segmented iteration is
        // facade-level only (merge uses per-segment readers directly).
        let mut items = Vec::with_capacity(self.entries.len());
        for (doc_id, field, idx) in &self.entries {
            if let Some(v) = self.readers[*idx].get_vector(*doc_id, field)? {
                items.push((*doc_id, field.clone(), v));
            }
        }
        Ok(Box::new(FacadeIterator { items, pos: 0 }))
    }

    fn metadata(&self) -> Result<VectorIndexMetadata> {
        Ok(VectorIndexMetadata {
            index_type: "segmented".to_string(),
            created_at: chrono::Utc::now(),
            modified_at: chrono::Utc::now(),
            version: "1".to_string(),
            build_config: serde_json::Value::Null,
            custom_metadata: std::collections::HashMap::new(),
        })
    }

    fn validate(&self) -> Result<ValidationReport> {
        let mut errors = Vec::new();
        for reader in &self.readers {
            let report = reader.validate()?;
            errors.extend(report.errors);
        }
        Ok(ValidationReport {
            is_valid: errors.is_empty(),
            errors,
            warnings: Vec::new(),
            repair_suggestions: Vec::new(),
        })
    }
}

/// Iterator over the facade's materialized newest-wins entries.
#[derive(Debug)]
struct FacadeIterator {
    items: Vec<(u64, String, Vector)>,
    pos: usize,
}

impl VectorIterator for FacadeIterator {
    fn next(&mut self) -> Result<Option<(u64, String, Vector)>> {
        if self.pos >= self.items.len() {
            return Ok(None);
        }
        let item = self.items[self.pos].clone();
        self.pos += 1;
        Ok(Some(item))
    }

    fn skip_to(&mut self, doc_id: u64, field_name: &str) -> Result<bool> {
        while self.pos < self.items.len() {
            let (d, f, _) = &self.items[self.pos];
            if *d == doc_id && f == field_name {
                return Ok(true);
            }
            self.pos += 1;
        }
        Ok(false)
    }

    fn reset(&mut self) -> Result<()> {
        self.pos = 0;
        Ok(())
    }

    fn position(&self) -> (u64, String) {
        if self.pos < self.items.len() {
            let (d, f, _) = &self.items[self.pos];
            (*d, f.clone())
        } else {
            (u64::MAX, String::new())
        }
    }
}

/// Multi-segment fan-out searcher (#880 newest-wins semantics, #883
/// expanding-refill), shared across index types.
///
/// Per-segment concrete searcher construction is supplied by the caller as
/// `make_searcher`, since that's the only index-type-specific piece (e.g.
/// HNSW's `ef_search`-carrying constructor vs. IVF's `n_probe`-carrying
/// one).
/// Builds the concrete per-segment searcher for one reader.
type MakeSearcher =
    Box<dyn Fn(Arc<dyn VectorIndexReader>) -> Result<Box<dyn VectorIndexSearcher>> + Send + Sync>;

pub struct SegmentFanoutSearcher {
    /// Sealed readers, newest generation first, deletion bitmap attached.
    readers: Vec<Arc<dyn VectorIndexReader>>,
    /// The shared deletion bitmap (also attached to the readers), used by
    /// [`Self::count`] to exclude soft-deleted docs.
    bitmap: Option<Arc<DeletionBitmap>>,
    /// Builds the concrete per-segment searcher for one reader.
    make_searcher: MakeSearcher,
}

impl std::fmt::Debug for SegmentFanoutSearcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SegmentFanoutSearcher")
            .field("segments", &self.readers.len())
            .field("has_bitmap", &self.bitmap.is_some())
            .finish_non_exhaustive()
    }
}

impl SegmentFanoutSearcher {
    /// Build a fan-out searcher over `readers` (newest generation first).
    ///
    /// `make_searcher` is invoked once per segment per `search`/`count` call
    /// to build that segment's concrete searcher.
    pub fn new(
        readers: Vec<Arc<dyn VectorIndexReader>>,
        bitmap: Option<Arc<DeletionBitmap>>,
        make_searcher: impl Fn(Arc<dyn VectorIndexReader>) -> Result<Box<dyn VectorIndexSearcher>>
        + Send
        + Sync
        + 'static,
    ) -> Self {
        Self {
            readers,
            bitmap,
            make_searcher: Box::new(make_searcher),
        }
    }

    /// Whether any reader NEWER than `idx` contains `(doc_id, field)` — the
    /// containment mask that makes the newest copy win (#880).
    fn shadowed(&self, idx: usize, doc_id: u64, field: &str) -> bool {
        self.readers[..idx]
            .iter()
            .any(|r| r.contains_vector(doc_id, field))
    }
}

impl VectorIndexSearcher for SegmentFanoutSearcher {
    fn search(&self, request: &VectorIndexQuery) -> Result<VectorIndexQueryResults> {
        let started = crate::util::time::Timer::now();
        let limit = request.params.top_k;
        let mut merged = VectorIndexQueryResults::new();
        if limit == 0 || self.readers.is_empty() {
            return Ok(merged);
        }

        // Over-fetch per segment: containment masking drops shadowed hits
        // AFTER the per-segment top-k, so stale copies would otherwise
        // consume result slots (#880). Start at 2x for the common case; if
        // masking dropped this segment below the requested limit AND the pass
        // was truncated at the budget (i.e. live hits may sit below the cut
        // behind a band of masked stale copies), EXPAND the budget
        // geometrically and re-query until enough live hits survive or the
        // segment is exhausted (#883). A fixed multiplier cannot bound recall
        // against an arbitrarily deep stale band — the true nearest neighbour
        // could sit below any constant cut — so the budget doubles each round.
        // Bounded: top_k reaches the segment size in O(log n) queries, and
        // the common (no-pollution) case stops after one pass.
        for (idx, reader) in self.readers.iter().enumerate() {
            let searcher = (self.make_searcher)(reader.clone())?;

            let mut probe = request.clone();
            probe.params.top_k = limit.saturating_mul(2);
            let mut kept: Vec<crate::vector::search::searcher::VectorIndexQueryResult>;
            loop {
                let results = searcher.search(&probe)?;
                merged.candidates_examined += results.candidates_examined;
                let returned = results.results.len();
                kept = results
                    .results
                    .into_iter()
                    .filter(|hit| !self.shadowed(idx, hit.doc_id, &hit.field_name))
                    .collect();

                // Enough live hits survived masking, or the segment returned
                // fewer than requested (nothing deeper to fetch) — done.
                if kept.len() >= limit || returned < probe.params.top_k {
                    break;
                }
                let next = probe.params.top_k.saturating_mul(2);
                if next == probe.params.top_k {
                    break; // budget saturated (overflow guard)
                }
                probe.params.top_k = next;
            }

            merged.results.append(&mut kept);
        }

        merged
            .results
            .sort_unstable_by(|a, b| b.similarity.total_cmp(&a.similarity));
        merged.results.truncate(limit);
        merged.search_time_ms = started.elapsed_ms() as f64;
        Ok(merged)
    }

    fn count(&self, request: VectorIndexQuery) -> Result<u64> {
        // Count distinct live `(doc_id, field)` keys across segments with
        // the same newest-wins masking as `search`, excluding soft-deleted
        // docs.
        let mut count = 0u64;
        for (idx, reader) in self.readers.iter().enumerate() {
            for (doc_id, field) in reader.vector_ids()? {
                if let Some(ref field_name) = request.field_name
                    && &field != field_name
                {
                    continue;
                }
                if let Some(bitmap) = &self.bitmap
                    && bitmap.is_deleted(doc_id)
                {
                    continue;
                }
                if !self.shadowed(idx, doc_id, &field) {
                    count += 1;
                }
            }
        }
        Ok(count)
    }
}
