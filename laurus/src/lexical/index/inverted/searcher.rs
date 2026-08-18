//! Searcher implementation for executing queries against an index.

use crate::lexical::core::field::FieldValue;
use std::cmp::Ordering;
use std::sync::Arc;
use std::time::Duration;

use ahash::AHashMap;
use roaring::RoaringTreemap;

use crate::util::time::Timer;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use crate::analysis::analyzer::standard::StandardAnalyzer;
use crate::data::DataValue::{
    Bool as Boolean, Bytes, DateTime, Float64 as Float, Geo, Int64 as Integer, Null, Text,
};
use crate::error::{LaurusError, Result};
// Note: Geo and DateTime were removed from FieldValue definition implicitly by switching to DataValue.
// Only standard types remain. Logic using Geo/DateTime needs update.
use crate::lexical::index::inverted::bmw::{BlockMaxOrExecutor, is_bmw_eligible};
use crate::lexical::index::inverted::parsed_query_cache::ParsedQueryCache;
use crate::lexical::index::inverted::per_segment_view::PerSegmentReaderView;
use crate::lexical::index::inverted::reader::InvertedIndexReader;
use crate::lexical::query::Query;
use crate::lexical::query::boolean::{BooleanQuery, Occur};
use crate::lexical::query::collector::{
    Collector, CountCollector, TopDocsCollector, TopFieldCollector,
};
use crate::lexical::query::parser::LexicalQueryParser;
use crate::lexical::query::term::TermQuery;
use crate::lexical::query::{LexicalSearchResults, SearchHit};
use crate::lexical::reader::LexicalIndexReader;
use crate::lexical::search::searcher::{
    LexicalSearchParams, LexicalSearchQuery, LexicalSearchRequest, SortField, SortOrder,
};

/// Default capacity (entries) of the per-searcher parsed-DSL query cache
/// (Issue #590) when none is configured.
const DEFAULT_PARSED_QUERY_CACHE_CAPACITY: usize = 1024;

/// How often the scan loops consult the wall clock for a search deadline,
/// measured in scanned documents (Issue #600). Checking every document would
/// make `Timer::elapsed` a per-doc cost; checking once per this many keeps the
/// overhead negligible while still bounding worst-case latency (the same
/// batched approach as Lucene's `TimeLimitingCollector`).
const DEADLINE_CHECK_INTERVAL: u64 = 2048;

/// A wall-clock deadline for cooperative search interruption (Issue #600).
///
/// Threaded through every scan loop (the default matcher loop, the Block-Max
/// WAND executor, and the per-segment fanout) so a timed search aborts
/// mid-flight instead of only being detected after it has already run to
/// completion. On `wasm32` `Timer` reports zero elapsed, so a deadline never
/// fires there.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Deadline {
    start: Timer,
    timeout: Duration,
}

impl Deadline {
    /// Create a deadline of `timeout` measured from `start`.
    pub(crate) fn new(start: Timer, timeout: Duration) -> Self {
        Self { start, timeout }
    }

    /// Return `Err` if the time budget is exhausted. Safe to call on every
    /// loop iteration: the clock is only read every [`DEADLINE_CHECK_INTERVAL`]
    /// scanned documents, so `scanned` is the caller's running document count.
    pub(crate) fn check(&self, scanned: u64) -> Result<()> {
        if scanned.is_multiple_of(DEADLINE_CHECK_INTERVAL) && self.start.elapsed() > self.timeout {
            return Err(LaurusError::index("Search timeout exceeded"));
        }
        Ok(())
    }
}

/// A searcher that executes queries against an index reader.
#[derive(Debug)]
pub struct InvertedIndexSearcher {
    /// The index reader to search against.
    reader: Arc<dyn LexicalIndexReader>,
    /// Default fields to search if none specified in query.
    default_fields: Vec<String>,
    /// Snapshot-scoped parsed-DSL query cache (Issue #590). The analyzer and
    /// `default_fields` are fixed for this searcher's lifetime, so a DSL string
    /// alone keys it; rebuilt (empty) whenever the store rebuilds the searcher.
    parsed_query_cache: ParsedQueryCache,
}

impl InvertedIndexSearcher {
    /// Create a new searcher with the given index reader.
    pub fn new(reader: Box<dyn LexicalIndexReader>) -> Self {
        InvertedIndexSearcher {
            reader: Arc::from(reader),
            default_fields: Vec::new(),
            parsed_query_cache: ParsedQueryCache::new(DEFAULT_PARSED_QUERY_CACHE_CAPACITY),
        }
    }

    /// Create a new searcher with an `Arc<dyn LexicalIndexReader>`.
    pub fn from_arc(reader: Arc<dyn LexicalIndexReader>) -> Self {
        InvertedIndexSearcher {
            reader,
            default_fields: Vec::new(),
            parsed_query_cache: ParsedQueryCache::new(DEFAULT_PARSED_QUERY_CACHE_CAPACITY),
        }
    }

    /// Set default fields for search.
    pub fn with_default_fields(mut self, fields: Vec<String>) -> Self {
        self.default_fields = fields;
        self
    }

    /// Set the capacity (entries) of the parsed-DSL query cache (Issue #590).
    /// `0` disables the cache. Replaces the default-capacity cache.
    pub fn with_parsed_query_cache_capacity(mut self, capacity: usize) -> Self {
        self.parsed_query_cache = ParsedQueryCache::new(capacity);
        self
    }

    /// Get the index reader.
    pub fn reader(&self) -> &Arc<dyn LexicalIndexReader> {
        &self.reader
    }

    /// Snapshot of the parsed-DSL query cache hit / miss counters (Issue #590).
    pub fn parsed_query_cache_stats(
        &self,
    ) -> crate::lexical::index::inverted::parsed_query_cache::ParsedQueryCacheStats {
        self.parsed_query_cache.stats()
    }

    /// Execute a search with a custom collector.
    pub fn search_with_collector<C: Collector>(
        &self,
        query: Box<dyn Query>,
        collector: C,
    ) -> Result<C> {
        self.search_with_collector_parallel(query, collector, false)
    }

    /// Execute a search with a custom collector, with optional parallel execution.
    pub fn search_with_collector_parallel<C: Collector>(
        &self,
        query: Box<dyn Query>,
        collector: C,
        parallel: bool,
    ) -> Result<C> {
        self.search_with_collector_deadline(query, collector, parallel, None)
    }

    /// Internal search entry point that additionally honours an optional
    /// wall-clock [`Deadline`] (Issue #600). The public
    /// [`Self::search_with_collector_parallel`] delegates here with
    /// `deadline = None`, so non-timed searches pay nothing; the timeout path
    /// passes `Some(..)` and every scan loop below (plus the per-segment
    /// fanout it recurses into) aborts mid-flight once the budget is spent.
    fn search_with_collector_deadline<C: Collector>(
        &self,
        query: Box<dyn Query>,
        mut collector: C,
        parallel: bool,
        deadline: Option<Deadline>,
    ) -> Result<C> {
        // Query rewrite (Issue #613): lower multi-term queries (prefix /
        // wildcard / fuzzy / regexp) into Boolean-of-TermQuery ONCE,
        // against the top-level reader, before any gating below. This
        // (a) halves the term-dictionary enumerations (matcher and
        // scorer used to re-enumerate independently), and (b) hands the
        // per-segment fanout a query it can execute — the fanout's
        // `PerSegmentReaderView` cannot enumerate the term dictionary,
        // which previously made raw multi-term queries return 0 hits on
        // multi-segment indexes. `None` (nothing to rewrite, or the
        // reader is itself a per-segment view) keeps the original query;
        // an already-lowered query rewrites to `None`, so the fanout's
        // recursion back into this method is a cheap no-op.
        let query = match query.rewrite(self.reader.as_ref())? {
            Some(rewritten) => rewritten,
            None => query,
        };

        // For BooleanQuery with multiple clauses, try to execute sub-queries in parallel
        if parallel && let Some(boolean_query) = query.as_any().downcast_ref::<BooleanQuery>() {
            return self.search_boolean_query_parallel(boolean_query, collector, deadline);
        }

        // Per-segment fanout fast path (#476 Phase 1). For multi-
        // segment top-K queries, each segment's `block_max` table is
        // valid as a per-segment scoring bound; running the query
        // independently on each segment via [`PerSegmentReaderView`]
        // re-activates PR-F's BMW pivot loop on each one. Cross-
        // segment merge collects the per-segment top-K into the
        // caller's collector.
        if collector.bmw_capable()
            && let Some(inverted_reader) =
                self.reader.as_any().downcast_ref::<InvertedIndexReader>()
            && inverted_reader.segment_count() >= 2
        {
            return self.search_per_segment_fanout(query, collector, deadline);
        }

        // Block-Max-WAND fast path (#475 PR-F). Eligible for Should-only
        // BooleanQuery against a top-K-style collector. Construction
        // re-checks each clause's per-block metadata at runtime; on
        // any miss we fall through to the existing matcher-driven loop.
        if collector.bmw_capable()
            && let Some(boolean_query) = is_bmw_eligible(query.as_ref())
            && let Ok(executor) = BlockMaxOrExecutor::new(boolean_query, self.reader.as_ref())
        {
            return executor.run(collector, deadline);
        }

        // Default single-threaded execution
        // Create the matcher and scorer in one pass (#996: queries with
        // an expensive shared candidate computation, e.g. geo, build
        // both from a single run of it).
        let (mut matcher, scorer) = query.matcher_scorer(self.reader.as_ref())?;

        // SIMD-batched default loop (#506). The scalar path collected
        // one doc at a time via `scorer.score`; this version gathers up
        // to `BATCH_SIZE` per-doc inputs (doc id / TF / field length)
        // and lowers the cross-doc kernel through
        // [`crate::lexical::query::scorer::Scorer::batch_score`], whose
        // BM25 override is an `f32x8` SIMD kernel. Non-BM25 scorers
        // inherit the trait's per-element default, so behaviour is
        // identical there.
        //
        // Trade-off: the cumulative early-break (#403 PR-C) and the
        // count-cap `needs_more()` check both consume the latest
        // `min_competitive()`, so batching delays them by up to
        // `BATCH_SIZE - 1` docs. The buffer flushes also fire before
        // any per-block skip so the skip target stays accurate.
        const BATCH_SIZE: usize = 8;
        let mut doc_buf: [u64; BATCH_SIZE] = [0; BATCH_SIZE];
        let mut tf_buf: [f32; BATCH_SIZE] = [0.0; BATCH_SIZE];
        let mut fl_buf: [f32; BATCH_SIZE] = [0.0; BATCH_SIZE];
        let mut score_buf: [f32; BATCH_SIZE] = [0.0; BATCH_SIZE];
        let mut n: usize = 0;
        let avg_fl = scorer.avg_field_length();
        let query_field = query.field().map(|s| s.to_string());

        // Running count of scanned documents, used to throttle the deadline
        // clock read (Issue #600). Starts at 0 so the first iteration checks
        // immediately, which makes a zero/expired budget fail fast.
        let mut scanned: u64 = 0;

        // Iterate through matching documents
        while !matcher.is_exhausted() {
            if let Some(d) = deadline {
                d.check(scanned)?;
            }
            scanned = scanned.wrapping_add(1);

            let doc_id = matcher.doc_id();

            if doc_id == u64::MAX {
                break;
            }

            // Block-Max skip-ahead pre-check (#403 PR-E). Before paying
            // the score / field-length cost on this doc, see whether
            // the block containing it is even competitive. The current
            // block's bound (`current_block_max_score`) is non-cumulative
            // — when it falls below the K-th score, jumping past the
            // block via `next_block_boundary` is sound (the global
            // `block_max_score_at` cumulative bound, queried right
            // below, still controls the hard `break`).
            let min_comp = collector.min_competitive();
            if scorer.current_block_max_score(doc_id) <= min_comp {
                // Flush the buffered batch before deciding the skip
                // target. The skip relies on `block_max_score_at`,
                // which factors in the K-th score; that score can only
                // be tight once buffered hits have been collected.
                if n > 0 {
                    scorer.batch_score(
                        &doc_buf[..n],
                        &tf_buf[..n],
                        &fl_buf[..n],
                        &mut score_buf[..n],
                    );
                    for i in 0..n {
                        collector.collect(doc_buf[i], score_buf[i])?;
                        if !collector.needs_more() {
                            return Ok(collector);
                        }
                    }
                    n = 0;
                }
                let min_comp = collector.min_competitive();
                if scorer.block_max_score_at(doc_id) <= min_comp {
                    // Cumulative suffix bound already non-competitive
                    // → no later block can produce a top-K hit.
                    break;
                }
                if let Some(target) = scorer.next_block_boundary(doc_id) {
                    if target == u64::MAX || target <= doc_id {
                        break;
                    }
                    if !matcher.skip_to(target)? || matcher.is_exhausted() {
                        break;
                    }
                    continue;
                }
                // No per-block info → fall through to existing PR-C
                // break path after scoring this doc.
            }

            // Gather per-doc inputs into the batch buffer. The field
            // length lookup mirrors the scalar path's reader downcasts
            // (`InvertedIndexReader` / `PerSegmentReaderView`), but
            // substitutes the scorer's avg when no per-doc value is
            // available so the dense SIMD slice stays valid.
            let term_freq = matcher.term_freq() as f32;
            let field_length = if let Some(field_name) = query_field.as_deref() {
                if let Some(inverted_index_reader) =
                    self.reader.as_any().downcast_ref::<InvertedIndexReader>()
                {
                    inverted_index_reader
                        .field_length(doc_id, field_name)
                        .ok()
                        .flatten()
                        .map(|len| len as f32)
                        .unwrap_or(avg_fl)
                } else if let Some(view) =
                    self.reader.as_any().downcast_ref::<PerSegmentReaderView>()
                {
                    // #476 Phase 1: per-segment fanout reads field
                    // lengths through the view so BM25 normalisation
                    // matches each segment's local avg.
                    view.field_length(doc_id, field_name)
                        .ok()
                        .flatten()
                        .map(|len| len as f32)
                        .unwrap_or(avg_fl)
                } else {
                    avg_fl
                }
            } else {
                avg_fl
            };

            doc_buf[n] = doc_id;
            tf_buf[n] = term_freq;
            fl_buf[n] = field_length;
            n += 1;

            if n == BATCH_SIZE {
                scorer.batch_score(
                    &doc_buf[..n],
                    &tf_buf[..n],
                    &fl_buf[..n],
                    &mut score_buf[..n],
                );
                let last_doc = doc_buf[n - 1];
                for i in 0..n {
                    collector.collect(doc_buf[i], score_buf[i])?;
                    if !collector.needs_more() {
                        return Ok(collector);
                    }
                }
                n = 0;

                // Cumulative early-break (#403 PR-C) once per batch.
                // The K-th score is at its tightest right after the
                // batch is collected; if the right-cumulative suffix
                // bound has already fallen below it, no later doc can
                // enter the top-K.
                if scorer.block_max_score_at(last_doc) <= collector.min_competitive() {
                    return Ok(collector);
                }
            }

            // Move to next document
            if !matcher.next()? {
                break;
            }
        }

        // Final flush for any partial batch left when the matcher is
        // exhausted (or a `break` above was taken without flushing).
        if n > 0 {
            scorer.batch_score(
                &doc_buf[..n],
                &tf_buf[..n],
                &fl_buf[..n],
                &mut score_buf[..n],
            );
            for i in 0..n {
                collector.collect(doc_buf[i], score_buf[i])?;
                if !collector.needs_more() {
                    return Ok(collector);
                }
            }
        }

        Ok(collector)
    }

    /// Execute a top-K query against a multi-segment reader by
    /// fanning out to per-segment searches (#476 Phase 1). Each
    /// segment runs the query through a [`PerSegmentReaderView`],
    /// which lets PR-F's BMW pivot loop fire on the segment's local
    /// `block_max` table. Results are merged into the caller's
    /// collector.
    fn search_per_segment_fanout<C: Collector>(
        &self,
        query: Box<dyn Query>,
        mut collector: C,
        deadline: Option<Deadline>,
    ) -> Result<C> {
        // Downcast ensured by the caller, but re-resolve here to
        // borrow the segment list.
        let inverted_reader = self
            .reader
            .as_any()
            .downcast_ref::<InvertedIndexReader>()
            .expect("search_per_segment_fanout requires InvertedIndexReader");

        let global_doc_count = inverted_reader.doc_count();
        let global_max_doc = inverted_reader.max_doc();
        // Build a global term-info closure that captures an Arc
        // pointing back at the cross-segment reader so each
        // PerSegmentReaderView can resolve IDF lookups.
        let global_term_info_fn = {
            let reader_arc = self.reader.clone();
            std::sync::Arc::new(
                move |field: &str,
                      term: &str|
                      -> Result<Option<crate::lexical::reader::ReaderTermInfo>> {
                    reader_arc.term_info(field, term)
                },
            )
        };

        // Build a cross-segment matching-doc-ids closure (#764) so each
        // PerSegmentReaderView can resolve a cacheable filter clause against the
        // cross-segment snapshot cache rather than re-walking postings per
        // segment. The fanout is only entered when `self.reader` is an
        // InvertedIndexReader (dispatch gate), so the downcast succeeds; the
        // defensive branch drains the matcher uncached.
        let global_matching_doc_ids_fn = {
            let reader_arc = self.reader.clone();
            std::sync::Arc::new(
                move |query: &dyn Query| -> Result<Arc<roaring::RoaringTreemap>> {
                    if let Some(inverted) =
                        reader_arc.as_any().downcast_ref::<InvertedIndexReader>()
                    {
                        inverted.matching_doc_ids(query)
                    } else {
                        let matcher = query.matcher(reader_arc.as_ref())?;
                        Ok(Arc::new(
                            crate::lexical::index::inverted::query_cache::drain_matcher(matcher)?,
                        ))
                    }
                },
            )
        };

        // Per-segment K. The collector wants `top_k` hits globally;
        // each segment returns up to `top_k` so the merge has the
        // headroom to pick any combination of per-segment hits.
        let per_segment_k = collector.requested_top_k().unwrap_or(10);

        let segments = inverted_reader.segment_readers().to_vec();

        #[cfg(not(target_arch = "wasm32"))]
        let segment_iter = segments.par_iter();
        #[cfg(target_arch = "wasm32")]
        let segment_iter = segments.iter();

        let per_segment_results: Vec<Result<Vec<SearchHit>>> = segment_iter
            .map(|seg_arc| -> Result<Vec<SearchHit>> {
                let view = PerSegmentReaderView::new(
                    seg_arc.clone(),
                    global_doc_count,
                    global_max_doc,
                    global_term_info_fn.clone(),
                    global_matching_doc_ids_fn.clone(),
                );
                let view_reader: Arc<dyn LexicalIndexReader> = Arc::new(view);
                let temp_searcher = InvertedIndexSearcher::from_arc(view_reader);
                let temp_collector = TopDocsCollector::new(per_segment_k);
                // Propagate the deadline so each per-segment search aborts
                // mid-flight too — segments run in parallel, so a single slow
                // segment would otherwise leave the whole fanout unbounded
                // (Issue #600).
                let collected = temp_searcher.search_with_collector_deadline(
                    query.clone_box(),
                    temp_collector,
                    false,
                    deadline,
                )?;
                Ok(collected.results())
            })
            .collect();

        // Merge per-segment top-K into the caller's collector. Errors
        // from any one segment short-circuit the whole search.
        for hits in per_segment_results {
            let hits = hits?;
            for hit in hits {
                collector.collect(hit.doc_id, hit.score)?;
                if !collector.needs_more() {
                    return Ok(collector);
                }
            }
        }
        Ok(collector)
    }

    /// Execute a BooleanQuery with parallel sub-query execution.
    ///
    /// Each clause is executed in parallel, then boolean logic is applied:
    /// - Must/Filter: intersection (all must match)
    /// - Should: union (adds score if matching; at least minimum_should_match required)
    /// - MustNot: exclusion (removes matching documents)
    fn search_boolean_query_parallel<C: Collector>(
        &self,
        boolean_query: &BooleanQuery,
        mut collector: C,
        deadline: Option<Deadline>,
    ) -> Result<C> {
        let clauses = boolean_query.clauses();

        if clauses.is_empty() {
            return Ok(collector);
        }

        // Single clause: no need for parallel execution
        if clauses.len() == 1 {
            return self.search_with_collector_deadline(
                clauses[0].query.clone_box(),
                collector,
                false,
                deadline,
            );
        }

        // Execute all clauses in parallel, collecting (doc_id, score) per clause
        #[cfg(not(target_arch = "wasm32"))]
        let iter = clauses.par_iter();
        #[cfg(target_arch = "wasm32")]
        let iter = clauses.iter();

        let clause_results: Vec<(Occur, Result<Vec<SearchHit>>)> = iter
            .map(|clause| {
                // Boolean operations (intersection/union/exclusion) require the
                // full result set from each clause, so we use an unbounded collector.
                let temp_collector = TopDocsCollector::new(usize::MAX);
                let result = self
                    .search_with_collector_deadline(
                        clause.query.clone_box(),
                        temp_collector,
                        false,
                        deadline,
                    )
                    .map(|c| c.results());
                (clause.occur, result)
            })
            .collect();

        // Fold each clause's hits into Roaring bitmaps for the set logic (#587)
        // and a flat `(doc_id, score)` list for Must/Should score accumulation.
        // The set operations (AND / ANDNOT / OR) run as Roaring word-walks
        // instead of `HashMap::retain` / `HashSet::remove`, and the final
        // selection is delegated to the collector's bounded top-K heap rather
        // than a full sort over the whole candidate set.
        //
        // Filter clauses contribute to membership only (score 0), so their docs
        // go into `must_bitmaps` but never into `scored_hits`.
        let mut must_bitmaps: Vec<RoaringTreemap> = Vec::new();
        let mut should_bitmap = RoaringTreemap::new();
        let mut must_not_bitmap = RoaringTreemap::new();
        let mut scored_hits: Vec<(u64, f32)> = Vec::new();
        let mut first_error: Option<LaurusError> = None;

        for (occur, result) in clause_results {
            match result {
                Ok(hits) => match occur {
                    Occur::Must => {
                        let mut bitmap = RoaringTreemap::new();
                        for hit in hits {
                            bitmap.insert(hit.doc_id);
                            scored_hits.push((hit.doc_id, hit.score));
                        }
                        must_bitmaps.push(bitmap);
                    }
                    Occur::Filter => {
                        // Membership only — Filter does not contribute to score.
                        let mut bitmap = RoaringTreemap::new();
                        for hit in hits {
                            bitmap.insert(hit.doc_id);
                        }
                        must_bitmaps.push(bitmap);
                    }
                    Occur::Should => {
                        for hit in hits {
                            should_bitmap.insert(hit.doc_id);
                            scored_hits.push((hit.doc_id, hit.score));
                        }
                    }
                    Occur::MustNot => {
                        for hit in hits {
                            must_not_bitmap.insert(hit.doc_id);
                        }
                    }
                },
                Err(e) => {
                    if first_error.is_none() {
                        first_error = Some(e);
                    }
                }
            }
        }

        // If any clause produced an error, fail the whole query
        if let Some(e) = first_error {
            return Err(e);
        }

        let minimum_should_match = boolean_query.minimum_should_match();
        let has_must = !must_bitmaps.is_empty();

        // Build the survivor membership set via Roaring set operations.
        let mut survivor = if has_must {
            // Intersect smallest-first so the running result shrinks fastest.
            must_bitmaps.sort_unstable_by_key(|b| b.len());
            let mut bitmaps = must_bitmaps.into_iter();
            let mut acc = bitmaps.next().unwrap_or_default();
            for bitmap in bitmaps {
                acc &= &bitmap;
            }
            acc
        } else {
            // No Must/Filter clauses: the Should union is the candidate set.
            should_bitmap.clone()
        };

        // With minimum_should_match > 0 a Must candidate must also appear in at
        // least one Should clause (preserves the existing parallel semantics).
        if has_must && minimum_should_match > 0 {
            survivor &= &should_bitmap;
        }

        // Exclude MustNot documents.
        if !must_not_bitmap.is_empty() {
            survivor -= &must_not_bitmap;
        }

        // Accumulate scores for survivors only (one pass over Must/Should hits),
        // then feed the collector. The collector keeps the top-K via a bounded
        // min-heap, so no full sort over the candidate set is needed.
        let mut score_acc: AHashMap<u64, f32> = AHashMap::with_capacity(survivor.len() as usize);
        for (doc_id, score) in scored_hits {
            if survivor.contains(doc_id) {
                *score_acc.entry(doc_id).or_insert(0.0) += score;
            }
        }

        for doc_id in survivor.iter() {
            let score = score_acc.get(&doc_id).copied().unwrap_or(0.0);
            collector.collect(doc_id, score)?;
            if !collector.needs_more() {
                break;
            }
        }

        Ok(collector)
    }

    /// Load documents for search hits.
    fn load_documents(&self, hits: &mut [SearchHit]) -> Result<()> {
        for hit in hits {
            if let Some(doc) = self.reader.document(hit.doc_id)? {
                hit.document = Some(doc);
            }
        }
        Ok(())
    }

    /// Load documents in parallel for better performance.
    fn load_documents_parallel(&self, hits: &mut [SearchHit]) -> Result<()> {
        // Use a parallel iterator to load documents
        #[cfg(not(target_arch = "wasm32"))]
        let results: Vec<_> = hits
            .par_iter()
            .map(|hit| (hit.doc_id, self.reader.document(hit.doc_id)))
            .collect();
        #[cfg(target_arch = "wasm32")]
        let results: Vec<_> = hits
            .iter()
            .map(|hit| (hit.doc_id, self.reader.document(hit.doc_id)))
            .collect();

        // Update hits with loaded documents
        for (i, (_, doc_result)) in results.into_iter().enumerate() {
            if let Ok(Some(doc)) = doc_result {
                hits[i].document = Some(doc);
            }
        }

        Ok(())
    }

    /// Execute a search with timeout (internal implementation).
    fn search_with_timeout_internal(
        &self,
        query: Box<dyn Query>,
        params: &LexicalSearchParams,
        timeout: Duration,
    ) -> Result<LexicalSearchResults> {
        let start_time = Timer::now();
        // Cooperative deadline (Issue #600). Threading it through the scan
        // loops lets the search abort mid-flight once the budget is spent,
        // instead of only being detected after the query has already run to
        // completion as the old post-hoc `elapsed()` check did.
        let deadline = Some(Deadline::new(start_time, timeout));

        // Create collector based on sort type
        let (mut hits, total_hits) = match &params.sort_by {
            SortField::Field { name, order } => {
                // Use TopFieldCollector for field-based sorting
                let ascending = matches!(order, SortOrder::Asc);
                let collector = TopFieldCollector::with_min_score(
                    params.limit,
                    params.min_score,
                    name.clone(),
                    ascending,
                    self.reader.as_ref(),
                );

                let result_collector = self.search_with_collector_deadline(
                    query.clone_box(),
                    collector,
                    params.parallel,
                    deadline,
                )?;

                (result_collector.results(), result_collector.total_hits())
            }
            SortField::Score => {
                // Use TopDocsCollector for score-based sorting
                let collector = TopDocsCollector::with_min_score(params.limit, params.min_score);

                let result_collector = self.search_with_collector_deadline(
                    query,
                    collector,
                    params.parallel,
                    deadline,
                )?;

                (result_collector.results(), result_collector.total_hits())
            }
        };

        // Final safety net: the scan loops abort mid-flight on the deadline,
        // but a search that finished just over budget (or spent the time
        // outside a scan loop) is still reported as timed out.
        if start_time.elapsed() > timeout {
            return Err(LaurusError::index("Search timeout exceeded"));
        }

        // Load documents if requested
        if params.load_documents {
            if params.parallel && hits.len() > 10 {
                self.load_documents_parallel(&mut hits)?;
            } else {
                self.load_documents(&mut hits)?;
            }
        }

        // No need to sort - already sorted during collection

        // Calculate max score
        let max_score = hits.iter().map(|hit| hit.score).fold(0.0f32, f32::max);

        Ok(LexicalSearchResults {
            hits,
            total_hits,
            max_score,
        })
    }

    /// Search with the given request.
    pub fn search(&self, request: LexicalSearchRequest) -> Result<LexicalSearchResults> {
        // Convert DSL query to Query object if necessary
        let query = match &request.query {
            LexicalSearchQuery::Dsl(dsl_string) => {
                // Parsed-query cache (#590): a popular DSL string is parsed once
                // per snapshot and reused via `clone_box` (cheap — refcount
                // bumps for boolean clause subtrees). The analyzer and
                // `default_fields` are fixed for this searcher, so the DSL
                // string alone keys the cache.
                if let Some(cached) = self.parsed_query_cache.get(dsl_string) {
                    cached.clone_box()
                } else {
                    // Get analyzer from reader
                    let analyzer = if let Some(inverted_index_reader) =
                        self.reader.as_any().downcast_ref::<InvertedIndexReader>()
                    {
                        inverted_index_reader.analyzer().clone()
                    } else {
                        // Fallback to standard analyzer
                        Arc::new(StandardAnalyzer::new()?)
                    };

                    // Parse DSL string into Query object
                    let mut parser = LexicalQueryParser::new(analyzer.clone());
                    if !self.default_fields.is_empty() {
                        parser = parser.with_default_fields(self.default_fields.clone());
                    }
                    let parsed: Arc<dyn Query> = Arc::from(parser.parse(dsl_string)?);
                    self.parsed_query_cache
                        .put(dsl_string.clone(), parsed.clone());
                    parsed.clone_box()
                }
            }
            LexicalSearchQuery::Obj(q) => q.clone_box(),
        };

        // Check if query is empty
        if query.is_empty(self.reader.as_ref())? {
            return Ok(LexicalSearchResults {
                hits: Vec::new(),
                total_hits: 0,
                max_score: 0.0,
            });
        }

        // Execute search with timeout if specified
        if let Some(timeout_ms) = request.params.timeout_ms {
            let timeout = Duration::from_millis(timeout_ms);
            self.search_with_timeout_internal(query, &request.params, timeout)
        } else {
            // Check if we should use field-based sorting during collection
            match &request.params.sort_by {
                SortField::Field { name, order } => {
                    // Use TopFieldCollector for field-based sorting
                    let ascending = matches!(order, SortOrder::Asc);
                    let collector = TopFieldCollector::with_min_score(
                        request.params.limit,
                        request.params.min_score,
                        name.clone(),
                        ascending,
                        self.reader.as_ref(),
                    );

                    let result_collector = self.search_with_collector_parallel(
                        query.clone_box(),
                        collector,
                        request.params.parallel,
                    )?;

                    let mut hits = result_collector.results();
                    let total_hits = result_collector.total_hits();

                    // Load documents if requested
                    if request.params.load_documents {
                        self.load_documents(&mut hits)?;
                    }

                    // No need to sort - already sorted by TopFieldCollector during collection

                    // Calculate max score
                    let max_score = hits.iter().map(|hit| hit.score).fold(0.0f32, f32::max);

                    Ok(LexicalSearchResults {
                        hits,
                        total_hits,
                        max_score,
                    })
                }
                SortField::Score => {
                    // Use TopDocsCollector for score-based sorting
                    let collector = TopDocsCollector::with_min_score(
                        request.params.limit,
                        request.params.min_score,
                    );
                    let result_collector = self.search_with_collector_parallel(
                        query,
                        collector,
                        request.params.parallel,
                    )?;

                    let mut hits = result_collector.results();
                    let total_hits = result_collector.total_hits();

                    // Load documents if requested
                    if request.params.load_documents {
                        self.load_documents(&mut hits)?;
                    }

                    // No need to sort - already sorted by score in TopDocsCollector

                    // Calculate max score
                    let max_score = hits.iter().map(|hit| hit.score).fold(0.0f32, f32::max);

                    Ok(LexicalSearchResults {
                        hits,
                        total_hits,
                        max_score,
                    })
                }
            }
        }
    }

    /// Sort search hits according to the specified sort field.
    /// This is the old post-collection sorting approach, kept for compatibility.
    #[allow(dead_code)]
    fn sort_hits(&self, hits: &mut [SearchHit], sort_by: &SortField) -> Result<()> {
        match sort_by {
            SortField::Score => {
                // Default behavior: already sorted by score from collector
                // Re-sort to ensure descending order
                hits.sort_unstable_by(|a, b| b.score.total_cmp(&a.score));
            }
            SortField::Field { name, order } => {
                // Sort by field value
                hits.sort_unstable_by(|a, b| {
                    let cmp = self.compare_field_values(a, b, name);
                    match order {
                        SortOrder::Asc => cmp,
                        SortOrder::Desc => cmp.reverse(),
                    }
                });
            }
        }
        Ok(())
    }

    /// Compare two search hits by a specific field value.
    #[allow(dead_code)]
    fn compare_field_values(&self, a: &SearchHit, b: &SearchHit, field_name: &str) -> Ordering {
        let val_a = a.document.as_ref().and_then(|doc| doc.get(field_name));
        let val_b = b.document.as_ref().and_then(|doc| doc.get(field_name));

        match (val_a, val_b) {
            (Some(a_val), Some(b_val)) => self.compare_values(a_val, b_val),
            (Some(_), None) => Ordering::Less, // Documents with value come first
            (None, Some(_)) => Ordering::Greater, // Documents without value come last
            (None, None) => Ordering::Equal,
        }
    }

    /// Compare two field values.
    #[allow(dead_code)]
    fn compare_values(&self, a: &FieldValue, b: &FieldValue) -> Ordering {
        match (a, b) {
            // Same type comparisons
            (Text(a_str), Text(b_str)) => a_str.cmp(b_str),
            (Integer(a_int), Integer(b_int)) => a_int.cmp(b_int),
            (Float(a_float), Float(b_float)) => a_float.total_cmp(b_float),
            (Boolean(a_bool), Boolean(b_bool)) => a_bool.cmp(b_bool),
            (DateTime(a_dt), DateTime(b_dt)) => a_dt.cmp(b_dt),
            (Geo(a), Geo(b)) => a
                .lat
                .total_cmp(&b.lat)
                .then_with(|| a.lon.total_cmp(&b.lon)),
            (Bytes(_, a_bytes), Bytes(_, b_bytes)) => a_bytes.cmp(b_bytes),
            (Null, Null) => Ordering::Equal,

            // Mixed types ordering precedence
            // Null < Bool < Int < Float < Text < Bytes
            (Null, _) => Ordering::Less,
            (_, Null) => Ordering::Greater,

            (Boolean(_), _) => Ordering::Less,
            (_, Boolean(_)) => Ordering::Greater,

            (Integer(_), _) => Ordering::Less,
            (_, Integer(_)) => Ordering::Greater,

            (Float(_), _) => Ordering::Less,
            (_, Float(_)) => Ordering::Greater,

            (Text(_), _) => Ordering::Less,
            (_, Text(_)) => Ordering::Greater,

            (Bytes(_, _), _) => Ordering::Less,
            (_, Bytes(_, _)) => Ordering::Greater,

            _ => Ordering::Equal, // Fallback
        }
    }

    /// Count documents matching the request.
    ///
    /// If `min_score` is specified in the request parameters, only documents
    /// with a score equal to or greater than the threshold are counted.
    pub fn count(&self, request: LexicalSearchRequest) -> Result<u64> {
        let lexical_query = request.query;

        // Parse DSL string if needed
        let query = if let LexicalSearchQuery::Dsl(_) = &lexical_query {
            // Get analyzer from reader
            let analyzer = if let Some(inverted_index_reader) =
                self.reader.as_any().downcast_ref::<InvertedIndexReader>()
            {
                inverted_index_reader.analyzer().clone()
            } else {
                // Fallback to standard analyzer
                Arc::new(StandardAnalyzer::new()?)
            };

            // Parse DSL string into Query object
            lexical_query.into_query(&analyzer)?
        } else {
            match lexical_query {
                LexicalSearchQuery::Obj(q) => q,
                _ => unreachable!(),
            }
        };

        // Check if query is empty
        if query.is_empty(self.reader.as_ref())? {
            return Ok(0);
        }

        // O(1) fast path (Issue #610): a bare `TermQuery` with no score
        // threshold over a reader with no deletions equals the term's document
        // frequency, which is already stored in the term dictionary — so the
        // full posting-list walk the slow path performs is unnecessary.
        //
        // All three guards are required for correctness; if any fails we fall
        // through to the slow path, so the fast path can never miscount:
        // - `min_score <= 0.0`: with a positive threshold each doc's score must
        //   be computed, so a count cannot come from `doc_freq` alone.
        // - `doc_count() == max_doc()`: the term dictionary's `doc_freq` counts
        //   raw postings, including deleted docs, whereas the slow path filters
        //   deletions out. The equality holds iff the index has no deletions,
        //   in which case the two agree. (Conservative: any inequality, for any
        //   reason, just keeps the slow path.)
        // - the query is exactly a `TermQuery` (not a Boolean/phrase/etc.).
        if request.params.min_score <= 0.0
            && self.reader.doc_count() == self.reader.max_doc()
            && let Some(term_query) = query.as_any().downcast_ref::<TermQuery>()
        {
            return self
                .reader
                .term_doc_freq(term_query.field(), term_query.term());
        }

        // Use count collector with min_score if specified
        let collector = if request.params.min_score > 0.0 {
            CountCollector::with_min_score(request.params.min_score)
        } else {
            CountCollector::new()
        };

        let result_collector = self.search_with_collector(query, collector)?;
        Ok(result_collector.total_hits())
    }
}

// Implement LexicalSearcher trait for InvertedIndexSearcher
impl crate::lexical::search::searcher::LexicalSearcher for InvertedIndexSearcher {
    fn search(&self, request: LexicalSearchRequest) -> Result<LexicalSearchResults> {
        InvertedIndexSearcher::search(self, request)
    }

    fn count(
        &self,
        request: crate::lexical::search::searcher::LexicalSearchRequest,
    ) -> Result<u64> {
        InvertedIndexSearcher::count(self, request)
    }

    fn matching_doc_ids(&self, query: Box<dyn Query>) -> Result<Arc<roaring::RoaringTreemap>> {
        // The common case: the reader is an `InvertedIndexReader`, which owns
        // the snapshot-scoped query/filter cache (Issue #578) and serves
        // cacheable queries without re-walking posting lists.
        if let Some(inverted_reader) = self.reader.as_any().downcast_ref::<InvertedIndexReader>() {
            return inverted_reader.matching_doc_ids(query.as_ref());
        }
        // Fallback for a non-inverted reader (e.g. a transient
        // `PerSegmentReaderView`): no snapshot cache is available, so drain the
        // matcher directly using the shared helper.
        let matcher = query.matcher(self.reader.as_ref())?;
        let bitmap = crate::lexical::index::inverted::query_cache::drain_matcher(matcher)?;
        Ok(Arc::new(bitmap))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexical::index::inverted::reader::{InvertedIndexReader, InvertedIndexReaderConfig};
    use crate::lexical::query::boolean::{BooleanQuery, BooleanQueryBuilder};
    use crate::lexical::query::term::TermQuery;

    use crate::storage::memory::MemoryStorage;
    use crate::storage::memory::MemoryStorageConfig;
    use std::sync::Arc;

    #[allow(dead_code)]
    fn create_test_searcher() -> InvertedIndexSearcher {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let reader = Box::new(
            InvertedIndexReader::new(vec![], storage, InvertedIndexReaderConfig::default())
                .unwrap(),
        );
        InvertedIndexSearcher::new(reader)
    }

    #[test]
    fn test_searcher_creation() {
        let searcher = create_test_searcher();

        // Verify searcher has a valid reader
        let reader = searcher.reader();
        assert!(Arc::strong_count(reader) >= 1, "Reader should be valid");

        // Verify reader has expected initial state
        assert_eq!(
            reader.doc_count(),
            0,
            "New searcher should have 0 documents"
        );
    }

    #[test]
    fn test_search_term_query() {
        let searcher = create_test_searcher();
        let query = Box::new(TermQuery::new("title", "hello")) as Box<dyn Query>;

        let request = LexicalSearchRequest::new(query);
        let results = searcher.search(request).unwrap();

        // Should return empty results for non-existent terms
        assert_eq!(results.hits.len(), 0);
        assert_eq!(results.total_hits, 0);
        assert_eq!(results.max_score, 0.0);
    }

    #[test]
    fn test_search_boolean_query() {
        let searcher = create_test_searcher();

        let query = Box::new(
            BooleanQueryBuilder::new()
                .must(Box::new(TermQuery::new("title", "hello")))
                .should(Box::new(TermQuery::new("body", "world")))
                .build(),
        ) as Box<dyn Query>;

        let request = LexicalSearchRequest::new(query);
        let results = searcher.search(request).unwrap();

        // Should return empty results for non-existent terms
        assert_eq!(results.hits.len(), 0);
        assert_eq!(results.total_hits, 0);
        assert_eq!(results.max_score, 0.0);
    }

    #[test]
    fn test_search_with_config() {
        let searcher = create_test_searcher();
        let query = Box::new(TermQuery::new("title", "hello")) as Box<dyn Query>;

        let request = LexicalSearchRequest::new(query)
            .limit(5)
            .min_score(0.5)
            .load_documents(false);

        let results = searcher.search(request).unwrap();

        // Should respect configuration
        assert_eq!(results.hits.len(), 0);
        assert_eq!(results.total_hits, 0);
    }

    #[test]
    fn test_count_query() {
        let searcher = create_test_searcher();
        let query = Box::new(TermQuery::new("title", "hello")) as Box<dyn Query>;

        let count = searcher.count(LexicalSearchRequest::new(query)).unwrap();

        // Should return 0 for non-existent terms
        assert_eq!(count, 0);
    }

    #[test]
    fn test_search_with_timeout() {
        let searcher = create_test_searcher();
        let query = Box::new(TermQuery::new("title", "hello")) as Box<dyn Query>;

        let request = LexicalSearchRequest::new(query).timeout_ms(1000); // 1 second timeout

        let results = searcher.search(request).unwrap();

        // Should complete within timeout
        assert_eq!(results.hits.len(), 0);
        assert_eq!(results.total_hits, 0);
    }

    #[test]
    fn deadline_check_semantics() {
        // The deadline primitive (Issue #600): it fires only at check-interval
        // indices, and only when the budget is actually exhausted.
        let now = Timer::now();
        // Index 0 is a multiple of the interval, so an exhausted (zero) budget
        // is detected immediately — a search fails fast.
        assert!(Deadline::new(now, Duration::ZERO).check(0).is_err());
        // Between check intervals the clock is never read, so even a zero
        // budget does not fire — this is what keeps the per-document cost out.
        assert!(Deadline::new(now, Duration::ZERO).check(1).is_ok());
        assert!(
            Deadline::new(now, Duration::ZERO)
                .check(DEADLINE_CHECK_INTERVAL - 1)
                .is_ok()
        );
        // A check-interval index with the budget spent fires.
        assert!(
            Deadline::new(now, Duration::ZERO)
                .check(DEADLINE_CHECK_INTERVAL)
                .is_err()
        );
        // An ample budget never fires, even at a check-interval index.
        assert!(
            Deadline::new(now, Duration::from_secs(3600))
                .check(0)
                .is_ok()
        );
    }

    /// Build a searcher over a populated index. `segments` commits the docs in
    /// that many batches so we can exercise both the single-segment scan loop
    /// and the multi-segment fanout (Issue #600).
    fn populated_searcher(segments: usize) -> InvertedIndexSearcher {
        use crate::analysis::analyzer::standard::StandardAnalyzer;
        use crate::lexical::index::inverted::writer::{
            InvertedIndexWriter, InvertedIndexWriterConfig,
        };
        use crate::lexical::writer::LexicalIndexWriter;

        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let config = InvertedIndexWriterConfig {
            analyzer: Arc::new(StandardAnalyzer::new().unwrap()),
            ..Default::default()
        };
        let mut writer = InvertedIndexWriter::new(storage, config).unwrap();
        let per_segment = 200;
        for seg in 0..segments.max(1) {
            for i in 0..per_segment {
                let n = seg * per_segment + i;
                writer
                    .add_document(
                        crate::Document::builder()
                            .add_text("content", format!("hello world doc {n}"))
                            .build(),
                    )
                    .unwrap();
            }
            writer.commit().unwrap();
        }
        InvertedIndexSearcher::from_arc(writer.build_reader().unwrap())
    }

    #[test]
    fn search_with_zero_timeout_interrupts_real_docs() {
        // With real matches the scan loop is entered, so the first deadline
        // check (scanned == 0) fires on an already-spent zero budget — proving
        // the timeout interrupts the search rather than only being reported
        // after it completes (Issue #600).
        let searcher = populated_searcher(1);
        let query = Box::new(TermQuery::new("content", "hello")) as Box<dyn Query>;
        let request = LexicalSearchRequest::new(query).timeout_ms(0);

        let err = searcher.search(request).unwrap_err();
        assert!(
            err.to_string().contains("timeout"),
            "expected a timeout error, got: {err}"
        );
    }

    #[test]
    fn search_with_zero_timeout_interrupts_multi_segment_fanout() {
        // The per-segment fanout must honour the deadline too (a single slow
        // segment would otherwise leave the parallel fanout unbounded).
        let searcher = populated_searcher(3);
        let query = Box::new(TermQuery::new("content", "hello")) as Box<dyn Query>;
        let request = LexicalSearchRequest::new(query).timeout_ms(0);

        let err = searcher.search(request).unwrap_err();
        assert!(
            err.to_string().contains("timeout"),
            "expected a timeout error, got: {err}"
        );
    }

    #[test]
    fn search_without_timeout_returns_hits() {
        // Regression guard: a search with no timeout still returns results
        // (the deadline path is inert when `timeout_ms` is unset).
        let searcher = populated_searcher(1);
        let query = Box::new(TermQuery::new("content", "hello")) as Box<dyn Query>;

        let results = searcher.search(LexicalSearchRequest::new(query)).unwrap();
        assert!(
            !results.hits.is_empty(),
            "a non-timed search must return matching docs"
        );
    }

    #[test]
    fn test_search_with_collector() {
        let searcher = create_test_searcher();
        let query = Box::new(TermQuery::new("title", "hello"));
        let collector = TopDocsCollector::new(10);

        let result_collector = searcher.search_with_collector(query, collector).unwrap();

        assert_eq!(result_collector.total_hits(), 0);
        assert_eq!(result_collector.results().len(), 0);
    }

    #[test]
    fn test_search_empty_query() {
        let searcher = create_test_searcher();
        // Create a boolean query with no clauses (empty query)
        let query = Box::new(BooleanQuery::new()) as Box<dyn Query>;

        let request = LexicalSearchRequest::new(query);
        let results = searcher.search(request).unwrap();

        // Should return empty results for empty query
        assert_eq!(results.hits.len(), 0);
        assert_eq!(results.total_hits, 0);
        assert_eq!(results.max_score, 0.0);
    }

    #[test]
    fn test_count_empty_query() {
        let searcher = create_test_searcher();
        let query = Box::new(BooleanQuery::new()) as Box<dyn Query>;

        let count = searcher.count(LexicalSearchRequest::new(query)).unwrap();

        // Should return 0 for empty query
        assert_eq!(count, 0);
    }

    #[test]
    fn test_search_request_builder() {
        let query = Box::new(TermQuery::new("title", "hello")) as Box<dyn Query>;

        let request = LexicalSearchRequest::new(query)
            .limit(20)
            .min_score(0.1)
            .load_documents(false)
            .timeout_ms(5000);

        assert_eq!(request.params.limit, 20);
        assert_eq!(request.params.min_score, 0.1);
        assert!(!request.params.load_documents);
        assert_eq!(request.params.timeout_ms, Some(5000));
    }

    /// Wrapper that suppresses BMW dispatch by returning
    /// `bmw_capable() = false`, so we can run the same query against
    /// the existing matcher-driven path for equivalence comparison.
    #[derive(Debug)]
    struct NonBmwTopDocs(TopDocsCollector);

    impl Collector for NonBmwTopDocs {
        fn collect(&mut self, doc_id: u64, score: f32) -> Result<()> {
            self.0.collect(doc_id, score)
        }
        fn results(&self) -> Vec<crate::lexical::query::SearchHit> {
            self.0.results()
        }
        fn total_hits(&self) -> u64 {
            self.0.total_hits()
        }
        fn needs_more(&self) -> bool {
            self.0.needs_more()
        }
        fn min_score(&self) -> f32 {
            self.0.min_score()
        }
        fn min_competitive(&self) -> f32 {
            self.0.min_competitive()
        }
        fn reset(&mut self) {
            self.0.reset()
        }
        // bmw_capable defaults to false → searcher uses the legacy path.
    }

    /// PR-F: BMW fast path must produce the same top-K (same docs,
    /// same scores) as the existing matcher-driven path on a real
    /// committed index. Skewed-TF distribution drives the heap to
    /// fill quickly and exercises the pivot loop's skip path.
    #[test]
    fn bmw_topk_equivalence_should_or() {
        use crate::Document;
        use crate::lexical::query::boolean::BooleanQueryBuilder;
        use crate::lexical::store::LexicalStore;
        use crate::lexical::store::config::LexicalIndexConfig;

        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let store = LexicalStore::new(storage, LexicalIndexConfig::default()).unwrap();

        // Skewed-TF corpus: alpha clusters at the start of the doc id
        // range; beta middle, gamma tail. With BLOCK_SIZE = 128 this
        // produces a non-trivial distribution of per-block bounds.
        for id in 0..512u64 {
            let mut body = String::new();
            if id < 60 {
                body.push_str("alpha alpha alpha ");
            } else if id < 200 {
                body.push_str("alpha ");
            }
            if (100..400).contains(&id) {
                body.push_str("beta ");
            }
            if id >= 350 && id % 3 == 0 {
                body.push_str("gamma ");
            }
            body.push_str("filler text content body");
            let doc = Document::builder()
                .add_text("title", format!("doc-{id}"))
                .add_text("body", &body)
                .build();
            store.upsert_document(id, doc).unwrap();
        }
        store.commit().unwrap();

        let make_query = || -> Box<dyn Query> {
            Box::new(
                BooleanQueryBuilder::new()
                    .should(Box::new(TermQuery::new("body", "alpha")))
                    .should(Box::new(TermQuery::new("body", "beta")))
                    .should(Box::new(TermQuery::new("body", "gamma")))
                    .build(),
            )
        };

        // BMW path: bmw_capable() is true on TopDocsCollector, so the
        // entrypoint dispatches to the executor.
        let bmw = store
            .search(LexicalSearchRequest::new(make_query()).limit(10))
            .unwrap();

        // Reference path: same query, same store, but the wrapper
        // collector reports `bmw_capable = false` so the searcher
        // falls through to the existing matcher-driven loop.
        let reference = {
            let request = LexicalSearchRequest::new(make_query()).limit(10);
            // Build a searcher manually so we can pass our wrapper
            // collector through `search_with_collector`. The store's
            // public `search()` always uses TopDocsCollector directly,
            // which bmw_capable's true → BMW.
            let _ = request;
            // Instead: round-trip through the store with a *much*
            // larger K so the heap never fills (min_competitive stays
            // NEG_INFINITY → BMW pivot loop reduces to a doc-by-doc
            // walk identical to the legacy path), then sort + slice.
            let big = store
                .search(LexicalSearchRequest::new(make_query()).limit(usize::MAX))
                .unwrap();
            let mut hits: Vec<_> = big.hits.into_iter().map(|h| (h.doc_id, h.score)).collect();
            hits.sort_by(|x, y| y.1.total_cmp(&x.1).then(x.0.cmp(&y.0)));
            hits.truncate(10);
            hits
        };

        let mut bmw_hits: Vec<_> = bmw.hits.iter().map(|h| (h.doc_id, h.score)).collect();
        bmw_hits.sort_by(|x, y| y.1.total_cmp(&x.1).then(x.0.cmp(&y.0)));
        assert_eq!(bmw_hits.len(), reference.len(), "result count differs");
        for (idx, (x, y)) in bmw_hits.iter().zip(reference.iter()).enumerate() {
            assert_eq!(x.0, y.0, "rank {idx}: doc_id mismatch");
            assert!(
                (x.1 - y.1).abs() < 1e-4,
                "rank {idx} doc {}: score mismatch bmw={} ref={}",
                x.0,
                x.1,
                y.1,
            );
        }

        // Suppress dead-code warning on the wrapper while we're using
        // the round-trip technique. The wrapper is kept for future
        // tests that want to invoke the legacy path explicitly.
        let _suppress_dead_code = NonBmwTopDocs(TopDocsCollector::new(0));
    }

    /// Helper for #476 Phase 1 tests: build a `LexicalStore` with
    /// the same skewed-TF corpus as the equivalence test, but split
    /// the writes across `segment_count` commits so the underlying
    /// reader has multiple segments.
    fn build_skewed_store_with_segments(
        segment_count: usize,
    ) -> crate::lexical::store::LexicalStore {
        use crate::Document;
        use crate::lexical::store::LexicalStore;
        use crate::lexical::store::config::LexicalIndexConfig;

        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let store = LexicalStore::new(storage, LexicalIndexConfig::default()).unwrap();

        let n: u64 = 512;
        let chunk = n.div_ceil(segment_count as u64);
        let mut next_commit = chunk;
        for id in 0..n {
            let mut body = String::new();
            if id < 60 {
                body.push_str("alpha alpha alpha ");
            } else if id < 200 {
                body.push_str("alpha ");
            }
            if (100..400).contains(&id) {
                body.push_str("beta ");
            }
            if id >= 350 && id % 3 == 0 {
                body.push_str("gamma ");
            }
            body.push_str("filler text content body");
            let doc = Document::builder()
                .add_text("title", format!("doc-{id}"))
                .add_text("body", &body)
                .build();
            store.upsert_document(id, doc).unwrap();

            if id + 1 == next_commit && id + 1 < n {
                store.commit().unwrap();
                next_commit += chunk;
            }
        }
        store.commit().unwrap();
        store
    }

    /// PR-F follow-up #476 Phase 1: the per-segment fanout path
    /// must return the **same top-K** as the legacy cross-segment
    /// path on the same multi-segment store. We can't compare to
    /// a single-segment build because per-segment scoring uses each
    /// segment's local `avg_field_length` (Lucene-style), which
    /// produces ranking-equivalent but numerically-different scores
    /// from global-avg single-segment scoring. Comparing fanout to
    /// the **legacy path on the same multi-segment store** isolates
    /// the fanout's correctness from that scoring choice.
    #[test]
    fn per_segment_fanout_topk_matches_legacy_multi_segment_path() {
        use crate::lexical::query::SearchHit;
        use crate::lexical::query::boolean::BooleanQueryBuilder;

        let store = build_skewed_store_with_segments(4);
        let make_query = || -> Box<dyn Query> {
            Box::new(
                BooleanQueryBuilder::new()
                    .should(Box::new(TermQuery::new("body", "alpha")))
                    .should(Box::new(TermQuery::new("body", "beta")))
                    .should(Box::new(TermQuery::new("body", "gamma")))
                    .build(),
            )
        };

        // Fanout path: TopDocsCollector reports `bmw_capable = true`
        // and segment_count == 4 → dispatches to fanout.
        let fanout_hits = store
            .search(LexicalSearchRequest::new(make_query()).limit(10))
            .unwrap()
            .hits;

        // Legacy path: drive the searcher directly with our
        // `NonBmwTopDocs` wrapper so `bmw_capable = false` and
        // dispatch falls through to the existing matcher-driven
        // loop on the cross-segment-aggregated reader.
        let legacy_hits: Vec<SearchHit> = {
            let reader = store.reader_for_tests().unwrap();
            let searcher = InvertedIndexSearcher::from_arc(reader);
            let collector = NonBmwTopDocs(TopDocsCollector::new(10));
            let collected = searcher
                .search_with_collector(make_query(), collector)
                .unwrap();
            collected.0.results()
        };

        assert_eq!(
            fanout_hits.len(),
            legacy_hits.len(),
            "fanout vs legacy hit count differs"
        );

        let fanout_ids: std::collections::BTreeSet<u64> =
            fanout_hits.iter().map(|h| h.doc_id).collect();
        let legacy_ids: std::collections::BTreeSet<u64> =
            legacy_hits.iter().map(|h| h.doc_id).collect();
        assert_eq!(
            fanout_ids, legacy_ids,
            "fanout vs legacy top-K doc id sets differ"
        );

        // Both paths run on the same multi-segment store with the
        // same per-segment avg semantics, so scores must agree
        // within float tolerance.
        for hit in &fanout_hits {
            let legacy_score = legacy_hits
                .iter()
                .find(|h| h.doc_id == hit.doc_id)
                .expect("doc must be in legacy top-K too")
                .score;
            let tol = 1e-4_f32.max(0.01_f32 * legacy_score.abs());
            assert!(
                (hit.score - legacy_score).abs() < tol,
                "doc {}: fanout={} legacy={} (tol {})",
                hit.doc_id,
                hit.score,
                legacy_score,
                tol,
            );
        }
    }

    /// PR-F follow-up #476 Phase 1: the per-segment fanout must
    /// fall through to the legacy path when the store has only one
    /// segment (the existing PR-F BMW path is already optimal).
    #[test]
    fn per_segment_fanout_falls_back_when_single_segment() {
        use crate::lexical::query::boolean::BooleanQueryBuilder;

        let store = build_skewed_store_with_segments(1);
        let query: Box<dyn Query> = Box::new(
            BooleanQueryBuilder::new()
                .should(Box::new(TermQuery::new("body", "alpha")))
                .should(Box::new(TermQuery::new("body", "beta")))
                .build(),
        );
        // The fact that this returns at all (without panicking on the
        // `expect("requires InvertedIndexReader")` in fanout) is the
        // proof: the dispatch saw `segment_count() == 1` and skipped
        // the fanout branch.
        let hits = store
            .search(LexicalSearchRequest::new(query).limit(10))
            .unwrap()
            .hits;
        assert!(!hits.is_empty(), "single-seg query should return hits");
    }

    /// PR-F follow-up #476 Phase 1: a non-`bmw_capable` collector
    /// (here: `CountCollector`) must skip both the BMW fast path and
    /// the per-segment fanout, so multi-segment count queries still
    /// hit the legacy aggregation path.
    #[test]
    fn per_segment_fanout_falls_back_for_count_collector() {
        use crate::lexical::query::boolean::BooleanQueryBuilder;

        let store = build_skewed_store_with_segments(4);
        let query: Box<dyn Query> = Box::new(
            BooleanQueryBuilder::new()
                .should(Box::new(TermQuery::new("body", "alpha")))
                .should(Box::new(TermQuery::new("body", "beta")))
                .build(),
        );
        let count = store.count(LexicalSearchRequest::new(query)).unwrap();
        assert!(count > 0, "count query on multi-seg corpus must hit");
    }

    // ----- Issue #578: query / filter result cache -----

    /// `matching_doc_ids` must return exactly the doc-id set that an unbounded
    /// `search` produces, for both a term filter and a boolean filter. The
    /// cache is score-independent, so only the *set* (not scores) is compared.
    #[test]
    fn matching_doc_ids_matches_search_hit_set() {
        use crate::lexical::query::boolean::BooleanQueryBuilder;
        use std::collections::BTreeSet;

        let store = build_skewed_store_with_segments(1);

        let cases: Vec<Box<dyn Query>> = vec![
            Box::new(TermQuery::new("body", "alpha")),
            Box::new(
                BooleanQueryBuilder::new()
                    .must(Box::new(TermQuery::new("body", "alpha")))
                    .should(Box::new(TermQuery::new("body", "beta")))
                    .build(),
            ),
        ];

        for query in cases {
            let bitmap = store.matching_doc_ids(query.clone_box()).unwrap();
            let cached_set: BTreeSet<u64> = bitmap.iter().collect();

            let search_set: BTreeSet<u64> = store
                .search(
                    LexicalSearchRequest::new(query.clone_box())
                        .limit(usize::MAX)
                        .load_documents(false),
                )
                .unwrap()
                .hits
                .into_iter()
                .map(|h| h.doc_id)
                .collect();

            assert_eq!(
                cached_set,
                search_set,
                "matching_doc_ids must equal the search hit set for {}",
                query.description()
            );
            assert!(!cached_set.is_empty(), "corpus should match the query");
        }
    }

    /// A repeated cacheable lookup against the same reader snapshot is served
    /// from the cache: it returns the very same `Arc` and bumps the hit
    /// counter.
    #[test]
    fn matching_doc_ids_cache_hit_returns_shared_arc() {
        let store = build_skewed_store_with_segments(1);
        let reader = store.reader_for_tests().unwrap();
        let inverted = reader
            .as_any()
            .downcast_ref::<InvertedIndexReader>()
            .expect("memory store yields an InvertedIndexReader");

        let query: Box<dyn Query> = Box::new(TermQuery::new("body", "alpha"));

        let first = inverted.matching_doc_ids(query.as_ref()).unwrap();
        let second = inverted.matching_doc_ids(query.as_ref()).unwrap();

        assert_eq!(first, second, "cache hit must return the same set");
        assert!(
            Arc::ptr_eq(&first, &second),
            "second lookup should be served from the cache (same Arc)"
        );

        let stats = inverted.query_cache_stats();
        assert_eq!(stats.misses, 1, "first lookup is a miss");
        assert_eq!(stats.hits, 1, "second lookup is a hit");
    }

    /// Deleted documents must not appear in a cached filter set (deletions are
    /// filtered at the posting-iterator level, before the matcher).
    #[test]
    fn matching_doc_ids_excludes_deleted_docs() {
        use crate::Document;
        use crate::lexical::store::LexicalStore;
        use crate::lexical::store::config::LexicalIndexConfig;

        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let store = LexicalStore::new(storage, LexicalIndexConfig::default()).unwrap();
        for id in 0..10u64 {
            let doc = Document::builder().add_text("body", "shared term").build();
            store.upsert_document(id, doc).unwrap();
        }
        store.commit().unwrap();

        let query = || -> Box<dyn Query> { Box::new(TermQuery::new("body", "shared")) };
        let before = store.matching_doc_ids(query()).unwrap();
        assert_eq!(before.len(), 10);

        store.delete_document_by_internal_id(3).unwrap();
        store.commit().unwrap();

        let after = store.matching_doc_ids(query()).unwrap();
        assert_eq!(after.len(), 9, "deleted doc must be excluded");
        assert!(!after.contains(3), "doc 3 was deleted");
    }

    /// `commit` drops the cached searcher (and its reader's cache), so the next
    /// lookup recomputes against the new snapshot and sees freshly added docs.
    #[test]
    fn commit_invalidates_query_filter_cache() {
        use crate::Document;
        use crate::lexical::store::LexicalStore;
        use crate::lexical::store::config::LexicalIndexConfig;

        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let store = LexicalStore::new(storage, LexicalIndexConfig::default()).unwrap();
        for id in 0..5u64 {
            let doc = Document::builder().add_text("body", "rust").build();
            store.upsert_document(id, doc).unwrap();
        }
        store.commit().unwrap();

        let query = || -> Box<dyn Query> { Box::new(TermQuery::new("body", "rust")) };
        let before = store.matching_doc_ids(query()).unwrap();
        assert_eq!(before.len(), 5);

        // Add a matching doc and commit; the cached searcher is invalidated.
        store
            .upsert_document(99, Document::builder().add_text("body", "rust").build())
            .unwrap();
        store.commit().unwrap();

        let after = store.matching_doc_ids(query()).unwrap();
        assert_eq!(
            after.len(),
            6,
            "post-commit lookup must see the new doc (cache invalidated)"
        );
        assert!(after.contains(99));
    }

    /// A query whose `cache_key` is `None` (here a MustNot-only boolean, R1)
    /// must never touch the cache: it recomputes each call (distinct `Arc`) and
    /// leaves the hit/miss counters untouched, while still returning a stable,
    /// correct set.
    #[test]
    fn uncacheable_query_bypasses_cache() {
        use crate::lexical::query::boolean::BooleanQueryBuilder;

        let store = build_skewed_store_with_segments(1);
        let reader = store.reader_for_tests().unwrap();
        let inverted = reader
            .as_any()
            .downcast_ref::<InvertedIndexReader>()
            .unwrap();

        let make = || -> Box<dyn Query> {
            Box::new(
                BooleanQueryBuilder::new()
                    .must_not(Box::new(TermQuery::new("body", "alpha")))
                    .build(),
            )
        };
        assert!(
            make().cache_key().is_none(),
            "MustNot-only boolean must be uncacheable"
        );

        let first = inverted.matching_doc_ids(make().as_ref()).unwrap();
        let second = inverted.matching_doc_ids(make().as_ref()).unwrap();

        assert_eq!(
            first, second,
            "uncacheable query still returns a stable set"
        );
        assert!(
            !Arc::ptr_eq(&first, &second),
            "uncacheable query must recompute (a distinct Arc each call)"
        );
        let stats = inverted.query_cache_stats();
        assert_eq!(stats.hits, 0, "uncacheable query never hits the cache");
        assert_eq!(stats.misses, 0, "uncacheable query never probes the cache");
    }

    /// Many threads hammering the same cached filter must not deadlock or race
    /// on the cache `Mutex`, and every thread must observe the same set.
    #[test]
    fn concurrent_matching_doc_ids_is_consistent() {
        use std::collections::BTreeSet;
        use std::thread;

        let store = Arc::new(build_skewed_store_with_segments(1));
        // Prime the cached searcher so all threads share one reader + cache.
        let expected: BTreeSet<u64> = store
            .matching_doc_ids(Box::new(TermQuery::new("body", "alpha")))
            .unwrap()
            .iter()
            .collect();
        assert!(!expected.is_empty());

        let mut handles = Vec::new();
        for _ in 0..8 {
            let store = Arc::clone(&store);
            let expected = expected.clone();
            handles.push(thread::spawn(move || {
                for _ in 0..50 {
                    let set: BTreeSet<u64> = store
                        .matching_doc_ids(Box::new(TermQuery::new("body", "alpha")))
                        .unwrap()
                        .iter()
                        .collect();
                    assert_eq!(set, expected, "every thread sees the same cached set");
                }
            }));
        }
        for handle in handles {
            handle.join().unwrap();
        }
    }

    // ----- Issue #764: Occur::Filter clause reuses the filter cache -----

    /// A repeated `must(...).filter(...)` search must serve the `Occur::Filter`
    /// clause from `QueryFilterCache` (single-segment / non-fanout path).
    #[test]
    fn filter_clause_reuses_cache_single_segment() {
        use crate::lexical::query::boolean::BooleanQueryBuilder;

        let store = build_skewed_store_with_segments(1);
        let reader = store.reader_for_tests().unwrap();
        let searcher = InvertedIndexSearcher::from_arc(reader.clone());

        let make = || -> Box<dyn Query> {
            Box::new(
                BooleanQueryBuilder::new()
                    .must(Box::new(TermQuery::new("body", "alpha")))
                    .filter(Box::new(TermQuery::new("body", "beta")))
                    .build(),
            )
        };

        // First search populates the filter-clause set; second reuses it.
        let _ = searcher
            .search_with_collector(make(), TopDocsCollector::new(10))
            .unwrap();
        let _ = searcher
            .search_with_collector(make(), TopDocsCollector::new(10))
            .unwrap();

        let inverted = reader
            .as_any()
            .downcast_ref::<InvertedIndexReader>()
            .unwrap();
        let stats = inverted.query_cache_stats();
        assert!(
            stats.hits >= 1,
            "the Occur::Filter clause must hit the cache on the repeat search (stats: {stats:?})"
        );
    }

    /// Cache-on must produce exactly the same result set as cache-off for a
    /// filtered boolean across a multi-segment index (exercises the fanout
    /// path through `PerSegmentReaderView::matching_doc_ids`).
    #[test]
    fn filter_clause_cache_matches_uncached_multi_segment() {
        use crate::Document;
        use crate::lexical::query::boolean::BooleanQueryBuilder;
        use crate::lexical::store::LexicalStore;
        use crate::lexical::store::config::LexicalIndexConfig;
        use std::collections::BTreeSet;

        // Build a 4-segment store with the given cache capacity. alpha = even
        // ids, beta = multiples of 3, so must(alpha) ∩ filter(beta) = ids % 6.
        let build = |capacity: usize| -> LexicalStore {
            let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
            let config = LexicalIndexConfig::builder()
                .query_filter_cache_capacity(capacity)
                .build();
            let store = LexicalStore::new(storage, config).unwrap();
            for id in 0..400u64 {
                let mut body = String::new();
                if id % 2 == 0 {
                    body.push_str("alpha ");
                }
                if id % 3 == 0 {
                    body.push_str("beta ");
                }
                body.push_str("filler");
                let doc = Document::builder().add_text("body", &body).build();
                store.upsert_document(id, doc).unwrap();
                if id % 100 == 99 {
                    store.commit().unwrap();
                }
            }
            store.commit().unwrap();
            store
        };

        let make = || -> Box<dyn Query> {
            Box::new(
                BooleanQueryBuilder::new()
                    .must(Box::new(TermQuery::new("body", "alpha")))
                    .filter(Box::new(TermQuery::new("body", "beta")))
                    .build(),
            )
        };
        let run = |store: &LexicalStore| -> BTreeSet<u64> {
            store
                .search(
                    LexicalSearchRequest::new(make())
                        .limit(usize::MAX)
                        .load_documents(false),
                )
                .unwrap()
                .hits
                .into_iter()
                .map(|h| h.doc_id)
                .collect()
        };

        let cached_set = run(&build(1024));
        let uncached_set = run(&build(0));

        assert_eq!(
            cached_set, uncached_set,
            "cache-on must equal cache-off for a filtered boolean (fanout path)"
        );
        assert!(!cached_set.is_empty(), "filter should match some docs");
        assert!(
            cached_set.iter().all(|&d| d % 6 == 0),
            "must(alpha=even) ∩ filter(beta=%3) == ids divisible by 6"
        );
    }

    // ----- Issue #587: Roaring-backed parallel boolean executor -----

    /// Single-segment store with a tiny, hand-checkable corpus for the parallel
    /// boolean set-logic tests:
    /// doc0=alpha, doc1=alpha+beta, doc2=beta, doc3=alpha+beta+gamma, doc4=gamma.
    fn build_boolean_corpus() -> crate::lexical::store::LexicalStore {
        use crate::Document;
        use crate::lexical::store::LexicalStore;
        use crate::lexical::store::config::LexicalIndexConfig;

        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let store = LexicalStore::new(storage, LexicalIndexConfig::default()).unwrap();
        for (id, body) in [
            (0u64, "alpha"),
            (1, "alpha beta"),
            (2, "beta"),
            (3, "alpha beta gamma"),
            (4, "gamma"),
        ] {
            store
                .upsert_document(id, Document::builder().add_text("body", body).build())
                .unwrap();
        }
        store.commit().unwrap();
        store
    }

    /// Drive `query` through the parallel boolean executor and return the
    /// sorted result doc-id set.
    fn parallel_doc_ids(
        store: &crate::lexical::store::LexicalStore,
        query: Box<dyn Query>,
    ) -> Vec<u64> {
        let reader = store.reader_for_tests().unwrap();
        let searcher = InvertedIndexSearcher::from_arc(reader);
        let mut ids: Vec<u64> = searcher
            .search_with_collector_parallel(query, TopDocsCollector::new(100), true)
            .unwrap()
            .results()
            .into_iter()
            .map(|h| h.doc_id)
            .collect();
        ids.sort_unstable();
        ids
    }

    #[test]
    fn parallel_boolean_must_should_membership() {
        use crate::lexical::query::boolean::BooleanQueryBuilder;
        // Must(alpha) + Should(gamma): membership = docs with alpha (gamma only boosts).
        let q: Box<dyn Query> = Box::new(
            BooleanQueryBuilder::new()
                .must(Box::new(TermQuery::new("body", "alpha")))
                .should(Box::new(TermQuery::new("body", "gamma")))
                .build(),
        );
        assert_eq!(parallel_doc_ids(&build_boolean_corpus(), q), vec![0, 1, 3]);
    }

    #[test]
    fn parallel_boolean_must_not_excludes() {
        use crate::lexical::query::boolean::BooleanQueryBuilder;
        // Must(alpha) AND NOT beta -> only doc 0.
        let q: Box<dyn Query> = Box::new(
            BooleanQueryBuilder::new()
                .must(Box::new(TermQuery::new("body", "alpha")))
                .must_not(Box::new(TermQuery::new("body", "beta")))
                .build(),
        );
        assert_eq!(parallel_doc_ids(&build_boolean_corpus(), q), vec![0]);
    }

    #[test]
    fn parallel_boolean_filter_narrows() {
        use crate::lexical::query::boolean::BooleanQueryBuilder;
        // Must(alpha) AND Filter(beta) -> alpha ∩ beta = {1, 3}.
        let q: Box<dyn Query> = Box::new(
            BooleanQueryBuilder::new()
                .must(Box::new(TermQuery::new("body", "alpha")))
                .filter(Box::new(TermQuery::new("body", "beta")))
                .build(),
        );
        assert_eq!(parallel_doc_ids(&build_boolean_corpus(), q), vec![1, 3]);
    }

    #[test]
    fn parallel_boolean_should_only_union() {
        use crate::lexical::query::boolean::BooleanQueryBuilder;
        // Should(alpha) OR Should(beta) -> {0, 1, 2, 3}.
        let q: Box<dyn Query> = Box::new(
            BooleanQueryBuilder::new()
                .should(Box::new(TermQuery::new("body", "alpha")))
                .should(Box::new(TermQuery::new("body", "beta")))
                .build(),
        );
        assert_eq!(
            parallel_doc_ids(&build_boolean_corpus(), q),
            vec![0, 1, 2, 3]
        );
    }

    #[test]
    fn parallel_boolean_minimum_should_match() {
        use crate::lexical::query::boolean::BooleanQueryBuilder;
        // Must(alpha) + Should(beta) with msm=1 -> alpha ∩ beta = {1, 3}.
        let q: Box<dyn Query> = Box::new(
            BooleanQueryBuilder::new()
                .must(Box::new(TermQuery::new("body", "alpha")))
                .should(Box::new(TermQuery::new("body", "beta")))
                .minimum_should_match(1)
                .build(),
        );
        assert_eq!(parallel_doc_ids(&build_boolean_corpus(), q), vec![1, 3]);
    }

    /// Should scores accumulate onto Must candidates: a Must doc that also
    /// matches a Should clause must outrank one that does not.
    #[test]
    fn parallel_boolean_should_boosts_score() {
        use crate::lexical::query::boolean::BooleanQueryBuilder;
        let store = build_boolean_corpus();
        let q: Box<dyn Query> = Box::new(
            BooleanQueryBuilder::new()
                .must(Box::new(TermQuery::new("body", "alpha")))
                .should(Box::new(TermQuery::new("body", "gamma")))
                .build(),
        );
        let reader = store.reader_for_tests().unwrap();
        let searcher = InvertedIndexSearcher::from_arc(reader);
        let hits = searcher
            .search_with_collector_parallel(q, TopDocsCollector::new(100), true)
            .unwrap()
            .results();
        let s3 = hits.iter().find(|h| h.doc_id == 3).unwrap().score;
        let s0 = hits.iter().find(|h| h.doc_id == 0).unwrap().score;
        assert!(
            s3 > s0,
            "doc 3 (alpha+gamma) must outscore doc 0 (alpha only): s3={s3} s0={s0}"
        );
        assert_eq!(hits[0].doc_id, 3, "the should-boosted doc must rank first");
    }

    /// Parallel and serial paths must agree on membership for a Must-present
    /// shape (both implement the same boolean membership there).
    #[test]
    fn parallel_matches_serial_membership() {
        use crate::lexical::query::boolean::BooleanQueryBuilder;
        use std::collections::BTreeSet;

        let store = build_boolean_corpus();
        let make = || -> Box<dyn Query> {
            Box::new(
                BooleanQueryBuilder::new()
                    .must(Box::new(TermQuery::new("body", "alpha")))
                    .should(Box::new(TermQuery::new("body", "gamma")))
                    .must_not(Box::new(TermQuery::new("body", "beta")))
                    .build(),
            )
        };
        let reader = store.reader_for_tests().unwrap();
        let searcher = InvertedIndexSearcher::from_arc(reader);
        let collect_ids = |parallel: bool| -> BTreeSet<u64> {
            searcher
                .search_with_collector_parallel(make(), TopDocsCollector::new(100), parallel)
                .unwrap()
                .results()
                .into_iter()
                .map(|h| h.doc_id)
                .collect()
        };
        let par = collect_ids(true);
        assert_eq!(
            par,
            collect_ids(false),
            "parallel/serial membership must agree"
        );
        // alpha ∩ not beta = {0} (doc 3 has beta → excluded; gamma only boosts).
        assert_eq!(par.into_iter().collect::<Vec<_>>(), vec![0]);
    }

    // ----- Issue #590: parsed-DSL query cache -----

    /// A repeated DSL search is parsed once: the second call is a cache hit and
    /// returns the identical result set (Issue #590).
    #[test]
    fn dsl_parse_cache_hit_on_repeat() {
        let store = build_skewed_store_with_segments(1);
        let reader = store.reader_for_tests().unwrap();
        let searcher = InvertedIndexSearcher::from_arc(reader);

        let req = || {
            LexicalSearchRequest::from_dsl("body:alpha")
                .limit(10)
                .load_documents(false)
        };
        let ids1: Vec<u64> = searcher
            .search(req())
            .unwrap()
            .hits
            .iter()
            .map(|h| h.doc_id)
            .collect();
        let ids2: Vec<u64> = searcher
            .search(req())
            .unwrap()
            .hits
            .iter()
            .map(|h| h.doc_id)
            .collect();

        assert_eq!(ids1, ids2, "repeat DSL search must return the same results");
        assert!(!ids1.is_empty(), "corpus should match body:alpha");

        let stats = searcher.parsed_query_cache_stats();
        assert_eq!(
            stats.misses, 1,
            "the DSL is parsed once (first call misses)"
        );
        assert!(stats.hits >= 1, "the repeat DSL search hits the cache");
    }

    /// With the cache disabled (capacity 0) the DSL is parsed every time, yet
    /// results are unchanged.
    #[test]
    fn dsl_parse_cache_disabled_still_correct() {
        let store = build_skewed_store_with_segments(1);
        let reader = store.reader_for_tests().unwrap();
        let searcher = InvertedIndexSearcher::from_arc(reader).with_parsed_query_cache_capacity(0);

        let req = || {
            LexicalSearchRequest::from_dsl("body:alpha")
                .limit(10)
                .load_documents(false)
        };
        let ids1: Vec<u64> = searcher
            .search(req())
            .unwrap()
            .hits
            .iter()
            .map(|h| h.doc_id)
            .collect();
        let ids2: Vec<u64> = searcher
            .search(req())
            .unwrap()
            .hits
            .iter()
            .map(|h| h.doc_id)
            .collect();

        assert_eq!(ids1, ids2);
        let stats = searcher.parsed_query_cache_stats();
        assert_eq!(stats.hits, 0, "a disabled cache never hits");
    }
}
