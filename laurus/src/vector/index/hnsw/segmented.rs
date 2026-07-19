//! Segment-per-commit HNSW vector index (Issue #634, PR-3 / #881).
//!
//! [`SegmentedHnswIndex`] implements [`VectorIndex`] over a set of immutable
//! per-segment `.hnsw` files registered in an atomic, checksummed
//! `segments.json` manifest (#879), instead of the monolithic single-file
//! layout of [`super::HnswIndex`]: each commit seals only the newly added
//! vectors as a new segment, turning the per-commit cost from O(index) into
//! O(new docs) — no full rewrite, no re-quantization of the existing corpus.
//!
//! Search fans out over the sealed segments newest-generation-first with
//! containment-based newest-wins deduplication (#880): a hit from an older
//! segment is dropped when any newer segment contains the same
//! `(doc_id, field)` — same-id upserts replayed from the WAL leave stale
//! copies behind until a merge physically collapses them. Uncommitted adds
//! are invisible until commit, matching the monolithic semantics.
//!
//! Deletions are logical: a shared index-level [`DeletionBitmap`] is marked
//! by [`VectorIndex::soft_delete_document`], persisted to `{name}.delmap` by
//! [`VectorIndex::persist_deletions`], filtered by every segment reader, and
//! physically reclaimed by [`VectorIndex::optimize`] (a force-merge). The
//! bitmap is doc-scoped, which is consistent here because a document's
//! fields always land in the same segment.
//!
//! Since #882 this is the DEFAULT HNSW layout ([`HnswIndexConfig::segmented`]
//! defaults to `true`): a legacy monolithic index is migrated zero-copy on
//! first open (its `.hnsw` becomes segment 0 of the manifest, no data
//! movement), and the manifest's `last_wal_seq` is published with the
//! durability ordering the engine's recovery relies on (data before
//! checkpoint, checkpoint before WAL truncate). Set the flag to `false` to
//! keep the monolithic layout.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use parking_lot::RwLock;

use crate::embedding::embedder::Embedder;
use crate::error::{LaurusError, Result};
use crate::maintenance::deletion::DeletionBitmap;
use crate::storage::Storage;
use crate::storage::structured::{StructReader, StructWriter};
use crate::vector::core::vector::Vector;
use crate::vector::index::config::HnswIndexConfig;
use crate::vector::index::hnsw::reader::HnswIndexReader;
use crate::vector::index::hnsw::searcher::HnswSearcher;
use crate::vector::index::hnsw::segment::manager::{
    ManagedSegmentInfo, MergeCandidate, SegmentManager, SegmentManagerConfig,
};
use crate::vector::index::hnsw::segment::merge_engine::{MergeConfig, MergeEngine};
use crate::vector::index::hnsw::segment::reader_cache::SegmentedReaderCache;
use crate::vector::index::hnsw::writer::HnswIndexWriter;
use crate::vector::index::{VectorIndex, VectorIndexStats};
use crate::vector::reader::{
    ValidationReport, VectorIndexMetadata, VectorIndexReader, VectorIterator, VectorStats,
};
use crate::vector::search::searcher::{
    VectorIndexQuery, VectorIndexQueryResults, VectorIndexSearcher,
};
use crate::vector::store::embedding_writer::EmbeddingVectorIndexWriter;
use crate::vector::writer::{VectorIndexWriter, VectorIndexWriterConfig};

/// State shared between the index handle, its writers, and its searchers.
#[derive(Debug)]
struct SegmentedShared {
    /// Index name — the prefix for the deletion bitmap file. Segment files
    /// are named by the manager (`segment_NNNNNN.hnsw`).
    name: String,

    /// Storage backend.
    storage: Arc<dyn Storage>,

    /// Segment registry (atomic manifest, #879).
    manager: Arc<SegmentManager>,

    /// Per-segment reader cache (#660); invalidated on merge.
    reader_cache: Arc<SegmentedReaderCache>,

    /// Index-level logical-deletion bitmap (doc-scoped). `None` means "not
    /// yet loaded / no deletions"; lazily loaded from `{name}.delmap`.
    deletion: RwLock<Option<Arc<DeletionBitmap>>>,

    /// Bumped when the bitmap is CREATED (first soft delete). Readers cached
    /// before that moment are not attached to it; the reader-building loop
    /// re-checks this epoch after populating from the cache and rebuilds if
    /// a create raced it (#881 review) — clearing the cache alone is not
    /// enough, because a concurrent build could repopulate it with
    /// bitmap-less readers after the clear.
    bitmap_epoch: std::sync::atomic::AtomicU64,

    /// Highest WAL sequence number applied to this index but NOT yet
    /// published to the manifest (#882). Published (and persisted) by
    /// [`VectorIndex::persist_deletions`] at the end of the store's commit
    /// ladder, once every covered mutation is durable.
    pending_wal_seq: std::sync::atomic::AtomicU64,
}

impl SegmentedShared {
    fn delmap_file_name(&self) -> String {
        format!("{}.delmap", self.name)
    }

    /// Load (or lazily create) the shared deletion bitmap.
    ///
    /// Mirrors `HnswIndex::load_or_get_bitmap`. When a bitmap is created for
    /// the first time, the reader cache is cleared so subsequently loaded
    /// segment readers attach it (cached readers hold the bitmap by `Arc`,
    /// so later marks on an *already attached* bitmap are always visible).
    fn load_or_get_bitmap(&self, create_if_missing: bool) -> Result<Option<Arc<DeletionBitmap>>> {
        if let Some(bitmap) = self.deletion.read().as_ref() {
            return Ok(Some(bitmap.clone()));
        }

        let mut guard = self.deletion.write();
        if let Some(bitmap) = guard.as_ref() {
            return Ok(Some(bitmap.clone()));
        }

        let file = self.delmap_file_name();
        if self.storage.file_exists(&file) {
            let input = self.storage.open_input(&file)?;
            let mut reader = StructReader::new(input)?;
            let bitmap = Arc::new(DeletionBitmap::read_from_storage(&mut reader)?);
            *guard = Some(bitmap.clone());
            return Ok(Some(bitmap));
        }

        if create_if_missing {
            let bitmap = Arc::new(DeletionBitmap::new(self.name.clone(), 0, u64::MAX - 1));
            *guard = Some(bitmap.clone());
            // Readers cached before the bitmap existed are not attached to
            // it; drop them and bump the epoch so a concurrent
            // `sealed_readers_newest_first` that already snapshotted a
            // `None` bitmap detects the race and rebuilds.
            self.reader_cache.clear();
            self.bitmap_epoch.fetch_add(1, Ordering::Release);
            return Ok(Some(bitmap));
        }

        Ok(None)
    }

    /// Sealed segment readers, newest generation first, with the deletion
    /// bitmap attached (#880 containment/dedup ordering).
    fn sealed_readers_newest_first(
        &self,
        config: &HnswIndexConfig,
    ) -> Result<Vec<Arc<HnswIndexReader>>> {
        let mut segments = self.manager.list_segments();
        segments.sort_by_key(|s| std::cmp::Reverse(s.generation));

        loop {
            let epoch = self.bitmap_epoch.load(Ordering::Acquire);
            let bitmap = self.load_or_get_bitmap(false)?;
            let mut readers = Vec::with_capacity(segments.len());
            for info in &segments {
                let storage = self.storage.clone();
                let metric = config.distance_metric;
                let bitmap_for_loader = bitmap.clone();
                let segment_id = info.segment_id.clone();
                let reader = self.reader_cache.get_or_load(&segment_id, || {
                    let mut r = HnswIndexReader::load(storage, &segment_id, metric)?;
                    if let Some(b) = bitmap_for_loader {
                        r.set_deletion_bitmap(b);
                    }
                    Ok(r)
                })?;
                readers.push(reader);
            }
            // A bitmap create that raced this build may have been missed by
            // readers loaded from our (pre-create) snapshot; the epoch bump
            // detects it. Creation happens at most once per index lifetime,
            // so the loop reruns at most once.
            if self.bitmap_epoch.load(Ordering::Acquire) == epoch {
                return Ok(readers);
            }
            self.reader_cache.clear();
        }
    }
}

/// Segment-per-commit HNSW index (see the module docs).
#[derive(Debug)]
pub struct SegmentedHnswIndex {
    shared: Arc<SegmentedShared>,
    config: HnswIndexConfig,
    closed: AtomicBool,
}

impl SegmentedHnswIndex {
    /// Open an existing segmented index or create a new one.
    ///
    /// # Arguments
    ///
    /// * `storage` - Storage backend.
    /// * `name` - Index name (bitmap file prefix; segment files are named by
    ///   the segment manager).
    /// * `config` - HNSW configuration (with [`HnswIndexConfig::segmented`]
    ///   set).
    ///
    /// A legacy monolithic index is migrated **zero-copy** (#882): the
    /// existing `{name}.hnsw` is registered verbatim as the first segment —
    /// its segment id is the index name, which every reader treats as a
    /// plain path prefix — with a single atomic manifest write and no data
    /// movement. A crash before the manifest save leaves the legacy layout
    /// intact, so the migration simply re-runs on the next open. The
    /// pre-existing `{name}.delmap` keeps its meaning: it is the index-level
    /// deletion bitmap in both layouts (and `delete_segment_files` never
    /// touches `.delmap` files for exactly this reason).
    ///
    /// # Errors
    ///
    /// Returns an error when the manifest fails to load (corruption fails
    /// loudly, #879), or when reading the legacy file's header during
    /// migration fails.
    pub fn open_or_create(
        storage: Arc<dyn Storage>,
        name: &str,
        config: HnswIndexConfig,
    ) -> Result<Self> {
        // The generated-segment namespace is reserved: an index named like
        // a generated id would collide with sealed segments (same file
        // names, orphan-sweep patterns, and merge GC) — reject it loudly
        // (#882 review).
        if let Some(ordinal) = name.strip_prefix("segment_")
            && !ordinal.is_empty()
            && ordinal.bytes().all(|b| b.is_ascii_digit())
        {
            return Err(LaurusError::invalid_config(format!(
                "index name '{name}' collides with the reserved segment-id \
                 namespace (segment_<digits>)"
            )));
        }

        let legacy_file = format!("{name}.hnsw");
        let migrate = storage.file_exists(&legacy_file) && !storage.file_exists("segments.json");

        // The manager's orphan sweep only touches `segment_NNNNNN.*` names,
        // so an unmigrated `{name}.hnsw` is never swept.
        let manager = Arc::new(SegmentManager::new(
            SegmentManagerConfig::default(),
            storage.clone(),
        )?);

        if migrate {
            // Read the committed vector count from the leading u64 of the
            // legacy file (same header every reader loads).
            let vector_count = {
                use std::io::Read;
                let mut input = storage.open_input(&legacy_file)?;
                let mut buf = [0u8; 8];
                input.read_exact(&mut buf)?;
                u64::from_le_bytes(buf)
            };
            // `add_segment` stamps generation 1, measures the on-storage
            // size, and saves the manifest atomically (#879) — the moment it
            // returns, the index is segmented; before that, it is still a
            // valid legacy index. A legacy delmap (same file in both
            // layouts) means the segment carries deletions.
            let mut info = ManagedSegmentInfo::new(name.to_string(), vector_count, 0, 0);
            info.has_deletions = storage.file_exists(&format!("{name}.delmap"));
            manager.add_segment(info)?;

            // Drop the monolithic index's now-stale `metadata.json` so the
            // factory's open/create routing can never again mistake this
            // directory for a monolithic index (#882 review: a stale
            // metadata.json flipped a migrated index back to the monolithic
            // view, silently hiding every post-migration segment).
            let _ = storage.delete_file("metadata.json");
        }

        Ok(Self {
            shared: Arc::new(SegmentedShared {
                name: name.to_string(),
                storage,
                manager,
                reader_cache: Arc::new(SegmentedReaderCache::new()),
                deletion: RwLock::new(None),
                bitmap_epoch: std::sync::atomic::AtomicU64::new(0),
                pending_wal_seq: std::sync::atomic::AtomicU64::new(0),
            }),
            config,
            closed: AtomicBool::new(false),
        })
    }

    fn check_closed(&self) -> Result<()> {
        if self.closed.load(Ordering::SeqCst) {
            return Err(LaurusError::InvalidOperation("Index is closed".to_string()));
        }
        Ok(())
    }

    /// Default `ef_search` for the multi-segment fan-out: the schema-level
    /// value when configured, else `ef_construction.max(50) * 2` — the
    /// compensation that kept multi-segment recall stable (#644 / #880).
    fn default_ef(&self) -> usize {
        self.config
            .default_ef_search
            .unwrap_or_else(|| self.config.ef_construction.max(50) * 2)
    }

    /// Merge one policy-selected window of segments (#882).
    ///
    /// Uses the generation-contiguous `SimpleMergePolicy` (#880) through
    /// the deletion-filtering, duplicate-collapsing merge engine; source
    /// readers are invalidated afterwards. A no-op when the policy finds no
    /// candidate.
    fn merge_once(&self) -> Result<()> {
        use crate::vector::index::hnsw::segment::merge_policy::SimpleMergePolicy;

        let Some(candidate) = self.shared.manager.check_merge(&SimpleMergePolicy::new()) else {
            return Ok(());
        };

        let mut engine = MergeEngine::new(
            MergeConfig::default(),
            self.shared.storage.clone(),
            self.config.clone(),
            VectorIndexWriterConfig::default(),
        );
        if let Some(bitmap) = self.shared.load_or_get_bitmap(false)? {
            engine.set_deletion_bitmap(bitmap);
        }

        let new_segment_id = self.shared.manager.generate_segment_id();
        let result = engine.merge_segments(candidate.segments.clone(), new_segment_id)?;

        let source_ids: Vec<String> = candidate
            .segments
            .iter()
            .map(|s| s.segment_id.clone())
            .collect();
        self.shared
            .manager
            .apply_merge(candidate, result.merged_segment)?;
        for id in &source_ids {
            self.shared.reader_cache.invalidate(id);
        }
        Ok(())
    }

    /// Drop the deletion state after a compaction physically reclaimed the
    /// deleted documents (mirrors `HnswIndex::clear_deletions`).
    fn clear_deletions(&self) -> Result<()> {
        *self.shared.deletion.write() = None;
        let file = self.shared.delmap_file_name();
        let _ = self.shared.storage.delete_file(&file);
        Ok(())
    }
}

impl VectorIndex for SegmentedHnswIndex {
    fn reader(&self) -> Result<Arc<dyn VectorIndexReader>> {
        self.check_closed()?;
        let readers = self.shared.sealed_readers_newest_first(&self.config)?;
        let bitmap = self.shared.load_or_get_bitmap(false)?;
        Ok(Arc::new(SegmentedReaderFacade::new(
            readers,
            bitmap,
            self.config.dimension,
            self.config.distance_metric,
        )))
    }

    fn writer(&self) -> Result<Box<dyn VectorIndexWriter>> {
        self.check_closed()?;

        // A fresh, EMPTY active-segment writer per call: constructing it does
        // not touch the existing segments — the whole point of the layout.
        let segment_id = self.shared.manager.generate_segment_id();
        let inner = HnswIndexWriter::with_storage(
            self.config.clone(),
            VectorIndexWriterConfig::default(),
            &segment_id,
            self.shared.storage.clone(),
        )?;
        let writer = SegmentedHnswWriter {
            shared: self.shared.clone(),
            segment_id,
            inner,
            sealed_len: None,
        };
        let embedder = self.embedder();
        Ok(Box::new(EmbeddingVectorIndexWriter::new(
            Box::new(writer),
            embedder,
        )))
    }

    fn storage(&self) -> &Arc<dyn Storage> {
        &self.shared.storage
    }

    fn close(&self) -> Result<()> {
        self.closed.store(true, Ordering::SeqCst);
        Ok(())
    }

    fn is_closed(&self) -> bool {
        self.closed.load(Ordering::SeqCst)
    }

    fn stats(&self) -> Result<VectorIndexStats> {
        self.check_closed()?;
        // `total` sums per-segment row counts: stale same-id upsert copies
        // awaiting a merge are counted once per segment, so this (and the
        // auto-compaction ratio derived from it) slightly over-counts until
        // the next merge. Exact distinct-key counting would cost O(total
        // ids) per call; the approximation is intentional (#881 review).
        let total = self.shared.manager.total_vectors();
        let deleted = match self.shared.load_or_get_bitmap(false)? {
            Some(bitmap) => bitmap.deleted_count.load(Ordering::Relaxed),
            None => 0,
        };
        let stats = self.shared.manager.stats();
        Ok(VectorIndexStats {
            vector_count: total.saturating_sub(deleted),
            dimension: self.config.dimension,
            total_size: stats.total_size,
            deleted_count: deleted,
            last_modified: 0,
        })
    }

    fn retain_writer_after_commit(&self) -> bool {
        // Retention existed to avoid the monolithic writer's O(index) reload
        // after commit (#864). A segmented writer starts EMPTY — there is
        // nothing to reload — so the store can drop it and construct a fresh
        // one per commit cycle; each commit then seals exactly one new
        // segment.
        false
    }

    fn optimize(&self) -> Result<()> {
        self.check_closed()?;

        let segments = self.shared.manager.list_segments();
        if segments.is_empty() {
            return Ok(());
        }
        // Force-merge everything into one segment; the merge engine filters
        // through the deletion bitmap (physical reclamation) and collapses
        // same-key duplicates newest-generation-first (#880).
        let mut engine = MergeEngine::new(
            MergeConfig::default(),
            self.shared.storage.clone(),
            self.config.clone(),
            VectorIndexWriterConfig::default(),
        );
        if let Some(bitmap) = self.shared.load_or_get_bitmap(false)? {
            engine.set_deletion_bitmap(bitmap);
        }

        let new_segment_id = self.shared.manager.generate_segment_id();
        let result = engine.merge_segments(segments.clone(), new_segment_id)?;

        let candidate = MergeCandidate {
            total_vectors: segments.iter().map(|s| s.vector_count).sum(),
            total_size: segments.iter().map(|s| s.size_bytes).sum(),
            segments,
        };
        let source_ids: Vec<String> = candidate
            .segments
            .iter()
            .map(|s| s.segment_id.clone())
            .collect();
        self.shared
            .manager
            .apply_merge(candidate, result.merged_segment)?;
        for id in &source_ids {
            self.shared.reader_cache.invalidate(id);
        }

        // Every logically deleted doc was physically dropped by the merge.
        self.clear_deletions()?;
        Ok(())
    }

    // `refresh` keeps its trait default (no-op), like the monolithic index:
    // sealed segment files are immutable, segment-set changes are picked up
    // from the shared manager on every `searcher()`/`reader()` call, and
    // merges invalidate their source readers explicitly. Clearing the reader
    // cache here would force an O(all segments) reload on every commit —
    // defeating the O(delta) design this index exists for (#660/#881).

    fn searcher(&self) -> Result<Box<dyn VectorIndexSearcher>> {
        self.check_closed()?;
        let readers = self.shared.sealed_readers_newest_first(&self.config)?;
        let bitmap = self.shared.load_or_get_bitmap(false)?;
        Ok(Box::new(SegmentedHnswSearcher {
            readers,
            bitmap,
            default_ef: self.default_ef(),
        }))
    }

    fn embedder(&self) -> Arc<dyn Embedder> {
        Arc::clone(&self.config.embedder)
    }

    fn last_wal_seq(&self) -> u64 {
        // The PUBLISHED checkpoint: the value loaded from (or last saved to)
        // the manifest — never the pending one, which may not be durable
        // yet. Recovery skips WAL records at or below this seq, so it must
        // only ever reflect state that is already on storage (#882).
        self.shared.manager.last_wal_seq()
    }

    fn set_last_wal_seq(&self, seq: u64) -> Result<()> {
        // Recorded as PENDING only. It is published into the manifest at the
        // end of `persist_deletions` — the last vector step of the store's
        // commit ladder — at which point the sealed segment files AND the
        // deletion bitmap for every record up to `seq` are durable, and the
        // engine has not yet truncated the WAL (#882 ordering; the #875
        // lesson: a checkpoint must never outrun the durability of the state
        // it covers).
        self.shared
            .pending_wal_seq
            .fetch_max(seq, Ordering::Release);
        Ok(())
    }

    fn supports_soft_delete(&self) -> bool {
        true
    }

    fn soft_delete_document(&self, doc_id: u64) -> Result<()> {
        self.check_closed()?;
        let bitmap = self
            .shared
            .load_or_get_bitmap(true)?
            .ok_or_else(|| LaurusError::internal("deletion bitmap unexpectedly missing"))?;
        let newly_deleted = bitmap.delete_document(doc_id)?;
        if newly_deleted && !self.shared.manager.list_segments().is_empty() {
            // Conservative-all flagging: segment infos carry no doc-id
            // range, and the flag only feeds merge-policy prioritization
            // (#880).
            self.shared.manager.mark_all_has_deletions()?;
        }
        Ok(())
    }

    fn persist_deletions(&self) -> Result<()> {
        let guard = self.shared.deletion.read();
        if let Some(bitmap) = guard.as_ref() {
            let file = self.shared.delmap_file_name();
            if bitmap.deleted_count.load(Ordering::Relaxed) > 0 {
                // Temp-then-rename for crash safety (#784); the payload is
                // CRC-32 protected by `StructWriter` (#684).
                let tmp = format!("{file}.tmp");
                let output = self.shared.storage.create_output(&tmp)?;
                let mut writer = StructWriter::new(output);
                bitmap.write_to_storage(&mut writer)?;
                writer.close()?;
                self.shared.storage.rename_file(&tmp, &file)?;
            } else if self.shared.storage.file_exists(&file) {
                // Undelete-to-zero (#881): the upsert dance's re-add can
                // clear every mark, and a previously persisted delmap would
                // then be STALE — on reopen it would mask the committed
                // upsert in every segment. "Nothing deleted" must persist as
                // "no delmap file", not "keep whatever file exists".
                self.shared.storage.delete_file(&file)?;
            }
        }
        drop(guard);

        // Publish the WAL checkpoint (#882). This is the last vector step of
        // the store's commit ladder: the sealed segment files (fsynced +
        // manifest-registered by `writer.commit()`) and the deletion bitmap
        // (written above) are durable for every record up to the pending
        // seq, and `Engine::commit` truncates the WAL only after all stores
        // finish — so a crash on either side of this save is safe: before
        // it, recovery replays idempotently from the old checkpoint; after
        // it, everything skipped is already on storage.
        let pending = self.shared.pending_wal_seq.load(Ordering::Acquire);
        if pending > self.shared.manager.last_wal_seq() {
            self.shared.manager.set_last_wal_seq(pending);
            self.shared.manager.save_state()?;
        }
        Ok(())
    }

    fn maybe_auto_compact(&self) -> Result<bool> {
        // Segment-count bound (#882): pure-append workloads never take the
        // deletion-ratio branch below, so without this the segment count —
        // and with it every search fan-out and the manifest size — would
        // grow unboundedly, one segment per commit. Merge one
        // generation-contiguous window (#880 policy) when the manager's
        // threshold is exceeded; the tiered policy lands with #883. Runs
        // regardless of `auto_compaction`, which gates deletion
        // *reclamation*, not the structural segment bound.
        if self.shared.manager.needs_merge() {
            self.merge_once()?;
            return Ok(true);
        }

        if !self.config.auto_compaction {
            return Ok(false);
        }
        let deleted = match self.shared.load_or_get_bitmap(false)? {
            Some(bitmap) => bitmap.deleted_count.load(Ordering::Relaxed),
            None => 0,
        };
        if deleted == 0 {
            return Ok(false);
        }
        let total = self.shared.manager.total_vectors();
        if total == 0 {
            return Ok(false);
        }
        let ratio = deleted as f64 / total as f64;
        if ratio >= self.config.compaction_threshold {
            self.optimize()?;
            return Ok(true);
        }
        Ok(false)
    }
}

/// Active-segment writer for [`SegmentedHnswIndex`].
///
/// Buffers new vectors in an ordinary [`HnswIndexWriter`] whose path is a
/// fresh `segment_NNNNNN`; [`VectorIndexWriter::commit`] seals it as a new
/// immutable segment file and registers it in the manifest exactly once.
/// A sealed writer is DONE: committing again with new changes is rejected
/// (#881 review) — re-writing a registered segment would carry its original
/// generation while newer segments may have sealed in between, inverting
/// the newest-wins ordering. The store never does this
/// (`retain_writer_after_commit` is false); direct API users get a loud
/// error instead of silent metadata corruption.
#[derive(Debug)]
struct SegmentedHnswWriter {
    shared: Arc<SegmentedShared>,
    segment_id: String,
    inner: HnswIndexWriter,
    /// Buffer length at the moment this writer's segment was sealed and
    /// registered; `None` until then. Used to make the post-commit
    /// `close()` a no-op (the inner buffer is retained after commit, so a
    /// plain "buffer non-empty" check would re-enter `commit`).
    sealed_len: Option<usize>,
}

impl VectorIndexWriter for SegmentedHnswWriter {
    fn next_vector_id(&self) -> u64 {
        self.inner.next_vector_id()
    }

    fn build(&mut self, vectors: Vec<(u64, String, Vector)>) -> Result<()> {
        self.add_vectors(vectors)
    }

    fn add_vectors(&mut self, vectors: Vec<(u64, String, Vector)>) -> Result<()> {
        // Same-id upsert completion (#880): the delete-first step marked the
        // ids so sealed copies stop matching; clear the marks so the NEW
        // copies are not shadowed by their own delete once this segment
        // seals. Revived stale copies are masked by newest-wins containment
        // at search and collapsed at merge.
        if let Some(bitmap) = self.shared.load_or_get_bitmap(false)? {
            for (doc_id, _, _) in &vectors {
                let _ = bitmap.undelete_document(*doc_id)?;
            }
        }
        self.inner.add_vectors(vectors)
    }

    fn finalize(&mut self) -> Result<()> {
        self.inner.finalize()
    }

    fn progress(&self) -> f32 {
        self.inner.progress()
    }

    fn estimated_memory_usage(&self) -> usize {
        self.inner.estimated_memory_usage()
    }

    fn vectors(&self) -> &[(u64, String, Vector)] {
        self.inner.vectors()
    }

    fn write(&self) -> Result<()> {
        if self.inner.vectors().is_empty() {
            // Nothing buffered — sealing would create an empty segment file
            // per commit. Deletion persistence is handled by
            // `persist_deletions` on the index.
            return Ok(());
        }
        // Writes `{segment_id}.hnsw`; manifest registration happens in
        // `commit()` (`&mut self`), which is the store's entry point.
        self.inner.write()
    }

    fn has_storage(&self) -> bool {
        self.inner.has_storage()
    }

    fn delete_document(&mut self, doc_id: u64) -> Result<()> {
        // Upsert delete-first (#880): remove any buffered copy AND mark the
        // shared bitmap so copies in sealed segments stop matching. A pure
        // delete stays marked; a following `add_vectors` with the same id
        // clears the mark.
        if let Some(bitmap) = self.shared.load_or_get_bitmap(true)? {
            let newly_deleted = bitmap.delete_document(doc_id)?;
            if newly_deleted && !self.shared.manager.list_segments().is_empty() {
                self.shared.manager.mark_all_has_deletions()?;
            }
        }
        let _ = self.inner.delete_document(doc_id);
        Ok(())
    }

    fn has_pending_changes(&self) -> bool {
        match self.sealed_len {
            // Sealed: pending only if the buffer changed since the seal.
            Some(sealed) => self.inner.vectors().len() != sealed,
            None => !self.inner.vectors().is_empty() || self.inner.has_pending_changes(),
        }
    }

    fn commit(&mut self) -> Result<()> {
        if self.inner.vectors().is_empty() {
            return Ok(());
        }
        if let Some(sealed) = self.sealed_len {
            if self.inner.vectors().len() == sealed {
                // Post-commit close(): the segment is already sealed and the
                // buffer is unchanged — nothing to do.
                return Ok(());
            }
            // See the struct docs: re-sealing a registered segment would
            // invert the newest-wins generation ordering.
            return Err(LaurusError::InvalidOperation(format!(
                "segment '{}' is already sealed; obtain a fresh writer for further changes",
                self.segment_id
            )));
        }
        self.inner.finalize()?;
        self.inner.write()?;
        let vector_count = self.inner.vectors().len() as u64;
        // generation 0 → the manager stamps max+1 (#879); size_bytes 0 →
        // measured from storage by `add_segment`.
        let info = ManagedSegmentInfo::new(self.segment_id.clone(), vector_count, 0, 0);
        self.shared.manager.add_segment(info)?;
        self.sealed_len = Some(self.inner.vectors().len());
        self.shared.reader_cache.invalidate(&self.segment_id);
        Ok(())
    }

    fn rollback(&mut self) -> Result<()> {
        // Rolled-back mutations are discarded, so the pending WAL
        // checkpoint must not keep covering their sequence numbers (#882):
        // roll it back to the published value; their WAL records then stay
        // replayable.
        self.shared
            .pending_wal_seq
            .store(self.shared.manager.last_wal_seq(), Ordering::Release);
        self.inner.rollback()
    }

    fn pending_docs(&self) -> u64 {
        self.inner.pending_docs()
    }

    fn close(&mut self) -> Result<()> {
        if self.has_pending_changes() {
            self.commit()?;
        }
        self.inner.close()
    }

    fn is_closed(&self) -> bool {
        self.inner.is_closed()
    }

    fn build_reader(&self) -> Result<Arc<dyn VectorIndexReader>> {
        self.inner.build_reader()
    }
}

impl Drop for SegmentedHnswWriter {
    fn drop(&mut self) {
        // Backstop (#882 review): dropping a writer whose buffered
        // mutations were never sealed discards the only in-process copy,
        // while the pending WAL checkpoint may already cover their
        // sequence numbers — publishing it later would hide the loss from
        // recovery. Roll the pending value back to the published
        // checkpoint so those records stay replayable. The store never
        // drops a dirty writer (its commit/optimize/add_field paths retain
        // it on failure), so this fires only on direct API use.
        if self.has_pending_changes() {
            self.shared
                .pending_wal_seq
                .store(self.shared.manager.last_wal_seq(), Ordering::Release);
        }
    }
}

/// Multi-segment fan-out searcher (#880 semantics).
#[derive(Debug)]
struct SegmentedHnswSearcher {
    /// Sealed readers, newest generation first, deletion bitmap attached.
    readers: Vec<Arc<HnswIndexReader>>,
    /// The shared deletion bitmap (also attached to the readers), used by
    /// [`Self::count`] to exclude soft-deleted docs.
    bitmap: Option<Arc<DeletionBitmap>>,
    /// Default `ef_search` (multi-segment compensation, #644).
    default_ef: usize,
}

impl SegmentedHnswSearcher {
    /// Whether any reader NEWER than `idx` contains `(doc_id, field)` — the
    /// containment mask that makes the newest copy win (#880).
    fn shadowed(&self, idx: usize, doc_id: u64, field: &str) -> bool {
        self.readers[..idx]
            .iter()
            .any(|r| r.vectors().contains(doc_id, field))
    }
}

impl VectorIndexSearcher for SegmentedHnswSearcher {
    fn search(&self, request: &VectorIndexQuery) -> Result<VectorIndexQueryResults> {
        let started = crate::util::time::Timer::now();
        let limit = request.params.top_k;
        let mut merged = VectorIndexQueryResults::new();
        if limit == 0 || self.readers.is_empty() {
            return Ok(merged);
        }

        // Over-fetch per segment: containment masking drops shadowed hits
        // AFTER the per-segment top-k, so stale copies would otherwise
        // consume result slots (#880). The 2x factor bounds the loss while
        // stale copies stay under half of a segment's local top list;
        // adaptive refill is planned with the merge-policy work (#883).
        let mut sub = request.clone();
        sub.params.top_k = limit.saturating_mul(2);

        for (idx, reader) in self.readers.iter().enumerate() {
            let searcher =
                HnswSearcher::with_default_ef_search(reader.clone(), Some(self.default_ef))?;
            let results = searcher.search(&sub)?;
            merged.candidates_examined += results.candidates_examined;
            for hit in results.results {
                if self.shadowed(idx, hit.doc_id, &hit.field_name) {
                    continue;
                }
                merged.results.push(hit);
            }
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

/// Read facade over the sealed segments (newest-wins, deletion-filtered).
///
/// Materializes the distinct live `(doc_id, field) -> segment` mapping once
/// at construction; per-vector data is fetched from the owning segment
/// reader on demand.
#[derive(Debug)]
struct SegmentedReaderFacade {
    /// Sealed readers, newest generation first.
    readers: Vec<Arc<HnswIndexReader>>,
    /// Distinct live keys with the owning (newest) reader's index, in
    /// first-seen (newest-segment) order.
    entries: Vec<(u64, String, usize)>,
    dimension: usize,
    metric: crate::vector::core::distance::DistanceMetric,
}

impl SegmentedReaderFacade {
    fn new(
        readers: Vec<Arc<HnswIndexReader>>,
        bitmap: Option<Arc<DeletionBitmap>>,
        dimension: usize,
        metric: crate::vector::core::distance::DistanceMetric,
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

    fn owner_of(&self, doc_id: u64, field_name: &str) -> Option<&Arc<HnswIndexReader>> {
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

    fn distance_metric(&self) -> crate::vector::core::distance::DistanceMetric {
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
            index_type: "hnsw-segmented".to_string(),
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
