//! Simple BKD Tree implementation for numeric range queries.
//!
//! This is a simplified version of Apache Lucene's BKD Tree data structure,
//! optimized for 1-dimensional numeric range filtering.

use super::aabb::AABB;
use super::visitor::{CellRelation, IntersectVisitor, RangeQueryVisitor};
use crate::error::Result;
use crate::storage::structured::{StructReader, StructWriter};
use crate::storage::{Storage, StorageInput, StorageOutput};
use std::cmp::Ordering;
use std::io::SeekFrom;
use std::sync::Arc;

/// Trait for BKD Tree implementations (in-memory or disk-based).
///
/// Implementations expose two query primitives:
///
/// - [`BKDTree::intersect`] is the low-level Lucene-style traversal: the
///   reader walks the tree once, calling the visitor's `compare` method on
///   each subtree's AABB (Inside / Outside / Crosses) and either pruning,
///   collecting, or descending accordingly. This is the building block for
///   sphere queries, k-NN, and any custom shape that fits the visitor API.
/// - [`BKDTree::range_search`] is the legacy axis-aligned range API. It is
///   provided as a default method that builds a [`RangeQueryVisitor`] and
///   delegates to `intersect`, so concrete `BKDTree` implementations only
///   need to supply `intersect`.
pub trait BKDTree: Send + Sync + std::fmt::Debug {
    /// Walk the tree, dispatching subtree pruning decisions and per-point
    /// candidates to `visitor`.
    ///
    /// Implementations are expected to honor the visitor's `compare` result
    /// faithfully:
    /// - `CellRelation::Outside` cells are skipped.
    /// - `CellRelation::Inside` cells contribute every doc id beneath them
    ///   via `visit_inside` (the visitor does not need the point bytes).
    /// - `CellRelation::Crosses` leaves expose every (doc_id, point) pair
    ///   via `visit` so the visitor can perform the final per-point check.
    fn intersect(&self, visitor: &mut dyn IntersectVisitor) -> Result<()>;

    /// Axis-aligned range search returning the matching doc ids in sorted
    /// and deduplicated order.
    ///
    /// `mins[d]` / `maxs[d]` may be `None` to leave a dimension unbounded.
    /// `include_min` / `include_max` control whether the boundary itself
    /// matches.
    ///
    /// The default implementation builds a [`RangeQueryVisitor`] and
    /// delegates to [`BKDTree::intersect`].
    fn range_search(
        &self,
        mins: &[Option<f64>],
        maxs: &[Option<f64>],
        include_min: bool,
        include_max: bool,
    ) -> Result<Vec<u64>> {
        let mut visitor = RangeQueryVisitor::new(mins, maxs, include_min, include_max);
        self.intersect(&mut visitor)?;
        let mut hits = visitor.into_hits();
        hits.sort_unstable();
        hits.dedup();
        Ok(hits)
    }
}

/// Magic number for BKD Tree files: ASCII "BKDT" in little-endian.
pub const BKD_MAGIC: u32 = 0x54444B42;

/// Current on-disk format version.
///
/// Version 2 (this revision): every internal index node and every leaf block
/// carries its own axis-aligned bounding box (AABB) so the reader can prune
/// subtrees with Inside/Outside/Crosses logic. The previous version 1 layout
/// (no per-node AABB) is no longer supported — laurus is pre-release, so the
/// format is broken intentionally rather than dual-supported.
///
/// File layout (all integers little-endian):
///
/// ```text
/// Header (fixed-prefix + 2 * num_dims * 8 bytes):
///   magic               u32
///   version             u32
///   num_dims            u32
///   bytes_per_dim       u32   (always 8 today: f64)
///   total_point_count   u64
///   num_blocks          u64
///   global_min          [f64; num_dims]
///   global_max          [f64; num_dims]
///   index_start_offset  u64
///   root_node_offset    u64
///
/// Leaf Block:
///   count               u32
///   leaf_min            [f64; num_dims]
///   leaf_max            [f64; num_dims]
///   point_values        [f64; count * num_dims]   (row-major)
///   doc_ids             [u64; count]
///
/// Internal Index Node (size = 28 + 32 * num_dims bytes):
///   split_dim           u32
///   split_value         f64
///   left_min            [f64; num_dims]
///   left_max            [f64; num_dims]
///   right_min           [f64; num_dims]
///   right_max           [f64; num_dims]
///   left_offset         u64
///   right_offset        u64
/// ```
pub const BKD_VERSION: u32 = 2;

/// BKD Tree File Header
#[derive(Debug, Clone)]
pub struct BKDFileHeader {
    pub magic: u32,
    pub version: u32,
    pub num_dims: u32,
    pub bytes_per_dim: u32,
    pub total_point_count: u64,
    pub num_blocks: u64,
    pub min_values: Vec<f64>,
    pub max_values: Vec<f64>,
    pub index_start_offset: u64,
    pub root_node_offset: u64,
}

/// Writer for BKD Trees.
pub struct BKDWriter<W: StorageOutput> {
    writer: StructWriter<W>,
    block_size: usize,
    num_blocks: u64,
    num_dims: u32,
    min_values: Vec<f64>,
    max_values: Vec<f64>,
    index_nodes: Vec<IndexNode>,
}

/// Internal index node for navigation.
///
/// Each node remembers the axis-aligned bounding box (`*_min`/`*_max`) of the
/// two child subtrees in addition to the split dimension and value, enabling
/// readers to prune entire subtrees when their AABB lies fully inside or
/// outside the query region.
#[derive(Debug, Clone)]
struct IndexNode {
    split_dim: u32,
    split_value: f64,
    left_min: Vec<f64>,
    left_max: Vec<f64>,
    right_min: Vec<f64>,
    right_max: Vec<f64>,
    left_offset: u64,
    right_offset: u64,
    // Helper to back-patch offsets during writing
    left_child_idx: Option<usize>,
    right_child_idx: Option<usize>,
}

/// Information returned by `BKDWriter::build_subtree` so the caller can fold
/// per-child AABBs into the parent index node.
struct SubtreeInfo {
    /// `Some(idx)` when the subtree is rooted at the internal node at index
    /// `idx` in `index_nodes`; `None` when the subtree is a single leaf
    /// (whose file offset was captured by the caller before recursion).
    node_idx: Option<usize>,
    /// Per-dimension minimum coordinates of all points in this subtree.
    min: Vec<f64>,
    /// Per-dimension maximum coordinates of all points in this subtree.
    max: Vec<f64>,
}

/// Borrowed view over the caller's flat point/doc_id buffers used during
/// recursive subtree construction. Holding only references avoids deep-copying
/// the input data while the builder permutes its private index array.
struct BuildContext<'a> {
    points: &'a [f64],
    doc_ids: &'a [u64],
    num_dims: usize,
}

impl BuildContext<'_> {
    /// Return the d-th coordinate of the point at slot `i` in the original
    /// (unpermuted) buffer.
    #[inline]
    fn value(&self, i: u32, d: usize) -> f64 {
        self.points[i as usize * self.num_dims + d]
    }
}

/// Compute the per-dimension axis-aligned bounding box that encloses every
/// point referenced by `indices` in the underlying buffer.
///
/// The returned `(min, max)` Vecs have length `ctx.num_dims`. Callers must
/// pass a non-empty `indices` slice; an empty slice would leave the bounds at
/// their `INFINITY` / `NEG_INFINITY` sentinels and propagate degenerate
/// AABBs into the index, so the caller is responsible for the precondition.
fn compute_aabb(ctx: &BuildContext<'_>, indices: &[u32]) -> (Vec<f64>, Vec<f64>) {
    let mut min = vec![f64::INFINITY; ctx.num_dims];
    let mut max = vec![f64::NEG_INFINITY; ctx.num_dims];
    for &i in indices {
        let base = i as usize * ctx.num_dims;
        for d in 0..ctx.num_dims {
            let v = ctx.points[base + d];
            if v < min[d] {
                min[d] = v;
            }
            if v > max[d] {
                max[d] = v;
            }
        }
    }
    (min, max)
}

impl<W: StorageOutput> BKDWriter<W> {
    pub fn new(writer: W, num_dims: u32) -> Self {
        BKDWriter {
            writer: StructWriter::new(writer),
            block_size: 512,
            num_blocks: 0,
            num_dims,
            min_values: vec![f64::MAX; num_dims as usize],
            max_values: vec![f64::MIN; num_dims as usize],
            index_nodes: Vec::new(),
        }
    }

    /// Set custom block size
    pub fn with_block_size(mut self, block_size: usize) -> Self {
        self.block_size = block_size;
        self
    }

    /// Write a BKD tree from flat point/doc_id buffers.
    ///
    /// The `points` buffer is laid out as a row-major matrix of
    /// `doc_ids.len()` rows by `num_dims` columns: the d-th coordinate of the
    /// i-th point lives at `points[i * num_dims + d]`. `points.len()` must
    /// therefore equal `doc_ids.len() * num_dims`.
    ///
    /// Internally the builder sorts an index permutation rather than the
    /// point/doc_id buffers themselves, so no per-point heap allocation is
    /// performed regardless of point count.
    ///
    /// # Arguments
    /// - `points`: flat row-major buffer of f64 coordinates.
    /// - `doc_ids`: parallel buffer of document ids.
    ///
    /// # Returns
    /// `Ok(())` on success, otherwise a `LaurusError::index` describing the
    /// dimensional mismatch or an underlying I/O error.
    pub fn write(&mut self, points: &[f64], doc_ids: &[u64]) -> Result<()> {
        let num_dims = self.num_dims as usize;
        let expected = doc_ids.len().checked_mul(num_dims).ok_or_else(|| {
            crate::error::LaurusError::index(
                "Point count overflows when multiplied by num_dims".to_string(),
            )
        })?;
        if points.len() != expected {
            return Err(crate::error::LaurusError::index(format!(
                "Point buffer size mismatch: expected {} doc_ids * {} dims = {} f64s, got {}",
                doc_ids.len(),
                num_dims,
                expected,
                points.len()
            )));
        }

        if doc_ids.is_empty() {
            // Write basic header for empty tree
            self.write_header(0, 0, 0)?;
            return Ok(());
        }

        // Calculate global min/max
        for i in 0..doc_ids.len() {
            let base = i * num_dims;
            for d in 0..num_dims {
                let v = points[base + d];
                self.min_values[d] = self.min_values[d].min(v);
                self.max_values[d] = self.max_values[d].max(v);
            }
        }

        let total_count = doc_ids.len() as u64;

        // Reserve space for header:
        // Magic(4) + Version(4) + num_dims(4) + bytes_per_dim(4) + total_count(8) + num_blocks(8)
        // + min_values(num_dims * 8) + max_values(num_dims * 8) + index_start(8) + root_offset(8)
        let header_size = 4 + 4 + 4 + 4 + 8 + 8 + (self.num_dims as u64 * 8 * 2) + 8 + 8;

        self.writer.write_u32(0)?; // Placeholder
        self.writer.seek(SeekFrom::Start(header_size))?;

        // Sort an index permutation instead of the data: this keeps the
        // point/doc_id buffers immutable and avoids per-point allocations.
        let mut indices: Vec<u32> = (0..doc_ids.len() as u32).collect();
        let ctx = BuildContext {
            points,
            doc_ids,
            num_dims,
        };
        let root_info = self.build_subtree(&ctx, &mut indices, 0)?;

        // Write index section after all leaves
        let index_start_offset = self.writer.stream_position()?;
        self.write_index()?;

        let node_size = Self::node_size(self.num_dims);
        let root_node_offset = if let Some(idx) = root_info.node_idx {
            index_start_offset + (idx as u64) * node_size
        } else {
            // Single-leaf tree: the leaf was written immediately after the
            // header, so the "root" address is just past the header bytes.
            header_size
        };

        // Go back and write real header
        self.writer.seek(SeekFrom::Start(0))?;
        self.write_header(total_count, index_start_offset, root_node_offset)?;

        // Go back to end
        self.writer.seek(SeekFrom::End(0))?;

        Ok(())
    }

    fn write_header(&mut self, total_count: u64, index_start: u64, root_offset: u64) -> Result<()> {
        self.writer.write_u32(BKD_MAGIC)?;
        self.writer.write_u32(BKD_VERSION)?;
        self.writer.write_u32(self.num_dims)?;
        self.writer.write_u32(8)?; // Bytes per dim (f64)
        self.writer.write_u64(total_count)?;
        self.writer.write_u64(self.num_blocks)?;
        for &v in &self.min_values {
            self.writer.write_f64(v)?;
        }
        for &v in &self.max_values {
            self.writer.write_f64(v)?;
        }
        self.writer.write_u64(index_start)?;
        self.writer.write_u64(root_offset)?;
        Ok(())
    }

    /// Returns the on-disk byte size of one internal index node.
    ///
    /// Each dimension contributes 32 bytes (left_min, left_max, right_min,
    /// right_max — four f64 values per dimension) on top of the fixed 28-byte
    /// split / offset header, matching the layout documented on
    /// [`BKD_VERSION`].
    #[inline]
    fn node_size(num_dims: u32) -> u64 {
        28 + 32 * num_dims as u64
    }

    /// Recursively build a subtree, writing leaves on the fly and recording
    /// internal nodes in `self.index_nodes` for back-patching. The slice
    /// `indices` is a permutation of point ids that this call owns and is
    /// allowed to reorder; recursion proceeds on the two halves of the
    /// permutation around the split position.
    ///
    /// In addition to the file mutation, the call returns a `SubtreeInfo`
    /// carrying the AABB of every point reachable through this subtree so the
    /// parent call can store per-child AABBs on its own index node.
    fn build_subtree(
        &mut self,
        ctx: &BuildContext<'_>,
        indices: &mut [u32],
        depth: u32,
    ) -> Result<SubtreeInfo> {
        if indices.is_empty() {
            // The recursion only descends into non-empty halves (we only split
            // when len > block_size, where the smaller half has at least one
            // element), so reaching this branch indicates a programmer error.
            return Err(crate::error::LaurusError::index(
                "build_subtree called with empty indices".to_string(),
            ));
        }

        if indices.len() <= self.block_size {
            let (leaf_min, leaf_max) = compute_aabb(ctx, indices);
            self.write_leaf_block(ctx, indices, &leaf_min, &leaf_max)?;
            self.num_blocks += 1;
            return Ok(SubtreeInfo {
                node_idx: None,
                min: leaf_min,
                max: leaf_max,
            });
        }

        // Split dimension: round robin (#293 will switch to widest-axis).
        let split_dim = depth % self.num_dims;
        let split_dim_us = split_dim as usize;

        // Sort the permutation by the split dimension to find the median.
        // The underlying point/doc_id buffers stay immutable; only `indices`
        // is reordered.
        indices.sort_by(|&a, &b| {
            ctx.value(a, split_dim_us)
                .partial_cmp(&ctx.value(b, split_dim_us))
                .unwrap_or(Ordering::Equal)
        });

        // Internal nodes are written AFTER all leaves; we track tree structure
        // in `index_nodes` here and back-patch offsets in `write_index`.
        let mid = indices.len() / 2;
        let (left_indices, right_indices) = indices.split_at_mut(mid);
        let split_value = ctx.value(right_indices[0], split_dim_us);

        // Reserve a slot for this internal node now so the index_nodes vector
        // is stable across recursive calls. AABB / offset / child fields are
        // back-patched once both children have been built.
        let node_idx = self.index_nodes.len();
        let nd = ctx.num_dims;
        self.index_nodes.push(IndexNode {
            split_dim,
            split_value,
            left_min: Vec::with_capacity(nd),
            left_max: Vec::with_capacity(nd),
            right_min: Vec::with_capacity(nd),
            right_max: Vec::with_capacity(nd),
            left_offset: 0,
            right_offset: 0,
            left_child_idx: None,
            right_child_idx: None,
        });

        let left_file_pos_before = self.writer.stream_position()?;
        let left_info = self.build_subtree(ctx, left_indices, depth + 1)?;
        let left_is_leaf = left_info.node_idx.is_none();

        let right_file_pos_before = self.writer.stream_position()?;
        let right_info = self.build_subtree(ctx, right_indices, depth + 1)?;
        let right_is_leaf = right_info.node_idx.is_none();

        // Compute the union AABB BEFORE moving the child AABB Vecs into the
        // node, so the borrow checker can see that we only consume the child
        // info once.
        let mut union_min = Vec::with_capacity(nd);
        let mut union_max = Vec::with_capacity(nd);
        for d in 0..nd {
            union_min.push(left_info.min[d].min(right_info.min[d]));
            union_max.push(left_info.max[d].max(right_info.max[d]));
        }

        // Update the previously reserved node slot.
        let node = &mut self.index_nodes[node_idx];
        node.left_child_idx = left_info.node_idx;
        node.right_child_idx = right_info.node_idx;
        node.left_min = left_info.min;
        node.left_max = left_info.max;
        node.right_min = right_info.min;
        node.right_max = right_info.max;
        if left_is_leaf {
            node.left_offset = left_file_pos_before;
        }
        if right_is_leaf {
            node.right_offset = right_file_pos_before;
        }

        Ok(SubtreeInfo {
            node_idx: Some(node_idx),
            min: union_min,
            max: union_max,
        })
    }

    fn write_leaf_block(
        &mut self,
        ctx: &BuildContext<'_>,
        indices: &[u32],
        leaf_min: &[f64],
        leaf_max: &[f64],
    ) -> Result<()> {
        let count = indices.len() as u32;
        self.writer.write_u32(count)?;

        // Per-leaf AABB, used by the reader for subtree pruning starting
        // from #292.
        for &v in leaf_min {
            self.writer.write_f64(v)?;
        }
        for &v in leaf_max {
            self.writer.write_f64(v)?;
        }

        // Write values for all dimensions, gathered through the permutation.
        for &i in indices {
            let base = i as usize * ctx.num_dims;
            for d in 0..ctx.num_dims {
                self.writer.write_f64(ctx.points[base + d])?;
            }
        }

        // Write doc ids in the same order
        for &i in indices {
            self.writer.write_u64(ctx.doc_ids[i as usize])?;
        }

        Ok(())
    }

    fn write_index(&mut self) -> Result<()> {
        let start_pos = self.writer.stream_position()?;
        let node_size = Self::node_size(self.num_dims);

        for i in 0..self.index_nodes.len() {
            let left_idx = self.index_nodes[i].left_child_idx;
            if let Some(idx) = left_idx {
                self.index_nodes[i].left_offset = start_pos + (idx as u64) * node_size;
            }

            let right_idx = self.index_nodes[i].right_child_idx;
            if let Some(idx) = right_idx {
                self.index_nodes[i].right_offset = start_pos + (idx as u64) * node_size;
            }
        }

        // Write nodes in the layout documented on `BKD_VERSION`.
        for node in &self.index_nodes {
            self.writer.write_u32(node.split_dim)?;
            self.writer.write_f64(node.split_value)?;
            for &v in &node.left_min {
                self.writer.write_f64(v)?;
            }
            for &v in &node.left_max {
                self.writer.write_f64(v)?;
            }
            for &v in &node.right_min {
                self.writer.write_f64(v)?;
            }
            for &v in &node.right_max {
                self.writer.write_f64(v)?;
            }
            self.writer.write_u64(node.left_offset)?;
            self.writer.write_u64(node.right_offset)?;
        }

        Ok(())
    }

    /// Finish writing and return the underlying writer.
    pub fn finish(self) -> Result<()> {
        self.writer.close()
    }
}

/// Reader for BKD Trees.
#[derive(Debug)]
pub struct BKDReader {
    header: BKDFileHeader,
    storage: Arc<dyn Storage>,
    path: String,
}

impl BKDReader {
    /// Open a BKD tree from storage and path.
    pub fn open(storage: Arc<dyn Storage>, path: &str) -> Result<Self> {
        let input = storage.open_input(path)?;
        let mut reader = StructReader::new(input)?;

        // Read header
        let magic = reader.read_u32()?;
        if magic != BKD_MAGIC {
            return Err(crate::error::LaurusError::storage(format!(
                "Invalid BKD magic: {:x}",
                magic
            )));
        }

        let version = reader.read_u32()?;
        if version != BKD_VERSION {
            return Err(crate::error::LaurusError::storage(format!(
                "Unsupported BKD version: {} (expected {}). Pre-release format \
                 changes do not support older revisions; rebuild the index.",
                version, BKD_VERSION
            )));
        }
        let num_dims = reader.read_u32()?;
        let bytes_per_dim = reader.read_u32()?;
        let total_point_count = reader.read_u64()?;
        let num_blocks = reader.read_u64()?;
        let mut min_values = Vec::with_capacity(num_dims as usize);
        for _ in 0..num_dims {
            min_values.push(reader.read_f64()?);
        }
        let mut max_values = Vec::with_capacity(num_dims as usize);
        for _ in 0..num_dims {
            max_values.push(reader.read_f64()?);
        }
        let index_start_offset = reader.read_u64()?;
        let root_node_offset = reader.read_u64()?;

        let header = BKDFileHeader {
            magic,
            version,
            num_dims,
            bytes_per_dim,
            total_point_count,
            num_blocks,
            min_values,
            max_values,
            index_start_offset,
            root_node_offset,
        };

        Ok(BKDReader {
            header,
            storage,
            path: path.to_string(),
        })
    }

    /// Read `num_dims` `f64` values for `min` followed by `num_dims` for
    /// `max`, returning the constructed AABB.
    fn read_child_aabb<R: StorageInput>(
        reader: &mut StructReader<R>,
        num_dims: usize,
    ) -> Result<AABB> {
        let mut min = Vec::with_capacity(num_dims);
        for _ in 0..num_dims {
            min.push(reader.read_f64()?);
        }
        let mut max = Vec::with_capacity(num_dims);
        for _ in 0..num_dims {
            max.push(reader.read_f64()?);
        }
        AABB::new(min, max)
    }

    /// Walk the subtree rooted at `offset`, dispatching pruning decisions
    /// to `visitor`. Internal nodes consult `visitor.compare` on each
    /// child's AABB; leaves either short-circuit (Outside / Inside) or
    /// stream every (doc_id, point) candidate through `visitor.visit`.
    fn intersect_subtree<R: StorageInput>(
        &self,
        reader: &mut StructReader<R>,
        offset: u64,
        visitor: &mut dyn IntersectVisitor,
    ) -> Result<()> {
        if offset < self.header.index_start_offset {
            return self.intersect_leaf(reader, offset, visitor);
        }
        let num_dims = self.header.num_dims as usize;
        reader.seek(SeekFrom::Start(offset))?;
        let _split_dim = reader.read_u32()?;
        let _split_value = reader.read_f64()?;
        let left_aabb = Self::read_child_aabb(reader, num_dims)?;
        let right_aabb = Self::read_child_aabb(reader, num_dims)?;
        let left_offset = reader.read_u64()?;
        let right_offset = reader.read_u64()?;

        match visitor.compare(&left_aabb) {
            CellRelation::Outside => {}
            CellRelation::Inside => self.collect_subtree(reader, left_offset, visitor)?,
            CellRelation::Crosses => self.intersect_subtree(reader, left_offset, visitor)?,
        }
        match visitor.compare(&right_aabb) {
            CellRelation::Outside => {}
            CellRelation::Inside => self.collect_subtree(reader, right_offset, visitor)?,
            CellRelation::Crosses => self.intersect_subtree(reader, right_offset, visitor)?,
        }
        Ok(())
    }

    /// Walk a leaf at `offset`, classifying it via `visitor.compare` on the
    /// stored leaf AABB and dispatching points accordingly.
    fn intersect_leaf<R: StorageInput>(
        &self,
        reader: &mut StructReader<R>,
        offset: u64,
        visitor: &mut dyn IntersectVisitor,
    ) -> Result<()> {
        reader.seek(SeekFrom::Start(offset))?;
        let count = reader.read_u32()? as usize;
        let num_dims = self.header.num_dims as usize;
        let leaf_aabb = Self::read_child_aabb(reader, num_dims)?;

        match visitor.compare(&leaf_aabb) {
            CellRelation::Outside => Ok(()),
            CellRelation::Inside => {
                // Skip the point bytes; we only need the doc ids.
                let point_bytes = (count as u64) * (num_dims as u64) * 8;
                reader.seek(SeekFrom::Current(point_bytes as i64))?;
                for _ in 0..count {
                    let doc_id = reader.read_u64()?;
                    visitor.visit_inside(doc_id);
                }
                Ok(())
            }
            CellRelation::Crosses => {
                let mut points = vec![0.0f64; count * num_dims];
                for slot in points.iter_mut() {
                    *slot = reader.read_f64()?;
                }
                for i in 0..count {
                    let doc_id = reader.read_u64()?;
                    let point = &points[i * num_dims..(i + 1) * num_dims];
                    visitor.visit(doc_id, point);
                }
                Ok(())
            }
        }
    }

    /// Walk a subtree whose root the caller has already classified as
    /// `Inside`. No `compare` calls are made — every doc is reported via
    /// `visit_inside` and the leaf point bytes are skipped entirely.
    fn collect_subtree<R: StorageInput>(
        &self,
        reader: &mut StructReader<R>,
        offset: u64,
        visitor: &mut dyn IntersectVisitor,
    ) -> Result<()> {
        if offset < self.header.index_start_offset {
            return self.collect_leaf(reader, offset, visitor);
        }
        let num_dims = self.header.num_dims as usize;
        reader.seek(SeekFrom::Start(offset))?;
        let _split_dim = reader.read_u32()?;
        let _split_value = reader.read_f64()?;
        // Skip both child AABBs.
        let aabb_bytes = (num_dims as u64) * 16 * 2;
        reader.seek(SeekFrom::Current(aabb_bytes as i64))?;
        let left_offset = reader.read_u64()?;
        let right_offset = reader.read_u64()?;
        self.collect_subtree(reader, left_offset, visitor)?;
        self.collect_subtree(reader, right_offset, visitor)?;
        Ok(())
    }

    /// Walk a leaf whose enclosing cell has already been classified as
    /// `Inside`. The leaf AABB and point bytes are skipped; only doc ids
    /// are read and reported via `visit_inside`.
    fn collect_leaf<R: StorageInput>(
        &self,
        reader: &mut StructReader<R>,
        offset: u64,
        visitor: &mut dyn IntersectVisitor,
    ) -> Result<()> {
        reader.seek(SeekFrom::Start(offset))?;
        let count = reader.read_u32()? as usize;
        let num_dims = self.header.num_dims as usize;
        // Skip leaf AABB (min + max) and all point bytes.
        let skip_bytes = (num_dims as u64) * 16 + (count as u64) * (num_dims as u64) * 8;
        reader.seek(SeekFrom::Current(skip_bytes as i64))?;
        for _ in 0..count {
            let doc_id = reader.read_u64()?;
            visitor.visit_inside(doc_id);
        }
        Ok(())
    }
}

impl BKDTree for BKDReader {
    fn intersect(&self, visitor: &mut dyn IntersectVisitor) -> Result<()> {
        if self.header.total_point_count == 0 {
            return Ok(());
        }
        let input = self.storage.open_input(&self.path)?;
        let mut reader = StructReader::new(input)?;
        let root_offset = self.header.root_node_offset;
        if root_offset < self.header.index_start_offset {
            // Single-leaf tree: the root "address" is just past the header.
            self.intersect_leaf(&mut reader, root_offset, visitor)
        } else {
            self.intersect_subtree(&mut reader, root_offset, visitor)
        }
    }
}

/// A simple BKD Tree for efficient range queries.
///
/// Slated for removal in #295 once the disk-backed `BKDReader` covers all
/// remaining call sites.
#[derive(Debug, Clone)]
pub struct SimpleBKDTree {
    /// Sorted array of (points, doc_id) pairs.
    entries: Vec<(Vec<f64>, u64)>,
    /// Retained for diagnostic accessors only; the new `intersect`-based
    /// implementation derives the per-point dimensionality from each entry.
    #[allow(dead_code)]
    num_dims: u32,
    field_name: String,
}

impl BKDTree for SimpleBKDTree {
    /// Linear scan across `entries`. The simple tree carries no AABB and
    /// performs no pruning, so every point is reported via `visitor.visit`
    /// (the `Crosses` path) and the visitor is responsible for filtering.
    fn intersect(&self, visitor: &mut dyn IntersectVisitor) -> Result<()> {
        for (vals, doc_id) in &self.entries {
            visitor.visit(*doc_id, vals);
        }
        Ok(())
    }
}

impl SimpleBKDTree {
    /// Create a new BKD Tree from unsorted (points, doc_id) pairs.
    pub fn new(field_name: String, num_dims: u32, mut entries: Vec<(Vec<f64>, u64)>) -> Self {
        // Simple sorting by first dimension for consistent ordering
        entries.sort_by(|a, b| {
            if a.0.is_empty() || b.0.is_empty() {
                std::cmp::Ordering::Equal
            } else {
                a.0[0]
                    .partial_cmp(&b.0[0])
                    .unwrap_or(std::cmp::Ordering::Equal)
            }
        });

        SimpleBKDTree {
            entries,
            num_dims,
            field_name,
        }
    }

    /// Create an empty BKD Tree.
    pub fn empty(field_name: String, num_dims: u32) -> Self {
        SimpleBKDTree {
            entries: Vec::new(),
            num_dims,
            field_name,
        }
    }

    /// Get the field name this tree is built for.
    pub fn field_name(&self) -> &str {
        &self.field_name
    }

    /// Get the number of entries in this tree.
    pub fn size(&self) -> usize {
        self.entries.len()
    }

    /// Check if the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::Storage;
    use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};
    use std::sync::Arc;

    fn create_test_tree() -> SimpleBKDTree {
        let entries = vec![
            (vec![1.0], 10), // doc 10, value 1.0
            (vec![3.0], 20), // doc 20, value 3.0
            (vec![2.0], 30), // doc 30, value 2.0
            (vec![5.0], 40), // doc 40, value 5.0
            (vec![4.0], 50), // doc 50, value 4.0
            (vec![1.5], 60), // doc 60, value 1.5
        ];
        SimpleBKDTree::new("test_field".to_string(), 1, entries)
    }

    #[test]
    fn test_bkd_writer_reader_2d() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        // Flat row-major buffer: [pt0_x, pt0_y, pt1_x, pt1_y, pt2_x, pt2_y]
        let points: Vec<f64> = vec![10.0, 20.0, 15.0, 25.0, 20.0, 30.0];
        let doc_ids: Vec<u64> = vec![1, 2, 3];

        // Write
        {
            let output = storage.create_output("test_2d.bkd").unwrap();
            let mut writer = BKDWriter::new(output, 2);
            writer.write(&points, &doc_ids).unwrap();
            writer.finish().unwrap();
        }

        // Read
        {
            let reader = BKDReader::open(storage.clone(), "test_2d.bkd").unwrap();
            assert_eq!(reader.header.num_dims, 2);

            // Search [10, 10] to [15, 25]
            let results = reader
                .range_search(
                    &[Some(10.0), Some(10.0)],
                    &[Some(15.0), Some(25.0)],
                    true,
                    true,
                )
                .unwrap();
            assert_eq!(results, vec![1, 2]);
        }
    }

    #[test]
    fn test_bkd_writer_empty() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let points: Vec<f64> = vec![];
        let doc_ids: Vec<u64> = vec![];

        {
            let output = storage.create_output("empty.bkd").unwrap();
            let mut writer = BKDWriter::new(output, 2);
            writer.write(&points, &doc_ids).unwrap();
            writer.finish().unwrap();
        }

        let reader = BKDReader::open(storage.clone(), "empty.bkd").unwrap();
        assert_eq!(reader.header.total_point_count, 0);
        let results = reader
            .range_search(&[None, None], &[None, None], true, true)
            .unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_bkd_writer_size_mismatch_rejected() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        // 2 doc_ids in 2D would require 4 f64s, but we pass 3.
        let points: Vec<f64> = vec![1.0, 2.0, 3.0];
        let doc_ids: Vec<u64> = vec![10, 20];

        let output = storage.create_output("bad.bkd").unwrap();
        let mut writer = BKDWriter::new(output, 2);
        let err = writer.write(&points, &doc_ids).unwrap_err();
        assert!(
            format!("{err:?}").contains("Point buffer size mismatch"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn test_bkd_writer_reader_1d_multi_block() {
        // Exercise the recursive build path with more points than the leaf
        // block size so the index/leaf split is actually visited.
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let n: usize = 2_000;
        let points: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let doc_ids: Vec<u64> = (0..n as u64).collect();

        {
            let output = storage.create_output("range1d.bkd").unwrap();
            let mut writer = BKDWriter::new(output, 1).with_block_size(128);
            writer.write(&points, &doc_ids).unwrap();
            writer.finish().unwrap();
        }

        let reader = BKDReader::open(storage.clone(), "range1d.bkd").unwrap();
        let results = reader
            .range_search(&[Some(100.0)], &[Some(200.0)], true, true)
            .unwrap();
        let expected: Vec<u64> = (100u64..=200u64).collect();
        assert_eq!(results, expected);
    }

    #[test]
    fn test_bkd_writer_reader_3d_multi_block() {
        // 3D round-trip with multiple leaf blocks: the new per-node /
        // per-leaf AABB layout must round-trip without misaligning offsets.
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let n: usize = 1_000;
        let mut points: Vec<f64> = Vec::with_capacity(n * 3);
        let mut doc_ids: Vec<u64> = Vec::with_capacity(n);
        for i in 0..n {
            let v = i as f64;
            points.push(v);
            points.push(v + 1000.0);
            points.push(v + 2000.0);
            doc_ids.push(i as u64);
        }

        {
            let output = storage.create_output("range3d.bkd").unwrap();
            let mut writer = BKDWriter::new(output, 3).with_block_size(64);
            writer.write(&points, &doc_ids).unwrap();
            writer.finish().unwrap();
        }

        let reader = BKDReader::open(storage.clone(), "range3d.bkd").unwrap();
        assert_eq!(reader.header.num_dims, 3);
        assert_eq!(reader.header.version, BKD_VERSION);

        // Half-open bound on the first axis with an upper-only bound on the
        // second axis to make sure visit_node consumes the AABB bytes
        // correctly even with mixed bounded/unbounded dimensions.
        let results = reader
            .range_search(
                &[Some(100.0), None, None],
                &[Some(150.0), Some(1200.0), None],
                true,
                true,
            )
            .unwrap();
        let expected: Vec<u64> = (100u64..=150u64)
            .filter(|&i| (i as f64) + 1000.0 <= 1200.0)
            .collect();
        assert_eq!(results, expected);
    }

    #[test]
    fn test_bkd_reader_rejects_version_mismatch() {
        // Hand-craft a header that claims version 1 (the previous on-disk
        // format) and confirm the reader refuses to open it.
        use crate::storage::structured::StructWriter;

        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        {
            let output = storage.create_output("v1.bkd").unwrap();
            let mut writer = StructWriter::new(output);
            writer.write_u32(BKD_MAGIC).unwrap();
            writer.write_u32(1).unwrap(); // legacy version
            writer.write_u32(2).unwrap(); // num_dims
            writer.write_u32(8).unwrap(); // bytes_per_dim
            writer.write_u64(0).unwrap(); // total_count
            writer.write_u64(0).unwrap(); // num_blocks
            writer.write_f64(0.0).unwrap(); // global_min[0]
            writer.write_f64(0.0).unwrap(); // global_min[1]
            writer.write_f64(0.0).unwrap(); // global_max[0]
            writer.write_f64(0.0).unwrap(); // global_max[1]
            writer.write_u64(0).unwrap(); // index_start
            writer.write_u64(0).unwrap(); // root_offset
            writer.close().unwrap();
        }

        let err = BKDReader::open(storage.clone(), "v1.bkd").unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.contains("Unsupported BKD version"),
            "unexpected error: {msg}"
        );
    }

    /// Visitor that segregates hits by which BKD code path produced them
    /// (`visit_inside` vs `visit`), used to assert that `Inside` cells avoid
    /// per-point filtering and `Crosses` leaves go through it.
    struct TracingVisitor {
        query: AABB,
        inside_hits: Vec<u64>,
        crosses_hits: Vec<u64>,
    }

    impl TracingVisitor {
        fn new(query: AABB) -> Self {
            Self {
                query,
                inside_hits: Vec::new(),
                crosses_hits: Vec::new(),
            }
        }
    }

    impl IntersectVisitor for TracingVisitor {
        fn compare(&self, cell: &AABB) -> CellRelation {
            // Conservative compare: cell vs query (closed intervals).
            let qmin = self.query.min();
            let qmax = self.query.max();
            let cmin = cell.min();
            let cmax = cell.max();
            for d in 0..cell.num_dims() {
                if cmax[d] < qmin[d] || cmin[d] > qmax[d] {
                    return CellRelation::Outside;
                }
            }
            for d in 0..cell.num_dims() {
                if cmin[d] < qmin[d] || cmax[d] > qmax[d] {
                    return CellRelation::Crosses;
                }
            }
            CellRelation::Inside
        }
        fn visit_inside(&mut self, doc_id: u64) {
            self.inside_hits.push(doc_id);
        }
        fn visit(&mut self, doc_id: u64, point: &[f64]) {
            if self.query.contains_point(point) {
                self.crosses_hits.push(doc_id);
            }
        }
    }

    /// A wrapper that also records every `compare` outcome (including
    /// `Outside`) by using a `Cell` for interior mutability.
    struct RecordingVisitor {
        query: AABB,
        relations: std::cell::RefCell<Vec<CellRelation>>,
        hits: Vec<u64>,
    }

    impl RecordingVisitor {
        fn new(query: AABB) -> Self {
            Self {
                query,
                relations: std::cell::RefCell::new(Vec::new()),
                hits: Vec::new(),
            }
        }
    }

    impl IntersectVisitor for RecordingVisitor {
        fn compare(&self, cell: &AABB) -> CellRelation {
            let qmin = self.query.min();
            let qmax = self.query.max();
            let cmin = cell.min();
            let cmax = cell.max();
            let mut relation = CellRelation::Inside;
            for d in 0..cell.num_dims() {
                if cmax[d] < qmin[d] || cmin[d] > qmax[d] {
                    relation = CellRelation::Outside;
                    break;
                }
            }
            if !matches!(relation, CellRelation::Outside) {
                for d in 0..cell.num_dims() {
                    if cmin[d] < qmin[d] || cmax[d] > qmax[d] {
                        relation = CellRelation::Crosses;
                        break;
                    }
                }
            }
            self.relations.borrow_mut().push(relation);
            relation
        }
        fn visit_inside(&mut self, doc_id: u64) {
            self.hits.push(doc_id);
        }
        fn visit(&mut self, doc_id: u64, point: &[f64]) {
            // For Crosses cells, accept the point only if it actually lies
            // inside the query.
            if self.query.contains_point(point) {
                self.hits.push(doc_id);
            }
        }
    }

    #[test]
    fn intersect_inside_avoids_per_point_filter() {
        // Build a 1D tree with 4 leaf blocks; query the entire range so the
        // root subtree is `Inside` and every doc is reported via
        // `visit_inside`, never `visit`.
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let n: usize = 256;
        let points: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let doc_ids: Vec<u64> = (0..n as u64).collect();
        {
            let output = storage.create_output("inside.bkd").unwrap();
            let mut writer = BKDWriter::new(output, 1).with_block_size(32);
            writer.write(&points, &doc_ids).unwrap();
            writer.finish().unwrap();
        }

        let reader = BKDReader::open(storage.clone(), "inside.bkd").unwrap();
        let query = AABB::new(vec![-1e9], vec![1e9]).unwrap();
        let mut v = TracingVisitor::new(query);
        reader.intersect(&mut v).unwrap();

        // Every hit came through visit_inside: the query bounds wholly
        // enclose every cell, so no point ever needed per-coordinate
        // filtering.
        assert_eq!(v.inside_hits.len(), n);
        assert!(v.crosses_hits.is_empty());
        v.inside_hits.sort_unstable();
        let expected: Vec<u64> = (0..n as u64).collect();
        assert_eq!(v.inside_hits, expected);
    }

    #[test]
    fn intersect_outside_prunes_subtree() {
        // Query that lies entirely above every point; expect zero hits and
        // at least one Outside compare result.
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let n: usize = 128;
        let points: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let doc_ids: Vec<u64> = (0..n as u64).collect();
        {
            let output = storage.create_output("outside.bkd").unwrap();
            let mut writer = BKDWriter::new(output, 1).with_block_size(16);
            writer.write(&points, &doc_ids).unwrap();
            writer.finish().unwrap();
        }

        let reader = BKDReader::open(storage.clone(), "outside.bkd").unwrap();
        let query = AABB::new(vec![1000.0], vec![2000.0]).unwrap();
        let mut v = RecordingVisitor::new(query);
        reader.intersect(&mut v).unwrap();

        assert!(v.hits.is_empty());
        assert!(
            v.relations
                .borrow()
                .iter()
                .any(|r| matches!(r, CellRelation::Outside)),
            "expected at least one Outside compare, got {:?}",
            v.relations.borrow()
        );
    }

    #[test]
    fn intersect_crosses_filters_per_point() {
        // Query that overlaps a leaf boundary; expect Crosses leaves and
        // hits accumulated via visit (per-point filtering).
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let n: usize = 200;
        let points: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let doc_ids: Vec<u64> = (0..n as u64).collect();
        {
            let output = storage.create_output("crosses.bkd").unwrap();
            let mut writer = BKDWriter::new(output, 1).with_block_size(16);
            writer.write(&points, &doc_ids).unwrap();
            writer.finish().unwrap();
        }

        let reader = BKDReader::open(storage.clone(), "crosses.bkd").unwrap();
        let query = AABB::new(vec![50.5], vec![100.5]).unwrap();
        let mut v = TracingVisitor::new(query);
        reader.intersect(&mut v).unwrap();

        let expected: Vec<u64> = (51u64..=100u64).collect();
        let mut got = v.crosses_hits.clone();
        got.append(&mut v.inside_hits.clone());
        got.sort_unstable();
        got.dedup();
        assert_eq!(got, expected);
        // At least some hits arrived via the `Crosses` path because the
        // query bounds (50.5 / 100.5) cut through leaf blocks.
        assert!(!v.crosses_hits.is_empty());
    }

    #[test]
    fn range_search_default_impl_matches_legacy_semantics() {
        // The trait's default `range_search` should still produce the same
        // sorted/deduped doc-id list it always has, now via `intersect`.
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let n: usize = 500;
        let points: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let doc_ids: Vec<u64> = (0..n as u64).collect();
        {
            let output = storage.create_output("legacy.bkd").unwrap();
            let mut writer = BKDWriter::new(output, 1).with_block_size(64);
            writer.write(&points, &doc_ids).unwrap();
            writer.finish().unwrap();
        }

        let reader = BKDReader::open(storage.clone(), "legacy.bkd").unwrap();

        // Inclusive bounds.
        let inclusive = reader
            .range_search(&[Some(100.0)], &[Some(200.0)], true, true)
            .unwrap();
        assert_eq!(inclusive, (100u64..=200u64).collect::<Vec<_>>());

        // Exclusive bounds: 100 < x < 200.
        let exclusive = reader
            .range_search(&[Some(100.0)], &[Some(200.0)], false, false)
            .unwrap();
        assert_eq!(exclusive, (101u64..=199u64).collect::<Vec<_>>());

        // Unbounded upper.
        let lower_only = reader
            .range_search(&[Some(490.0)], &[None], true, true)
            .unwrap();
        assert_eq!(lower_only, (490u64..n as u64).collect::<Vec<_>>());
    }

    #[test]
    fn test_bkd_writer_reader_2d_single_leaf_aabb() {
        // Single-leaf tree: exercises the leaf-only write/read path that
        // skips the index section entirely. The new leaf AABB must still be
        // written and consumed.
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let points: Vec<f64> = vec![1.0, 100.0, 2.0, 200.0, 3.0, 300.0];
        let doc_ids: Vec<u64> = vec![10, 20, 30];

        {
            let output = storage.create_output("single.bkd").unwrap();
            let mut writer = BKDWriter::new(output, 2);
            writer.write(&points, &doc_ids).unwrap();
            writer.finish().unwrap();
        }

        let reader = BKDReader::open(storage.clone(), "single.bkd").unwrap();
        let results = reader
            .range_search(
                &[Some(2.0), Some(150.0)],
                &[Some(3.0), Some(250.0)],
                true,
                true,
            )
            .unwrap();
        assert_eq!(results, vec![20]);
    }

    #[test]
    fn test_bkd_tree_creation() {
        let tree = create_test_tree();

        assert_eq!(tree.size(), 6);
        assert_eq!(tree.field_name(), "test_field");
        assert!(!tree.is_empty());

        // Verify entries are sorted by value
        let expected_order = vec![
            (vec![1.0], 10),
            (vec![1.5], 60),
            (vec![2.0], 30),
            (vec![3.0], 20),
            (vec![4.0], 50),
            (vec![5.0], 40),
        ];
        assert_eq!(tree.entries, expected_order);
    }

    #[test]
    fn test_empty_tree() {
        let tree = SimpleBKDTree::empty("empty_field".to_string(), 1);

        assert_eq!(tree.size(), 0);
        assert!(tree.is_empty());
        assert_eq!(
            tree.range_search(&[Some(1.0)], &[Some(5.0)], true, true)
                .unwrap(),
            Vec::<u64>::new()
        );
    }

    #[test]
    fn test_range_search_exact_bounds() {
        let tree = create_test_tree();

        // Range [2.0, 4.0] should match docs 30, 20, 50
        let results = tree
            .range_search(&[Some(2.0)], &[Some(4.0)], true, true)
            .unwrap();
        let mut expected = vec![30, 20, 50]; // docs with values 2.0, 3.0, 4.0
        expected.sort();

        assert_eq!(results, expected);
    }
}
