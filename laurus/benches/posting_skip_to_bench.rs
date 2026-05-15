//! Microbenchmark for `PostingIterator::skip_to` (#503).
//!
//! Bypasses the full search engine to measure the skip_to cost in
//! isolation. Two implementations are compared on identical inputs:
//!
//! - `old`: the pre-#503 linear-walk `block_cache` style — scans
//!   block-min/max metadata sequentially from index 0 on every call,
//!   then linear-scans within the matching block. This is the exact
//!   algorithm the iterator used before the multi-level skip table
//!   landed.
//! - `new`: the current `InvertedIndexPostingIterator::skip_to`, which
//!   uses the Lucene-90-style multi-level skip table built by
//!   `build_skip_levels`.
//!
//! The workload simulates an AND-conjunction matcher: a "dense" side
//! with `N` postings (= the iterator under test) is repeatedly
//! `skip_to`-advanced past a "rare" side's doc ids. The rare side has
//! roughly 6 % of N postings — the same shape `lexical/seek_skewed`
//! uses but without the engine machinery, so the measured time is
//! dominated by the seek itself.

use std::time::Duration;

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use laurus::lexical::index::inverted::core::posting::{Posting, build_skip_levels};
use laurus::lexical::index::inverted::reader::InvertedIndexPostingIterator;
use laurus::lexical::reader::PostingIterator;

/// One block of the legacy single-level skip cache. Mirrors the
/// removed `reader::PostingBlock` so the microbench can drive the
/// pre-#503 algorithm on the same input the new iterator uses.
#[derive(Debug, Clone)]
struct OldBlock {
    min_doc_id: u64,
    max_doc_id: u64,
    start_position: usize,
}

fn make_old_blocks(doc_ids: &[u32], block_size: usize) -> Vec<OldBlock> {
    let mut blocks = Vec::new();
    let mut start = 0;
    while start < doc_ids.len() {
        let end = (start + block_size).min(doc_ids.len());
        blocks.push(OldBlock {
            min_doc_id: doc_ids[start] as u64,
            max_doc_id: doc_ids[end - 1] as u64,
            start_position: start,
        });
        start = end;
    }
    blocks
}

/// Exact replica of the pre-#503 `find_block` linear scan, kept here
/// so the microbench compares apples-to-apples against the live code.
fn old_find_block(blocks: &[OldBlock], target: u64) -> Option<usize> {
    for (i, block) in blocks.iter().enumerate() {
        if target >= block.min_doc_id && target <= block.max_doc_id {
            return Some(i);
        }
        if target < block.min_doc_id {
            return Some(i);
        }
    }
    if !blocks.is_empty() {
        Some(blocks.len() - 1)
    } else {
        None
    }
}

/// Exact replica of the pre-#503 `skip_to` algorithm: find_block
/// linearly + tail linear scan inside the block. Operates on raw
/// `doc_ids` so this bench does not depend on the now-deleted
/// `PostingBlock` type.
fn old_skip_to(doc_ids: &[u32], blocks: &[OldBlock], position: &mut usize, target: u64) -> bool {
    if let Some(block_idx) = old_find_block(blocks, target) {
        *position = blocks[block_idx].start_position;
    }
    while *position < doc_ids.len() {
        if doc_ids[*position] as u64 >= target {
            return true;
        }
        *position += 1;
    }
    false
}

/// Build a "dense" doc_id list of `n` entries with stride 3 — this is
/// the same shape as the seek_skewed engine bench's `search` posting
/// list at the algorithmic level.
fn build_dense_doc_ids(n: usize) -> Vec<u32> {
    (0..n as u32).map(|i| i.saturating_mul(3) + 7).collect()
}

/// Pick rare-side targets at roughly 6 % the density of the dense
/// side. With `n = 1_000_000` and `rare_ratio = 16` this produces
/// 62_500 targets — comparable to a real `lattice`-vs-`search`
/// conjunction at 1 M docs.
fn build_rare_targets(dense: &[u32], rare_ratio: usize) -> Vec<u64> {
    dense
        .iter()
        .step_by(rare_ratio)
        .map(|&v| v as u64)
        .collect()
}

fn bench_skip_to(c: &mut Criterion) {
    let mut group = c.benchmark_group("posting/skip_to");
    // Bench each posting-list size at a few corpus scales — at 100k
    // both algorithms fit in L2 so the difference is small; at ≥ 1M
    // the legacy linear `find_block` walk falls out of cache and the
    // multi-level skip table's constant cost wins decisively.
    let sizes: &[usize] = &[100_000, 1_000_000];
    let rare_ratio: usize = 16;
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(20);

    for &n in sizes {
        let doc_ids = build_dense_doc_ids(n);
        let targets = build_rare_targets(&doc_ids, rare_ratio);
        let blocks = make_old_blocks(&doc_ids, 64);

        // Build a NEW iterator backed by the SAME doc_ids so the two
        // arms run on identical input.
        let postings: Vec<Posting> = doc_ids
            .iter()
            .map(|&v| Posting::with_frequency(v as u64, 1))
            .collect();

        group.bench_with_input(BenchmarkId::new("old_find_block_linear", n), &n, |b, _| {
            b.iter(|| {
                let mut position = 0usize;
                let mut hits = 0u64;
                for &target in &targets {
                    if old_skip_to(&doc_ids, &blocks, &mut position, target) {
                        hits += 1;
                    }
                }
                black_box(hits)
            });
        });

        group.bench_with_input(BenchmarkId::new("new_multi_level_skip", n), &n, |b, _| {
            // Build a fresh iterator inside `b.iter` so its internal
            // `position` cursor resets between samples — matches the
            // OLD arm's `let mut position = 0` reset.
            b.iter(|| {
                let mut iter = InvertedIndexPostingIterator::new(postings.clone());
                let mut hits = 0u64;
                for &target in &targets {
                    if iter.skip_to(target).unwrap_or(false) {
                        hits += 1;
                    }
                }
                black_box(hits)
            });
        });

        // Sanity: ensure the two arms agree on the hit count so the
        // bench is comparing correct algorithms, not coincidentally
        // identical timings on divergent outputs.
        let mut old_pos = 0usize;
        let mut old_hits = 0u64;
        for &target in &targets {
            if old_skip_to(&doc_ids, &blocks, &mut old_pos, target) {
                old_hits += 1;
            }
        }
        let mut iter = InvertedIndexPostingIterator::new(postings.clone());
        let mut new_hits = 0u64;
        for &target in &targets {
            if iter.skip_to(target).unwrap_or(false) {
                new_hits += 1;
            }
        }
        assert_eq!(
            old_hits, new_hits,
            "old vs new disagree on hit count at n={n}"
        );

        // Also verify `build_skip_levels` produced a non-trivial
        // table — otherwise the "new" arm is effectively measuring
        // the tail linear scan, not the multi-level skip.
        let levels = build_skip_levels(&doc_ids);
        assert!(
            !levels.is_empty(),
            "build_skip_levels returned empty levels at n={n}"
        );
    }

    group.finish();
}

criterion_group!(benches, bench_skip_to);
criterion_main!(benches);
