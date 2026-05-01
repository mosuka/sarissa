//! Criterion benchmarks for the BKD tree.
//!
//! Covers `BKDReader::range_search` (the legacy axis-aligned API) and
//! `BKDReader::intersect` (the visitor-driven primitive that 3D distance,
//! k-NN, and other custom-shape queries will be built on) across 1D, 2D,
//! and 3D datasets at 10k / 100k / 1M points.
//!
//! # Running
//!
//! Run every bench in this file:
//!
//! ```sh
//! cargo bench --bench bkd_bench
//! ```
//!
//! Filter by group / case (substring match against the criterion id):
//!
//! ```sh
//! cargo bench --bench bkd_bench -- range_search/3d/100000
//! cargo bench --bench bkd_bench -- intersect_counting
//! ```
//!
//! Compile-only smoke check (skips the runtime, used by CI):
//!
//! ```sh
//! cargo bench --bench bkd_bench --no-run
//! ```
//!
//! All trees are built once per `(num_dims, n)` combination and reused
//! across iterations. Synthetic data is generated with an inline LCG so
//! the benches stay deterministic without pulling in `rand`.

mod common;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use laurus::lexical::index::structures::aabb::AABB;
use laurus::lexical::index::structures::bkd_tree::{BKDReader, BKDTree, BKDWriter};
use laurus::lexical::index::structures::visitor::{CellRelation, IntersectVisitor};
use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use std::hint::black_box;
use std::sync::Arc;

use common::{DEFAULT_SEED, lcg_next};

/// Build an in-memory BKD tree of `n` random points in `num_dims` dimensions
/// with coordinates uniformly distributed in `[0, 1000)`. Returns the open
/// `BKDReader` ready for query benchmarking.
fn build_tree(num_dims: usize, n: usize) -> BKDReader {
    let storage: Arc<MemoryStorage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let mut points: Vec<f64> = Vec::with_capacity(n * num_dims);
    let mut doc_ids: Vec<u64> = Vec::with_capacity(n);
    let mut rng_state: u64 = DEFAULT_SEED;
    for i in 0..n {
        for _ in 0..num_dims {
            points.push(lcg_next(&mut rng_state));
        }
        doc_ids.push(i as u64);
    }
    let path = format!("bench_{num_dims}d_{n}.bkd");
    let output = storage.create_output(&path).unwrap();
    let mut writer = BKDWriter::new(output, num_dims as u32);
    writer.write(&points, &doc_ids).unwrap();
    writer.finish().unwrap();
    BKDReader::open(storage, &path).unwrap()
}

/// Visitor that just counts hits, used to benchmark the raw `intersect`
/// path without paying the `RangeQueryVisitor` boundary-handling cost.
struct CountingVisitor {
    query: AABB,
    count: u64,
}

impl IntersectVisitor for CountingVisitor {
    fn compare(&self, cell: &AABB) -> CellRelation {
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
    fn visit_inside(&mut self, _doc_id: u64) {
        self.count += 1;
    }
    fn visit(&mut self, _doc_id: u64, point: &[f64]) {
        if self.query.contains_point(point) {
            self.count += 1;
        }
    }
}

/// Sizes to sweep across the dimensionality axis.
const SIZES: &[usize] = &[10_000, 100_000, 1_000_000];

/// Dimensionalities to sweep.
const DIMS: &[usize] = &[1, 2, 3];

/// Narrow (selective) query: 1% of the per-dimension data range.
/// Hit counts: ~1% × n in 1D, ~0.01% × n in 2D, ~0.0001% × n in 3D.
const QUERY_NARROW_LO: f64 = 100.0;
const QUERY_NARROW_HI: f64 = 110.0;

/// Wide query: 50% of the per-dimension data range.
/// Hit counts: ~50% × n in 1D, ~25% × n in 2D, ~12.5% × n in 3D.
const QUERY_WIDE_LO: f64 = 100.0;
const QUERY_WIDE_HI: f64 = 600.0;

fn bench_range_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("bkd_range_search");
    for &num_dims in DIMS {
        for &n in SIZES {
            let reader = build_tree(num_dims, n);
            for (label, lo, hi) in [
                ("narrow", QUERY_NARROW_LO, QUERY_NARROW_HI),
                ("wide", QUERY_WIDE_LO, QUERY_WIDE_HI),
            ] {
                let mins: Vec<Option<f64>> = (0..num_dims).map(|_| Some(lo)).collect();
                let maxs: Vec<Option<f64>> = (0..num_dims).map(|_| Some(hi)).collect();
                let id = format!("{num_dims}d/{label}/{n}");
                group.throughput(Throughput::Elements(n as u64));
                group.bench_with_input(BenchmarkId::from_parameter(id), &(mins, maxs), |b, qs| {
                    let (mins, maxs) = qs;
                    b.iter(|| {
                        let hits = reader
                            .range_search(black_box(mins), black_box(maxs), true, true)
                            .unwrap();
                        black_box(hits);
                    });
                });
            }
        }
    }
    group.finish();
}

fn bench_intersect_counting(c: &mut Criterion) {
    let mut group = c.benchmark_group("bkd_intersect_counting");
    for &num_dims in DIMS {
        for &n in SIZES {
            let reader = build_tree(num_dims, n);
            for (label, lo, hi) in [
                ("narrow", QUERY_NARROW_LO, QUERY_NARROW_HI),
                ("wide", QUERY_WIDE_LO, QUERY_WIDE_HI),
            ] {
                let qmin: Vec<f64> = (0..num_dims).map(|_| lo).collect();
                let qmax: Vec<f64> = (0..num_dims).map(|_| hi).collect();
                let id = format!("{num_dims}d/{label}/{n}");
                group.throughput(Throughput::Elements(n as u64));
                group.bench_with_input(BenchmarkId::from_parameter(id), &(qmin, qmax), |b, qs| {
                    let (qmin, qmax) = qs;
                    b.iter(|| {
                        let mut v = CountingVisitor {
                            query: AABB::new(qmin.clone(), qmax.clone()).unwrap(),
                            count: 0,
                        };
                        reader.intersect(&mut v).unwrap();
                        black_box(v.count);
                    });
                });
            }
        }
    }
    group.finish();
}

fn bench_build(c: &mut Criterion) {
    // Tree construction itself, useful as a baseline and to gauge the
    // amortized cost of widest-axis splitting + AABB computation
    // introduced in #291–#293.
    let mut group = c.benchmark_group("bkd_build");
    // Smaller sweep: building 1M points × 3 dims repeatedly is slow.
    for &num_dims in DIMS {
        for &n in &[10_000usize, 100_000] {
            let id = format!("{num_dims}d/{n}");
            group.throughput(Throughput::Elements(n as u64));
            group.bench_function(BenchmarkId::from_parameter(id), |b| {
                // Pre-generate input outside the iter loop so we measure
                // build, not data generation.
                let mut points: Vec<f64> = Vec::with_capacity(n * num_dims);
                let mut doc_ids: Vec<u64> = Vec::with_capacity(n);
                let mut rng_state: u64 = DEFAULT_SEED;
                for i in 0..n {
                    for _ in 0..num_dims {
                        points.push(lcg_next(&mut rng_state));
                    }
                    doc_ids.push(i as u64);
                }
                b.iter(|| {
                    let storage: Arc<MemoryStorage> =
                        Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
                    let path = "build.bkd";
                    let output = storage.create_output(path).unwrap();
                    let mut writer = BKDWriter::new(output, num_dims as u32);
                    writer.write(&points, &doc_ids).unwrap();
                    writer.finish().unwrap();
                    black_box(storage);
                });
            });
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_range_search,
    bench_intersect_counting,
    bench_build
);
criterion_main!(benches);
