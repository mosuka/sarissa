//! Issue #498: reproducible SIFT (TEXMEX) sweep for Stage 2 rerank.
//!
//! Loads a SIFT base / query pair, computes brute-force Cosine
//! ground-truth top-10, and runs a `(ef_search × rerank_factor × HNSW
//! config)` sweep over the int8 + rerank Stage 2 path. Each cell
//! reports Recall@10 and average per-query latency.
//!
//! This example is committed so users (and CI machines) can reproduce
//! the Issue #498 Phase 0 evidence on their own hardware without
//! standing up an external ANN benchmarking framework.
//!
//! # Usage
//!
//! ```sh
//! ./scripts/fetch-sift.sh --small               # fetch siftsmall (~5MB)
//! ./scripts/fetch-sift.sh --large               # fetch SIFT1M  (~478MB)
//! cargo run --release --example sift_rerank_probe -- --dataset siftsmall
//! cargo run --release --example sift_rerank_probe -- --dataset sift --subsample 50000
//! ```
//!
//! Output cells are tagged `*PASS*` when **both** Recall@10 ≥ 0.99 and
//! the in-process brute-force-f32 speedup ≥ 3× hold. The brute-force-
//! f32 number is an absolute baseline (recall = 1.0); the Issue #498
//! acceptance gate (≥ 1.5× vs the pre-Stage-1 main commit's HNSW f32
//! path) is measured separately via `cargo bench` in the
//! `vector_search_bench` suite — see the docs for the cross-branch
//! protocol.

use std::env;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::time::Instant;

use laurus::storage::StorageConfig;
use laurus::storage::StorageFactory;
use laurus::storage::memory::MemoryStorageConfig;
use laurus::vector::core::distance::DistanceMetric;
use laurus::vector::core::rerank::RerankStorageKind;
use laurus::vector::core::vector::Vector;
use laurus::vector::index::VectorIndex;
use laurus::vector::index::config::HnswIndexConfig;
use laurus::vector::index::hnsw::HnswIndex;
use laurus::vector::index::hnsw::searcher::HnswSearcher;
use laurus::vector::search::searcher::{VectorIndexQuery, VectorIndexSearcher};

const DIM: usize = 128;
const TOP_K: usize = 10;
/// Subsample the queries for the sweep so iteration finishes fast.
/// Full siftsmall query set is 100; SIFT1M is 10 000. 50 queries are
/// enough to detect a recall surface that meets the 0.99 gate.
const N_QUERIES_DEFAULT: usize = 50;

/// `.fvecs` format: per vector, [dim: u32 LE][values: f32 LE × dim].
fn read_fvecs(path: &Path, expect_dim: usize, max: Option<usize>) -> Vec<Vec<f32>> {
    let file = File::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    let mut reader = BufReader::new(file);
    let mut out = Vec::new();
    let mut hdr = [0u8; 4];
    let mut vec_buf = vec![0u8; expect_dim * 4];
    loop {
        if reader.read_exact(&mut hdr).is_err() {
            break;
        }
        let dim = i32::from_le_bytes(hdr) as usize;
        assert_eq!(dim, expect_dim, "dim mismatch in {}", path.display());
        reader.read_exact(&mut vec_buf).expect("vec body");
        let mut v = Vec::with_capacity(dim);
        for chunk in vec_buf.chunks_exact(4) {
            v.push(f32::from_le_bytes(chunk.try_into().unwrap()));
        }
        out.push(v);
        if let Some(cap) = max
            && out.len() >= cap
        {
            break;
        }
    }
    out
}

/// L2-normalise so Cosine distance is well-defined; SIFT vectors are
/// non-negative integer histograms that have non-zero norms.
fn normalise(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

fn exact_cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0_f32;
    let mut na = 0.0_f32;
    let mut nb = 0.0_f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom == 0.0 {
        1.0
    } else {
        let cos = (dot / denom).clamp(-1.0, 1.0);
        1.0 - cos
    }
}

fn exact_top_k(corpus: &[Vec<f32>], q: &[f32], k: usize) -> Vec<u64> {
    let mut scored: Vec<(u64, f32)> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u64, exact_cosine_distance(q, v)))
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().take(k).map(|(id, _)| id).collect()
}

fn recall_at_k(exact: &[u64], approx: &[u64], k: usize) -> f32 {
    let exact_set: std::collections::HashSet<u64> = exact.iter().copied().collect();
    let approx_set: std::collections::HashSet<u64> = approx.iter().copied().collect();
    exact_set.intersection(&approx_set).count() as f32 / k as f32
}

struct Cell {
    label: &'static str,
    ef_search: usize,
    rerank_factor: Option<usize>,
    m: usize,
    ef_construction: usize,
}

fn build_index(corpus: &[Vec<f32>], rerank: bool, m: usize, ef_construction: usize) -> HnswIndex {
    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))
        .expect("memory storage");
    let config = HnswIndexConfig {
        dimension: DIM,
        m,
        ef_construction,
        distance_metric: DistanceMetric::Cosine,
        rerank_storage: if rerank {
            Some(RerankStorageKind::F32)
        } else {
            None
        },
        ..Default::default()
    };
    let index = HnswIndex::create(storage, "probe_index", config).expect("create index");
    let mut writer = index.writer().expect("writer");
    let docs: Vec<(u64, String, Vector)> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u64, "embedding".to_string(), Vector::new(v.clone())))
        .collect();
    writer.build(docs).expect("build");
    writer.finalize().expect("finalize");
    writer.commit().expect("commit");
    index
}

fn measure_brute_force_f32_latency(corpus: &[Vec<f32>], queries: &[Vec<f32>]) -> f64 {
    let start = Instant::now();
    let mut sink: u64 = 0;
    for q in queries {
        let mut scored: Vec<(u64, f32)> = corpus
            .iter()
            .enumerate()
            .map(|(i, v)| (i as u64, exact_cosine_distance(q, v)))
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        // Touch the top result so the optimiser can't elide work.
        sink = sink.wrapping_add(scored[0].0);
    }
    let elapsed = start.elapsed().as_secs_f64() / queries.len() as f64 * 1e6;
    // Use sink so the loop is not optimised out.
    std::hint::black_box(sink);
    elapsed
}

fn measure_cell(
    corpus: &[Vec<f32>],
    queries: &[Vec<f32>],
    exact_truth: &[Vec<u64>],
    cell: &Cell,
) -> (f32, f64) {
    let rerank = cell.rerank_factor.is_some();
    let index = build_index(corpus, rerank, cell.m, cell.ef_construction);
    let reader = index.reader().expect("reader");
    let mut searcher = HnswSearcher::new(reader).expect("searcher");
    searcher.set_ef_search(cell.ef_search);

    // Warm-up
    for q in queries.iter().take(3) {
        let mut req = VectorIndexQuery::new(Vector::new(q.clone()))
            .top_k(TOP_K)
            .field_name("embedding".to_string());
        if let Some(rf) = cell.rerank_factor {
            req = req.rerank_factor(rf);
        }
        let _ = searcher.search(&req).expect("warmup");
    }

    let mut total_recall = 0.0_f32;
    let start = Instant::now();
    for (qi, q) in queries.iter().enumerate() {
        let mut req = VectorIndexQuery::new(Vector::new(q.clone()))
            .top_k(TOP_K)
            .field_name("embedding".to_string());
        if let Some(rf) = cell.rerank_factor {
            req = req.rerank_factor(rf);
        }
        let res = searcher.search(&req).expect("search");
        let approx: Vec<u64> = res.results.iter().map(|r| r.doc_id).collect();
        total_recall += recall_at_k(&exact_truth[qi], &approx, TOP_K);
    }
    let elapsed_us = start.elapsed().as_secs_f64() / queries.len() as f64 * 1e6;
    let recall = total_recall / queries.len() as f32;
    (recall, elapsed_us)
}

fn parse_args() -> (String, usize, usize) {
    let mut dataset = String::from("siftsmall");
    let mut subsample: usize = 0;
    let mut n_queries = N_QUERIES_DEFAULT;
    let args: Vec<String> = env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--dataset" => {
                dataset = args[i + 1].clone();
                i += 2;
            }
            "--subsample" => {
                subsample = args[i + 1].parse().expect("subsample u64");
                i += 2;
            }
            "--queries" => {
                n_queries = args[i + 1].parse().expect("queries u64");
                i += 2;
            }
            other => panic!("unknown arg {other}"),
        }
    }
    (dataset, subsample, n_queries)
}

fn dataset_paths(name: &str) -> (PathBuf, PathBuf) {
    let cache = Path::new("./.cache/sift").to_path_buf();
    match name {
        "siftsmall" => (
            cache.join("siftsmall/siftsmall_base.fvecs"),
            cache.join("siftsmall/siftsmall_query.fvecs"),
        ),
        "sift" => (
            cache.join("sift/sift_base.fvecs"),
            cache.join("sift/sift_query.fvecs"),
        ),
        other => panic!("unknown dataset {other}"),
    }
}

fn main() {
    let (dataset, subsample, n_queries) = parse_args();
    let (base_path, query_path) = dataset_paths(&dataset);

    println!(
        "=== Issue #498 Phase 0 probe ===\n\
         dataset = {dataset}\n\
         base    = {}\n\
         query   = {}\n\
         dim     = {DIM} top_k = {TOP_K} queries = {n_queries}",
        base_path.display(),
        query_path.display()
    );

    let cap = if subsample > 0 { Some(subsample) } else { None };
    let mut corpus = read_fvecs(&base_path, DIM, cap);
    let mut queries = read_fvecs(&query_path, DIM, Some(n_queries));
    for v in corpus.iter_mut() {
        normalise(v);
    }
    for v in queries.iter_mut() {
        normalise(v);
    }
    println!(
        "loaded corpus = {}  queries = {}",
        corpus.len(),
        queries.len()
    );

    println!("computing brute-force Cosine ground truth top-{TOP_K}...");
    let start = Instant::now();
    let truth: Vec<Vec<u64>> = queries
        .iter()
        .map(|q| exact_top_k(&corpus, q, TOP_K))
        .collect();
    println!(
        "ground truth built in {:.2}s",
        start.elapsed().as_secs_f64()
    );

    println!("measuring brute-force f32 baseline latency...");
    let bf_us = measure_brute_force_f32_latency(&corpus, &queries);
    println!("  brute-force f32: {bf_us:>9.2} µs/query (absolute baseline, exact recall)");

    println!("\nsweep cells (m, efc, ef_search, rerank_factor):");
    let cells = vec![
        // m=16 ef_construction=200 (default HNSW config), no rerank
        Cell {
            label: "m16efc200 no-rerank        ",
            ef_search: 200,
            rerank_factor: None,
            m: 16,
            ef_construction: 200,
        },
        Cell {
            label: "m16efc200 no-rerank        ",
            ef_search: 400,
            rerank_factor: None,
            m: 16,
            ef_construction: 200,
        },
        // m=16 ef_construction=200, with rerank
        Cell {
            label: "m16efc200 rerank=5         ",
            ef_search: 50,
            rerank_factor: Some(5),
            m: 16,
            ef_construction: 200,
        },
        Cell {
            label: "m16efc200 rerank=5         ",
            ef_search: 100,
            rerank_factor: Some(5),
            m: 16,
            ef_construction: 200,
        },
        Cell {
            label: "m16efc200 rerank=5         ",
            ef_search: 200,
            rerank_factor: Some(5),
            m: 16,
            ef_construction: 200,
        },
        Cell {
            label: "m16efc200 rerank=10        ",
            ef_search: 100,
            rerank_factor: Some(10),
            m: 16,
            ef_construction: 200,
        },
        Cell {
            label: "m16efc200 rerank=10        ",
            ef_search: 200,
            rerank_factor: Some(10),
            m: 16,
            ef_construction: 200,
        },
        // m=32 ef_construction=500 (stronger graph), with rerank
        Cell {
            label: "m32efc500 rerank=5         ",
            ef_search: 50,
            rerank_factor: Some(5),
            m: 32,
            ef_construction: 500,
        },
        Cell {
            label: "m32efc500 rerank=5         ",
            ef_search: 100,
            rerank_factor: Some(5),
            m: 32,
            ef_construction: 500,
        },
        Cell {
            label: "m32efc500 rerank=5         ",
            ef_search: 200,
            rerank_factor: Some(5),
            m: 32,
            ef_construction: 500,
        },
        Cell {
            label: "m32efc500 rerank=10        ",
            ef_search: 50,
            rerank_factor: Some(10),
            m: 32,
            ef_construction: 500,
        },
        Cell {
            label: "m32efc500 rerank=10        ",
            ef_search: 100,
            rerank_factor: Some(10),
            m: 32,
            ef_construction: 500,
        },
        // Strong graph, no rerank (control)
        Cell {
            label: "m32efc500 no-rerank        ",
            ef_search: 200,
            rerank_factor: None,
            m: 32,
            ef_construction: 500,
        },
    ];

    println!(
        "  {:<30}  {:>9}  {:>11}  {:>9}  {:>11}  {:>9}",
        "config", "ef_search", "rerank_fac", "recall@10", "lat µs/qry", "speedup"
    );
    for c in &cells {
        let (recall, us) = measure_cell(&corpus, &queries, &truth, c);
        let rf = c
            .rerank_factor
            .map(|r| r.to_string())
            .unwrap_or_else(|| "—".to_string());
        let speedup = bf_us / us;
        let recall_ok = recall >= 0.99;
        let speed_ok = speedup >= 3.0;
        let mark = match (recall_ok, speed_ok) {
            (true, true) => " *PASS*",
            (true, false) => " recall ok",
            (false, true) => " speed ok",
            _ => "",
        };
        println!(
            "  {label:<30}  {ef:>9}  {rf:>11}  {recall:>9.4}  {us:>11.2}  {speedup:>8.2}×{mark}",
            label = c.label,
            ef = c.ef_search,
            rf = rf,
            recall = recall,
            us = us,
            speedup = speedup,
        );
    }
    println!(
        "\n(speedup is vs brute-force f32 above; cross-branch f32 HNSW \
         baseline must be taken separately on pre-Stage-1 main — see \
         `bench_hnsw_graph_search_rerank_real_data` for the Criterion \
         protocol.)"
    );
}
