//! Issue #481 Stage 3 POC: brute-force Product Quantization (PQ) sweep
//! on SIFT.
//!
//! Validates whether PQ can reach the Stage 3 acceptance target
//! (`≥ 5× speedup at Recall@10 ≥ 0.95` vs the pre-Stage-1 f32 HNSW
//! baseline) **before** committing to a full implementation that
//! integrates PQ into the HNSW graph, extends the LVS1 segment format,
//! and wires every binding through.
//!
//! # Approach
//!
//! 1. Load SIFT (TEXMEX) `.fvecs` files. siftsmall for smoke, sift
//!    (1M / subsampled to 50k) for the Stage 3 acceptance grid.
//! 2. L2-normalise so Cosine ≈ L2² / 2 — the same trick laurus uses in
//!    `DistanceMetric::prepare_query` for Cosine on the int8 path.
//! 3. Train a per-segment PQ codebook (M sub-vectors × K = 256 centroids,
//!    Lloyd k-means with k-means++ init, 25 iterations).
//! 4. Encode every corpus vector to M bytes (M-byte PQ codes).
//! 5. For each query: build an ADC look-up table (M × K floats), score
//!    every candidate by `Σ_m lut[m][codes[m]]`, take top-10.
//! 6. Compare against brute-force f32 ground-truth top-10 for Recall@10
//!    and time the loop for latency.
//!
//! No HNSW graph involvement — this measures the ADC kernel's ceiling.
//! If even brute-force PQ falls short of the 5× / 0.95 target on real
//! data, integrating PQ into HNSW cannot rescue it; that signal is what
//! we want before committing to the full Stage 3 PR.
//!
//! # Usage
//!
//! ```sh
//! ./scripts/fetch-sift.sh --small               # siftsmall (~5 MB)
//! ./scripts/fetch-sift.sh --large               # SIFT1M  (~478 MB)
//! cargo run --release --example pq_brute_force_probe -- --dataset siftsmall
//! cargo run --release --example pq_brute_force_probe -- \
//!     --dataset sift --subsample 50000 --queries 200
//! ```
//!
//! Output cells are tagged `*PASS*` when **both** Recall@10 ≥ 0.95 and
//! the in-process brute-force-f32 speedup ≥ 5× hold. The brute-force-f32
//! baseline reported here is an absolute speed reference (recall = 1.0)
//! — for the Stage 3 acceptance gate we still need a cross-branch
//! Criterion measurement against the pre-Stage-1 f32 HNSW path, by
//! analogy with Issue #498.

use std::env;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::time::Instant;

const DIM: usize = 128;
const TOP_K: usize = 10;
const K: usize = 256;
const KMEANS_ITERS: usize = 25;

/// `.fvecs` reader.
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
        let dim = u32::from_le_bytes(hdr) as usize;
        assert_eq!(dim, expect_dim, "dim mismatch in {}", path.display());
        reader.read_exact(&mut vec_buf).expect("vec body");
        let mut v = Vec::with_capacity(dim);
        for chunk in vec_buf.as_chunks::<4>().0 {
            v.push(f32::from_le_bytes(*chunk));
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

/// L2-normalise so Cosine ≈ L2² / 2 over unit-norm inputs.
fn normalise(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Squared Euclidean distance between two slices of equal length.
#[inline]
fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut sum = 0.0_f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum
}

/// Cosine distance over normalised inputs: 1 - dot product (since L2 = 1).
#[inline]
fn cosine_distance_unit(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0_f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
    }
    1.0 - dot.clamp(-1.0, 1.0)
}

/// Brute-force Cosine top-K on unit-norm vectors.
fn exact_top_k(corpus: &[Vec<f32>], q: &[f32], k: usize) -> Vec<u64> {
    let mut scored: Vec<(u64, f32)> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u64, cosine_distance_unit(q, v)))
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().take(k).map(|(id, _)| id).collect()
}

/// Brute-force L2² top-K — used in `--no-normalise` mode where the
/// corpus stays in its native SIFT distribution and PQ runs natively
/// against L2 distance instead of via the Cosine = L2² / 2 identity.
fn exact_top_k_l2(corpus: &[Vec<f32>], q: &[f32], k: usize) -> Vec<u64> {
    let mut scored: Vec<(u64, f32)> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u64, l2_squared(q, v)))
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().take(k).map(|(id, _)| id).collect()
}

/// Recall@k between two top-k id lists.
fn recall_at_k(exact: &[u64], approx: &[u64], k: usize) -> f32 {
    let exact_set: std::collections::HashSet<u64> = exact.iter().copied().collect();
    let approx_set: std::collections::HashSet<u64> = approx.iter().copied().collect();
    exact_set.intersection(&approx_set).count() as f32 / k as f32
}

/// Deterministic LCG-based pseudo-random index pick from `[0, n)`.
///
/// Used for k-means++ initialisation only — the probe is reproducible
/// across machines so result tables compare cleanly.
fn lcg_pick(state: &mut u64, n: usize) -> usize {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*state >> 33) as usize) % n
}

/// k-means++ initialisation: pick `k` centroids spread across `data`
/// proportional to their squared distance to the already-picked set.
///
/// Returns `k` centroids (each `data[0].len()` long) as a 2D vector.
fn kmeans_pp_init(data: &[Vec<f32>], k: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut state = seed;
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(k);
    centroids.push(data[lcg_pick(&mut state, data.len())].clone());
    let mut min_d2: Vec<f32> = data.iter().map(|v| l2_squared(v, &centroids[0])).collect();
    for _ in 1..k {
        let total: f32 = min_d2.iter().sum();
        if total == 0.0 {
            // All points collapsed onto existing centroids; pick uniformly.
            centroids.push(data[lcg_pick(&mut state, data.len())].clone());
        } else {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let r = ((state >> 32) as f32 / u32::MAX as f32) * total;
            let mut acc = 0.0_f32;
            let mut chosen = data.len() - 1;
            for (i, &d) in min_d2.iter().enumerate() {
                acc += d;
                if acc >= r {
                    chosen = i;
                    break;
                }
            }
            centroids.push(data[chosen].clone());
            // Update min squared distance.
            let new = &centroids[centroids.len() - 1];
            for (i, v) in data.iter().enumerate() {
                let d = l2_squared(v, new);
                if d < min_d2[i] {
                    min_d2[i] = d;
                }
            }
        }
    }
    centroids
}

/// Lloyd iterations for a single sub-vector k-means run.
///
/// `data` is the sub-vector slice (each row `sub_dim` long). Returns
/// the trained centroids (`k` rows of `sub_dim` floats each).
fn kmeans_train(data: &[Vec<f32>], k: usize, iters: usize, seed: u64) -> Vec<Vec<f32>> {
    let sub_dim = data[0].len();
    let mut centroids = kmeans_pp_init(data, k, seed);
    for _ in 0..iters {
        // Assign every point to its nearest centroid.
        let mut sums: Vec<Vec<f32>> = vec![vec![0.0; sub_dim]; k];
        let mut counts: Vec<u32> = vec![0; k];
        for v in data {
            let mut best = 0usize;
            let mut best_d = l2_squared(v, &centroids[0]);
            for (j, c) in centroids.iter().enumerate().skip(1) {
                let d = l2_squared(v, c);
                if d < best_d {
                    best_d = d;
                    best = j;
                }
            }
            counts[best] += 1;
            for d in 0..sub_dim {
                sums[best][d] += v[d];
            }
        }
        // Recompute centroids; empty cells keep the previous centroid.
        for j in 0..k {
            if counts[j] > 0 {
                let inv = 1.0 / counts[j] as f32;
                for d in 0..sub_dim {
                    centroids[j][d] = sums[j][d] * inv;
                }
            }
        }
    }
    centroids
}

/// Train a full PQ codebook: M independent k-means on each sub-vector
/// stride. Returns `M × K × sub_dim` floats stored as `Vec<Vec<Vec<f32>>>`.
fn train_pq_codebook(corpus: &[Vec<f32>], m: usize, dim: usize) -> Vec<Vec<Vec<f32>>> {
    assert_eq!(dim % m, 0, "dim {dim} must be divisible by M {m}");
    let sub_dim = dim / m;
    let n = corpus.len();
    let mut codebook: Vec<Vec<Vec<f32>>> = Vec::with_capacity(m);
    for sub in 0..m {
        // Project corpus onto this sub-vector stride.
        let mut sub_data: Vec<Vec<f32>> = Vec::with_capacity(n);
        for v in corpus {
            sub_data.push(v[sub * sub_dim..(sub + 1) * sub_dim].to_vec());
        }
        // Mix M into the seed so each sub-vector gets a different init
        // sequence while remaining reproducible.
        let seed: u64 = 0xCAFE_F00D_DEAD_BEEF ^ ((sub as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let cents = kmeans_train(&sub_data, K, KMEANS_ITERS, seed);
        codebook.push(cents);
    }
    codebook
}

/// Encode one vector against a trained codebook: `M` byte codes (each
/// the index of the nearest centroid for that sub-vector).
fn encode(v: &[f32], codebook: &[Vec<Vec<f32>>], m: usize, dim: usize) -> Vec<u8> {
    let sub_dim = dim / m;
    let mut codes = Vec::with_capacity(m);
    for sub in 0..m {
        let q = &v[sub * sub_dim..(sub + 1) * sub_dim];
        let mut best = 0u8;
        let mut best_d = l2_squared(q, &codebook[sub][0]);
        for (j, c) in codebook[sub].iter().enumerate().skip(1) {
            let d = l2_squared(q, c);
            if d < best_d {
                best_d = d;
                best = j as u8;
            }
        }
        codes.push(best);
    }
    codes
}

/// Build the per-query ADC look-up table: `lut[m][k] = ||q_m -
/// codebook[m][k]||²`.
///
/// Stored row-major (`m * K + k`) so the hot path can `lut[m * K + code]`
/// without re-indexing.
fn build_lut(query: &[f32], codebook: &[Vec<Vec<f32>>], m: usize, dim: usize) -> Vec<f32> {
    let sub_dim = dim / m;
    let mut lut = vec![0.0_f32; m * K];
    for sub in 0..m {
        let q = &query[sub * sub_dim..(sub + 1) * sub_dim];
        for k in 0..K {
            lut[sub * K + k] = l2_squared(q, &codebook[sub][k]);
        }
    }
    lut
}

/// PQ ADC distance for one candidate: `Σ_m lut[m][codes[m]]`.
#[inline]
fn distance_pq(lut: &[f32], codes: &[u8], m: usize) -> f32 {
    let mut sum = 0.0_f32;
    for sub in 0..m {
        sum += lut[sub * K + codes[sub] as usize];
    }
    sum
}

/// Brute-force PQ top-K over the whole encoded corpus.
fn pq_search(encoded: &[Vec<u8>], lut: &[f32], m: usize, k: usize) -> Vec<u64> {
    let mut scored: Vec<(u64, f32)> = encoded
        .iter()
        .enumerate()
        .map(|(i, codes)| (i as u64, distance_pq(lut, codes, m)))
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().take(k).map(|(id, _)| id).collect()
}

/// Measure brute-force f32 latency over the same corpus — the absolute
/// reference speed for the *PASS* marker (5× of this is the in-process
/// upper bound; the Stage 3 gate compares against pre-Stage-1 f32 HNSW
/// via a separate cross-branch Criterion run).
fn measure_brute_force_f32_latency(corpus: &[Vec<f32>], queries: &[Vec<f32>], use_l2: bool) -> f64 {
    let start = Instant::now();
    let mut sink: u64 = 0;
    for q in queries {
        let mut scored: Vec<(u64, f32)> = corpus
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let d = if use_l2 {
                    l2_squared(q, v)
                } else {
                    cosine_distance_unit(q, v)
                };
                (i as u64, d)
            })
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        sink = sink.wrapping_add(scored[0].0);
    }
    std::hint::black_box(sink);
    start.elapsed().as_secs_f64() / queries.len() as f64 * 1e6
}

fn parse_args() -> (String, usize, usize, bool) {
    let mut dataset = String::from("siftsmall");
    let mut subsample: usize = 0;
    let mut n_queries: usize = 50;
    let mut skip_normalise = false;
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
            "--no-normalise" => {
                skip_normalise = true;
                i += 1;
            }
            other => panic!("unknown arg {other}"),
        }
    }
    (dataset, subsample, n_queries, skip_normalise)
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
    let (dataset, subsample, n_queries, skip_normalise) = parse_args();
    let (base_path, query_path) = dataset_paths(&dataset);

    println!(
        "=== Issue #481 Stage 3 PQ POC ===\n\
         dataset = {dataset}{}\n\
         base    = {}\n\
         query   = {}\n\
         dim     = {DIM} top_k = {TOP_K} K = {K} kmeans_iters = {KMEANS_ITERS} queries = {n_queries}",
        if skip_normalise {
            " (raw, no L2 normalise — native L2 distance)"
        } else {
            " (L2-normalised — Cosine via dot product)"
        },
        base_path.display(),
        query_path.display()
    );

    let cap = if subsample > 0 { Some(subsample) } else { None };
    let mut corpus = read_fvecs(&base_path, DIM, cap);
    let mut queries = read_fvecs(&query_path, DIM, Some(n_queries));
    if !skip_normalise {
        for v in corpus.iter_mut() {
            normalise(v);
        }
        for v in queries.iter_mut() {
            normalise(v);
        }
    }
    println!(
        "loaded corpus = {}  queries = {}",
        corpus.len(),
        queries.len()
    );

    println!("computing brute-force ground truth top-{TOP_K}...");
    let t0 = Instant::now();
    let truth: Vec<Vec<u64>> = if skip_normalise {
        queries
            .iter()
            .map(|q| exact_top_k_l2(&corpus, q, TOP_K))
            .collect()
    } else {
        queries
            .iter()
            .map(|q| exact_top_k(&corpus, q, TOP_K))
            .collect()
    };
    println!("ground truth built in {:.2}s", t0.elapsed().as_secs_f64());

    println!("measuring brute-force f32 baseline latency...");
    let bf_us = measure_brute_force_f32_latency(&corpus, &queries, skip_normalise);
    println!("  brute-force f32: {bf_us:>9.2} µs/query (in-process baseline)");

    println!(
        "\nsweep cells (M, sub_dim) — K = {K}, k-means iters = {KMEANS_ITERS}, top-{TOP_K}, n = {}:",
        corpus.len()
    );
    println!(
        "  {:>3}  {:>7}  {:>9}  {:>11}  {:>11}  {:>9}",
        "M", "sub_dim", "train s", "encode ms", "search µs", "recall@10"
    );

    for &m in &[8usize, 16, 32, 64] {
        assert_eq!(DIM % m, 0);
        let sub_dim = DIM / m;

        let t0 = Instant::now();
        let codebook = train_pq_codebook(&corpus, m, DIM);
        let train_s = t0.elapsed().as_secs_f64();

        let t0 = Instant::now();
        let encoded: Vec<Vec<u8>> = corpus
            .iter()
            .map(|v| encode(v, &codebook, m, DIM))
            .collect();
        let encode_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Warm-up
        for q in queries.iter().take(3) {
            let lut = build_lut(q, &codebook, m, DIM);
            let _ = pq_search(&encoded, &lut, m, TOP_K);
        }

        let mut total_recall = 0.0_f32;
        let t0 = Instant::now();
        for (qi, q) in queries.iter().enumerate() {
            let lut = build_lut(q, &codebook, m, DIM);
            let approx = pq_search(&encoded, &lut, m, TOP_K);
            total_recall += recall_at_k(&truth[qi], &approx, TOP_K);
        }
        let search_us = t0.elapsed().as_secs_f64() / queries.len() as f64 * 1e6;
        let recall = total_recall / queries.len() as f32;

        let speedup = bf_us / search_us;
        let mark = if recall >= 0.95 && speedup >= 5.0 {
            " *PASS*"
        } else if recall >= 0.95 {
            " recall ok"
        } else if speedup >= 5.0 {
            " speed ok"
        } else {
            ""
        };

        println!(
            "  {m:>3}  {sub_dim:>7}  {train_s:>9.2}  {encode_ms:>11.2}  {search_us:>11.2}  {recall:>9.4} \
             ({speedup:.2}x f32 brute){mark}"
        );
    }

    println!(
        "\n(speedup column is vs the in-process brute-force f32 baseline above. \
         For the Stage 3 acceptance gate, a cross-branch Criterion run against \
         the pre-Stage-1 f32 HNSW path is needed — see Issue #498 for the \
         worktree protocol.)"
    );
}
