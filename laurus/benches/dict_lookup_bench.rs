//! Criterion benchmarks for the lexical term dictionary.
//!
//! Exercises the active term dictionary implementation at scales
//! (10k / 100k / 1M unique terms) that the rest of the bench suite
//! does not reach.
//!
//! Issue #487 ports the dictionary from a parallel-array
//! `(Vec<String>, Vec<TermInfo>)` representation
//! (`HybridTermDictionary`, removed in Phase 9) to a Lucene
//! `BlockTreeTermsWriter`-style block + FST structure
//! ([`BlockTermDictionary`]). This bench is the
//! before-and-after measurement: same probes, same term corpus,
//! comparing the new dictionary against the `pre-fst-port`
//! criterion baseline saved on `main`.
//!
//! # Scope
//!
//! Three access patterns, three dictionary scales:
//!
//! | Group                              | Pattern                                   |
//! |------------------------------------|-------------------------------------------|
//! | `lexical/dict_lookup/get_hit`      | exact-match lookup on a term in the dict  |
//! | `lexical/dict_lookup/get_miss`     | exact-match lookup on a term **not** in dict |
//! | `lexical/dict_lookup/iter`         | full sequential scan via `iter()`         |
//! | `lexical/dict_lookup/find_prefix`  | prefix scan with one- and two-letter prefixes |
//!
//! # Corpus
//!
//! Terms are 5–10 byte ASCII strings drawn from `[a-z]` by the shared LCG
//! (deterministic across runs — see [`crate::common`]). Duplicates are
//! discarded via `BTreeSet` so the resulting dictionaries hold exactly
//! `n` unique terms. The distribution is uniform random; this is enough
//! to expose the structural cost difference between the current
//! representation and a future BlockTreeTerms-style port without
//! requiring a real-corpus fixture.
//!
//! # Run
//!
//! Whole suite:
//!
//! ```sh
//! cargo bench --bench dict_lookup_bench
//! ```
//!
//! Single group:
//!
//! ```sh
//! cargo bench --bench dict_lookup_bench -- "lexical/dict_lookup/get_hit"
//! ```
//!
//! Single corpus size:
//!
//! ```sh
//! cargo bench --bench dict_lookup_bench -- "lexical/dict_lookup/iter/100000"
//! ```
//!
//! For perf-PR comparison runs use `--save-baseline` / `--baseline` —
//! the global `SAMPLE_SIZE_SLOW = 30` (PR #486) gives a within-run IQR
//! of roughly ±1 % at the larger sizes here.

#![allow(dead_code)]

mod common;

use std::collections::BTreeSet;
use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use common::{DEFAULT_SEED, SAMPLE_SIZE_SLOW};

use laurus::lexical::index::structures::dictionary::{
    BlockTermDictionary, TermDictionaryBuilder, TermInfo,
};

/// Corpus sizes the dict bench sweeps. Chosen to bracket the crossover
/// where BlockTreeTerms-style indirection starts to dominate the
/// current parallel-array representation:
///
/// - `10_000` — small corpus, baseline win for a flat sorted Vec.
/// - `100_000` — typical mid-sized index; expected crossover region.
/// - `1_000_000` — Lucene's documented sweet spot for FST + block-tree.
/// - `10_000_000` — production-target scale (10M+ terms / segment) for
///   #487 evaluation. Adds ~1 minute per group to the run time.
const CORPUS_SIZES: &[usize] = &[10_000, 100_000, 1_000_000, 10_000_000];

/// Number of probe terms drawn for the `get_*` and prefix microbenches.
/// Picked at 100 so each `b.iter` body amortises the criterion harness
/// overhead across a meaningful number of dict ops.
const PROBE_COUNT: usize = 100;

/// LCG step used by the bench fixture. Mirrors [`common::lcg_next`] but
/// returns the raw `u32` upper half for byte-by-byte ASCII generation.
fn lcg_next_u32(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 32) as u32
}

/// Generate a deterministic ASCII term of length in `min_len..=max_len`.
/// Characters are uniformly sampled from `[a-z]`. The LCG state is
/// advanced once for the length and once per character so the corpus
/// is reproducible across runs.
fn lcg_term(state: &mut u64, min_len: usize, max_len: usize) -> String {
    let span = (max_len - min_len + 1) as u32;
    let len = min_len + (lcg_next_u32(state) % span) as usize;
    let mut s = String::with_capacity(len);
    for _ in 0..len {
        let c = b'a' + (lcg_next_u32(state) as u8 % 26);
        s.push(c as char);
    }
    s
}

/// Build a dictionary of exactly `n` unique terms plus `probe_count`
/// hit-path probes drawn from the dictionary.
///
/// Returns the constructed dictionary and a `Vec<String>` of probe
/// terms (each guaranteed to be present in the dictionary). Probes are
/// spaced at roughly even ordinals so the lookup pattern doesn't
/// concentrate on a single hash bucket / FST branch.
fn build_dict_with_hit_probes(n: usize, probe_count: usize) -> (BlockTermDictionary, Vec<String>) {
    assert!(n >= probe_count, "n must be at least probe_count");

    // Generate `n` unique terms via BTreeSet (dedupes the LCG output).
    let mut state = DEFAULT_SEED;
    let mut terms: BTreeSet<String> = BTreeSet::new();
    while terms.len() < n {
        terms.insert(lcg_term(&mut state, 5, 10));
    }
    let terms: Vec<String> = terms.into_iter().collect();

    // Build the dictionary. Each term gets a synthetic TermInfo whose
    // `posting_offset` is its sort ordinal (×16) — the absolute values
    // don't matter for lookup benches, but giving each entry a unique
    // payload guards against the optimizer collapsing reads.
    let mut builder = TermDictionaryBuilder::new();
    for (i, term) in terms.iter().enumerate() {
        builder.add_term(term.clone(), TermInfo::new(i as u64 * 16, 64, 1, 1));
    }
    let dict = builder.build().expect("build BlockTermDictionary");

    // Probe sampling: spread evenly across the dictionary by ordinal so
    // every probe lands at a different position in the sorted layout.
    let step = n / probe_count.max(1);
    let probes: Vec<String> = (0..probe_count)
        .map(|i| terms[(i * step).min(n - 1)].clone())
        .collect();

    (dict, probes)
}

/// Generate `count` terms guaranteed not to appear in
/// [`build_dict_with_hit_probes`]'s output: every term begins with
/// `'~'` (`0x7E`), which sorts above the entire `[a-z]` range used by
/// the hit-path corpus.
fn miss_probes(count: usize) -> Vec<String> {
    (0..count).map(|i| format!("~miss_{i:08}")).collect()
}

fn bench_get_hit(c: &mut Criterion) {
    let mut group = c.benchmark_group("lexical/dict_lookup/get_hit");
    group.sample_size(SAMPLE_SIZE_SLOW);

    for &n in CORPUS_SIZES {
        let (dict, probes) = build_dict_with_hit_probes(n, PROBE_COUNT);
        // Sanity: every probe must hit. Caught here so a regression in
        // the fixture surfaces before the bench loop muddies the
        // diagnostic.
        for term in &probes {
            assert!(
                dict.get(term).is_some(),
                "hit probe {term:?} unexpectedly missed at n={n}"
            );
        }

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                for term in &probes {
                    black_box(dict.get(term));
                }
            });
        });
    }
    group.finish();
}

fn bench_get_miss(c: &mut Criterion) {
    let mut group = c.benchmark_group("lexical/dict_lookup/get_miss");
    group.sample_size(SAMPLE_SIZE_SLOW);

    let probes = miss_probes(PROBE_COUNT);
    for &n in CORPUS_SIZES {
        let (dict, _) = build_dict_with_hit_probes(n, 1);
        for term in &probes {
            assert!(
                dict.get(term).is_none(),
                "miss probe {term:?} unexpectedly hit at n={n}"
            );
        }

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                for term in &probes {
                    black_box(dict.get(term));
                }
            });
        });
    }
    group.finish();
}

fn bench_iter(c: &mut Criterion) {
    let mut group = c.benchmark_group("lexical/dict_lookup/iter");
    group.sample_size(SAMPLE_SIZE_SLOW);

    for &n in CORPUS_SIZES {
        let (dict, _) = build_dict_with_hit_probes(n, 1);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let mut count = 0u64;
                for (_, info) in dict.iter() {
                    // Touch each entry so the optimizer can't elide
                    // the iteration body.
                    count = count.wrapping_add(info.doc_frequency);
                }
                black_box(count)
            });
        });
    }
    group.finish();
}

fn bench_find_prefix(c: &mut Criterion) {
    let mut group = c.benchmark_group("lexical/dict_lookup/find_prefix");
    group.sample_size(SAMPLE_SIZE_SLOW);

    for &n in CORPUS_SIZES {
        let (dict, _) = build_dict_with_hit_probes(n, 1);

        // 1-letter prefix on a uniform `[a-z]` corpus matches ~1/26 of
        // the dict — gives a sense of large-result-set scan cost.
        group.bench_with_input(BenchmarkId::new("one_letter_a", n), &n, |b, _| {
            b.iter(|| {
                let results = dict.find_prefix("a");
                black_box(results.len())
            });
        });

        // 2-letter prefix narrows the result set to ~1/676 of the
        // dict — closer to a realistic prefix-query workload.
        group.bench_with_input(BenchmarkId::new("two_letter_ab", n), &n, |b, _| {
            b.iter(|| {
                let results = dict.find_prefix("ab");
                black_box(results.len())
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_get_hit,
    bench_get_miss,
    bench_iter,
    bench_find_prefix,
);
criterion_main!(benches);
