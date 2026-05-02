//! Criterion benchmarks for the synonym dictionary.
//!
//! Measures the cost of [`SynonymDictionary::get_synonyms`] (per-token
//! lookup performed during analysis when synonym expansion is enabled) and
//! the build cost of the dictionary itself.
//!
//! # Scope
//!
//! - Single lookup at three dictionary sizes: 100, 1k, 10k synonym groups.
//! - Batch lookup of 100 distinct terms against the 10k dictionary.
//! - Dictionary build at 1k groups.
//! - Inputs are deterministic — group keys are `term_<i>` and lookups query
//!   known-present terms.
//!
//! # Run
//!
//! ```sh
//! cargo bench --bench synonym_bench
//! ```
//!
//! Filter by case:
//!
//! ```sh
//! cargo bench --bench synonym_bench -- lookup_large_10k
//! ```
//!
//! Compile-only smoke check:
//!
//! ```sh
//! cargo bench --bench synonym_bench --no-run
//! ```
//!
//! See `benches/common.rs` for the suite-wide hygiene rules.

mod common;

use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};

use laurus::analysis::synonym::dictionary::SynonymDictionary;

/// Build a [`SynonymDictionary`] populated with `num_groups` synonym groups.
/// Each group contains a primary term and two synonyms named deterministically.
fn create_test_dictionary(num_groups: usize) -> SynonymDictionary {
    let mut dict = SynonymDictionary::new(None).unwrap();
    for i in 0..num_groups {
        dict.add_synonym_group(vec![
            format!("term_{i}"),
            format!("synonym_a_{i}"),
            format!("synonym_b_{i}"),
        ]);
    }
    dict
}

fn bench_synonym_dictionary(c: &mut Criterion) {
    let mut group = c.benchmark_group("synonym_dictionary");

    let small_dict = create_test_dictionary(100);
    let medium_dict = create_test_dictionary(1000);
    let large_dict = create_test_dictionary(10_000);

    // One-time sanity check: a known-present term must return a non-empty
    // synonym list at every size.
    assert!(
        small_dict.get_synonyms("term_50").is_some(),
        "small_dict must contain term_50"
    );
    assert!(
        medium_dict.get_synonyms("term_500").is_some(),
        "medium_dict must contain term_500"
    );
    assert!(
        large_dict.get_synonyms("term_5000").is_some(),
        "large_dict must contain term_5000"
    );

    group.bench_function("lookup_small_100", |b| {
        b.iter(|| {
            let result = small_dict.get_synonyms(black_box("term_50"));
            black_box(result)
        })
    });

    group.bench_function("lookup_medium_1k", |b| {
        b.iter(|| {
            let result = medium_dict.get_synonyms(black_box("term_500"));
            black_box(result)
        })
    });

    group.bench_function("lookup_large_10k", |b| {
        b.iter(|| {
            let result = large_dict.get_synonyms(black_box("term_5000"));
            black_box(result)
        })
    });

    group.throughput(Throughput::Elements(100));
    group.bench_function("batch_lookup_100", |b| {
        b.iter(|| {
            for i in 0..100 {
                let term = format!("term_{i}");
                let result = large_dict.get_synonyms(black_box(&term));
                black_box(result);
            }
        })
    });

    group.bench_function("build_dict_1k", |b| {
        b.iter(|| {
            let dict = create_test_dictionary(1000);
            black_box(dict)
        })
    });

    group.finish();
}

criterion_group!(benches, bench_synonym_dictionary);
criterion_main!(benches);
