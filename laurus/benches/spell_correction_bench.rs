//! Criterion benchmarks for the spelling corrector.
//!
//! Measures the per-call cost of [`SpellingCorrector::correct`] for typical
//! single-word and small-batch misspelling inputs.
//!
//! # Scope
//!
//! - Single misspelled word (`correct_single_word`).
//! - Small batch of 5 common misspellings (`correct_batch_words`,
//!   `Throughput::Elements`).
//! - Each timed iteration receives a **fresh** [`SpellingCorrector`] via
//!   [`Bencher::iter_batched`]. This is the **cold-state** measurement —
//!   `query_history` accumulation across queries is not in scope here. For a
//!   warm-state measurement (corrector pre-populated with N queries), a
//!   separate benchmark would need to pre-warm the corrector before timing.
//! - The previous shared-`&mut SpellingCorrector` pattern in `bench.rs`
//!   conflated cold and warm-cache paths because `correct()` mutates
//!   `query_history` on every call.
//!
//! # Run
//!
//! ```sh
//! cargo bench --bench spell_correction_bench
//! ```
//!
//! Filter by case:
//!
//! ```sh
//! cargo bench --bench spell_correction_bench -- correct_single_word
//! ```
//!
//! Compile-only smoke check:
//!
//! ```sh
//! cargo bench --bench spell_correction_bench --no-run
//! ```
//!
//! See `benches/common.rs` for the suite-wide hygiene rules.

mod common;

use std::hint::black_box;

use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};

use laurus::spelling::corrector::SpellingCorrector;

fn bench_spell_correction(c: &mut Criterion) {
    let mut group = c.benchmark_group("spell_correction");

    let misspellings = ["searc", "engin", "documnet", "qurey", "algortihm"];

    // One-time sanity check: a fresh corrector must produce a non-trivial
    // CorrectionResult for the probe word. If this fires, the corrector
    // initial state is broken.
    let probe = SpellingCorrector::new().correct("searc");
    assert!(
        !probe.original.is_empty(),
        "spelling probe must populate original query"
    );

    group.bench_function("correct_single_word", |b| {
        b.iter_batched(
            SpellingCorrector::new,
            |mut corrector| {
                let result = corrector.correct(black_box("searc"));
                black_box(result);
            },
            BatchSize::PerIteration,
        );
    });

    group.throughput(Throughput::Elements(misspellings.len() as u64));
    group.bench_function("correct_batch_words", |b| {
        b.iter_batched(
            SpellingCorrector::new,
            |mut corrector| {
                for word in &misspellings {
                    let result = corrector.correct(black_box(word));
                    black_box(result);
                }
            },
            BatchSize::PerIteration,
        );
    });

    group.finish();
}

criterion_group!(benches, bench_spell_correction);
criterion_main!(benches);
