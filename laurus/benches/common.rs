//! Shared utilities and hygiene policy for the laurus benchmark suite.
//!
//! This module is included by every bench file via `mod common;` so the suite
//! shares a single source of truth for deterministic randomness, sample sizes,
//! and the contract every bench is expected to follow.
//!
//! # Hygiene rules (apply to every bench file in this directory)
//!
//! 1. **Deterministic input**: never use `rand::rng()` (OS-seeded). Use the
//!    LCG helpers in this module so two consecutive runs produce comparable
//!    numbers.
//! 2. **File-level doc comment**: each bench file starts with a `//!` block
//!    listing scope, scenarios, how to run, and how to filter.
//! 3. **One-time sanity assert**: each top-level bench function calls the
//!    measured code once before `b.iter` and asserts on the shape of the
//!    result (e.g. result count > 0). This catches regressions that produce
//!    empty output without affecting the timed loop.
//! 4. **`sample_size` policy**: use [`SAMPLE_SIZE_FAST`] for cheap operations
//!    (sub-50 ms per iter) and [`SAMPLE_SIZE_SLOW`] for slow construction
//!    paths. Pick one of the two; do not invent intermediate values.
//!
//! # Recommended environment
//!
//! For stable measurements, run benches with the CPU governor set to
//! `performance` and turbo boost disabled. Background load skews
//! short-running benches significantly. If a bench's two consecutive runs
//! diverge by more than 5 %, suspect environment noise before code.
//!
//! # Suppress unused warnings
//!
//! Each bench is its own compile unit and may use only a subset of these
//! helpers. `#![allow(dead_code)]` keeps clippy quiet across the suite.

#![allow(dead_code)]

use std::sync::Arc;

use laurus::storage::file::FileStorageConfig;
use laurus::storage::memory::MemoryStorageConfig;
use laurus::storage::{Storage, StorageConfig, StorageFactory};

/// Default seed for deterministic LCG state. Used as the starting point in
/// every bench so two runs of `cargo bench` produce identical inputs.
pub const DEFAULT_SEED: u64 = 0xDEAD_BEEF_CAFE_F00D;

/// `sample_size` for fast operations (search, distance, scoring loops). This
/// is Criterion's default; spell it out explicitly so the policy is visible
/// at the call site.
pub const SAMPLE_SIZE_FAST: usize = 100;

/// `sample_size` for slow construction paths (HNSW build, IVF training,
/// engine population at large scale). Lowered from the default to keep
/// `cargo bench` runtimes manageable.
pub const SAMPLE_SIZE_SLOW: usize = 10;

/// Inline LCG (numerical recipes constants) advancing `state` by one step
/// and returning a deterministic value in `[0, 1000)`.
///
/// Used in places where the bench wants drop-in replacement for the existing
/// `[0, 1000)` data range (e.g. BKD point coordinates).
pub fn lcg_next(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 32) as u32;
    (bits as f64) * (1000.0 / (u32::MAX as f64))
}

/// Select the storage backend for a bench based on `LAURUS_BENCH_DISK`.
///
/// - **Default (env unset)**: returns an in-memory storage. This is what
///   every bench used before #444 and is the right choice for
///   microbenchmarks that want to remove I/O variance.
/// - **`LAURUS_BENCH_DISK=1`**: returns a file-backed storage rooted in a
///   freshly-created temp directory. Each call yields a distinct
///   directory so concurrent benches do not collide.
///
/// The temp directory created in the disk-backed branch is intentionally
/// **leaked** — `tempfile::TempDir::keep` keeps it on disk after the
/// returned `Storage` is dropped. Bench runs accumulate a handful of
/// directories under `$TMPDIR` and rely on OS-level `/tmp` cleanup
/// (`systemd-tmpfiles`, reboots, etc.) rather than per-iteration cleanup.
/// This is acceptable for benchmark runs and avoids the lifetime
/// gymnastics of returning `(Storage, TempDir)` everywhere.
///
/// Disk numbers are sensitive to the host filesystem and OS page cache.
/// They are useful for **comparing** in-tree changes (e.g. before / after
/// a perf PR), not for absolute throughput claims. Document this caveat
/// in any PR that posts numbers from `LAURUS_BENCH_DISK=1`.
pub fn select_storage() -> Arc<dyn Storage> {
    if std::env::var("LAURUS_BENCH_DISK").is_ok() {
        let dir = tempfile::tempdir().expect("create temp dir for disk-backed bench storage");
        let path = dir.keep();
        let config = FileStorageConfig::new(path);
        StorageFactory::create(StorageConfig::File(config))
            .expect("instantiate file-backed storage for bench")
    } else {
        StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))
            .expect("instantiate memory-backed storage for bench")
    }
}

/// Inline LCG variant returning a deterministic `f32` in `[0, 1)`.
///
/// Suited to vector-component generation (cosine / dot-product workloads
/// expect bounded magnitudes; the `[0, 1)` range mirrors what `rand::rng()`
/// previously produced via the `Rng::random::<f32>()` call).
pub fn lcg_next_unit(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 32) as u32;
    (bits as f32) / (u32::MAX as f32)
}

/// Generate a deterministic `Vec<f32>` of length `dim` advancing the
/// caller-provided LCG state. Components are in `[0, 1)`.
pub fn lcg_vec_unit(state: &mut u64, dim: usize) -> Vec<f32> {
    (0..dim).map(|_| lcg_next_unit(state)).collect()
}
