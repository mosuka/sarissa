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
//! Keep `cargo bench` as the default invocation; the bumped
//! [`SAMPLE_SIZE_SLOW`] (30 samples — Criterion's documented minimum for a
//! reliable t-test) is what brings the within-run interquartile spread on
//! `topk_or_skewed_tf/should_or_topk10/100000` down from "noise dominates"
//! to roughly ±1 % of the median. That is enough for `--baseline` runs to
//! tell real changes from jitter on a typical desktop without further
//! intervention.
//!
//! For micro-isolation (e.g. tracking a single hot loop where a 2-3 % win is
//! the headline), the optional wrapper at `scripts/bench-stable.sh` pins the
//! cargo bench process to one CPU and raises its priority. Caveat: pinning
//! constrains any parallel work the bench fixture does to the chosen core,
//! so the absolute timings shift relative to unpinned runs and historical
//! baselines saved without pinning are not directly comparable. Use it only
//! when the extra control is worth losing that comparability — for the
//! perf-PR style baselines this suite is built around, plain
//! `cargo bench` is the right tool.
//!
//! Other knobs that help when even that isn't enough: set the CPU governor
//! to `performance`, disable turbo boost (so the chip can't drop a sample
//! into a thermally-throttled window), and stop browsers / IDEs before
//! kicking off a comparison run.
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
/// engine population at large scale, top-K queries on ≥ 10k corpora).
/// Sized at 30 — Criterion's documented minimum for a stable t-test, which
/// is what gates the "performance has improved / regressed" decision in
/// `--baseline` runs. Lower values (the previous 10) make the reported
/// change percentage flip sign across two runs of identical code at the
/// larger sizes, which makes perf PRs hard to evaluate honestly.
pub const SAMPLE_SIZE_SLOW: usize = 30;

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

// ============================================================================
// .fvecs loader (Issue #498 real ANN benchmark data)
// ============================================================================

/// Read vectors from a `.fvecs` file (TEXMEX format used by SIFT1M /
/// GIST / GloVe distributions). Each vector record on disk is laid out
/// as:
///
/// ```text
/// [dim: u32 LE]                4 bytes
/// [values: f32 LE × dim]       dim * 4 bytes
/// ```
///
/// # Arguments
///
/// * `path` - File path to the `.fvecs` file.
/// * `expect_dim` - Required dimension; the function panics if any
///   record disagrees so a corrupted fixture fails loudly.
/// * `max` - Optional cap on the number of vectors to read; useful for
///   sub-sampling (e.g. the SIFT1M-50k case is `max = Some(50_000)`).
///
/// # Returns
///
/// A `Vec<Vec<f32>>` of length up to `max` (or the full file).
pub fn load_fvecs(
    path: &std::path::Path,
    expect_dim: usize,
    max: Option<usize>,
) -> std::io::Result<Vec<Vec<f32>>> {
    use std::io::{BufReader, Read};
    let file = std::fs::File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut out = Vec::new();
    let mut hdr = [0u8; 4];
    let mut vec_buf = vec![0u8; expect_dim * 4];
    loop {
        if reader.read_exact(&mut hdr).is_err() {
            break;
        }
        let dim = u32::from_le_bytes(hdr) as usize;
        if dim != expect_dim {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "{}: per-vector dim {} != expected {}",
                    path.display(),
                    dim,
                    expect_dim
                ),
            ));
        }
        reader.read_exact(&mut vec_buf)?;
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
    Ok(out)
}

/// L2-normalise a vector in place so Cosine distance is well-defined.
/// SIFT vectors are non-negative integer histograms (norms > 0); this
/// rescales them onto the unit hypersphere to match the distribution
/// the Cosine searcher assumes.
pub fn l2_normalise(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Repo-relative path to the `.cache/sift/` directory populated by
/// `scripts/fetch-sift.sh`. The path is anchored at this crate's
/// manifest dir so it works from `cargo test`, `cargo bench`, or any
/// other workspace member's working directory.
pub fn sift_cache_dir() -> std::path::PathBuf {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    // CARGO_MANIFEST_DIR for the laurus crate is `<repo>/laurus`; the
    // .cache directory lives at the repo root one level up.
    manifest
        .parent()
        .expect("workspace root is one level up from laurus/")
        .join(".cache")
        .join("sift")
}

// ============================================================================
// Cold-cache bench harness
// ============================================================================
//
// Cold-cache benches simulate "first access from disk" by evicting the
// OS page cache for an index file before each measured iteration.
// Without eviction, the kernel keeps the file warm in the page cache
// after construction (or after the previous sample), so any read into
// the same byte range hits RAM and the measurement collapses to a
// memcpy benchmark.
//
// On Linux, `posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)` is the
// canonical knob for this — it requests that the kernel drop the
// file's clean pages from the page cache. It is non-destructive
// (dirty pages stay), needs no root, and is precisely scoped to the
// file being benchmarked.
//
// Other operating systems do not expose an equivalent at this level
// (macOS' `F_NOCACHE` applies to a fd's future I/O rather than evicting
// existing pages; Windows has no public API for it). On those hosts
// the helpers below are best-effort no-ops and the cold-cache numbers
// will silently conflate with warm-cache state — call this out
// explicitly in any PR that posts numbers measured off Linux.

/// Evict the OS page cache for a single file. Best-effort: real eviction
/// on Linux via `posix_fadvise(POSIX_FADV_DONTNEED)`; no-op on other
/// hosts.
///
/// # Arguments
///
/// * `path` - Absolute or relative path of the file to evict.
///
/// # Errors
///
/// Returns the underlying `std::io::Error` if the file cannot be
/// opened or the syscall fails.
pub fn evict_file_cache<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<()> {
    evict_file_cache_impl(path.as_ref())
}

#[cfg(target_os = "linux")]
fn evict_file_cache_impl(path: &std::path::Path) -> std::io::Result<()> {
    use std::os::unix::io::AsRawFd;

    let file = std::fs::File::open(path)?;
    // SAFETY: `posix_fadvise` is FFI-safe. The fd is alive for the
    // duration of the call (held by `file`); offset 0 + len 0 is the
    // documented "to end of file" form; `POSIX_FADV_DONTNEED` is a
    // valid advice constant. The return value is the errno (0 on
    // success), not a pointer, so there is nothing to dereference.
    let r = unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
    if r != 0 {
        return Err(std::io::Error::from_raw_os_error(r));
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn evict_file_cache_impl(_path: &std::path::Path) -> std::io::Result<()> {
    // Best-effort no-op. See module-level note above.
    Ok(())
}

/// Evict the OS page cache for every regular file under `dir`,
/// recursively. Used by cold-cache benches that touch a whole
/// segment directory (`*.dict`, `*.post`, `*.docs`, etc. are all
/// flushed in one call).
///
/// # Arguments
///
/// * `dir` - Directory to walk recursively.
///
/// # Errors
///
/// Returns on the first error from either `read_dir` or
/// [`evict_file_cache`]. Partial eviction may have already happened
/// when the error is returned; for cold-cache bench correctness, treat
/// any error from this helper as "unable to guarantee cold cache" and
/// abort the bench.
pub fn evict_directory_cache<P: AsRef<std::path::Path>>(dir: P) -> std::io::Result<()> {
    fn walk(dir: &std::path::Path) -> std::io::Result<()> {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            let file_type = entry.file_type()?;
            if file_type.is_dir() {
                walk(&path)?;
            } else if file_type.is_file() {
                evict_file_cache(&path)?;
            }
        }
        Ok(())
    }
    walk(dir.as_ref())
}

/// Returns `true` if the host supports real page-cache eviction via
/// the helpers above. Cold-cache benches should consult this and emit
/// a one-line warning when running on a host where eviction is a
/// no-op, so PR readers don't trust the resulting numbers as
/// "cold-cache" silently.
pub const fn cold_cache_eviction_supported() -> bool {
    cfg!(target_os = "linux")
}
