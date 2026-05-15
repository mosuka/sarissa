//! Unix-family storage platform knobs (Issue #504, #508).
//!
//! On Unix the kernel keeps mmap-backed inodes alive across
//! `unlink` / `truncate` / overwrite, so the file storage backend
//! can default `use_mmap` to `true` without colliding with the
//! segment-overwrite path. Operators opt out via the
//! `LAURUS_NO_MMAP=1` environment variable for debug sessions or
//! hosts where mmap misbehaves.

/// Returns the platform default for [`crate::storage::file::FileStorageConfig::use_mmap`].
///
/// Reads `LAURUS_NO_MMAP` at construction time and returns `false`
/// when it is set to `"1"`. Otherwise returns `true`.
///
/// Reading the env var here (instead of at every call site) keeps
/// the toggle close to the policy decision and lets tests / fixtures
/// pick either path without code changes.
pub fn default_use_mmap() -> bool {
    !matches!(std::env::var("LAURUS_NO_MMAP").as_deref(), Ok("1"))
}
