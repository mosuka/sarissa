//! Windows storage platform knobs (Issue #504, #508).
//!
//! On Windows the OS holds an exclusive lock on memory-mapped files
//! (`ERROR_USER_MAPPED_FILE`, os error 1224) which prevents the
//! writer from truncating / deleting a segment file while a reader
//! still holds an mmap. The current segment-file lifecycle is
//! incompatible with that lock, so the file storage backend defaults
//! `use_mmap` to `false` on Windows. Read-only / read-mostly
//! workloads can opt in via `LAURUS_USE_MMAP=1`.
//!
//! Full Windows mmap support (lifecycle coordination or atomic
//! segment rotation, the Tantivy / Lucene pattern) is tracked in
//! [Issue #508](https://github.com/mosuka/laurus/issues/508).

/// Returns the platform default for [`crate::storage::file::FileStorageConfig::use_mmap`].
///
/// Reads `LAURUS_USE_MMAP` at construction time and returns `true`
/// only when it is set to `"1"`. Otherwise returns `false`.
///
/// Reading the env var here (instead of at every call site) keeps
/// the toggle close to the policy decision and lets tests / fixtures
/// pick either path without code changes.
pub fn default_use_mmap() -> bool {
    matches!(std::env::var("LAURUS_USE_MMAP").as_deref(), Ok("1"))
}
