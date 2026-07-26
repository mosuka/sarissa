//! Shared test helpers for integration tests (Issue #889 PR-4).
//!
//! `tests/common/mod.rs` (not `tests/common.rs`) is the standard Cargo idiom
//! for code shared between integration test binaries without being picked
//! up as its own test target.

use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use laurus::storage::{Storage, StorageInput, StorageOutput};

/// Storage decorator counting every byte written through `create_output`
/// (including rewrites — file sizes alone cannot see them).
///
/// Originated in the #634 HNSW segment-per-commit campaign
/// (`vector_segmented_index_test.rs`); moved here in #889 PR-4 so Flat's
/// (and IVF's, in PR-6) segmented gates can reuse it instead of copying it
/// a third time.
#[derive(Debug)]
pub struct ByteCountingStorage {
    inner: Arc<dyn Storage>,
    pub written: Arc<AtomicU64>,
}

impl ByteCountingStorage {
    pub fn new(inner: Arc<dyn Storage>) -> Self {
        Self {
            inner,
            written: Arc::new(AtomicU64::new(0)),
        }
    }
}

#[derive(Debug)]
struct CountingOutput {
    inner: Box<dyn StorageOutput>,
    written: Arc<AtomicU64>,
}

impl Write for CountingOutput {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.written.fetch_add(n as u64, Ordering::Relaxed);
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

impl std::io::Seek for CountingOutput {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        self.inner.seek(pos)
    }
}

impl StorageOutput for CountingOutput {
    fn flush_and_sync(&mut self) -> laurus::Result<()> {
        self.inner.flush_and_sync()
    }
    fn position(&self) -> laurus::Result<u64> {
        self.inner.position()
    }
    fn close(&mut self) -> laurus::Result<()> {
        self.inner.close()
    }
}

impl Storage for ByteCountingStorage {
    fn open_input(&self, name: &str) -> laurus::Result<Box<dyn StorageInput>> {
        self.inner.open_input(name)
    }
    fn create_output(&self, name: &str) -> laurus::Result<Box<dyn StorageOutput>> {
        Ok(Box::new(CountingOutput {
            inner: self.inner.create_output(name)?,
            written: self.written.clone(),
        }))
    }
    fn create_output_append(&self, name: &str) -> laurus::Result<Box<dyn StorageOutput>> {
        Ok(Box::new(CountingOutput {
            inner: self.inner.create_output_append(name)?,
            written: self.written.clone(),
        }))
    }
    fn delete_file(&self, name: &str) -> laurus::Result<()> {
        self.inner.delete_file(name)
    }
    fn file_exists(&self, name: &str) -> bool {
        self.inner.file_exists(name)
    }
    fn list_files(&self) -> laurus::Result<Vec<String>> {
        self.inner.list_files()
    }
    fn file_size(&self, name: &str) -> laurus::Result<u64> {
        self.inner.file_size(name)
    }
    fn rename_file(&self, from: &str, to: &str) -> laurus::Result<()> {
        self.inner.rename_file(from, to)
    }
    fn metadata(&self, name: &str) -> laurus::Result<laurus::storage::FileMetadata> {
        self.inner.metadata(name)
    }
    fn create_temp_output(&self, prefix: &str) -> laurus::Result<(String, Box<dyn StorageOutput>)> {
        let (name, output) = self.inner.create_temp_output(prefix)?;
        Ok((
            name,
            Box::new(CountingOutput {
                inner: output,
                written: self.written.clone(),
            }),
        ))
    }
    fn sync(&self) -> laurus::Result<()> {
        self.inner.sync()
    }
    fn close(&mut self) -> laurus::Result<()> {
        Ok(())
    }
}
