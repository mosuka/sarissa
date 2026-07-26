//! Integration tests for crash-safe atomic writes of Flat and IVF index
//! files (Issue #889 PR-3, matching HNSW's existing #784/#786 pattern —
//! see `vector_hnsw_atomic_write_test.rs`).
//!
//! `FlatIndexWriter::write()` and `IvfIndexWriter::write()` now write to a
//! `.flat.tmp`/`.ivf.tmp` file and atomically `rename_file` it into place,
//! instead of writing `storage.create_output(&file_name)` directly. A crash
//! between writing the temp file and the rename therefore leaves the
//! previously committed file untouched.
//!
//! Unlike HNSW's segmented layout, Flat and IVF are still monolithic at
//! this point in the #889 campaign (segmentation for them lands in PR-4/
//! PR-6) — there is no manifest and no orphan sweep, so these tests only
//! assert that a stray `.tmp` file cannot corrupt or replace the
//! previously committed data, not that it gets cleaned up.

use std::cell::Cell;
use std::io::{Seek, SeekFrom, Write};
use std::sync::{Arc, Mutex};

use laurus::Result;
use laurus::storage::memory::MemoryStorage;
use laurus::storage::{FileMetadata, Storage, StorageInput, StorageOutput};
use laurus::vector::core::distance::DistanceMetric;
use laurus::vector::index::ivf::writer::IvfIndexWriter;
use laurus::vector::{
    FlatIndexConfig, FlatIndexWriter, IvfIndexConfig, Vector, VectorIndexWriter,
    VectorIndexWriterConfig,
};

/// A [`Storage`] decorator whose next `create_output` call returns a writer
/// that fails after a configured number of `write` calls — simulating a
/// crash partway through `write()`'s sequence of `write_all` calls, the way
/// a real process kill or disk-full error would.
#[derive(Debug)]
struct FlakyStorage {
    inner: Arc<dyn Storage>,
    fail_after_writes: Mutex<Option<usize>>,
}

impl FlakyStorage {
    fn new(inner: Arc<dyn Storage>) -> Self {
        Self {
            inner,
            fail_after_writes: Mutex::new(None),
        }
    }

    /// Arm the *next* `create_output` call to fail on its `(n+1)`-th write.
    fn arm_next_write_failure(&self, after_n_writes: usize) {
        *self.fail_after_writes.lock().unwrap() = Some(after_n_writes);
    }
}

/// `StorageOutput` need not be `Sync` (only `Send`), so a plain `Cell` is
/// fine here even though `FlakyStorage` itself must be `Sync`.
#[derive(Debug)]
struct FlakyOutput {
    inner: Box<dyn StorageOutput>,
    remaining: Cell<usize>,
}

impl Write for FlakyOutput {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let remaining = self.remaining.get();
        if remaining == 0 {
            return Err(std::io::Error::other("simulated crash mid-write"));
        }
        self.remaining.set(remaining - 1);
        self.inner.write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

impl Seek for FlakyOutput {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        self.inner.seek(pos)
    }
}

impl StorageOutput for FlakyOutput {
    fn flush_and_sync(&mut self) -> Result<()> {
        self.inner.flush_and_sync()
    }
    fn position(&self) -> Result<u64> {
        self.inner.position()
    }
    fn close(&mut self) -> Result<()> {
        self.inner.close()
    }
}

impl Storage for FlakyStorage {
    fn open_input(&self, name: &str) -> Result<Box<dyn StorageInput>> {
        self.inner.open_input(name)
    }

    fn create_output(&self, name: &str) -> Result<Box<dyn StorageOutput>> {
        let inner = self.inner.create_output(name)?;
        if let Some(n) = self.fail_after_writes.lock().unwrap().take() {
            Ok(Box::new(FlakyOutput {
                inner,
                remaining: Cell::new(n),
            }))
        } else {
            Ok(inner)
        }
    }

    fn create_output_append(&self, name: &str) -> Result<Box<dyn StorageOutput>> {
        self.inner.create_output_append(name)
    }

    fn file_exists(&self, name: &str) -> bool {
        self.inner.file_exists(name)
    }

    fn delete_file(&self, name: &str) -> Result<()> {
        self.inner.delete_file(name)
    }

    fn list_files(&self) -> Result<Vec<String>> {
        self.inner.list_files()
    }

    fn file_size(&self, name: &str) -> Result<u64> {
        self.inner.file_size(name)
    }

    fn metadata(&self, name: &str) -> Result<FileMetadata> {
        self.inner.metadata(name)
    }

    fn rename_file(&self, old_name: &str, new_name: &str) -> Result<()> {
        self.inner.rename_file(old_name, new_name)
    }

    fn create_temp_output(&self, prefix: &str) -> Result<(String, Box<dyn StorageOutput>)> {
        self.inner.create_temp_output(prefix)
    }

    fn sync(&self) -> Result<()> {
        self.inner.sync()
    }

    fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

const DIM: usize = 4;
const N: u64 = 20;

fn doc_vec(i: u64) -> Vector {
    let t = i as f32 * 0.1;
    Vector::new(vec![t.cos(), t.sin(), (t * 2.0).cos(), (t * 2.0).sin()])
}

fn build_and_commit_flat(storage: Arc<dyn Storage>, name: &str) {
    let config = FlatIndexConfig {
        dimension: DIM,
        distance_metric: DistanceMetric::Cosine,
        ..FlatIndexConfig::default()
    };
    let mut writer =
        FlatIndexWriter::with_storage(config, VectorIndexWriterConfig::default(), name, storage)
            .unwrap();
    let vectors: Vec<_> = (0..N).map(|i| (i, "v".to_string(), doc_vec(i))).collect();
    writer.add_vectors(vectors).unwrap();
    writer.finalize().unwrap();
    writer.write().unwrap();
}

fn build_and_commit_ivf(storage: Arc<dyn Storage>, name: &str) {
    let config = IvfIndexConfig {
        dimension: DIM,
        distance_metric: DistanceMetric::Cosine,
        n_clusters: 2,
        n_probe: 2,
        normalize_vectors: false,
        ..IvfIndexConfig::default()
    };
    let mut writer =
        IvfIndexWriter::with_storage(config, VectorIndexWriterConfig::default(), name, storage)
            .unwrap();
    let vectors: Vec<_> = (0..N).map(|i| (i, "v".to_string(), doc_vec(i))).collect();
    writer.add_vectors(vectors).unwrap();
    writer.finalize().unwrap();
    writer.write().unwrap();
}

fn read_all_flat(storage: Arc<dyn Storage>, name: &str) -> Vec<u64> {
    use laurus::vector::reader::VectorIndexReader;
    let reader = laurus::vector::index::flat::reader::FlatVectorIndexReader::load(
        storage,
        name,
        DistanceMetric::Cosine,
    )
    .unwrap();
    let mut ids: Vec<u64> = reader
        .vector_ids()
        .unwrap()
        .into_iter()
        .map(|(d, _)| d)
        .collect();
    ids.sort_unstable();
    ids
}

fn read_all_ivf(storage: Arc<dyn Storage>, name: &str) -> Vec<u64> {
    use laurus::vector::reader::VectorIndexReader;
    let reader = laurus::vector::index::ivf::reader::IvfIndexReader::load(
        storage,
        name,
        DistanceMetric::Cosine,
    )
    .unwrap();
    let mut ids: Vec<u64> = reader
        .vector_ids()
        .unwrap()
        .into_iter()
        .map(|(d, _)| d)
        .collect();
    ids.sort_unstable();
    ids
}

#[test]
fn flat_commit_leaves_no_temp_file() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::default());
    build_and_commit_flat(storage.clone(), "flat_atomic");

    assert!(
        storage.file_exists("flat_atomic.flat"),
        "committed file must exist"
    );
    let temps: Vec<String> = storage
        .list_files()
        .unwrap()
        .into_iter()
        .filter(|f| f.ends_with(".tmp"))
        .collect();
    assert!(
        temps.is_empty(),
        "a successful write must not leave any .tmp behind, got {temps:?}"
    );
    assert_eq!(read_all_flat(storage, "flat_atomic").len(), N as usize);
}

#[test]
fn ivf_commit_leaves_no_temp_file() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::default());
    build_and_commit_ivf(storage.clone(), "ivf_atomic");

    assert!(
        storage.file_exists("ivf_atomic.ivf"),
        "committed file must exist"
    );
    let temps: Vec<String> = storage
        .list_files()
        .unwrap()
        .into_iter()
        .filter(|f| f.ends_with(".tmp"))
        .collect();
    assert!(
        temps.is_empty(),
        "a successful write must not leave any .tmp behind, got {temps:?}"
    );
    assert_eq!(read_all_ivf(storage, "ivf_atomic").len(), N as usize);
}

/// The strongest proof of atomicity: a real interruption *during* the
/// second write's byte stream (not just a stray leftover `.tmp` file).
/// Under the pre-#889 code (`create_output(&file_name)` directly), this
/// early-returns via `?` and the writer's `Drop` still flushes/closes the
/// partially-written buffer, publishing truncated garbage over the
/// previously-committed file. Under the fix, only `{name}.flat.tmp` is
/// affected — `{name}.flat` is untouched until the final `rename_file`,
/// which never happens because `write()` returned an error first.
#[test]
fn flat_second_write_failure_does_not_corrupt_first_commit() {
    let base: Arc<dyn Storage> = Arc::new(MemoryStorage::default());
    build_and_commit_flat(base.clone(), "flat_atomic");
    let before = read_all_flat(base.clone(), "flat_atomic");
    assert_eq!(before.len(), N as usize);

    let flaky = Arc::new(FlakyStorage::new(base.clone()));
    flaky.arm_next_write_failure(5);
    let config = FlatIndexConfig {
        dimension: DIM,
        distance_metric: DistanceMetric::Cosine,
        ..FlatIndexConfig::default()
    };
    // Reload (picks up the 20 committed docs), add one more, then attempt a
    // second write that fails partway through.
    let mut writer = FlatIndexWriter::with_storage(
        config,
        VectorIndexWriterConfig::default(),
        "flat_atomic",
        flaky as Arc<dyn Storage>,
    )
    .unwrap();
    writer
        .add_vectors(vec![(N, "v".to_string(), doc_vec(N))])
        .unwrap();
    writer.finalize().unwrap();
    let write_result = writer.write();
    assert!(
        write_result.is_err(),
        "the simulated mid-write failure must propagate"
    );
    drop(writer);

    assert!(
        base.file_exists("flat_atomic.flat"),
        "the previously committed file must still exist"
    );
    let after = read_all_flat(base, "flat_atomic");
    assert_eq!(
        after, before,
        "a write failure partway through must leave the previously committed \
         data completely intact — not truncated or partially overwritten"
    );
}

/// IVF analogue of [`flat_second_write_failure_does_not_corrupt_first_commit`].
#[test]
fn ivf_second_write_failure_does_not_corrupt_first_commit() {
    let base: Arc<dyn Storage> = Arc::new(MemoryStorage::default());
    build_and_commit_ivf(base.clone(), "ivf_atomic");
    let before = read_all_ivf(base.clone(), "ivf_atomic");
    assert_eq!(before.len(), N as usize);

    let flaky = Arc::new(FlakyStorage::new(base.clone()));
    flaky.arm_next_write_failure(5);
    let config = IvfIndexConfig {
        dimension: DIM,
        distance_metric: DistanceMetric::Cosine,
        n_clusters: 2,
        n_probe: 2,
        normalize_vectors: false,
        ..IvfIndexConfig::default()
    };
    let mut writer = IvfIndexWriter::with_storage(
        config,
        VectorIndexWriterConfig::default(),
        "ivf_atomic",
        flaky as Arc<dyn Storage>,
    )
    .unwrap();
    writer
        .add_vectors(vec![(N, "v".to_string(), doc_vec(N))])
        .unwrap();
    writer.finalize().unwrap();
    let write_result = writer.write();
    assert!(
        write_result.is_err(),
        "the simulated mid-write failure must propagate"
    );
    drop(writer);

    assert!(
        base.file_exists("ivf_atomic.ivf"),
        "the previously committed file must still exist"
    );
    let after = read_all_ivf(base, "ivf_atomic");
    assert_eq!(
        after, before,
        "a write failure partway through must leave the previously committed \
         data completely intact — not truncated or partially overwritten"
    );
}

#[test]
fn flat_orphaned_temp_from_crashed_write_does_not_corrupt_committed_data() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::default());
    build_and_commit_flat(storage.clone(), "flat_atomic");
    let before = read_all_flat(storage.clone(), "flat_atomic");
    assert_eq!(before.len(), N as usize);

    // Simulate a crash *during* a later write: the temp file was created
    // but the atomic rename never happened.
    {
        let mut out = storage.create_output("flat_atomic.flat.tmp").unwrap();
        out.write_all(b"partially-written-garbage-from-a-crashed-write")
            .unwrap();
        out.close().unwrap();
    }
    assert!(
        storage.file_exists("flat_atomic.flat"),
        "committed file still present"
    );

    let after = read_all_flat(storage, "flat_atomic");
    assert_eq!(
        after, before,
        "the committed data must survive an orphaned temp file from a crashed write"
    );
}

#[test]
fn ivf_orphaned_temp_from_crashed_write_does_not_corrupt_committed_data() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::default());
    build_and_commit_ivf(storage.clone(), "ivf_atomic");
    let before = read_all_ivf(storage.clone(), "ivf_atomic");
    assert_eq!(before.len(), N as usize);

    {
        let mut out = storage.create_output("ivf_atomic.ivf.tmp").unwrap();
        out.write_all(b"partially-written-garbage-from-a-crashed-write")
            .unwrap();
        out.close().unwrap();
    }
    assert!(
        storage.file_exists("ivf_atomic.ivf"),
        "committed file still present"
    );

    let after = read_all_ivf(storage, "ivf_atomic");
    assert_eq!(
        after, before,
        "the committed data must survive an orphaned temp file from a crashed write"
    );
}
