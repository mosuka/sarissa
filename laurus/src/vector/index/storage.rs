use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::sync::Arc;

use crate::error::{LaurusError, Result};
use crate::storage::Storage;
use crate::vector::core::quantization::{QuantizedVectorMeta, ScalarQuantParams};
use crate::vector::core::vector::Vector;

/// Storage for vectors (in-memory or on-demand from disk).
///
/// # Thread Safety
///
/// - The `Owned` variant holds an immutable `Arc<HashMap>` that is freely
///   shareable across threads.
/// - The `OnDemand` variant stores a reference to the underlying
///   [`Storage`] and the file name so that each call to [`get`](Self::get)
///   opens an independent file handle.  This eliminates the previous
///   `Mutex`-based serialization and allows fully concurrent reads.
#[derive(Debug, Clone)]
pub enum VectorStorage {
    /// All vectors are loaded into memory.
    Owned(Arc<HashMap<(u64, String), Vector>>),
    /// Vectors are read from disk on demand.
    ///
    /// Each [`get`](Self::get) call opens a fresh [`StorageInput`](crate::storage::StorageInput)
    /// via [`Storage::open_input`], performs a single seek + read, and closes
    /// the handle.  For mmap-backed storage this is essentially free; for
    /// file-backed storage the OS typically caches the file descriptor.
    ///
    /// `quant_params` controls how the per-vector data section is decoded:
    /// `Some(params)` -> Issue #481 Stage 1 quantized format
    /// (int8 + per-vector meta, dequantized on read);
    /// `None` -> legacy f32 format (still in use by Flat / IVF until
    /// Step 7 of #481 Stage 1 migrates them).
    OnDemand {
        /// Reference to the storage backend (e.g. file system, mmap).
        storage: Arc<dyn Storage>,
        /// Name of the vector index file within the storage.
        file_name: String,
        /// Pre-built mapping from `(doc_id, field_name)` to byte offset.
        offsets: Arc<HashMap<(u64, String), u64>>,
        /// Per-segment quantization params, if the on-disk vector
        /// format is the Stage-1 quantized layout. `None` means the
        /// legacy f32 layout.
        quant_params: Option<ScalarQuantParams>,
    },
}

impl VectorStorage {
    /// Returns all keys stored in this vector storage.
    pub fn keys(&self) -> Vec<(u64, String)> {
        match self {
            VectorStorage::Owned(map) => map.keys().cloned().collect(),
            VectorStorage::OnDemand { offsets, .. } => offsets.keys().cloned().collect(),
        }
    }

    /// Returns the number of vectors stored.
    pub fn len(&self) -> usize {
        match self {
            VectorStorage::Owned(map) => map.len(),
            VectorStorage::OnDemand { offsets, .. } => offsets.len(),
        }
    }

    /// Returns `true` if no vectors are stored.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns `true` if a vector with the given key exists.
    ///
    /// # Arguments
    ///
    /// * `key` - A `(doc_id, field_name)` tuple identifying the vector.
    pub fn contains_key(&self, key: &(u64, String)) -> bool {
        match self {
            VectorStorage::Owned(map) => map.contains_key(key),
            VectorStorage::OnDemand { offsets, .. } => offsets.contains_key(key),
        }
    }

    /// Retrieves a vector by its key.
    ///
    /// For the `Owned` variant the vector is cloned (O(1) due to `Arc`
    /// wrapping).  For the `OnDemand` variant a fresh file handle is opened,
    /// the reader seeks to the recorded offset, and the vector data is read
    /// directly.
    ///
    /// # Arguments
    ///
    /// * `key` - A `(doc_id, field_name)` tuple identifying the vector.
    /// * `dimension` - The expected number of dimensions (used to size the read buffer).
    ///
    /// # Returns
    ///
    /// `Ok(Some(vector))` if the key exists, `Ok(None)` otherwise.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError`] on I/O failure.
    pub fn get(&self, key: &(u64, String), dimension: usize) -> Result<Option<Vector>> {
        match self {
            VectorStorage::Owned(map) => Ok(map.get(key).cloned()),
            VectorStorage::OnDemand {
                storage,
                file_name,
                offsets,
                quant_params,
            } => {
                let Some(&offset) = offsets.get(key) else {
                    return Ok(None);
                };
                let mut input = storage.open_input(file_name).map_err(|e| {
                    LaurusError::internal(format!("Failed to open vector file: {e}"))
                })?;

                input
                    .seek(SeekFrom::Start(offset))
                    .map_err(LaurusError::Io)?;

                // Skip doc_id (8 bytes) + field_name (4 bytes length + variable)
                let mut doc_id_buf = [0u8; 8];
                input.read_exact(&mut doc_id_buf)?;

                let mut field_name_len_buf = [0u8; 4];
                input.read_exact(&mut field_name_len_buf)?;
                let field_name_len = u32::from_le_bytes(field_name_len_buf) as usize;
                let mut field_name_buf = vec![0u8; field_name_len];
                input.read_exact(&mut field_name_buf)?;

                // Read vector data — branch on the on-disk format.
                let values = match quant_params {
                    Some(params) => {
                        // Stage-1 quantized: int8 payload + per-vector meta.
                        // Dequantize back to f32 here so callers see the same
                        // Vector type as the legacy path. Step 6 of #481
                        // Stage 1 will offer an int8-native read path for the
                        // search hot loop.
                        let mut int8_buf = vec![0u8; dimension];
                        input.read_exact(&mut int8_buf)?;
                        let mut meta_buf = [0u8; QuantizedVectorMeta::SERIALIZED_SIZE];
                        input.read_exact(&mut meta_buf)?;
                        int8_buf
                            .iter()
                            .map(|&b| params.dequantize_value(b))
                            .collect()
                    }
                    None => {
                        // Legacy f32 layout (Flat / IVF until Step 7).
                        let mut values = vec![0.0f32; dimension];
                        for value in &mut values {
                            let mut value_buf = [0u8; 4];
                            input.read_exact(&mut value_buf)?;
                            *value = f32::from_le_bytes(value_buf);
                        }
                        values
                    }
                };
                Ok(Some(Vector::new(values)))
            }
        }
    }
}
