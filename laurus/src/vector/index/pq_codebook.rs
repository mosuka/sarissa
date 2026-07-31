//! Standalone shared PQ codebook file (Issue #631, part of the #631
//! campaign).
//!
//! A codebook trained once (via [`train_and_write_pq_codebook`],
//! typically driven by `Engine::train_pq_codebook`/the
//! `laurus train pq-codebook` CLI command in a later PR) and reused by
//! every segment's `write()` instead of retraining k-means from
//! scratch on every commit and every merge — HNSW's `write()`
//! currently does that unconditionally, measured at multiple seconds
//! per call and recurring at every tier of the segment-per-commit
//! merge hierarchy.
//!
//! # File format
//!
//! Reuses [`VectorSegmentHeader::product_quantization`]'s existing
//! serialization verbatim (so the params/codebook bytes are laid out
//! exactly as they would be inline in a `.hnsw` segment header, with
//! no vector records and the default field dictionary), followed by a
//! CRC-32 footer — the same shape as
//! [`crate::vector::index::rerank_sidecar`]'s file, just carrying a PQ
//! header instead of a rerank payload.

use std::io::{Read, Write};

use crate::error::{LaurusError, Result};
use crate::storage::Storage;
use crate::storage::checksum::{CrcReader, CrcWriter};
use crate::vector::core::quantization::{PqParams, pq_train_codebook};
use crate::vector::core::vector::Vector;
use crate::vector::index::format::{QuantHeader, VectorSegmentHeader};

/// Magic value for the codebook file's trailing footer ("PQCB" ASCII),
/// distinguishing it from other footer-bearing formats in this crate
/// (HNSW's own segment footer, the rerank sidecar's footer).
const PQ_CODEBOOK_FOOTER_MAGIC: u32 = 0x5051_4342;

/// Footer size in bytes: magic (`u32`) + CRC-32 (`u32`).
const FOOTER_SIZE: usize = 8;

/// Byte length of the fixed PQ header prefix (the 16-byte
/// [`VectorSegmentHeader`] fixed header plus the 8-byte
/// `m`/`k`/`sub_dim`/padding block). The truncation test uses it to cut
/// a file mid-codebook; the allocation bound itself lives inside
/// [`VectorSegmentHeader::read_from`] since Issue #921.
#[cfg(test)]
const PQ_HEADER_PREFIX_SIZE: usize = 24;

/// A trained PQ codebook loaded from (or about to be persisted to) a
/// standalone file, shared across many segments instead of being
/// retrained inline by every `write()` call.
#[derive(Debug, Clone)]
pub struct SharedPqCodebook {
    /// PQ geometry (`m`, `k`, `sub_dim`) this codebook was trained for.
    pub params: PqParams,
    /// Row-major codebook, `params.codebook_len()` entries.
    pub codebook: Vec<f32>,
}

impl SharedPqCodebook {
    /// Confirm this codebook can be used to encode `dimension`-d
    /// vectors split into `subvector_count` sub-vectors.
    ///
    /// # Arguments
    ///
    /// * `dimension` - The caller's configured vector dimension.
    /// * `subvector_count` - The caller's configured PQ `m`.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError::InvalidOperation`] if the codebook's
    /// geometry does not match the caller's expectations, or if its
    /// stored length is inconsistent with its own params (defensive;
    /// [`read_pq_codebook`] already guarantees this on the load path).
    pub fn validate_for(&self, dimension: usize, subvector_count: usize) -> Result<()> {
        if self.params.original_dim() != dimension {
            return Err(LaurusError::InvalidOperation(format!(
                "shared PQ codebook dimension {} does not match the configured \
                 dimension {dimension}",
                self.params.original_dim()
            )));
        }
        if self.params.m as usize != subvector_count {
            return Err(LaurusError::InvalidOperation(format!(
                "shared PQ codebook subvector_count {} does not match the configured \
                 subvector_count {subvector_count}",
                self.params.m
            )));
        }
        if self.codebook.len() != self.params.codebook_len() {
            return Err(LaurusError::InvalidOperation(format!(
                "shared PQ codebook has {} entries, expected {} for params {:?}",
                self.codebook.len(),
                self.params.codebook_len(),
                self.params
            )));
        }
        Ok(())
    }
}

/// Default storage-relative file name for a field's shared codebook
/// (e.g. `"embedding.pqcb"`).
pub fn default_codebook_name(field: &str) -> String {
    format!("{field}.pqcb")
}

/// Persist `params`/`codebook` to `name` in `storage`, via a
/// temp-file + fsync + atomic rename (mirroring the `.hnsw` segment
/// write pattern, Issue #784) so a crash mid-write cannot leave a torn
/// file behind.
///
/// # Errors
///
/// Any I/O error from `storage`.
pub fn write_pq_codebook(
    storage: &dyn Storage,
    name: &str,
    params: PqParams,
    codebook: &[f32],
) -> Result<()> {
    let tmp_name = format!("{name}.tmp");
    let mut output = CrcWriter::new(storage.create_output(&tmp_name)?);
    VectorSegmentHeader::product_quantization(params, codebook.to_vec()).write_to(&mut output)?;
    let content_crc = output.checksum();
    let mut inner = output.into_inner();
    inner.write_all(&PQ_CODEBOOK_FOOTER_MAGIC.to_le_bytes())?;
    inner.write_all(&content_crc.to_le_bytes())?;
    inner.close()?;
    storage.rename_file(&tmp_name, name)?;
    Ok(())
}

/// Load a codebook previously written by [`write_pq_codebook`],
/// verifying its CRC-32 footer.
///
/// # Allocation safety
///
/// The codebook allocation inside the header parse is bounded against
/// `file_size` by [`VectorSegmentHeader::read_from`] itself (Issue
/// #921) — a header whose `m`/`sub_dim` declare more codebook entries
/// than the file can physically hold is rejected as corrupted before
/// anything is reserved. (This function originally carried its own
/// prefix-peek pre-check because `read_from` had no budget parameter;
/// #921 moved the bound inside, protecting every `.hnsw` PQ segment
/// too, so the bespoke guard is gone.)
///
/// # Errors
///
/// * [`LaurusError::Index`] if the file is truncated, the footer magic
///   is wrong, the checksum does not match, or the header declares a
///   codebook larger than the file can hold.
/// * Any I/O error from `storage`.
pub fn read_pq_codebook(storage: &dyn Storage, name: &str) -> Result<SharedPqCodebook> {
    let file_size = storage.file_size(name)?;
    let mut crc_reader = CrcReader::new(storage.open_input(name)?);
    let header = VectorSegmentHeader::read_from(&mut crc_reader, file_size)
        .map_err(|e| LaurusError::index(format!("shared PQ codebook '{name}': {e}")))?;
    let (params, codebook) = match header.quant {
        QuantHeader::ProductQuantization { params, codebook } => (params, codebook),
        other => {
            return Err(LaurusError::index(format!(
                "shared PQ codebook '{name}' has an unexpected quantization kind: {other:?}"
            )));
        }
    };

    let computed = crc_reader.checksum();
    let inner = crc_reader.get_mut();
    let mut footer = [0u8; FOOTER_SIZE];
    inner.read_exact(&mut footer)?;
    let magic = u32::from_le_bytes([footer[0], footer[1], footer[2], footer[3]]);
    if magic != PQ_CODEBOOK_FOOTER_MAGIC {
        return Err(LaurusError::index(format!(
            "shared PQ codebook '{name}' footer magic mismatch: file is corrupted"
        )));
    }
    let stored_crc = u32::from_le_bytes([footer[4], footer[5], footer[6], footer[7]]);
    if stored_crc != computed {
        return Err(LaurusError::index(format!(
            "shared PQ codebook '{name}' checksum mismatch: file is corrupted"
        )));
    }

    Ok(SharedPqCodebook { params, codebook })
}

/// Train a PQ codebook on `vectors` (a representative sample) and
/// persist it to `name` in `storage`.
///
/// # Arguments
///
/// * `storage` - The vector index's storage namespace to persist into.
/// * `name` - Storage-relative file name (see [`default_codebook_name`]).
/// * `dimension` - Original vector dimension.
/// * `subvector_count` - PQ `m` (must divide `dimension`).
/// * `normalize` - Must match the field's `HnswIndexConfig::normalize_vectors`
///   (i.e. `true` for Cosine distance). Training on a different scale
///   than the segments that will later encode against this codebook
///   produces centroids on the wrong scale, silently degrading recall
///   — the same trap as Issue #794.
/// * `vectors` - The training sample.
///
/// # Errors
///
/// * [`LaurusError::InvalidOperation`] if `vectors` is empty, has
///   mixed dimensions, or `subvector_count` does not divide `dimension`.
/// * Any error from [`write_pq_codebook`].
pub fn train_and_write_pq_codebook(
    storage: &dyn Storage,
    name: &str,
    dimension: usize,
    subvector_count: usize,
    normalize: bool,
    vectors: &[Vector],
) -> Result<SharedPqCodebook> {
    let params = PqParams::from_dim_and_m(dimension, subvector_count)?;

    let normalized;
    let training_set: &[Vector] = if normalize {
        let mut owned = vectors.to_vec();
        for v in &mut owned {
            v.normalize();
        }
        normalized = owned;
        &normalized
    } else {
        vectors
    };

    let codebook = pq_train_codebook(dimension, params, training_set)?;
    write_pq_codebook(storage, name, params, &codebook)?;
    Ok(SharedPqCodebook { params, codebook })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};

    fn storage() -> MemoryStorage {
        MemoryStorage::new(MemoryStorageConfig::default())
    }

    fn sample_vectors(count: usize, dim: usize) -> Vec<Vector> {
        let mut state: u64 = 0x1234_5678_9ABC_DEF0;
        (0..count)
            .map(|_| {
                let data: Vec<f32> = (0..dim)
                    .map(|_| {
                        state = state
                            .wrapping_mul(6_364_136_223_846_793_005)
                            .wrapping_add(1_442_695_040_888_963_407);
                        ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
                    })
                    .collect();
                Vector::new(data)
            })
            .collect()
    }

    #[test]
    fn write_read_roundtrip_preserves_params_and_codebook() {
        let storage = storage();
        let vectors = sample_vectors(300, 32);
        let trained =
            train_and_write_pq_codebook(&storage, "field.pqcb", 32, 4, false, &vectors).unwrap();

        let loaded = read_pq_codebook(&storage, "field.pqcb").unwrap();
        assert_eq!(loaded.params, trained.params);
        assert_eq!(loaded.codebook, trained.codebook);
    }

    #[test]
    fn corrupted_payload_byte_fails_checksum() {
        let storage = storage();
        let vectors = sample_vectors(300, 32);
        train_and_write_pq_codebook(&storage, "field.pqcb", 32, 4, false, &vectors).unwrap();

        // Flip one byte inside the codebook payload (after the 24-byte
        // prefix) and confirm the CRC catches it.
        let mut input = storage.open_input("field.pqcb").unwrap();
        let mut bytes = Vec::new();
        input.read_to_end(&mut bytes).unwrap();
        bytes[30] ^= 0xFF;
        let mut output = storage.create_output("field.pqcb").unwrap();
        output.write_all(&bytes).unwrap();
        output.close().unwrap();

        let err = read_pq_codebook(&storage, "field.pqcb").unwrap_err();
        assert!(
            matches!(&err, LaurusError::Index(msg) if msg.contains("checksum")),
            "expected a checksum-mismatch Index error, got {err:?}"
        );
    }

    #[test]
    fn truncated_file_is_rejected_before_allocating() {
        let storage = storage();
        let vectors = sample_vectors(300, 32);
        train_and_write_pq_codebook(&storage, "field.pqcb", 32, 4, false, &vectors).unwrap();

        // Truncate to just past the fixed prefix -- the declared
        // codebook (m=4, k=256, sub_dim=8 -> 8192 floats = 32768 bytes)
        // cannot possibly fit, so this must fail on the size check, not
        // attempt a matching allocation.
        let mut input = storage.open_input("field.pqcb").unwrap();
        let mut bytes = Vec::new();
        input.read_to_end(&mut bytes).unwrap();
        bytes.truncate(PQ_HEADER_PREFIX_SIZE + 4);
        let mut output = storage.create_output("field.pqcb").unwrap();
        output.write_all(&bytes).unwrap();
        output.close().unwrap();

        let err = read_pq_codebook(&storage, "field.pqcb").unwrap_err();
        assert!(
            matches!(&err, LaurusError::Index(msg) if msg.contains("corrupted")),
            "expected a size-mismatch Index error, got {err:?}"
        );
    }

    #[test]
    fn validate_for_rejects_dimension_and_subvector_mismatch() {
        let storage = storage();
        let vectors = sample_vectors(300, 32);
        let cb =
            train_and_write_pq_codebook(&storage, "field.pqcb", 32, 4, false, &vectors).unwrap();

        assert!(cb.validate_for(32, 4).is_ok());
        assert!(
            cb.validate_for(64, 4).is_err(),
            "dimension mismatch must be rejected"
        );
        assert!(
            cb.validate_for(32, 8).is_err(),
            "subvector_count mismatch must be rejected"
        );
    }

    #[test]
    fn train_and_write_normalizes_training_sample_when_requested() {
        let storage = storage();
        let vectors = sample_vectors(300, 32);

        let cb_raw =
            train_and_write_pq_codebook(&storage, "raw.pqcb", 32, 4, false, &vectors).unwrap();
        let cb_normalized =
            train_and_write_pq_codebook(&storage, "norm.pqcb", 32, 4, true, &vectors).unwrap();

        // Same input, different `normalize` flag: the trained codebooks
        // must differ (proves normalization actually ran, rather than
        // being silently ignored).
        assert_ne!(cb_raw.codebook, cb_normalized.codebook);
    }
}
