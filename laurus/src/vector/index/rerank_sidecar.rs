//! On-disk format for the rerank sidecar (Issue #481 Stage 2).
//!
//! Stage 1 stored vectors as int8 only (LVS1 segment, see
//! [`super::format`]). Stage 2 adds an *optional* per-field sidecar
//! that carries the original full-precision vectors so the searcher
//! can do a wide candidate fetch over int8 (cheap) and then rescore
//! the top `top_k * rerank_factor` candidates against the original
//! values (accurate).
//!
//! The sidecar lives in a separate file (`*.<index_ext>.f32` for
//! [`crate::vector::core::rerank::RerankStorageKind::F32`]) and is
//! produced/consumed in lockstep with the LVS1 segment: vectors are
//! written in the same `(doc_id, field_name)` order as the matching
//! LVS1 segment so a (sidecar position) → (LVS1 position) mapping is
//! the identity. This keeps the sidecar self-contained — the LVS1
//! segment never references it.
//!
//! # On-disk layout
//!
//! ```text
//! offset  size  field
//! ------  ----  -------------------------------------------
//!      0     4  magic         ASCII "LRS1"
//!      4     2  version       u16 LE  (current = 1)
//!      6     2  storage_kind  u16 LE  (1 = F32; 0 reserved for "no
//!                                       sidecar"; 2.. reserved for
//!                                       future bf16 / fp16)
//!      8     8  reserved      zero-padded
//!     16     4  dim           u32 LE
//!     20     4  vector_count  u32 LE
//!     24     -  payload       vector_count × dim × bytes_per_element
//!                              of the configured RerankStorageKind
//!    end     8  footer        magic u32 LE ("LRC1") + CRC-32 u32 LE
//!                              over header + payload (Issue #788)
//! ```
//!
//! All multi-byte integers and IEEE-754 floats are little-endian.
//!
//! # Checksum footer (Issue #788)
//!
//! New sidecars end with an 8-byte footer
//! `[RERANK_SIDECAR_FOOTER_MAGIC u32 LE][crc-32 u32 LE]` whose CRC-32
//! covers every preceding byte (header + payload), mirroring the
//! `.hnsw` footer added by Issue #786. Because the header fully
//! determines the content length, [`read_sidecar`] detects the footer
//! by the number of bytes remaining after the payload — no file-size
//! heuristics: zero remaining bytes is a legacy footer-less sidecar
//! (verification is skipped), exactly eight is a footer (verified),
//! anything else is corruption.
//!
//! # Backward compatibility
//!
//! Pre-Stage-2 segments do not have a sidecar file at all. Readers
//! must treat "sidecar file missing" as Stage 1 and fall back to the
//! int8 LVS1 path. This module never sees that case — it only knows
//! how to read a sidecar that exists.
//!
//! Sidecars written before Issue #788 lack the checksum footer and
//! still load (the footer is optional on read). Conversely, readers
//! that predate the footer stop at the payload end and never see the
//! trailing bytes, so new files remain readable by old code; the
//! header version stays at 1.
//!
//! One consequence of the optional footer: a footer-carrying sidecar
//! whose trailing 8 bytes are lost is byte-identical to a legacy file
//! and loads with verification skipped (the payload itself is still
//! bit-exact in that case — only the integrity guarantee is
//! downgraded). Partial footer loss (1–7 trailing bytes) is detected.
//! The same limitation applies to the `.hnsw` footer from Issue #786;
//! atomic temp+rename writes (Issue #784) make such tail loss
//! improbable in practice.

use std::io::{Read, Write};

use crate::error::{LaurusError, Result};
use crate::vector::core::rerank::RerankStorageKind;

/// 4-byte ASCII magic at offset 0 of every rerank sidecar file.
pub const RERANK_SIDECAR_MAGIC: [u8; 4] = *b"LRS1";

/// Current sidecar header version. Reader rejects anything else.
pub const CURRENT_VERSION: u16 = 1;

/// Size in bytes of the fixed magic+version+kind+reserved portion of
/// the sidecar header (the part shared with [`super::format`]).
pub const FIXED_HEADER_SIZE: usize = 16;

/// Size in bytes of the dim + vector_count fields that follow the
/// fixed header.
pub const DIM_AND_COUNT_SIZE: usize = 8;

/// Total serialized size of a [`RerankSidecarHeader`]: fixed header
/// plus the dim/count pair.
pub const HEADER_SIZE: usize = FIXED_HEADER_SIZE + DIM_AND_COUNT_SIZE;

/// Magic marker for the CRC-32 footer appended to a rerank sidecar
/// (Issue #788). Spelled "LRC1" (Laurus Rerank Checksum v1); distinct
/// from both the sidecar header magic `LRS1` and the `.hnsw` footer
/// magic `LVC1` so a footer can never be mistaken for another format.
pub const RERANK_SIDECAR_FOOTER_MAGIC: u32 = 0x4C52_4331; // "LRC1"

/// Byte length of the sidecar CRC footer: `magic u32` + `crc-32 u32`.
pub const FOOTER_SIZE: usize = 8;

/// Reserved `storage_kind` numeric values stored in the sidecar header.
pub mod storage_kind {
    /// Reserved for "no rerank storage". Sidecars with this value
    /// must not be written; callers expressing "no sidecar" should
    /// simply omit the file. Reader rejects this value.
    pub const NONE: u16 = 0;
    /// Full IEEE-754 single-precision floats. Stage 2.
    pub const F32: u16 = 1;
}

/// Header parsed from / written to a rerank sidecar file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RerankSidecarHeader {
    /// Header format version. Always [`CURRENT_VERSION`] when written
    /// by this build.
    pub version: u16,
    /// On-disk encoding of each stored element.
    pub storage_kind: RerankStorageKind,
    /// Vector dimension (must match the matching LVS1 segment).
    pub dim: u32,
    /// Number of vectors stored in this sidecar (must match the
    /// matching LVS1 segment's vector count).
    pub vector_count: u32,
}

impl RerankSidecarHeader {
    /// Build a header for a Stage 2 sidecar with the given parameters.
    pub fn new(storage_kind: RerankStorageKind, dim: u32, vector_count: u32) -> Self {
        Self {
            version: CURRENT_VERSION,
            storage_kind,
            dim,
            vector_count,
        }
    }

    /// Bytes occupied by the payload that follows the header.
    pub fn payload_size(&self) -> usize {
        self.vector_count as usize * self.dim as usize * self.storage_kind.bytes_per_element()
    }

    /// Write the header to `writer`.
    ///
    /// All multi-byte fields are little-endian. The reserved 8 bytes
    /// at offset 8 are zero-filled.
    pub fn write_to<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_all(&RERANK_SIDECAR_MAGIC)?;
        writer.write_all(&self.version.to_le_bytes())?;
        writer.write_all(&self.storage_kind.tag().to_le_bytes())?;
        writer.write_all(&[0u8; 8])?;
        writer.write_all(&self.dim.to_le_bytes())?;
        writer.write_all(&self.vector_count.to_le_bytes())?;
        Ok(())
    }

    /// Read and validate the header.
    ///
    /// # Errors
    ///
    /// * [`LaurusError::IncompatibleFormat`] if the magic does not
    ///   match `LRS1`, the version is not [`CURRENT_VERSION`], or
    ///   `storage_kind` is unknown / reserved (0).
    /// * [`LaurusError::Io`] for any underlying read failure.
    pub fn read_from<R: Read>(reader: &mut R) -> Result<Self> {
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if magic != RERANK_SIDECAR_MAGIC {
            return Err(LaurusError::IncompatibleFormat(format!(
                "expected rerank sidecar magic {:?} (\"LRS1\"), found {:?}",
                RERANK_SIDECAR_MAGIC, magic
            )));
        }

        let mut version_bytes = [0u8; 2];
        reader.read_exact(&mut version_bytes)?;
        let version = u16::from_le_bytes(version_bytes);
        if version != CURRENT_VERSION {
            return Err(LaurusError::IncompatibleFormat(format!(
                "unsupported rerank sidecar header version {version} \
                 (this build supports {CURRENT_VERSION})"
            )));
        }

        let mut kind_bytes = [0u8; 2];
        reader.read_exact(&mut kind_bytes)?;
        let kind_code = u16::from_le_bytes(kind_bytes);
        let storage_kind = match kind_code {
            storage_kind::NONE => {
                return Err(LaurusError::IncompatibleFormat(
                    "rerank sidecar storage_kind = 0 is reserved (no storage); \
                     callers expressing 'no sidecar' must omit the file entirely"
                        .to_string(),
                ));
            }
            storage_kind::F32 => RerankStorageKind::F32,
            other => {
                return Err(LaurusError::IncompatibleFormat(format!(
                    "unknown rerank sidecar storage_kind = {other}"
                )));
            }
        };

        let mut reserved = [0u8; 8];
        reader.read_exact(&mut reserved)?;

        let mut dim_bytes = [0u8; 4];
        reader.read_exact(&mut dim_bytes)?;
        let dim = u32::from_le_bytes(dim_bytes);

        let mut count_bytes = [0u8; 4];
        reader.read_exact(&mut count_bytes)?;
        let vector_count = u32::from_le_bytes(count_bytes);

        Ok(Self {
            version,
            storage_kind,
            dim,
            vector_count,
        })
    }
}

/// Fill `buf` from `reader`, stopping early on EOF.
///
/// Unlike [`Read::read_exact`], a clean EOF before the buffer is full
/// is not an error — it is the signal that distinguishes a legacy
/// footer-less sidecar (zero trailing bytes) from a footer or
/// corruption. Transient [`std::io::ErrorKind::Interrupted`] reads are
/// retried.
///
/// # Arguments
///
/// * `reader` - The source to read from.
/// * `buf` - The buffer to fill.
///
/// # Returns
///
/// The number of bytes actually read (`0..=buf.len()`).
///
/// # Errors
///
/// Any non-`Interrupted` I/O error from `reader`.
fn read_up_to<R: Read>(reader: &mut R, buf: &mut [u8]) -> std::io::Result<usize> {
    let mut filled = 0;
    while filled < buf.len() {
        match reader.read(&mut buf[filled..]) {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(filled)
}

/// Write a complete sidecar (header + raw f32 payload + CRC-32 footer)
/// to `writer`.
///
/// `vectors` is a flat slice of length `vector_count * dim` containing
/// `vector_count` vectors stored back-to-back. The function infers
/// `vector_count` from the slice length and `dim`. The trailing footer
/// (`[RERANK_SIDECAR_FOOTER_MAGIC u32 LE][crc-32 u32 LE]`, Issue #788)
/// carries a CRC-32 over the header and payload so [`read_sidecar`]
/// can detect silent on-disk corruption.
///
/// # Errors
///
/// * [`LaurusError::InvalidOperation`] if `dim == 0` or `vectors.len()`
///   is not a multiple of `dim`.
/// * Any I/O error from `writer`.
pub fn write_sidecar<W: Write>(
    writer: &mut W,
    storage_kind: RerankStorageKind,
    dim: u32,
    vectors: &[f32],
) -> Result<()> {
    if dim == 0 {
        return Err(LaurusError::InvalidOperation(
            "rerank sidecar dim must be > 0".to_string(),
        ));
    }
    let dim_usize = dim as usize;
    if !vectors.len().is_multiple_of(dim_usize) {
        return Err(LaurusError::InvalidOperation(format!(
            "rerank sidecar payload length {} is not divisible by dim {dim}",
            vectors.len()
        )));
    }
    let vector_count = (vectors.len() / dim_usize) as u32;
    let header = RerankSidecarHeader::new(storage_kind, dim, vector_count);
    let content_crc = {
        let mut crc_writer = crate::storage::checksum::CrcWriter::new(&mut *writer);
        header.write_to(&mut crc_writer)?;
        match storage_kind {
            RerankStorageKind::F32 => {
                for v in vectors {
                    crc_writer.write_all(&v.to_le_bytes())?;
                }
            }
        }
        crc_writer.checksum()
    };
    // The footer itself is excluded from the CRC, mirroring the .hnsw
    // footer layout (Issue #786).
    writer.write_all(&RERANK_SIDECAR_FOOTER_MAGIC.to_le_bytes())?;
    writer.write_all(&content_crc.to_le_bytes())?;
    Ok(())
}

/// Read a complete sidecar (header + raw payload) from `reader`,
/// verifying the CRC-32 footer when present (Issue #788).
///
/// Returns `(header, payload_bytes)` where the byte length of
/// `payload_bytes` equals [`RerankSidecarHeader::payload_size`]. The
/// caller decides how to reinterpret the bytes (e.g. transmute to
/// `&[f32]` for [`RerankStorageKind::F32`]).
///
/// Since the header fully determines the content length, the bytes
/// remaining after the payload identify the file shape unambiguously:
///
/// * zero bytes — legacy footer-less sidecar; verification is skipped
///   (back-compat).
/// * exactly [`FOOTER_SIZE`] bytes starting with
///   [`RERANK_SIDECAR_FOOTER_MAGIC`] — the CRC-32 accumulated over the
///   header and payload is compared against the stored value.
/// * anything else — the file is corrupted.
///
/// # Errors
///
/// * Any error from [`RerankSidecarHeader::read_from`].
/// * [`LaurusError::Io`] if the payload bytes can't be fully read.
/// * [`LaurusError::Index`] if the checksum does not match the
///   content, the footer magic is wrong, the footer is truncated, or
///   data trails a valid footer (all corruption).
pub fn read_sidecar<R: Read>(reader: &mut R) -> Result<(RerankSidecarHeader, Vec<u8>)> {
    let mut crc_reader = crate::storage::checksum::CrcReader::new(&mut *reader);
    let header = RerankSidecarHeader::read_from(&mut crc_reader)?;
    let mut payload = vec![0u8; header.payload_size()];
    crc_reader.read_exact(&mut payload)?;
    let computed = crc_reader.checksum();
    // The footer must not enter the running CRC, so read it from the
    // inner reader (the exact use case of `CrcReader::get_mut`).
    let inner = crc_reader.get_mut();
    let mut footer = [0u8; FOOTER_SIZE];
    let trailing = read_up_to(inner, &mut footer)?;
    match trailing {
        // Legacy sidecar written before Issue #788: no footer, no
        // verification (back-compat).
        0 => Ok((header, payload)),
        FOOTER_SIZE => {
            let magic = u32::from_le_bytes([footer[0], footer[1], footer[2], footer[3]]);
            if magic != RERANK_SIDECAR_FOOTER_MAGIC {
                return Err(LaurusError::index(
                    "rerank sidecar has unexpected trailing bytes: .hnsw.f32 file is corrupted",
                ));
            }
            let stored = u32::from_le_bytes([footer[4], footer[5], footer[6], footer[7]]);
            if stored != computed {
                return Err(LaurusError::index(
                    "rerank sidecar checksum mismatch: .hnsw.f32 file is corrupted",
                ));
            }
            let mut extra = [0u8; 1];
            if read_up_to(inner, &mut extra)? != 0 {
                return Err(LaurusError::index(
                    "rerank sidecar has trailing bytes after the checksum footer: \
                     .hnsw.f32 file is corrupted",
                ));
            }
            Ok((header, payload))
        }
        _ => Err(LaurusError::index(
            "rerank sidecar has a truncated checksum footer: .hnsw.f32 file is corrupted",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn sample_header() -> RerankSidecarHeader {
        RerankSidecarHeader::new(RerankStorageKind::F32, 8, 4)
    }

    #[test]
    fn roundtrip_header() {
        let header = sample_header();
        let mut buf: Vec<u8> = Vec::new();
        header.write_to(&mut buf).unwrap();
        assert_eq!(buf.len(), HEADER_SIZE);
        let parsed = RerankSidecarHeader::read_from(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(parsed, header);
    }

    #[test]
    fn header_layout_starts_with_magic_then_version_then_kind() {
        let header = sample_header();
        let mut buf: Vec<u8> = Vec::new();
        header.write_to(&mut buf).unwrap();
        assert_eq!(&buf[0..4], b"LRS1");
        assert_eq!(u16::from_le_bytes([buf[4], buf[5]]), CURRENT_VERSION);
        assert_eq!(u16::from_le_bytes([buf[6], buf[7]]), storage_kind::F32);
        assert_eq!(&buf[8..16], &[0u8; 8]);
        assert_eq!(u32::from_le_bytes([buf[16], buf[17], buf[18], buf[19]]), 8);
        assert_eq!(u32::from_le_bytes([buf[20], buf[21], buf[22], buf[23]]), 4);
    }

    #[test]
    fn payload_size_is_count_times_dim_times_element_bytes() {
        let header = RerankSidecarHeader::new(RerankStorageKind::F32, 16, 100);
        assert_eq!(header.payload_size(), 100 * 16 * 4);
    }

    #[test]
    fn missing_magic_returns_incompatible_format() {
        let buf = [0u8; HEADER_SIZE];
        let err = RerankSidecarHeader::read_from(&mut Cursor::new(&buf[..])).unwrap_err();
        match err {
            LaurusError::IncompatibleFormat(msg) => {
                assert!(msg.contains("LRS1"), "message should mention LRS1");
            }
            other => panic!("expected IncompatibleFormat, got {other:?}"),
        }
    }

    #[test]
    fn unsupported_version_returns_incompatible_format() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"LRS1");
        buf.extend_from_slice(&99u16.to_le_bytes());
        buf.extend_from_slice(&storage_kind::F32.to_le_bytes());
        buf.extend_from_slice(&[0u8; 8]);
        buf.extend_from_slice(&8u32.to_le_bytes());
        buf.extend_from_slice(&4u32.to_le_bytes());

        let err = RerankSidecarHeader::read_from(&mut Cursor::new(&buf)).unwrap_err();
        match err {
            LaurusError::IncompatibleFormat(msg) => {
                assert!(msg.contains("99"), "message should mention the version");
            }
            other => panic!("expected IncompatibleFormat, got {other:?}"),
        }
    }

    #[test]
    fn storage_kind_zero_is_rejected() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"LRS1");
        buf.extend_from_slice(&CURRENT_VERSION.to_le_bytes());
        buf.extend_from_slice(&storage_kind::NONE.to_le_bytes());
        buf.extend_from_slice(&[0u8; 8]);
        buf.extend_from_slice(&8u32.to_le_bytes());
        buf.extend_from_slice(&4u32.to_le_bytes());

        let err = RerankSidecarHeader::read_from(&mut Cursor::new(&buf)).unwrap_err();
        assert!(matches!(err, LaurusError::IncompatibleFormat(_)));
    }

    #[test]
    fn unknown_storage_kind_is_rejected() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"LRS1");
        buf.extend_from_slice(&CURRENT_VERSION.to_le_bytes());
        buf.extend_from_slice(&999u16.to_le_bytes());
        buf.extend_from_slice(&[0u8; 8]);
        buf.extend_from_slice(&8u32.to_le_bytes());
        buf.extend_from_slice(&4u32.to_le_bytes());

        let err = RerankSidecarHeader::read_from(&mut Cursor::new(&buf)).unwrap_err();
        match err {
            LaurusError::IncompatibleFormat(msg) => {
                assert!(msg.contains("999"));
            }
            other => panic!("expected IncompatibleFormat, got {other:?}"),
        }
    }

    #[test]
    fn truncated_header_returns_io_error() {
        let buf = b"LRS1".to_vec();
        let err = RerankSidecarHeader::read_from(&mut Cursor::new(&buf)).unwrap_err();
        assert!(matches!(err, LaurusError::Io(_)));
    }

    #[test]
    fn write_then_read_sidecar_roundtrips_f32_payload() {
        let dim = 4u32;
        let vectors: Vec<f32> = vec![
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, -1.0, -2.0, -3.0, -4.0,
        ];
        let mut buf: Vec<u8> = Vec::new();
        write_sidecar(&mut buf, RerankStorageKind::F32, dim, &vectors).unwrap();

        let (header, payload) = read_sidecar(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(header.dim, dim);
        assert_eq!(header.vector_count, 3);
        assert_eq!(header.storage_kind, RerankStorageKind::F32);
        assert_eq!(payload.len(), header.payload_size());

        for (i, expected) in vectors.iter().enumerate() {
            let lo = i * 4;
            let actual = f32::from_le_bytes([
                payload[lo],
                payload[lo + 1],
                payload[lo + 2],
                payload[lo + 3],
            ]);
            assert_eq!(actual, *expected, "element {i}");
        }
    }

    #[test]
    fn write_sidecar_rejects_zero_dim() {
        let mut buf: Vec<u8> = Vec::new();
        let err = write_sidecar(&mut buf, RerankStorageKind::F32, 0, &[]).unwrap_err();
        assert!(matches!(err, LaurusError::InvalidOperation(_)));
    }

    #[test]
    fn write_sidecar_rejects_misaligned_payload() {
        let mut buf: Vec<u8> = Vec::new();
        let err = write_sidecar(&mut buf, RerankStorageKind::F32, 4, &[1.0, 2.0, 3.0]).unwrap_err();
        assert!(matches!(err, LaurusError::InvalidOperation(_)));
    }

    #[test]
    fn header_constants_are_consistent() {
        assert_eq!(FIXED_HEADER_SIZE, 16);
        assert_eq!(DIM_AND_COUNT_SIZE, 8);
        assert_eq!(HEADER_SIZE, 24);
    }

    /// Build a complete footer-carrying sidecar byte buffer for the
    /// corruption tests below.
    fn sample_sidecar_bytes() -> Vec<u8> {
        let vectors: Vec<f32> = vec![0.5, -1.5, 2.5, -3.5, 4.5, 5.5, -6.5, 7.5];
        let mut buf: Vec<u8> = Vec::new();
        write_sidecar(&mut buf, RerankStorageKind::F32, 4, &vectors).unwrap();
        buf
    }

    fn assert_index_error(err: LaurusError, expected_fragment: &str) {
        match err {
            LaurusError::Index(msg) => {
                assert!(
                    msg.contains(expected_fragment),
                    "message {msg:?} should contain {expected_fragment:?}"
                );
            }
            other => panic!("expected Index error, got {other:?}"),
        }
    }

    #[test]
    fn write_sidecar_appends_crc_footer() {
        let buf = sample_sidecar_bytes();
        let payload_len = 8 * 4; // 8 f32 elements
        assert_eq!(buf.len(), HEADER_SIZE + payload_len + FOOTER_SIZE);

        let content_end = buf.len() - FOOTER_SIZE;
        let magic = u32::from_le_bytes([
            buf[content_end],
            buf[content_end + 1],
            buf[content_end + 2],
            buf[content_end + 3],
        ]);
        assert_eq!(magic, RERANK_SIDECAR_FOOTER_MAGIC);

        let stored = u32::from_le_bytes([
            buf[content_end + 4],
            buf[content_end + 5],
            buf[content_end + 6],
            buf[content_end + 7],
        ]);
        assert_eq!(stored, crc32fast::hash(&buf[..content_end]));
    }

    #[test]
    fn read_sidecar_accepts_legacy_footerless_payload() {
        // Hand-assemble the pre-#788 layout: header + raw payload, no
        // footer. It must load with verification skipped.
        let vectors: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let header = RerankSidecarHeader::new(RerankStorageKind::F32, 4, 1);
        let mut buf: Vec<u8> = Vec::new();
        header.write_to(&mut buf).unwrap();
        for v in &vectors {
            buf.extend_from_slice(&v.to_le_bytes());
        }

        let (parsed, payload) = read_sidecar(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(parsed, header);
        assert_eq!(payload.len(), parsed.payload_size());
    }

    #[test]
    fn corrupted_payload_with_footer_is_rejected() {
        let mut buf = sample_sidecar_bytes();
        let payload_mid = HEADER_SIZE + (buf.len() - HEADER_SIZE - FOOTER_SIZE) / 2;
        buf[payload_mid] ^= 0xff;

        let err = read_sidecar(&mut Cursor::new(&buf)).unwrap_err();
        assert_index_error(err, "checksum mismatch");
    }

    #[test]
    fn corrupted_header_with_footer_is_rejected() {
        // The reserved region (offset 8..16) parses fine whatever its
        // contents, so a flip there is only caught if the CRC covers
        // the header — which this test locks in.
        let mut buf = sample_sidecar_bytes();
        buf[10] ^= 0xff;

        let err = read_sidecar(&mut Cursor::new(&buf)).unwrap_err();
        assert_index_error(err, "checksum mismatch");
    }

    #[test]
    fn footer_with_wrong_magic_is_rejected() {
        let mut buf = sample_sidecar_bytes();
        let magic_pos = buf.len() - FOOTER_SIZE;
        buf[magic_pos] ^= 0xff;

        let err = read_sidecar(&mut Cursor::new(&buf)).unwrap_err();
        assert_index_error(err, "unexpected trailing bytes");
    }

    #[test]
    fn truncated_footer_is_rejected() {
        let buf = sample_sidecar_bytes();
        for keep in 1..FOOTER_SIZE {
            let truncated = &buf[..buf.len() - FOOTER_SIZE + keep];
            let err = read_sidecar(&mut Cursor::new(truncated)).unwrap_err();
            assert_index_error(err, "truncated checksum footer");
        }
    }

    #[test]
    fn trailing_bytes_after_footer_are_rejected() {
        let mut buf = sample_sidecar_bytes();
        buf.push(0u8);

        let err = read_sidecar(&mut Cursor::new(&buf)).unwrap_err();
        assert_index_error(err, "trailing bytes after the checksum footer");
    }
}
