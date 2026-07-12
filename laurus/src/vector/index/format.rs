//! On-disk format header shared by all quantized vector segments
//! (HNSW / Flat / IVF) introduced in Issue #481 Stage 1, extended in
//! Stage 3 with the Product Quantization variant.
//!
//! # Layout
//!
//! Every vector segment file written by laurus v0.10+ starts with a
//! fixed 16-byte header followed by a variable-size quantization
//! metadata block. The reader inspects the header first to dispatch
//! on quantization kind:
//!
//! ```text
//! offset  size  field
//! ------  ----  -------------------------------------------
//!      0     4  magic         ASCII "LVS1"
//!      4     2  version       u16 LE  (current = 1)
//!      6     2  quant_kind    u16 LE  (1 = Scalar8Bit, 2 = PQ;
//!                                       0 reserved for "no quant")
//!      8     8  reserved      zero-padded
//!     16     -  quantization metadata (variant-specific, see below)
//!      :     -  vector data
//! ```
//!
//! For `quant_kind = 1` (Scalar8Bit, Stage 1):
//!
//! ```text
//!     16     8  scalar_8bit_params
//!                              offset: f32 LE
//!                              scale:  f32 LE
//!     24     -  vector data (per-vector: dim bytes int8 +
//!                            QuantizedVectorMeta::SERIALIZED_SIZE)
//! ```
//!
//! For `quant_kind = 2` (Product Quantization, Stage 3):
//!
//! ```text
//!     16     2  m             u16 LE  (number of sub-vectors)
//!     18     2  k             u16 LE  (centroids per sub-vector; = 256)
//!     20     2  sub_dim       u16 LE  (dim / m)
//!     22     2  padding       u16 LE  (zero, alignment)
//!     24     -  codebook      m * k * sub_dim * 4 bytes (f32 LE,
//!                              row-major: codebook[m][k][sub_dim])
//!      :     -  vector data (per-vector: m bytes of u8 codes)
//! ```
//!
//! Endianness is **little-endian** for all multi-byte integers and
//! IEEE-754 floats.
//!
//! # Forward compatibility
//!
//! `quant_kind = 0` is reserved for a future re-introduction of an
//! unquantized variant. The reader currently rejects it with
//! [`LaurusError::IncompatibleFormat`].
//!
//! # Backward compatibility
//!
//! Pre-Stage-1 (`f32`-only) segments do **not** have this header.
//! Reading one yields an [`LaurusError::IncompatibleFormat`] with a
//! message instructing the user to rebuild the vector index.

use std::io::{Read, Write};

use crate::error::{LaurusError, Result};
use crate::vector::core::quantization::{PqParams, ScalarQuantParams};

/// 4-byte ASCII magic at offset 0 of every quantized vector segment.
pub const VECTOR_SEGMENT_MAGIC: [u8; 4] = *b"LVS1";

/// Default header format version stamped by writers.
///
/// Flat and IVF segments still write version 1 (their layout is
/// unchanged), so older builds keep reading them. Only the HNSW writer
/// opts into [`VERSION_ORDINAL_GRAPH`] via
/// [`VectorSegmentHeader::with_version`].
pub const CURRENT_VERSION: u16 = 1;

/// Header version for HNSW segments whose graph block stores
/// segment-local u32 ordinals instead of u64 doc ids (Issue #686).
///
/// The ordinal of a vector is its rank in the ascending, deduplicated
/// record doc-id sequence — derivable for both versions because records
/// are always written sorted by doc id.
pub const VERSION_ORDINAL_GRAPH: u16 = 2;

/// Highest header version this build can read. The reader accepts the
/// inclusive range `CURRENT_VERSION..=MAX_SUPPORTED_VERSION`.
pub const MAX_SUPPORTED_VERSION: u16 = VERSION_ORDINAL_GRAPH;

/// `quant_kind` numeric values stored in the header.
///
/// Kept as a module-level constant set rather than a Rust enum so the
/// disk format can carry "unknown but reserved" values forward without
/// requiring a Rust-side enum extension. Reader code matches on these
/// constants explicitly.
pub mod quant_kind {
    /// Reserved for future "no quantization" variant. Reader currently
    /// rejects this value (no Rust-side `None` exists in [`super::super::super::core::quantization::QuantizationMethod`]).
    pub const NONE: u16 = 0;
    /// Per-segment global affine 8-bit scalar quantization. Stage 1.
    pub const SCALAR_8BIT: u16 = 1;
    /// Product quantization. Stage 3 — reader returns NotImplemented.
    pub const PRODUCT_QUANTIZATION: u16 = 2;
    /// FastScan Product Quantization (K=16 4-bit codes + SIMD LUT
    /// distance). Issue [#695](https://github.com/mosuka/laurus/issues/695)
    /// / part D of [#651](https://github.com/mosuka/laurus/issues/651).
    /// Same on-disk layout as [`PRODUCT_QUANTIZATION`] (m / k / sub_dim
    /// header followed by the codebook) but `k == 16` and the per-vector
    /// records are 4-bit packed instead of 8-bit AoS. Only available
    /// with the `pq-fastscan` cargo feature.
    #[cfg(feature = "pq-fastscan")]
    pub const PRODUCT_QUANTIZATION_FASTSCAN: u16 = 3;
}

/// Size in bytes of the fixed portion of the segment header (the part
/// before the quant-kind-specific metadata).
pub const FIXED_HEADER_SIZE: usize = 16;

/// Size in bytes of the [`QuantHeader::Scalar8Bit`] metadata block
/// (offset: f32 + scale: f32).
pub const SCALAR_8BIT_METADATA_SIZE: usize = 8;

/// Size in bytes of the fixed portion of the
/// [`QuantHeader::ProductQuantization`] metadata block (the `m`, `k`,
/// `sub_dim` + padding fields, before the codebook payload). The
/// codebook itself adds `m * k * sub_dim * 4` bytes.
pub const PQ_FIXED_METADATA_SIZE: usize = 8;

/// Centroids per sub-vector in the Stage 3 PQ format. Encoded in the
/// segment header so future 4-bit variants can be added without
/// breaking the layout, but the writer always emits this value today
/// (Issue #481 Stage 3 ships only 8-bit codes).
pub const PQ_CENTROIDS_PER_SUBVECTOR: u16 = 256;

/// Per-segment header read from / written to a vector segment file.
///
/// Combines the fixed 16-byte header (magic / version / quant_kind /
/// reserved) with the quantization-kind-specific metadata block. The
/// reader and writer treat both as a single unit so callers don't have
/// to track parsing state.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorSegmentHeader {
    /// Header format version. Always [`CURRENT_VERSION`] when written
    /// by this build.
    pub version: u16,
    /// Quantization kind discriminator plus its metadata payload.
    pub quant: QuantHeader,
}

/// Quantization-kind-specific portion of [`VectorSegmentHeader`].
///
/// New variants must update `quant_kind::*` constants and the
/// `read_from` / `write_to` dispatch.
#[derive(Debug, Clone, PartialEq)]
pub enum QuantHeader {
    /// `quant_kind = 1` — per-segment global affine SQ (Stage 1).
    Scalar8Bit(ScalarQuantParams),
    /// `quant_kind = 2` — Product Quantization (Stage 3).
    ///
    /// Carries the PQ parameters (`m`, `k`, `sub_dim`) plus the
    /// per-segment codebook (`m * k * sub_dim` floats, row-major:
    /// `codebook[m * k * sub_dim + k * sub_dim + d]`).
    ProductQuantization {
        /// Stage 3 quantizer parameters (M / K / sub_dim).
        params: PqParams,
        /// Per-segment codebook in row-major layout. The length is
        /// always `m * k * sub_dim` floats.
        codebook: Vec<f32>,
    },
    /// `quant_kind = 3` — FastScan Product Quantization (#695 / #651).
    ///
    /// Wire-identical to [`Self::ProductQuantization`] on disk (same
    /// m / k / sub_dim header + codebook); the distinguishing factor is
    /// `k == 16` (4-bit codes) and the per-vector record format used
    /// by the reader to build a
    /// [`crate::vector::index::pq_fastscan_storage::PqFastScanPool`]
    /// instead of a [`crate::vector::index::pq_storage::PqVectorPool`].
    /// Only constructible when the `pq-fastscan` cargo feature is on.
    #[cfg(feature = "pq-fastscan")]
    ProductQuantizationFastScan {
        /// FastScan quantizer parameters (M / K=16 / sub_dim).
        params: PqParams,
        /// Per-segment codebook in row-major layout. The length is
        /// always `m * 16 * sub_dim` floats.
        codebook: Vec<f32>,
    },
}

impl QuantHeader {
    /// Numeric `quant_kind` discriminator written to the header.
    pub fn kind_code(&self) -> u16 {
        match self {
            Self::Scalar8Bit(_) => quant_kind::SCALAR_8BIT,
            Self::ProductQuantization { .. } => quant_kind::PRODUCT_QUANTIZATION,
            #[cfg(feature = "pq-fastscan")]
            Self::ProductQuantizationFastScan { .. } => quant_kind::PRODUCT_QUANTIZATION_FASTSCAN,
        }
    }

    /// Size in bytes of this variant's metadata block following the
    /// fixed 16-byte header.
    pub fn metadata_size(&self) -> usize {
        match self {
            Self::Scalar8Bit(_) => SCALAR_8BIT_METADATA_SIZE,
            Self::ProductQuantization { params, .. } => {
                PQ_FIXED_METADATA_SIZE + params.codebook_byte_size()
            }
            #[cfg(feature = "pq-fastscan")]
            Self::ProductQuantizationFastScan { params, .. } => {
                PQ_FIXED_METADATA_SIZE + params.codebook_byte_size()
            }
        }
    }
}

impl VectorSegmentHeader {
    /// Build a Stage-1 header with the given quantization params.
    pub fn scalar_8bit(params: ScalarQuantParams) -> Self {
        Self {
            version: CURRENT_VERSION,
            quant: QuantHeader::Scalar8Bit(params),
        }
    }

    /// Build a Stage-3 (PQ) header with the given params and codebook.
    ///
    /// The codebook must contain exactly `params.codebook_len()` floats
    /// in row-major layout (`m × k × sub_dim`).
    pub fn product_quantization(params: PqParams, codebook: Vec<f32>) -> Self {
        debug_assert_eq!(codebook.len(), params.codebook_len());
        Self {
            version: CURRENT_VERSION,
            quant: QuantHeader::ProductQuantization { params, codebook },
        }
    }

    /// Build a FastScan PQ header (Issue #695 / part D).
    ///
    /// `params.k` must be 16. The codebook layout is the same as
    /// [`Self::product_quantization`] (row-major `m × k × sub_dim`
    /// floats) — only the per-vector record format on disk differs
    /// (4-bit packed instead of 8-bit AoS).
    #[cfg(feature = "pq-fastscan")]
    pub fn product_quantization_fastscan(params: PqParams, codebook: Vec<f32>) -> Self {
        debug_assert_eq!(params.k, 16, "FastScan requires k == 16");
        debug_assert_eq!(codebook.len(), params.codebook_len());
        Self {
            version: CURRENT_VERSION,
            quant: QuantHeader::ProductQuantizationFastScan { params, codebook },
        }
    }

    /// Return `self` with the header `version` replaced.
    ///
    /// Used by the HNSW writer to stamp [`VERSION_ORDINAL_GRAPH`] on
    /// segments whose graph block stores u32 ordinals (Issue #686);
    /// Flat/IVF writers keep the [`CURRENT_VERSION`] default.
    ///
    /// # Arguments
    ///
    /// * `version` - The header version to stamp.
    ///
    /// # Returns
    ///
    /// The header value with `version` set.
    pub fn with_version(mut self, version: u16) -> Self {
        self.version = version;
        self
    }

    /// Total serialized size: fixed header + quant-kind metadata.
    pub fn serialized_size(&self) -> usize {
        FIXED_HEADER_SIZE + self.quant.metadata_size()
    }

    /// Write the header (fixed portion + quant metadata) to `writer`.
    ///
    /// All multi-byte fields are little-endian. The reserved 8 bytes
    /// at offset 8 are zero-filled.
    pub fn write_to<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_all(&VECTOR_SEGMENT_MAGIC)?;
        writer.write_all(&self.version.to_le_bytes())?;
        writer.write_all(&self.quant.kind_code().to_le_bytes())?;
        writer.write_all(&[0u8; 8])?; // reserved
        match &self.quant {
            QuantHeader::Scalar8Bit(params) => {
                writer.write_all(&params.offset.to_le_bytes())?;
                writer.write_all(&params.scale.to_le_bytes())?;
            }
            QuantHeader::ProductQuantization { params, codebook } => {
                writer.write_all(&params.m.to_le_bytes())?;
                writer.write_all(&params.k.to_le_bytes())?;
                writer.write_all(&params.sub_dim.to_le_bytes())?;
                writer.write_all(&0u16.to_le_bytes())?; // padding
                debug_assert_eq!(codebook.len(), params.codebook_len());
                for &f in codebook {
                    writer.write_all(&f.to_le_bytes())?;
                }
            }
            #[cfg(feature = "pq-fastscan")]
            QuantHeader::ProductQuantizationFastScan { params, codebook } => {
                writer.write_all(&params.m.to_le_bytes())?;
                writer.write_all(&params.k.to_le_bytes())?;
                writer.write_all(&params.sub_dim.to_le_bytes())?;
                writer.write_all(&0u16.to_le_bytes())?; // padding
                debug_assert_eq!(codebook.len(), params.codebook_len());
                for &f in codebook {
                    writer.write_all(&f.to_le_bytes())?;
                }
            }
        }
        Ok(())
    }

    /// Read and validate the header (fixed portion + quant metadata).
    ///
    /// # Errors
    ///
    /// * [`LaurusError::IncompatibleFormat`] if the magic does not
    ///   match `LVS1`. Most commonly this means the segment was written
    ///   by a pre-quantization (f32-only) build of laurus and must be
    ///   rebuilt — Issue #481 Stage 1 is a deliberate format break.
    /// * [`LaurusError::IncompatibleFormat`] if the version is outside
    ///   the supported `CURRENT_VERSION..=MAX_SUPPORTED_VERSION` range
    ///   or if `quant_kind` is the reserved value 0.
    /// * [`LaurusError::NotImplemented`] if `quant_kind` is the
    ///   Product-Quantization value (2) — reserved for Stage 3.
    /// * [`LaurusError::Io`] for any underlying read failure.
    pub fn read_from<R: Read>(reader: &mut R) -> Result<Self> {
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if magic != VECTOR_SEGMENT_MAGIC {
            return Err(LaurusError::IncompatibleFormat(format!(
                "expected vector segment magic {:?} (\"LVS1\"), found {:?}. \
                 Pre-quantization (f32) segments must be rebuilt — \
                 Issue #481 Stage 1 introduced int8 scalar quantization \
                 with a new on-disk format.",
                VECTOR_SEGMENT_MAGIC, magic
            )));
        }

        let mut version_bytes = [0u8; 2];
        reader.read_exact(&mut version_bytes)?;
        let version = u16::from_le_bytes(version_bytes);
        if !(CURRENT_VERSION..=MAX_SUPPORTED_VERSION).contains(&version) {
            return Err(LaurusError::IncompatibleFormat(format!(
                "unsupported vector segment header version {version} \
                 (this build supports {CURRENT_VERSION}..={MAX_SUPPORTED_VERSION})"
            )));
        }

        let mut kind_bytes = [0u8; 2];
        reader.read_exact(&mut kind_bytes)?;
        let kind = u16::from_le_bytes(kind_bytes);

        let mut reserved = [0u8; 8];
        reader.read_exact(&mut reserved)?;

        let quant = match kind {
            quant_kind::SCALAR_8BIT => {
                let mut offset_bytes = [0u8; 4];
                let mut scale_bytes = [0u8; 4];
                reader.read_exact(&mut offset_bytes)?;
                reader.read_exact(&mut scale_bytes)?;
                QuantHeader::Scalar8Bit(ScalarQuantParams {
                    offset: f32::from_le_bytes(offset_bytes),
                    scale: f32::from_le_bytes(scale_bytes),
                })
            }
            quant_kind::PRODUCT_QUANTIZATION => {
                let mut buf2 = [0u8; 2];
                reader.read_exact(&mut buf2)?;
                let m = u16::from_le_bytes(buf2);
                reader.read_exact(&mut buf2)?;
                let k = u16::from_le_bytes(buf2);
                reader.read_exact(&mut buf2)?;
                let sub_dim = u16::from_le_bytes(buf2);
                reader.read_exact(&mut buf2)?; // padding
                let params = PqParams::new(m, k, sub_dim).map_err(|e| {
                    LaurusError::IncompatibleFormat(format!("invalid PQ params: {e}"))
                })?;
                let codebook_len = params.codebook_len();
                let mut codebook = Vec::with_capacity(codebook_len);
                let mut fbuf = [0u8; 4];
                for _ in 0..codebook_len {
                    reader.read_exact(&mut fbuf)?;
                    codebook.push(f32::from_le_bytes(fbuf));
                }
                QuantHeader::ProductQuantization { params, codebook }
            }
            #[cfg(feature = "pq-fastscan")]
            quant_kind::PRODUCT_QUANTIZATION_FASTSCAN => {
                let mut buf2 = [0u8; 2];
                reader.read_exact(&mut buf2)?;
                let m = u16::from_le_bytes(buf2);
                reader.read_exact(&mut buf2)?;
                let k = u16::from_le_bytes(buf2);
                reader.read_exact(&mut buf2)?;
                let sub_dim = u16::from_le_bytes(buf2);
                reader.read_exact(&mut buf2)?; // padding
                let params = PqParams::new(m, k, sub_dim).map_err(|e| {
                    LaurusError::IncompatibleFormat(format!("invalid PQ FastScan params: {e}"))
                })?;
                if params.k != 16 {
                    return Err(LaurusError::IncompatibleFormat(format!(
                        "PQ FastScan segment must declare k == 16, got k = {}",
                        params.k
                    )));
                }
                let codebook_len = params.codebook_len();
                let mut codebook = Vec::with_capacity(codebook_len);
                let mut fbuf = [0u8; 4];
                for _ in 0..codebook_len {
                    reader.read_exact(&mut fbuf)?;
                    codebook.push(f32::from_le_bytes(fbuf));
                }
                QuantHeader::ProductQuantizationFastScan { params, codebook }
            }
            quant_kind::NONE => {
                return Err(LaurusError::IncompatibleFormat(
                    "vector segment header reports quant_kind = 0 (no quantization), \
                     but the Rust-side QuantizationMethod has no None variant; \
                     this build cannot read unquantized segments"
                        .to_string(),
                ));
            }
            other => {
                return Err(LaurusError::IncompatibleFormat(format!(
                    "unknown vector segment quant_kind = {other}"
                )));
            }
        };

        Ok(Self { version, quant })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn sample_params() -> ScalarQuantParams {
        ScalarQuantParams {
            offset: -1.5,
            scale: 3.0 / 255.0,
        }
    }

    #[test]
    fn roundtrip_scalar_8bit_header() {
        let header = VectorSegmentHeader::scalar_8bit(sample_params());
        let mut buf: Vec<u8> = Vec::new();
        header.write_to(&mut buf).unwrap();
        assert_eq!(buf.len(), header.serialized_size());
        assert_eq!(buf.len(), FIXED_HEADER_SIZE + SCALAR_8BIT_METADATA_SIZE);

        let mut cursor = Cursor::new(&buf);
        let parsed = VectorSegmentHeader::read_from(&mut cursor).unwrap();
        assert_eq!(parsed, header);
    }

    #[test]
    fn roundtrip_v2_ordinal_graph_header() {
        let header =
            VectorSegmentHeader::scalar_8bit(sample_params()).with_version(VERSION_ORDINAL_GRAPH);
        let mut buf: Vec<u8> = Vec::new();
        header.write_to(&mut buf).unwrap();
        assert_eq!(
            u16::from_le_bytes([buf[4], buf[5]]),
            VERSION_ORDINAL_GRAPH,
            "with_version must stamp the version bytes at offset 4"
        );

        let mut cursor = Cursor::new(&buf);
        let parsed = VectorSegmentHeader::read_from(&mut cursor).unwrap();
        assert_eq!(parsed.version, VERSION_ORDINAL_GRAPH);
        assert_eq!(parsed, header);
    }

    #[test]
    fn rejects_header_version_above_max_supported() {
        let header = VectorSegmentHeader::scalar_8bit(sample_params())
            .with_version(MAX_SUPPORTED_VERSION + 1);
        let mut buf: Vec<u8> = Vec::new();
        header.write_to(&mut buf).unwrap();

        let mut cursor = Cursor::new(&buf);
        let err = VectorSegmentHeader::read_from(&mut cursor).unwrap_err();
        assert!(
            err.to_string().contains("unsupported vector segment"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn header_starts_with_magic_then_version_then_kind() {
        let header = VectorSegmentHeader::scalar_8bit(sample_params());
        let mut buf: Vec<u8> = Vec::new();
        header.write_to(&mut buf).unwrap();
        assert_eq!(&buf[0..4], b"LVS1");
        assert_eq!(u16::from_le_bytes([buf[4], buf[5]]), CURRENT_VERSION);
        assert_eq!(
            u16::from_le_bytes([buf[6], buf[7]]),
            quant_kind::SCALAR_8BIT
        );
        // reserved bytes are zero
        assert_eq!(&buf[8..16], &[0u8; 8]);
    }

    #[test]
    fn missing_magic_returns_incompatible_format() {
        // Simulate a pre-Stage-1 f32 segment: starts with raw float bytes.
        let f32_bytes = [
            0xCD, 0xCC, 0x4C, 0x3F, // 0.8 in LE f32
            0x00, 0x00, 0x80, 0x3F, // 1.0
            0x00, 0x00, 0x00, 0x40, // 2.0
            0x00, 0x00, 0x40, 0x40, // 3.0
        ];
        let mut cursor = Cursor::new(&f32_bytes[..]);
        let err = VectorSegmentHeader::read_from(&mut cursor).unwrap_err();
        match err {
            LaurusError::IncompatibleFormat(msg) => {
                assert!(msg.contains("LVS1"), "message should mention LVS1");
                assert!(
                    msg.contains("rebuilt"),
                    "message should instruct to rebuild"
                );
            }
            other => panic!("expected IncompatibleFormat, got {other:?}"),
        }
    }

    #[test]
    fn unsupported_version_returns_incompatible_format() {
        // Same magic but bumped version.
        let mut buf = Vec::new();
        buf.extend_from_slice(b"LVS1");
        buf.extend_from_slice(&99u16.to_le_bytes()); // unknown version
        buf.extend_from_slice(&quant_kind::SCALAR_8BIT.to_le_bytes());
        buf.extend_from_slice(&[0u8; 8]);
        buf.extend_from_slice(&0.0_f32.to_le_bytes());
        buf.extend_from_slice(&1.0_f32.to_le_bytes());

        let err = VectorSegmentHeader::read_from(&mut Cursor::new(&buf)).unwrap_err();
        match err {
            LaurusError::IncompatibleFormat(msg) => {
                assert!(msg.contains("99"), "message should mention the version");
            }
            other => panic!("expected IncompatibleFormat, got {other:?}"),
        }
    }

    #[test]
    fn quant_kind_zero_is_rejected_for_now() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"LVS1");
        buf.extend_from_slice(&CURRENT_VERSION.to_le_bytes());
        buf.extend_from_slice(&quant_kind::NONE.to_le_bytes());
        buf.extend_from_slice(&[0u8; 8]);

        let err = VectorSegmentHeader::read_from(&mut Cursor::new(&buf)).unwrap_err();
        assert!(matches!(err, LaurusError::IncompatibleFormat(_)));
    }

    fn sample_pq_params() -> PqParams {
        PqParams::new(4, 256, 2).expect("valid PQ params")
    }

    fn sample_pq_codebook(params: &PqParams) -> Vec<f32> {
        // Deterministic codebook so the roundtrip test compares
        // every byte: codebook[m_idx][k_idx][d] = m_idx * 100 +
        // k_idx + d as f32.
        let mut cb = Vec::with_capacity(params.codebook_len());
        for m in 0..params.m as usize {
            for k in 0..params.k as usize {
                for d in 0..params.sub_dim as usize {
                    cb.push((m * 100 + k) as f32 + d as f32 * 0.01);
                }
            }
        }
        cb
    }

    #[test]
    fn roundtrip_product_quantization_header() {
        let params = sample_pq_params();
        let codebook = sample_pq_codebook(&params);
        let header = VectorSegmentHeader::product_quantization(params, codebook);

        let mut buf: Vec<u8> = Vec::new();
        header.write_to(&mut buf).unwrap();
        assert_eq!(buf.len(), header.serialized_size());
        // Fixed header (16) + PQ fixed metadata (8) + codebook
        // (m * k * sub_dim * 4) = 16 + 8 + 4 * 256 * 2 * 4 = 24 +
        // 8192.
        assert_eq!(
            buf.len(),
            FIXED_HEADER_SIZE + PQ_FIXED_METADATA_SIZE + 4 * 256 * 2 * 4
        );

        let parsed = VectorSegmentHeader::read_from(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(parsed, header);
    }

    #[test]
    fn pq_header_serialised_starts_with_magic_and_kind_two() {
        let params = sample_pq_params();
        let codebook = sample_pq_codebook(&params);
        let header = VectorSegmentHeader::product_quantization(params, codebook);
        let mut buf: Vec<u8> = Vec::new();
        header.write_to(&mut buf).unwrap();
        assert_eq!(&buf[0..4], b"LVS1");
        assert_eq!(u16::from_le_bytes([buf[4], buf[5]]), CURRENT_VERSION);
        assert_eq!(
            u16::from_le_bytes([buf[6], buf[7]]),
            quant_kind::PRODUCT_QUANTIZATION
        );
        // PQ params start at offset 16.
        assert_eq!(u16::from_le_bytes([buf[16], buf[17]]), 4); // m
        assert_eq!(u16::from_le_bytes([buf[18], buf[19]]), 256); // k
        assert_eq!(u16::from_le_bytes([buf[20], buf[21]]), 2); // sub_dim
        // padding zero
        assert_eq!(u16::from_le_bytes([buf[22], buf[23]]), 0);
    }

    #[test]
    fn pq_header_metadata_size_matches_codebook() {
        let params = sample_pq_params();
        let header = QuantHeader::ProductQuantization {
            params,
            codebook: vec![0.0; params.codebook_len()],
        };
        assert_eq!(
            header.metadata_size(),
            PQ_FIXED_METADATA_SIZE + params.codebook_byte_size()
        );
    }

    #[test]
    fn pq_header_rejects_invalid_params() {
        // Construct a buffer with m = 0 (invalid).
        let mut buf = Vec::new();
        buf.extend_from_slice(b"LVS1");
        buf.extend_from_slice(&CURRENT_VERSION.to_le_bytes());
        buf.extend_from_slice(&quant_kind::PRODUCT_QUANTIZATION.to_le_bytes());
        buf.extend_from_slice(&[0u8; 8]);
        buf.extend_from_slice(&0u16.to_le_bytes()); // m = 0
        buf.extend_from_slice(&256u16.to_le_bytes());
        buf.extend_from_slice(&2u16.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes());
        let err = VectorSegmentHeader::read_from(&mut Cursor::new(&buf)).unwrap_err();
        assert!(matches!(err, LaurusError::IncompatibleFormat(_)));
    }

    #[test]
    fn unknown_quant_kind_is_rejected() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"LVS1");
        buf.extend_from_slice(&CURRENT_VERSION.to_le_bytes());
        buf.extend_from_slice(&999u16.to_le_bytes()); // unknown
        buf.extend_from_slice(&[0u8; 8]);

        let err = VectorSegmentHeader::read_from(&mut Cursor::new(&buf)).unwrap_err();
        match err {
            LaurusError::IncompatibleFormat(msg) => {
                assert!(msg.contains("999"));
            }
            other => panic!("expected IncompatibleFormat, got {other:?}"),
        }
    }

    #[test]
    fn truncated_header_returns_io_error() {
        // Magic only, no version → reader hits EOF on the version read.
        let buf = b"LVS1".to_vec();
        let err = VectorSegmentHeader::read_from(&mut Cursor::new(&buf)).unwrap_err();
        assert!(matches!(err, LaurusError::Io(_)));
    }

    #[test]
    fn serialized_size_constants_are_consistent() {
        let h = VectorSegmentHeader::scalar_8bit(sample_params());
        assert_eq!(h.serialized_size(), 24);
        assert_eq!(FIXED_HEADER_SIZE + SCALAR_8BIT_METADATA_SIZE, 24);
    }

    #[test]
    fn fixed_header_size_is_sixteen() {
        assert_eq!(FIXED_HEADER_SIZE, 16);
    }

    #[test]
    fn quant_header_kind_code_matches_constants() {
        let h = QuantHeader::Scalar8Bit(sample_params());
        assert_eq!(h.kind_code(), quant_kind::SCALAR_8BIT);
        assert_eq!(h.metadata_size(), SCALAR_8BIT_METADATA_SIZE);
    }
}
