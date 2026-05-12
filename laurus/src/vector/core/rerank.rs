//! Rerank storage configuration for two-stage vector search
//! (Issue #481 Stage 2).
//!
//! Stage 1 stored vectors as int8 only (per-segment global affine
//! quantization). Stage 2 adds an *optional* full-precision sidecar
//! per field so the search hot path can do a wide candidate fetch
//! over int8 (cheap) and then re-score the top `top_k * rerank_factor`
//! candidates against the original vectors (accurate).
//!
//! The sidecar is opt-in per field via
//! [`crate::vector::core::field::HnswOption::rerank_storage`] (and
//! the corresponding `FlatOption` / `IvfOption` fields). Fields that
//! leave it `None` stay on the Stage 1 int8-only path; queries that
//! set `rerank_factor` against such fields are silently treated as
//! Stage 1 (no rerank) -- the searcher cannot recover the
//! information that was discarded at index time.

use serde::{Deserialize, Serialize};

/// Storage backend used by the Stage 2 two-stage rerank flow.
///
/// Each variant fixes the on-disk encoding of the sidecar file
/// (`*.<index_ext>.f32` for [`Self::F32`]) and the in-memory
/// representation that the rerank kernel reads from. New variants
/// (e.g. bf16, fp16) can be added without changing the per-field
/// configuration shape; readers dispatch on the variant at load
/// time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RerankStorageKind {
    /// Full IEEE-754 single-precision floats. 4 bytes per dimension.
    /// Matches the Lucene 99 / FAISS rerank convention; gives
    /// numerically exact rerank distances at the cost of 4x sidecar
    /// size vs the int8 Stage 1 payload.
    F32,
}

impl RerankStorageKind {
    /// Bytes occupied by one stored element.
    #[inline]
    pub const fn bytes_per_element(self) -> usize {
        match self {
            Self::F32 => 4,
        }
    }

    /// Numeric tag persisted to the sidecar segment header.
    /// Reserved values:
    ///
    /// - `0`: reserved for "no rerank storage" (callers should use
    ///   `Option::None` at the Rust level instead; the on-disk format
    ///   never encodes 0).
    /// - `1`: [`Self::F32`].
    /// - `2..`: future (e.g. bf16, fp16).
    #[inline]
    pub const fn tag(self) -> u16 {
        match self {
            Self::F32 => 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_bytes_per_element_matches_ieee754() {
        assert_eq!(RerankStorageKind::F32.bytes_per_element(), 4);
    }

    #[test]
    fn f32_tag_is_one() {
        assert_eq!(RerankStorageKind::F32.tag(), 1);
    }

    #[test]
    fn serde_roundtrip_preserves_variant() {
        let kind = RerankStorageKind::F32;
        let json = serde_json::to_string(&kind).expect("serialize");
        let back: RerankStorageKind = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, kind);
    }
}
