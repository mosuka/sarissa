//! Integration test verifying the on-disk size of the new Lucene
//! `BlockTreeTermsWriter`-style term dictionary (#487 PR1).
//!
//! Builds a deterministic 100k-term corpus, writes it through
//! [`BlockTermDictionary::write_to_storage`], and asserts that the
//! resulting `.dict` byte size stays below the regression guard
//! threshold derived empirically from the new layout.
//!
//! Pre-port reference (legacy `HybridTermDictionary` parallel-array
//! representation, before #487 Phase 9 removed it):
//!
//! - 100k unique 5–10-byte ASCII terms × ~7.5 byte avg + 40 bytes /
//!   `TermInfo` (4×u64 + f32) = roughly **4.75 MB** of raw payload
//!   per copy, before any sharing or bit-packing
//!
//! New `BlockTermDictionary` `LTDD` layout target:
//!
//! - FST over per-block last terms (~ 800 entries for 100k @ 128
//!   block size) — typically 10–30 KB
//! - 800 blocks × (front-coded term bytes + bit-packed `TermInfoBlock`
//!   + `BlockMaxData`) — typically ≤ 2 KB / block at this corpus
//!
//! The regression guard below picks **1.6 MB** as a soft ceiling: a
//! conservative cut-off well above the expected ~1.0–1.4 MB for this
//! fixture but well below the legacy ~4.75 MB raw size, so a
//! regression that bloats the layout by 2× will trip the test.

use std::collections::BTreeSet;
use std::sync::Arc;

use laurus::lexical::index::structures::dictionary::{TermDictionaryBuilder, TermInfo};
use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::storage::structured::StructWriter;

/// Generates `n` unique deterministic ASCII terms of length 5..=10.
fn make_corpus(n: usize, seed: u64) -> Vec<String> {
    let mut state = seed;
    let mut next_u32 = || -> u32 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 32) as u32
    };
    let mut set = BTreeSet::new();
    while set.len() < n {
        let len = 5 + (next_u32() % 6) as usize;
        let mut s = String::with_capacity(len);
        for _ in 0..len {
            s.push((b'a' + (next_u32() as u8 % 26)) as char);
        }
        set.insert(s);
    }
    set.into_iter().collect()
}

#[test]
fn block_term_dictionary_file_size_100k_under_1_6mb() {
    const N: usize = 100_000;
    const SEED: u64 = 0x9E3779B97F4A7C15;
    const SOFT_CEILING_BYTES: usize = 1_600_000;

    let terms = make_corpus(N, SEED);
    assert_eq!(terms.len(), N);

    let mut builder = TermDictionaryBuilder::new();
    for (i, term) in terms.iter().enumerate() {
        builder.add_term(term.clone(), TermInfo::new(i as u64 * 16, 64, 1, 1));
    }
    let dict = builder.build().expect("build BlockTermDictionary");

    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    {
        let output = storage.create_output("100k.dict").unwrap();
        let mut writer = StructWriter::new(output);
        dict.write_to_storage(&mut writer).unwrap();
        writer.close().unwrap();
    }
    let size = storage.file_size("100k.dict").unwrap() as usize;
    let bytes_per_term = size as f64 / N as f64;

    eprintln!(
        "BlockTermDictionary 100k corpus: {} bytes ({:.2} bytes/term)",
        size, bytes_per_term
    );

    assert!(
        size < SOFT_CEILING_BYTES,
        "100k-term .dict is {size} bytes, exceeds soft ceiling {SOFT_CEILING_BYTES}: \
         either the layout regressed or the threshold needs revisiting"
    );
}
