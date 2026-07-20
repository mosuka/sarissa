//! WebAssembly bindings for the Laurus unified search library.
//!
//! Build with [wasm-pack](https://rustwasm.github.io/wasm-pack/):
//!
//! ```bash
//! cd laurus-wasm
//! wasm-pack build --target web
//! wasm-pack build --target bundler
//! ```

#![deny(clippy::all)]
#![allow(dead_code)]
#![allow(clippy::upper_case_acronyms)]
#![allow(clippy::enum_variant_names)]

mod analysis;
mod commit;
mod convert;
mod embedder;
mod errors;
mod index;
mod query;
mod schema;
mod search;
mod storage;
mod wal;
