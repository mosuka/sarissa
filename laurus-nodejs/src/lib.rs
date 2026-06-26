//! Node.js bindings for the Laurus unified search library.
//!
//! Build with [napi-rs](https://napi.rs):
//!
//! ```bash
//! cd laurus-nodejs
//! npm run build       # release build
//! npm run build:debug # debug build
//! ```

#![deny(clippy::all)]
#![allow(dead_code)]
#![allow(clippy::upper_case_acronyms)]
#![allow(clippy::enum_variant_names)]

mod analysis;
mod convert;
mod errors;
mod index;
mod query;
mod schema;
mod search;
mod wal;
