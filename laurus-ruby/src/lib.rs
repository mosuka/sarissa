//! Ruby bindings for the Laurus unified search library.
//!
//! Build with [rake-compiler](https://github.com/rake-compiler/rake-compiler):
//!
//! ```bash
//! cd laurus-ruby
//! bundle install
//! bundle exec rake compile
//! ```

#![deny(clippy::all)]
#![allow(dead_code)]

mod analysis;
mod commit_policy;
mod convert;
mod errors;
mod gvl;
mod index;
mod query;
mod schema;
mod search;
mod wal;

use magnus::{Error, Ruby};

/// Initialize the `Laurus` module and register all classes.
#[magnus::init]
fn init(ruby: &Ruby) -> Result<(), Error> {
    let module = ruby.define_module("Laurus")?;

    // Core
    index::define(ruby, &module)?;
    schema::define(ruby, &module)?;
    wal::define(ruby, &module)?;
    commit_policy::define(ruby, &module)?;

    // Search result & request, fusion algorithms
    search::define(ruby, &module)?;

    // Query types
    query::define(ruby, &module)?;

    // Analysis pipeline
    analysis::define(ruby, &module)?;

    Ok(())
}
