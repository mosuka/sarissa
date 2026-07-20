//! Ruby wrapper for the Laurus [`laurus::CommitPolicy`] value object.
//!
//! Exposes `Laurus::CommitPolicy` with the two construction class methods
//! `manual` and `every_docs`, mirroring the auto-commit policy plumbed through
//! [`laurus::EngineBuilder::commit_policy`]. The resulting value object is
//! passed to `Laurus::Index.new(commit_policy: ...)`.

use laurus::CommitPolicy;
use magnus::prelude::*;
use magnus::{Error, RModule, Ruby};

// ---------------------------------------------------------------------------
// CommitPolicy
// ---------------------------------------------------------------------------

/// Ruby value object wrapping a [`laurus::CommitPolicy`].
///
/// A commit policy controls when the engine automatically runs the commit
/// ladder during ingestion:
///
/// * `Laurus::CommitPolicy.manual` — the default. The caller drives every
///   `commit`. Identical to omitting `commit_policy:`.
/// * `Laurus::CommitPolicy.every_docs(n)` — auto-commit after every `n` applied
///   documents, across the singular and batch ingest APIs (and every `n`
///   documents within a single batch). `every_docs(0)` disables auto-commit
///   (equivalent to `manual`).
///
/// This is orthogonal to `Laurus::WalSyncPolicy`.
///
/// # Examples
///
/// ```ruby
/// require "laurus"
///
/// # Default manual commits.
/// policy = Laurus::CommitPolicy.manual
///
/// # Auto-commit every 1000 applied documents.
/// policy = Laurus::CommitPolicy.every_docs(1000)
///
/// index = Laurus::Index.new(commit_policy: policy)
/// ```
#[magnus::wrap(class = "Laurus::CommitPolicy")]
pub struct RbCommitPolicy {
    /// The wrapped core policy, forwarded verbatim to the engine builder.
    pub inner: CommitPolicy,
}

impl RbCommitPolicy {
    /// Build the manual (no auto-commit) policy
    /// ([`laurus::CommitPolicy::Manual`]).
    ///
    /// The engine commits only when `Laurus::Index#commit` is called
    /// explicitly. This is the engine default and matches the behavior when
    /// `commit_policy:` is omitted from `Laurus::Index.new`.
    ///
    /// # Returns
    ///
    /// A `Laurus::CommitPolicy` wrapping [`laurus::CommitPolicy::Manual`].
    fn manual() -> Self {
        Self {
            inner: CommitPolicy::Manual,
        }
    }

    /// Build the auto-commit-every-`n`-documents policy
    /// ([`laurus::CommitPolicy::EveryDocs`]).
    ///
    /// # Arguments
    ///
    /// * `n` (Integer): Commit after this many applied documents. `0` disables
    ///   auto-commit (equivalent to `manual`).
    ///
    /// # Returns
    ///
    /// A `Laurus::CommitPolicy` wrapping [`laurus::CommitPolicy::EveryDocs`].
    fn every_docs(n: usize) -> Self {
        Self {
            inner: CommitPolicy::EveryDocs(n),
        }
    }

    /// Human-readable representation for `inspect` / `to_s`.
    fn inspect(&self) -> String {
        match self.inner {
            CommitPolicy::Manual => "CommitPolicy(Manual)".to_string(),
            CommitPolicy::EveryDocs(n) => format!("CommitPolicy(EveryDocs {n})"),
            // `CommitPolicy` is #[non_exhaustive]; render a future variant
            // generically rather than failing to compile.
            _ => "CommitPolicy(<unknown>)".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Class registration
// ---------------------------------------------------------------------------

/// Register the `Laurus::CommitPolicy` class and its methods.
///
/// # Arguments
///
/// * `ruby` - Ruby interpreter handle.
/// * `module` - The `Laurus` module.
pub fn define(ruby: &Ruby, module: &RModule) -> Result<(), Error> {
    let class = module.define_class("CommitPolicy", ruby.class_object())?;
    class.define_singleton_method("manual", magnus::function!(RbCommitPolicy::manual, 0))?;
    class.define_singleton_method(
        "every_docs",
        magnus::function!(RbCommitPolicy::every_docs, 1),
    )?;
    class.define_method("inspect", magnus::method!(RbCommitPolicy::inspect, 0))?;
    class.define_method("to_s", magnus::method!(RbCommitPolicy::inspect, 0))?;
    Ok(())
}
