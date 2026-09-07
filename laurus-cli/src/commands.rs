//! CLI subcommand implementations.
//!
//! Each submodule corresponds to a top-level CLI subcommand:
//!
//! - [`add`] - Add a resource (field or document).
//! - [`commit`] - Commit pending changes.
//! - [`create`] - Create a resource (index or schema file).
//! - [`delete`] - Delete a resource (field or document).
//! - [`get`] - Retrieve a resource (index stats or document).
//! - [`mcp`] - MCP (Model Context Protocol) server on stdio.
//! - [`put`] - Put (upsert) a resource (document).
//! - [`repl`] - Interactive Read-Eval-Print Loop session.
//! - [`search`] - One-shot search query execution.
//! - [`serve`] - gRPC (and optional HTTP gateway) server.
//! - [`train`] - Train auxiliary index structures (shared PQ codebook).
//! - [`update`] - Change a resource (an existing field's type/options).

pub mod add;
pub mod bulk;
pub mod commit;
pub mod create;
pub mod delete;
pub mod get;
pub mod mcp;
pub mod put;
pub mod repl;
pub mod search;
pub mod serve;
pub mod train;
pub mod update;
