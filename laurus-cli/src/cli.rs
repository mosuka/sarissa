//! CLI argument definitions for the laurus command-line tool.
//!
//! This module defines the top-level [`Cli`] struct and all subcommand
//! structures parsed by [`clap`]. Each subcommand maps to a specific
//! operation such as creating an index, querying documents, or starting
//! a gRPC server.

use std::path::PathBuf;

use clap::{Parser, Subcommand};

use crate::output::OutputFormat;

/// Laurus - Unified search engine CLI
#[derive(Parser)]
#[command(name = "laurus", version, about)]
pub struct Cli {
    /// Path to the index directory.
    #[arg(long, env = "LAURUS_INDEX_DIR", default_value = "./laurus_index")]
    pub index_dir: PathBuf,

    /// Output format.
    #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
    pub format: OutputFormat,

    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Create a resource.
    Create(CreateCommand),
    /// Get a resource.
    Get(GetCommand),
    /// Add a resource.
    Add(AddCommand),
    /// Put (upsert) a resource.
    Put(PutCommand),
    /// Delete a resource.
    Delete(DeleteCommand),
    /// Commit pending changes.
    Commit,
    /// Execute a search query.
    Search(SearchCommand),
    /// Start an interactive REPL session.
    Repl,
    /// Start the gRPC server.
    Serve(ServeCommand),
    /// Start the MCP (Model Context Protocol) server on stdio.
    Mcp(McpCommand),
}

// --- Create ---

/// CLI arguments for the `create` subcommand.
///
/// Holds the target resource to create (e.g. an index or a schema file).
#[derive(Parser)]
pub struct CreateCommand {
    #[command(subcommand)]
    pub resource: CreateResource,
}

#[derive(Subcommand)]
pub enum CreateResource {
    /// Create a new index. If --schema is given, uses that TOML file;
    /// otherwise launches the interactive schema wizard.
    Index {
        /// Path to an existing schema TOML file. When omitted, the
        /// interactive schema wizard is launched instead.
        #[arg(long)]
        schema: Option<PathBuf>,
    },
    /// Interactively generate a schema TOML file.
    Schema {
        /// Output file path for the generated schema TOML.
        #[arg(long, default_value = "schema.toml")]
        output: PathBuf,
    },
}

// --- Get ---

/// CLI arguments for the `get` subcommand.
///
/// Holds the target resource to retrieve (e.g. index stats or a document).
#[derive(Parser)]
pub struct GetCommand {
    #[command(subcommand)]
    pub resource: GetResource,
}

#[derive(Subcommand)]
pub enum GetResource {
    /// Show index statistics.
    Stats,
    /// Show the current schema.
    Schema,
    /// Get all documents (including chunks) by ID.
    Docs {
        /// External document ID.
        #[arg(long)]
        id: String,
    },
}

// --- Add ---

/// CLI arguments for the `add` subcommand.
///
/// Holds the target resource to add (e.g. a document).
#[derive(Parser)]
pub struct AddCommand {
    #[command(subcommand)]
    pub resource: AddResource,
}

#[derive(Subcommand)]
pub enum AddResource {
    /// Add a document to the index.
    Doc {
        /// External document ID.
        #[arg(long)]
        id: String,
        /// Document data as a JSON string.
        #[arg(long)]
        data: String,
    },
    /// Bulk-add document chunks from a JSONL file (one entry per line).
    ///
    /// Each line is `{"id": "...", "document": {"fields": {...}}}` — the
    /// same document JSON shape as `add doc --data`. Unlike `put docs`,
    /// repeated ids accumulate as chunks. Commits automatically (per
    /// `--commit-every` and once at the end).
    Docs {
        /// Path to the JSONL file to ingest.
        #[arg(long)]
        file: std::path::PathBuf,
        /// Documents per engine batch call.
        #[arg(long, default_value_t = 1000)]
        batch_size: usize,
        /// Commit every N applied documents (0 = only the final commit).
        #[arg(long, default_value_t = 0)]
        commit_every: usize,
    },
    /// Dynamically add a new field to an existing index.
    Field {
        /// The name of the new field.
        #[arg(long)]
        name: String,
        /// Field configuration as a JSON string.
        ///
        /// Uses the same externally-tagged format as the schema TOML.
        /// Examples:
        ///   '{"Text": {"indexed": true, "stored": true}}'
        ///   '{"Hnsw": {"dimension": 384}}'
        ///   '{"Integer": {}}'
        #[arg(long)]
        field_option: String,
    },
}

// --- Put ---

/// CLI arguments for the `put` subcommand.
///
/// Holds the target resource to put (upsert).
#[derive(Parser)]
pub struct PutCommand {
    #[command(subcommand)]
    pub resource: PutResource,
}

#[derive(Subcommand)]
pub enum PutResource {
    /// Put (upsert) a document into the index.
    ///
    /// If a document with the same ID already exists, all its chunks are
    /// deleted before the new document is indexed.
    Doc {
        /// External document ID.
        #[arg(long)]
        id: String,
        /// Document data as a JSON string.
        #[arg(long)]
        data: String,
    },
    /// Bulk-upsert documents from a JSONL file (one entry per line).
    ///
    /// Each line is `{"id": "...", "document": {"fields": {...}}}` — the
    /// same document JSON shape as `put doc --data`. Entries are applied in
    /// order (duplicate ids dedup, last wins) with one WAL fsync per batch.
    /// Commits automatically (per `--commit-every` and once at the end).
    Docs {
        /// Path to the JSONL file to ingest.
        #[arg(long)]
        file: std::path::PathBuf,
        /// Documents per engine batch call.
        #[arg(long, default_value_t = 1000)]
        batch_size: usize,
        /// Commit every N applied documents (0 = only the final commit).
        #[arg(long, default_value_t = 0)]
        commit_every: usize,
    },
}

// --- Delete ---

/// CLI arguments for the `delete` subcommand.
///
/// Holds the target resource to delete (e.g. a document by ID).
#[derive(Parser)]
pub struct DeleteCommand {
    #[command(subcommand)]
    pub resource: DeleteResource,
}

#[derive(Subcommand)]
pub enum DeleteResource {
    /// Delete all documents (including chunks) by ID.
    Docs {
        /// External document ID.
        #[arg(long)]
        id: String,
    },
    /// Remove a field from the index schema.
    Field {
        /// The name of the field to delete.
        #[arg(long)]
        name: String,
    },
}

// --- Mcp ---

/// CLI arguments for the `mcp` subcommand.
///
/// Configures the MCP stdio server and its connection to a running
/// laurus-server instance.
#[derive(Parser)]
pub struct McpCommand {
    /// gRPC endpoint of a running laurus-server to connect to at startup.
    ///
    /// If omitted, the server starts without a connection.  Use the `connect`
    /// MCP tool to connect later.
    #[arg(long, env = "LAURUS_ENDPOINT")]
    pub endpoint: Option<String>,
}

// --- Serve ---

/// CLI arguments for the `serve` subcommand.
///
/// Configures the gRPC server (and optional HTTP gateway) including
/// listen address, ports, and an optional TOML configuration file.
/// Values can be supplied via CLI flags or environment variables.
/// Use the `RUST_LOG` environment variable to control log verbosity.
#[derive(Parser)]
pub struct ServeCommand {
    /// Path to the configuration file (TOML).
    #[arg(short = 'c', long = "config", env = "LAURUS_CONFIG")]
    pub config: Option<PathBuf>,

    /// Listen address.
    #[arg(short = 'H', long = "host", env = "LAURUS_HOST")]
    pub host: Option<String>,

    /// Listen port.
    #[arg(short = 'p', long = "port", env = "LAURUS_PORT")]
    pub port: Option<u16>,

    /// HTTP Gateway port. If set, starts an HTTP gateway alongside the gRPC server.
    #[arg(long = "http-port", env = "LAURUS_HTTP_PORT")]
    pub http_port: Option<u16>,
}

// --- Search ---

/// CLI arguments for the `search` subcommand.
///
/// Accepts a query string written in the Laurus query DSL along with
/// pagination parameters (`limit` and `offset`).
#[derive(Parser)]
pub struct SearchCommand {
    /// Search query string (Laurus query DSL).
    pub query: String,

    /// Maximum number of results.
    #[arg(long, default_value_t = 10)]
    pub limit: usize,

    /// Number of results to skip.
    #[arg(long, default_value_t = 0)]
    pub offset: usize,
}
