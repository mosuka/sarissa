//! # laurus-cli
//!
//! Command-line interface binary for the **laurus** search engine.
//!
//! This crate provides the `laurus` CLI executable, which supports creating and
//! managing search indices, adding/retrieving/deleting documents and fields,
//! executing search queries, running an interactive REPL session, and starting
//! a gRPC server.
//!
//! ## Usage
//!
//! ```shell
//! laurus <COMMAND> [OPTIONS]
//! ```
//!
//! Run `laurus --help` for a full list of available commands and options.

mod cli;
mod commands;
mod context;
mod output;

use anyhow::Result;
use clap::Parser;

use crate::cli::{
    AddResource, Cli, Command, CreateResource, DeleteResource, GetResource, McpCommand,
    PutResource, TrainResource,
};
use crate::commands::{
    add, bulk, commit, create, delete, get, mcp, put, repl, search, serve, train,
};

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let format = cli.format;
    let index_dir = cli.index_dir;

    match cli.command {
        Command::Create(cmd) => match cmd.resource {
            CreateResource::Index {
                schema,
                train_pq_codebook,
            } => {
                create::run_index(schema.as_deref(), train_pq_codebook.as_deref(), &index_dir).await
            }
            CreateResource::Schema { output } => create::run_schema(&output),
        },
        Command::Get(cmd) => match cmd.resource {
            GetResource::Stats => get::run_stats(&index_dir, format).await,
            GetResource::Schema => get::run_schema(&index_dir),
            GetResource::Docs { id } => get::run_docs(&id, &index_dir, format).await,
        },
        Command::Add(cmd) => match cmd.resource {
            AddResource::Doc { id, data } => add::run_doc(&id, &data, &index_dir).await,
            AddResource::Docs {
                file,
                batch_size,
                commit_every,
            } => {
                bulk::run(
                    &file,
                    bulk::BulkMode::Add,
                    batch_size,
                    commit_every,
                    &index_dir,
                )
                .await
            }
            AddResource::Field { name, field_option } => {
                add::run_field(&name, &field_option, &index_dir).await
            }
        },
        Command::Put(cmd) => match cmd.resource {
            PutResource::Doc { id, data } => put::run_doc(&id, &data, &index_dir).await,
            PutResource::Docs {
                file,
                batch_size,
                commit_every,
            } => {
                bulk::run(
                    &file,
                    bulk::BulkMode::Put,
                    batch_size,
                    commit_every,
                    &index_dir,
                )
                .await
            }
        },
        Command::Delete(cmd) => match cmd.resource {
            DeleteResource::Docs { id } => delete::run_docs(&id, &index_dir).await,
            DeleteResource::Field { name } => delete::run_field(&name, &index_dir).await,
        },
        Command::Commit => commit::run(&index_dir).await,
        Command::Train(cmd) => match cmd.resource {
            TrainResource::PqCodebook {
                field,
                input,
                from_index,
                sample_size,
                output,
                update_schema,
            } => {
                train::run_pq_codebook(
                    &field,
                    input.as_deref(),
                    from_index,
                    sample_size,
                    output.as_deref(),
                    update_schema,
                    &index_dir,
                )
                .await
            }
        },
        Command::Search(cmd) => search::run(cmd, &index_dir, format).await,
        Command::Repl => repl::run(&index_dir, format).await,
        Command::Serve(cmd) => serve::run(cmd, &index_dir).await,
        Command::Mcp(McpCommand { endpoint }) => mcp::run(endpoint.as_deref()).await,
    }
}
