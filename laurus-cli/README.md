# laurus-cli

[![Crates.io](https://img.shields.io/crates/v/laurus-cli.svg)](https://crates.io/crates/laurus-cli)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Command-line interface for the [Laurus](https://github.com/mosuka/laurus) search engine.

## Features

- **Index management** -- Create and inspect indexes from TOML schema files, with an interactive schema generator
- **Document CRUD** -- Add, put (upsert), retrieve, and delete documents via JSON
- **Search** -- Execute queries using the Laurus Query DSL
- **Dual output** -- Human-readable tables or machine-parseable JSON (`--format json`)
- **Interactive REPL** -- Explore your index in a live session with command history
- **Server integration** -- Start a gRPC server or MCP server directly from the CLI

## Installation

Prebuilt binaries (including fully static musl builds for Alpine/scratch)
are attached to each [GitHub release](https://github.com/mosuka/laurus/releases);
see [Installation](https://mosuka.github.io/laurus/laurus-cli/installation.html)
for the full target list and a download example.

```bash
cargo install laurus-cli
```

## Quick Start

```bash
# Create an index from a schema file
laurus --index-dir ./my_index create index --schema schema.toml

# Add a document
laurus --index-dir ./my_index add doc \
  --id doc1 --data '{"title":"Hello","body":"World"}'

# Put (upsert) a document
laurus --index-dir ./my_index put doc \
  --id doc1 --data '{"title":"Updated","body":"Content"}'

# Commit changes
laurus --index-dir ./my_index commit

# Search
laurus --index-dir ./my_index search "body:world"

# Get documents by ID
laurus --index-dir ./my_index get docs --id doc1

# Delete documents by ID
laurus --index-dir ./my_index delete docs --id doc1

# Start the interactive REPL
laurus --index-dir ./my_index repl
```

## Commands

| Command | Description |
| :--- | :--- |
| `create index [--schema <FILE>]` | Create a new index (interactive wizard if no schema given) |
| `create schema [--output <FILE>]` | Interactive schema generation wizard |
| `get stats` | Show index statistics |
| `get schema` | Show the current schema as JSON |
| `get docs --id <ID>` | Get all documents (including chunks) by ID |
| `add doc --id <ID> --data <JSON>` | Add a document as a new chunk (append) |
| `add field --name <NAME> --field-option <JSON>` | Dynamically add a field to the index |
| `put doc --id <ID> --data <JSON>` | Put (upsert) a document (replaces existing) |
| `delete docs --id <ID>` | Delete all documents (including chunks) by ID |
| `delete field --name <NAME>` | Remove a field from the schema |
| `commit` | Commit pending changes to disk |
| `search <QUERY> [--limit N] [--offset N]` | Execute a search query |
| `repl` | Start an interactive REPL session |
| `serve [OPTIONS]` | Start the gRPC server |
| `mcp [--endpoint <URL>]` | Start the MCP server on stdio |

## Documentation

- [CLI Guide](https://mosuka.github.io/laurus/laurus-cli.html)
- [Command Reference](https://mosuka.github.io/laurus/laurus-cli/commands.html)
- [REPL](https://mosuka.github.io/laurus/laurus-cli/repl.html)
- [Schema Format](https://mosuka.github.io/laurus/laurus-cli/schema_format.html)
- [Tutorial](https://mosuka.github.io/laurus/laurus-cli/tutorial.html)

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.
