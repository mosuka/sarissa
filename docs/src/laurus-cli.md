# CLI Overview

Laurus provides a command-line tool `laurus` that lets you create indexes, manage documents, and run search queries without writing code.

## Features

- **Index management** -- Create and inspect indexes from TOML schema files, with an interactive schema generator
- **Document CRUD** -- Add, retrieve, and delete documents via JSON
- **Search** -- Execute queries using the [Query DSL](concepts/query_dsl.md)
- **Dual output** -- Human-readable tables or machine-parseable JSON
- **Interactive REPL** -- Explore your index in a live session
- **gRPC server** -- Start a [gRPC server](laurus-server.md) with `laurus serve`

## Getting Started

```bash
# Install
cargo install laurus-cli

# Generate a schema interactively
laurus create schema

# Create an index from the schema
laurus --index-dir ./my_index create index --schema schema.toml

# Add a document
laurus --index-dir ./my_index add doc --id doc1 --data '{"fields":{"title":"Hello","body":"World"}}'

# Commit changes
laurus --index-dir ./my_index commit

# Search
laurus --index-dir ./my_index search "body:world"
```

See the sub-sections for detailed documentation:

- [Installation](laurus-cli/installation.md) -- How to install the CLI
- [Commands](laurus-cli/commands.md) -- Full command reference
- [Schema Format](laurus-cli/schema_format.md) -- Schema TOML format reference
- [REPL](laurus-cli/repl.md) -- Interactive mode
