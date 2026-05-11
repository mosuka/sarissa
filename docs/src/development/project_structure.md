# Project Structure

Laurus is organized as a Cargo workspace with nine crates: the core library, three first-party binaries (CLI, gRPC server, MCP server), and five language bindings.

## Workspace Layout

```text
laurus/                          # Repository root
├── Cargo.toml                   # Workspace definition (members + workspace.package)
├── laurus/                      # Core search engine library
│   ├── Cargo.toml
│   ├── src/
│   │   ├── lib.rs               # Public API and module declarations
│   │   ├── engine.rs            # Engine, EngineBuilder, SearchRequest
│   │   ├── analysis/            # Text analysis pipeline
│   │   ├── lexical/             # Inverted index and lexical search
│   │   ├── vector/              # Vector indexes (Flat, HNSW, IVF)
│   │   ├── embedding/           # Embedder implementations
│   │   ├── storage/             # Storage backends (memory, file, mmap)
│   │   ├── store/               # Document log (WAL)
│   │   ├── spelling/            # Spelling correction
│   │   ├── data/                # DataValue, Document types
│   │   └── error.rs             # LaurusError type
│   └── examples/                # Runnable examples
├── laurus-cli/                  # Command-line interface
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs              # CLI entry point (clap)
│       ├── cli.rs               # Subcommand definitions
│       └── commands/            # Per-subcommand implementations
├── laurus-server/               # gRPC server + HTTP gateway
│   ├── Cargo.toml
│   ├── proto/laurus/v1/         # Protobuf service definitions
│   └── src/
│       ├── lib.rs               # Server library
│       ├── config.rs            # TOML configuration
│       ├── service/             # gRPC service implementations (tonic)
│       └── gateway/             # HTTP/JSON gateway (axum)
├── laurus-mcp/                  # MCP (Model Context Protocol) stdio server
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── server.rs            # rmcp tool router (12 tools)
│       └── convert.rs           # JSON ↔ DataValue helpers
├── laurus-python/               # Python bindings (PyO3 + Maturin)
├── laurus-nodejs/               # Node.js bindings (NAPI-RS)
├── laurus-wasm/                 # WebAssembly bindings (wasm-bindgen)
├── laurus-ruby/                 # Ruby bindings (magnus + rb-sys)
├── laurus-php/                  # PHP bindings (ext-php-rs)
└── docs/                        # mdBook documentation
    ├── book.toml                # English mdBook config
    ├── src/                     # English source
    │   └── SUMMARY.md           # English table of contents
    └── ja/                      # Japanese mdBook (independent build)
        ├── book.toml
        └── src/
            └── SUMMARY.md
```

## Crate Responsibilities

| Crate | Type | Description |
| :--- | :--- | :--- |
| `laurus` | Library | Core search engine with lexical, vector, and hybrid search |
| `laurus-cli` | Binary | CLI tool for index management, document CRUD, search, REPL, and `serve` / `mcp` launchers |
| `laurus-server` | Library + Binary | gRPC server with optional HTTP/JSON gateway |
| `laurus-mcp` | Binary | MCP server on stdio that proxies tool calls to a running laurus-server |
| `laurus-python` | Dynamic library | Python package (PyPI) built with PyO3 / Maturin |
| `laurus-nodejs` | Dynamic library | Node.js package (npm) built with NAPI-RS |
| `laurus-wasm` | WebAssembly | npm package for browser and edge runtimes, built with wasm-bindgen |
| `laurus-ruby` | Dynamic library | Ruby gem built with magnus and rb-sys |
| `laurus-php` | PHP extension | PHP extension built with ext-php-rs (standalone — excluded from the workspace; see [Build & Test](build_and_test.md)) |

All binding crates and the three first-party binaries depend on `laurus`.

## Design Conventions

- **Module style**: File-based modules (Rust 2018 edition style), not `mod.rs`
  - `src/tokenizer.rs` + `src/tokenizer/dictionary.rs`
  - Not: `src/tokenizer/mod.rs`
- **Error handling**: `thiserror` for library error types, `anyhow` only in binary crates
- **No `unwrap()` / `expect()`** in production code (allowed in tests)
- **Async**: All public APIs use async/await with Tokio runtime
- **Unsafe**: Every `unsafe` block must have a `// SAFETY: ...` comment
- **Documentation**: All public types, functions, and enums must have doc comments (`///`)
- **Licensing**: Dependencies must be MIT or Apache-2.0 compatible
