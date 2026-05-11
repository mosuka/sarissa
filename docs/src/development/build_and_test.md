# Build & Test

## Prerequisites

- **Rust** 1.85 or later (edition 2024)
- **Cargo** (included with Rust)
- **protobuf compiler** (`protoc`) -- required for building `laurus-server`

## Building

```bash
# Build all crates
cargo build

# Build with specific features
cargo build --features embeddings-candle

# Build in release mode
cargo build --release
```

## Testing

```bash
# Run all workspace tests (default features)
cargo test

# Run a specific test by name
cargo test <test_name>

# Run tests for a specific crate
cargo test -p laurus
cargo test -p laurus-cli
cargo test -p laurus-server
cargo test -p laurus-mcp
```

### Language binding tests

Each language binding has its own toolchain (Python virtualenv, Node.js
npm, Ruby Bundler, PHP Composer, `wasm32-unknown-unknown` target). The
Makefile wraps these so each target sets up the toolchain before
running the suite:

```bash
make test-laurus-python   # cargo test -p laurus-python + pytest via Maturin
make test-laurus-nodejs   # npm run build:debug + npm test
make test-laurus-wasm     # cargo build -p laurus-wasm --target wasm32-unknown-unknown
make test-laurus-ruby     # cargo test -p laurus-ruby + Ruby minitest
make test-laurus-php      # cargo build -p laurus-php --release + PHPUnit
```

`laurus-php` is excluded from the Cargo workspace because of `links =
"clang"` conflicts with `laurus-ruby`; it builds and tests as a
standalone crate via the Makefile target above. See
[`Makefile`](../../../Makefile) for the full target list including the
matching `format-laurus-*` / `lint-laurus-*` / `build-laurus-*`
variants.

## Linting

```bash
# Run clippy with warnings as errors
cargo clippy -- -D warnings
```

## Formatting

```bash
# Check formatting
cargo fmt --check

# Apply formatting
cargo fmt
```

## Documentation

### API Documentation

```bash
# Generate and open Rust API docs
cargo doc --no-deps --open
```

### mdBook Documentation

```bash
# Build the documentation site
mdbook build docs

# Start a local preview server (http://localhost:3000)
mdbook serve docs

# Lint markdown files
markdownlint-cli2 "docs/src/**/*.md"
```
