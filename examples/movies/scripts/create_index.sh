#!/usr/bin/env bash
# Create the movies index from the schema definition.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
INDEX_DIR="$SCRIPT_DIR/../index"
SCHEMA="$SCRIPT_DIR/../schema.toml"

echo "==> Building laurus (release, embeddings-multimodal)..."
cargo build --manifest-path "$PROJECT_ROOT/Cargo.toml" --release --bin laurus \
  --features embeddings-multimodal
LAURUS="$PROJECT_ROOT/target/release/laurus"

echo "==> Creating movies index at $INDEX_DIR"
"$LAURUS" --index-dir "$INDEX_DIR" create index --schema "$SCHEMA"

echo "==> Done. Index created at $INDEX_DIR"
