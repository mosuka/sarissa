#!/usr/bin/env bash
# Index Aozora Bunko works into laurus.
#
# This script:
#   1. Builds the release laurus binary.
#   2. Runs build_dataset.py, which downloads the Aozora Bunko work list,
#      selects public-domain works, downloads and cleans their body text,
#      and writes a laurus `put docs` JSONL file.
#   3. Bulk-loads the JSONL file via `laurus put docs`.
#
# Usage:
#   bash index_aozora.sh [build_dataset.py options]
#
#   --limit N        Index only the first N works (default: 1000; 0 = all)
#   --ndc CODE       Only works whose NDC code contains CODE (e.g. 913)
#   --author NAME    Only works whose author name contains NAME
#   --parallel N     Concurrent body-text downloads (default: 4)
#   --sleep SECONDS  Delay between downloads per worker (default: 0.2)
#   --refresh-list   Re-download the work list CSV even if cached
#   --yes            Skip the confirmation delay for --limit 0
#
# All arguments are passed through to build_dataset.py — see that script's
# --help for the full list.
#
# Requires: python3, curl (via urllib, no external tool needed)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
EXAMPLE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
INDEX_DIR="$EXAMPLE_DIR/index"
DATA_DIR="$EXAMPLE_DIR/data"
JSONL="$DATA_DIR/aozora.jsonl"

if ! command -v python3 &>/dev/null; then
  echo "Error: python3 is required but not installed." >&2
  exit 1
fi

if [ ! -f "$INDEX_DIR/schema.toml" ]; then
  echo "Error: no index found at $INDEX_DIR. Run create_index.sh first." >&2
  exit 1
fi

echo "==> Building laurus (release)..."
cargo build --manifest-path "$PROJECT_ROOT/Cargo.toml" --release --bin laurus \
  --features embeddings-candle
LAURUS="$PROJECT_ROOT/target/release/laurus"

echo "==> Building the dataset (downloads from aozora.gr.jp)..."
mkdir -p "$DATA_DIR"
python3 "$SCRIPT_DIR/build_dataset.py" --data-dir "$DATA_DIR" --output "$JSONL" "$@"

COUNT=$(wc -l <"$JSONL" | tr -d ' ')
if [ "$COUNT" -eq 0 ]; then
  echo "Error: no works were written to $JSONL." >&2
  exit 1
fi

echo "==> Indexing $COUNT works into $INDEX_DIR"
"$LAURUS" --index-dir "$INDEX_DIR" put docs \
  --file "$JSONL" --batch-size 100 --commit-every 200

echo "==> Done. Indexed $COUNT works."
