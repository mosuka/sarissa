#!/usr/bin/env bash
# Create the Aozora Bunko index from the schema definition.
#
# Builds the release binary, fetches the Lindera IPADIC dictionary (see
# fetch_dict.sh), renders schema.toml's @IPADIC_DIR@ placeholder into an
# absolute path, and creates the index.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
EXAMPLE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
INDEX_DIR="$EXAMPLE_DIR/index"
DATA_DIR="$EXAMPLE_DIR/data"
TEMPLATE="$EXAMPLE_DIR/schema.toml"
RENDERED="$DATA_DIR/schema.generated.toml"

for cmd in curl unzip python3; do
  if ! command -v "$cmd" &>/dev/null; then
    echo "Error: $cmd is required but not installed." >&2
    exit 1
  fi
done

echo "==> Building laurus (release)..."
cargo build --manifest-path "$PROJECT_ROOT/Cargo.toml" --release --bin laurus \
  --features embeddings-candle
LAURUS="$PROJECT_ROOT/target/release/laurus"

echo "==> Preparing the Lindera IPADIC dictionary..."
DICT_PATH="$(bash "$SCRIPT_DIR/fetch_dict.sh" | tail -1)"

# Render the schema: replace the @IPADIC_DIR@ placeholder with an absolute
# path. Lindera opens `dict` as a plain filesystem path with no relative-
# path or environment-variable expansion, so a relative path here would
# break the moment `laurus` is invoked from a different working directory.
mkdir -p "$DATA_DIR"
python3 - "$TEMPLATE" "$RENDERED" "$DICT_PATH" <<'PY'
import pathlib
import sys

template_path, output_path, dict_dir = sys.argv[1], sys.argv[2], sys.argv[3]
content = pathlib.Path(template_path).read_text(encoding="utf-8")
if "@IPADIC_DIR@" not in content:
    sys.exit("Error: placeholder @IPADIC_DIR@ not found in schema.toml")
# Escape for TOML string literals (backslash and double-quote), in case the
# dictionary lives under a path containing either (e.g. Windows paths).
escaped = dict_dir.replace("\\", "\\\\").replace('"', '\\"')
pathlib.Path(output_path).write_text(content.replace("@IPADIC_DIR@", escaped), encoding="utf-8")
print(f"==> Rendered schema: {output_path}")
print(f"    dict = {dict_dir}")
PY

echo "==> Creating index at $INDEX_DIR"
"$LAURUS" --index-dir "$INDEX_DIR" create index --schema "$RENDERED"

echo "==> Verifying the analyzer wiring..."
"$LAURUS" --index-dir "$INDEX_DIR" get schema >/dev/null

echo "==> Done. Index created at $INDEX_DIR"
