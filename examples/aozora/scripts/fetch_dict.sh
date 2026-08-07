#!/usr/bin/env bash
# Download and extract the Lindera IPADIC dictionary used by this example.
#
# `laurus` does not enable Lindera's `embed-*` features in release builds
# (see laurus/Cargo.toml), so the `laurus` binary cannot resolve
# `dict = "embedded://ipadic"` and needs a real filesystem dictionary
# directory instead. This script fetches a pre-built one from the Lindera
# project's GitHub releases.
#
# Idempotent: exits early when the dictionary is already present. The
# version is resolved from `cargo metadata` so the dictionary binary
# format always matches the `lindera` crate version laurus links against
# (the same technique used by .github/workflows/deploy-docs.yml).
#
# Usage: bash fetch_dict.sh [--force]
#
# On success, prints the absolute path to the extracted dictionary
# directory as the last line of stdout.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
# Resolved via `cd && pwd` (not a plain `../dict` join) so the printed path
# has no `..` components — it is embedded verbatim into schema.toml as the
# Lindera `dict` path, which create_index.sh needs to be independent of
# any caller's working directory.
DICT_DIR="$(cd "$SCRIPT_DIR/.." && mkdir -p dict && cd dict && pwd)"
DICT_PATH="$DICT_DIR/lindera-ipadic"

FORCE=0
if [ "${1:-}" = "--force" ]; then
  FORCE=1
fi

for cmd in curl unzip python3 cargo; do
  if ! command -v "$cmd" &>/dev/null; then
    echo "Error: $cmd is required but not installed." >&2
    exit 1
  fi
done

# --- 1. Skip if already present (idempotent) ---
if [ "$FORCE" -eq 0 ] && [ -f "$DICT_PATH/dict.da" ] && [ -f "$DICT_PATH/metadata.json" ]; then
  echo "==> IPADIC already present at $DICT_PATH" >&2
  echo "$DICT_PATH"
  exit 0
fi

# --- 2. Resolve the lindera crate version from cargo metadata ---
LINDERA_VERSION=$(cargo metadata --manifest-path "$PROJECT_ROOT/Cargo.toml" --format-version 1 2>/dev/null \
  | python3 -c "
import json, sys
metadata = json.load(sys.stdin)
versions = sorted({p['version'] for p in metadata['packages'] if p['name'] == 'lindera'})
print(versions[-1] if versions else '')
")

if [ -z "$LINDERA_VERSION" ]; then
  echo "Error: failed to resolve the lindera version from cargo metadata." >&2
  echo "Hint: check the 'lindera' entry in laurus/Cargo.toml." >&2
  exit 1
fi
echo "==> Lindera version: $LINDERA_VERSION" >&2

# --- 3. Download (the zip itself is also cached) ---
ZIP="$DICT_DIR/lindera-ipadic-${LINDERA_VERSION}.zip"
URL="https://github.com/lindera/lindera/releases/download/v${LINDERA_VERSION}/lindera-ipadic-${LINDERA_VERSION}.zip"

mkdir -p "$DICT_DIR"
if [ "$FORCE" -eq 1 ] || [ ! -f "$ZIP" ]; then
  echo "==> Downloading $URL" >&2
  if ! curl -fsSL -o "$ZIP.part" "$URL"; then
    rm -f "$ZIP.part"
    echo "Error: download failed. Is there a release asset for v$LINDERA_VERSION?" >&2
    exit 1
  fi
  # Atomic rename: an interrupted download never leaves a corrupt file at
  # the expected cache path.
  mv "$ZIP.part" "$ZIP"
fi

# --- 4. Extract and verify ---
echo "==> Extracting to $DICT_DIR" >&2
unzip -q -o "$ZIP" -d "$DICT_DIR"

for f in metadata.json dict.da dict.vals dict.wordsidx dict.words matrix.mtx char_def.bin unk.bin; do
  if [ ! -f "$DICT_PATH/$f" ]; then
    echo "Error: missing $f in $DICT_PATH after extraction." >&2
    exit 1
  fi
done

echo "==> IPADIC ready at $DICT_PATH" >&2
echo "$DICT_PATH"
