#!/usr/bin/env bash
# Runs the fixture queries against an index and saves search / stats / doc
# results as JSON. Called identically from both sides of the cross-platform
# interop check (macOS "expected" generation and Linux "actual" generation)
# so no divergent code path can hide a real incompatibility.
#
# Usage: run-queries.sh <laurus-bin> <index-dir> <out-dir>
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <laurus-bin> <index-dir> <out-dir>" >&2
  exit 1
fi

LAURUS_BIN="$1"
INDEX_DIR="$2"
OUT_DIR="$3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUERIES_TSV="${SCRIPT_DIR}/queries.tsv"

# Representative document IDs whose stored fields are dumped verbatim, to
# exercise the document store (not just search ranking).
REPRESENTATIVE_IDS=(doc001 doc012 doc020 doc032)

mkdir -p "$OUT_DIR"

"$LAURUS_BIN" --index-dir "$INDEX_DIR" --format json get stats > "${OUT_DIR}/stats.json"

for id in "${REPRESENTATIVE_IDS[@]}"; do
  "$LAURUS_BIN" --index-dir "$INDEX_DIR" --format json get docs --id "$id" > "${OUT_DIR}/doc-${id}.json"
done

while IFS=$'\t' read -r name mode limit query; do
  # Skip blank lines.
  [ -z "$name" ] && continue
  echo "Running query '${name}' (${mode}, limit=${limit}): ${query}"
  "$LAURUS_BIN" --index-dir "$INDEX_DIR" --format json search "$query" --limit "$limit" \
    > "${OUT_DIR}/query-${name}.json"
done < "$QUERIES_TSV"

echo "Saved results for $(wc -l < "$QUERIES_TSV" | tr -d ' ') queries to ${OUT_DIR}"
