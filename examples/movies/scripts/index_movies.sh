#!/usr/bin/env bash
# Index the Meilisearch movies dataset into laurus.
# Uses the REPL with piped stdin to avoid per-document process startup overhead.
#
# This script:
#   1. Downloads poster images from TMDB to examples/movies/images/
#   2. Indexes movies with lexical fields AND poster_vec (CLIP embedding)
#
# Usage:
#   bash index_movies.sh [--limit N]
#
#   --limit N   Index only the first N movies (default: all)
#
# Requires: jq, curl, python3 (for binary-to-JSON-array conversion)
# Build feature: embeddings-multimodal (for CLIP-based poster embedding)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
INDEX_DIR="$SCRIPT_DIR/../index"
IMAGES_DIR="$SCRIPT_DIR/../images"
DATASET="$PROJECT_ROOT/../datasets/datasets/movies/movies.json"

# --- Parse arguments ---
LIMIT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    *)
      echo "Usage: $0 [--limit N]" >&2
      exit 1
      ;;
  esac
done

if [ ! -f "$DATASET" ]; then
  echo "Error: Dataset not found at $DATASET" >&2
  exit 1
fi

for cmd in jq curl python3; do
  if ! command -v "$cmd" &>/dev/null; then
    echo "Error: $cmd is required but not installed." >&2
    exit 1
  fi
done

# Build the release binary with multimodal embedding support.
echo "==> Building laurus (release, embeddings-multimodal)..."
cargo build --manifest-path "$PROJECT_ROOT/Cargo.toml" --release --bin laurus \
  --features embeddings-multimodal
LAURUS="$PROJECT_ROOT/target/release/laurus"

DATASET_TOTAL=$(jq length "$DATASET")
if [ -n "$LIMIT" ]; then
  TOTAL="$LIMIT"
  echo "==> Limiting to first $TOTAL of $DATASET_TOTAL movies"
else
  TOTAL="$DATASET_TOTAL"
fi
COMMIT_INTERVAL=1000
DOWNLOAD_PARALLEL=8

# --- Phase 1: Download poster images ---
mkdir -p "$IMAGES_DIR"

echo "==> Downloading poster images to $IMAGES_DIR (parallel=$DOWNLOAD_PARALLEL)"

# Generate download list: ID TAB URL (skip null posters).
# When --limit is set, only download images for the first N movies.
JQ_SLICE=".[]"
if [ -n "$LIMIT" ]; then
  JQ_SLICE=".[:${LIMIT}][]"
fi

jq -r "${JQ_SLICE} | select(.poster != null and .poster != \"\") | \"\(.id)\t\(.poster)\"" "$DATASET" \
  | while IFS=$'\t' read -r id url; do
    dest="$IMAGES_DIR/${id}.jpg"
    # Skip already-downloaded images (idempotent).
    if [ -f "$dest" ]; then
      continue
    fi
    echo "$url $dest"
  done \
  | xargs -P "$DOWNLOAD_PARALLEL" -L 1 bash -c '
    url="$0"; dest="$1"
    curl -sS -L --fail -o "$dest" "$url" 2>/dev/null || rm -f "$dest"
  '

DOWNLOADED=$(find "$IMAGES_DIR" -name '*.jpg' -type f | wc -l)
echo "==> Downloaded $DOWNLOADED poster images."

# --- Phase 2: Index documents ---
echo "==> Indexing $TOTAL movies into $INDEX_DIR"

# Helper: convert a binary file to a JSON integer array and write to a temp file.
# Usage: file_to_json_array <image_path> <output_path>
# Returns 0 on success, 1 on failure.
file_to_json_array() {
  python3 -c "
import sys, json
with open(sys.argv[1], 'rb') as f:
    data = f.read()
with open(sys.argv[2], 'w') as out:
    json.dump(list(data), out)
" "$1" "$2" 2>/dev/null
}

BYTES_TMPFILE=$(mktemp)
trap 'rm -f "$BYTES_TMPFILE"' EXIT

# Generate REPL commands from the dataset and pipe them into a single laurus process.
generate_commands() {
  local count=0
  jq -c "${JQ_SLICE}" "$DATASET" | while IFS= read -r movie; do
    ID=$(echo "$movie" | jq -r '.id')

    # Build the base document with lexical fields.
    BASE_DOC=$(echo "$movie" | jq -c '{
      title: {Text: .title},
      overview: {Text: .overview},
      genres: {Text: (.genres | join(", "))},
      poster: {Text: .poster},
      release_date: {Int64: .release_date}
    }')

    # If the poster image was downloaded, add poster_vec as Bytes field.
    # Use --slurpfile to read the byte array from a temp file instead of
    # --argjson, which would hit the shell argument length limit for large images.
    IMAGE_FILE="$IMAGES_DIR/${ID}.jpg"
    if [ -f "$IMAGE_FILE" ] && file_to_json_array "$IMAGE_FILE" "$BYTES_TMPFILE"; then
      DOC=$(echo "$BASE_DOC" | jq -c --slurpfile bytes "$BYTES_TMPFILE" \
        '. + {poster_vec: {Bytes: [$bytes[0], "image/jpeg"]}} | {fields: .}')
    else
      DOC=$(echo "$BASE_DOC" | jq -c '{fields: .}')
    fi

    echo "doc add $ID $DOC"

    count=$((count + 1))
    if [ $((count % COMMIT_INTERVAL)) -eq 0 ]; then
      echo "commit"
      echo "    Committed $count / $TOTAL documents" >&2
    fi
  done

  # Final commit and exit.
  echo "commit"
  echo "quit"
}

generate_commands | "$LAURUS" --index-dir "$INDEX_DIR" repl >/dev/null

echo "==> Done. Indexed $TOTAL movies."
