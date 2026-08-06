#!/usr/bin/env bash
# Example search queries against the movies index.
# Includes both lexical and multimodal (CLIP vector) searches.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
INDEX_DIR="$SCRIPT_DIR/../index"

echo "==> Building laurus (release, embeddings-multimodal)..."
cargo build --manifest-path "$PROJECT_ROOT/Cargo.toml" --release --bin laurus \
  --features embeddings-multimodal
LAURUS="$PROJECT_ROOT/target/release/laurus"

# --- Lexical searches ---

echo "=== Lexical Search: 'star wars' ==="
"$LAURUS" --index-dir "$INDEX_DIR" search "star wars" --limit 5
echo

echo "=== Lexical Search: title:nemo ==="
"$LAURUS" --index-dir "$INDEX_DIR" search "title:nemo" --limit 5
echo

echo "=== Lexical Search: genres:comedy ==="
"$LAURUS" --index-dir "$INDEX_DIR" search "genres:comedy" --limit 5
echo

echo '=== Lexical Search: overview:"robot" ==='
"$LAURUS" --index-dir "$INDEX_DIR" search "overview:robot" --limit 5
echo

echo "=== Lexical Search (JSON): 'star wars' ==="
"$LAURUS" --index-dir "$INDEX_DIR" --format json search "star wars" --limit 3
echo

# --- Multimodal (CLIP vector) searches ---

echo '=== Multimodal Search: poster_vec:"space adventure" ==='
"$LAURUS" --index-dir "$INDEX_DIR" search 'poster_vec:"space adventure"' --limit 5
echo

echo '=== Multimodal Search: poster_vec:"romantic couple" ==='
"$LAURUS" --index-dir "$INDEX_DIR" search 'poster_vec:"romantic couple"' --limit 5
echo

echo '=== Multimodal Search: poster_vec:"scary monster horror" ==='
"$LAURUS" --index-dir "$INDEX_DIR" search 'poster_vec:"scary monster horror"' --limit 5
