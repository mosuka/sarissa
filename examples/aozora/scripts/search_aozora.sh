#!/usr/bin/env bash
# Example search queries against the Aozora Bunko index.
#
# Each example is labeled and printed with `=== ... ===` so failures are
# easy to spot; a query returning zero hits does not stop the script
# (some examples are deliberately chosen to return zero hits, to
# demonstrate exact- vs partial-match semantics).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
INDEX_DIR="$SCRIPT_DIR/../index"

echo "==> Building laurus (release)..."
cargo build --manifest-path "$PROJECT_ROOT/Cargo.toml" --release --bin laurus \
  --features embeddings-candle
LAURUS="$PROJECT_ROOT/target/release/laurus"

run() { # run <label> <query> [extra laurus args...]
  local label="$1"
  local query="$2"
  shift 2
  echo "=== $label: $query ==="
  "$LAURUS" --index-dir "$INDEX_DIR" search "$query" --limit 5 "$@" || true
  echo
}

# --- 1. Basic full-text search (Lindera morphological analysis) -----------
run "全文検索" "羅生門"

# --- 2. Field-specific search -----------------------------------------------
run "フィールド指定（作品名）" "title:こころ"
run "フィールド指定（本文）" "body:蜘蛛の糸"

# --- 3. Author search: two analyzers, deliberately contrasted -------------
# author       … Lindera analysis → matches on the surname alone
# author_exact … keyword analyzer → exact match only
run "著者・部分一致（Lindera analyzer）" "author:芥川"
run "著者・完全一致（keyword analyzer）" "author_exact:芥川竜之介"
run "著者・完全一致は部分一致しない（意図的に 0 件）" "author_exact:芥川"

# --- 4. Phrase (strict adjacency) vs. unquoted (OR-relaxed) ---------------
# Quoted   → PhraseQuery: requires the exact morpheme sequence.
# Unquoted → BooleanQuery(Should): each analyzed morpheme is OR'd.
run "フレーズ検索（厳密・引用符あり）" 'title:"銀河鉄道の夜"'
run "引用符なし（OR 緩和）" "title:銀河鉄道の夜"

# --- 5. A phrase containing particles ---------------------------------------
# schema.toml's ja_ipadic analyzer keeps every morpheme (no stop-word
# filter), so a phrase with particles like "は"/"の" still matches exactly.
run "助詞を含むフレーズ検索" 'title:"吾輩は猫である"'

# --- 6. Natural-sentence search ---------------------------------------------
# Bare (unquoted) terms containing punctuation are a parse error — the
# query grammar accepts Unicode letters/numbers as bare terms, but not
# punctuation (。、「」など). Quote any sentence that includes it.
run "自然文検索（引用符で囲む）" '"ある日の暮方の事である"'

# --- 7. Boolean operators ---------------------------------------------------
run "AND" "body:恋愛 AND ndc:913"
run "OR" "title:こころ OR title:三四郎"
run "除外 (-)" "body:恋愛 -author:漱石"

# --- 8. Classification / numeric range filters ------------------------------
run "NDC 分類（913 = 日本の小説）" "ndc:913"
run "長編のみ（本文 5 万字以上）" "chars:[50000 TO *]"

# --- 9. Vector search / hybrid search (Candle BERT embedding) ---------------
# title_vec/body_vec are semantic (Hnsw) fields backed by
# sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (see
# schema.toml). Unlike the lexical fields above, these match by meaning,
# not by morpheme — a query can surface works that never contain the
# literal query text.
run "ベクトル検索（作品名の意味的類似）" 'title_vec:"人間の孤独と疎外感"'
run "ハイブリッド検索（レキシカル OR ベクトル、既定は RRF 融合）" 'title:こころ body_vec:"人間の孤独感"'
run "ハイブリッド検索（ベクトル節を + で必須化）" 'title:こころ +body_vec:"人間の孤独感"'

# --- 10. JSON output ----------------------------------------------------------
echo "=== JSON 出力: 羅生門 ==="
"$LAURUS" --index-dir "$INDEX_DIR" --format json search "羅生門" --limit 2 || true
echo
