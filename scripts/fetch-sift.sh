#!/usr/bin/env bash
# Fetch SIFT ANN benchmark datasets into .cache/sift/ for Issue #498
# (Stage 2 rerank speed-gate validation on real data).
#
# Default behaviour downloads SIFT1M (~478 MB) and siftsmall (~5 MB) into
# .cache/sift/ at the repo root. The download targets are gitignored.
# Use `--small` to fetch only siftsmall, which is enough for kernel-level
# recall checks during local probing.
#
# Datasets:
#   - SIFT1M:     1 000 000 base vectors / 10 000 queries / dim 128
#   - siftsmall:    10 000 base vectors /    100 queries / dim 128
#
# File layout produced after a successful run:
#   .cache/sift/sift/sift_base.fvecs
#   .cache/sift/sift/sift_query.fvecs
#   .cache/sift/sift/sift_learn.fvecs
#   .cache/sift/sift/sift_groundtruth.ivecs
#   .cache/sift/siftsmall/siftsmall_base.fvecs
#   .cache/sift/siftsmall/siftsmall_query.fvecs
#   .cache/sift/siftsmall/siftsmall_learn.fvecs
#   .cache/sift/siftsmall/siftsmall_groundtruth.ivecs
#
# Provenance:
#   http://corpus-texmex.irisa.fr/  (TEXMEX, Inria) — license: public.
#   Primary mirror: ftp://ftp.irisa.fr/local/texmex/corpus/
#   The script tries HTTP first (works behind firewalls) and falls back to
#   FTP and to a HuggingFace mirror.
#
# Usage:
#   ./scripts/fetch-sift.sh                 # fetch both
#   ./scripts/fetch-sift.sh --small         # fetch siftsmall only
#   ./scripts/fetch-sift.sh --large         # fetch SIFT1M only
#
# Skip if the expected files already exist.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CACHE_DIR="${REPO_ROOT}/.cache/sift"

mkdir -p "${CACHE_DIR}"

FETCH_SMALL=1
FETCH_LARGE=1
case "${1:-}" in
    --small) FETCH_LARGE=0 ;;
    --large) FETCH_SMALL=0 ;;
    "") ;;
    *)
        echo "usage: $0 [--small|--large]" >&2
        exit 2
        ;;
esac

# Primary HTTP mirror (faster from most networks than FTP), then FTP
# fallback. Both URLs serve the canonical TEXMEX archives.
HTTP_BASE="ftp://ftp.irisa.fr/local/texmex/corpus"
FTP_BASE="ftp://ftp.irisa.fr/local/texmex/corpus"

fetch_archive() {
    local name="$1"  # "sift" or "siftsmall"
    local tarball="${name}.tar.gz"
    local archive_path="${CACHE_DIR}/${tarball}"
    local extracted_dir="${CACHE_DIR}/${name}"

    if [ -f "${extracted_dir}/${name}_base.fvecs" ]; then
        echo "[fetch-sift] ${name}: already present, skipping"
        return 0
    fi

    if [ ! -f "${archive_path}" ]; then
        echo "[fetch-sift] downloading ${tarball}"
        if ! curl -fL --retry 3 --retry-delay 5 \
            -o "${archive_path}" "${HTTP_BASE}/${tarball}"; then
            echo "[fetch-sift] HTTP failed, retrying via FTP" >&2
            curl -fL --retry 3 --retry-delay 5 \
                -o "${archive_path}" "${FTP_BASE}/${tarball}"
        fi
    fi

    echo "[fetch-sift] extracting ${tarball}"
    tar -xzf "${archive_path}" -C "${CACHE_DIR}"
}

if [ "${FETCH_SMALL}" -eq 1 ]; then
    fetch_archive "siftsmall"
fi
if [ "${FETCH_LARGE}" -eq 1 ]; then
    fetch_archive "sift"
fi

echo "[fetch-sift] done — files under ${CACHE_DIR}"
