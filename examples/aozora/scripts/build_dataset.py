#!/usr/bin/env python3
"""Build a laurus-ready JSONL dataset from Aozora Bunko.

Downloads the Aozora Bunko full work list, selects public-domain works
with a body text URL, downloads and cleans each work's body text, and
writes one laurus `put docs` JSONL line per work.

Usage:
    python3 build_dataset.py --data-dir <dir> --output <jsonl> [options]

Options:
    --limit N        Index only the first N works, ordered by work ID
                      (ascending). 0 means all works. Default: 1000.
                      Aozora Bunko's low work IDs skew toward well-known
                      authors (Akutagawa, Miyazawa, Natsume, Dazai, ...),
                      so "first N" already tends to be a reasonable demo
                      selection.
    --ndc CODE        Only include works whose NDC classification code
                      contains CODE (e.g. "913" for Japanese novels).
    --author NAME     Only include works whose author name contains NAME
                      (substring match against "姓名").
    --parallel N      Number of concurrent body-text downloads. Default: 4.
    --sleep SECONDS   Delay between download requests, per worker.
                      Default: 0.2.
    --refresh-list    Re-download the work list CSV even if cached.
    --yes             Skip the confirmation delay for --limit 0 (all
                      works).

Requires only the Python standard library (csv, zipfile, urllib, re,
json, argparse, concurrent.futures, pathlib) — no extra pip installs, to
mirror the "jq + curl + python3" dependency footprint of examples/movies.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

WORK_LIST_URL = (
    "https://raw.githubusercontent.com/aozorabunko/aozorabunko/master/"
    "index_pages/list_person_all_extended_utf8.zip"
)
USER_AGENT = "laurus-example/0.9 (+https://github.com/mosuka/laurus)"

# Column indices in list_person_all_extended_utf8.csv (55 columns; verified
# against a live download). This file is a work x person join table: a
# translated work has one row per contributor (author, translator, ...).
COL_WORK_ID = 0
COL_TITLE = 1
COL_NDC = 8
COL_RIGHTS_FLAG = 10
COL_LAST_NAME = 15
COL_FIRST_NAME = 16
COL_ROLE = 23
COL_TEXT_URL = 45
COL_TEXT_ENCODING = 47

RULER = "-" * 55


def log(message: str) -> None:
    print(message, file=sys.stderr)


def fetch_work_list(data_dir: Path, refresh: bool) -> Path:
    """Download and extract the Aozora Bunko work list CSV, with caching."""
    zip_path = data_dir / "list_person_all_extended_utf8.zip"
    csv_path = data_dir / "list_person_all_extended_utf8.csv"

    if refresh or not zip_path.exists():
        log(f"==> Downloading work list from {WORK_LIST_URL}")
        request = urllib.request.Request(WORK_LIST_URL, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=60) as response:
            data_dir.mkdir(parents=True, exist_ok=True)
            zip_path.write_bytes(response.read())

    if refresh or not csv_path.exists():
        log(f"==> Extracting {zip_path.name}")
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(data_dir)

    if not csv_path.exists():
        sys.exit(f"Error: {csv_path} not found after extraction.")

    return csv_path


def load_public_domain_rows(csv_path: Path) -> list[list[str]]:
    """Read the work list CSV and return public-domain rows with a body URL."""
    rows: list[list[str]] = []
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        next(reader)  # header
        for line in reader:
            if len(line) <= COL_TEXT_ENCODING:
                continue
            if line[COL_RIGHTS_FLAG] != "なし":
                continue
            text_url = line[COL_TEXT_URL]
            if not text_url.startswith("https://www.aozora.gr.jp/cards/") or not text_url.endswith(".zip"):
                continue
            rows.append(line)
    return rows


def fold_into_works(rows: list[list[str]]) -> dict[str, dict]:
    """Fold work x person rows into one record per work ID.

    The CSV is a join table: a translated work appears as one row per
    contributor (author, translator, editor, ...). This groups by work ID
    and picks the "著者" (author) row as the representative contributor,
    falling back to the first row when no author role is present.
    """
    works: dict[str, list[list[str]]] = {}
    for row in rows:
        works.setdefault(row[COL_WORK_ID], []).append(row)

    folded: dict[str, dict] = {}
    for work_id, contributor_rows in works.items():
        author_row = next(
            (r for r in contributor_rows if r[COL_ROLE] == "著者"),
            contributor_rows[0],
        )
        author_name = f"{author_row[COL_LAST_NAME]} {author_row[COL_FIRST_NAME]}".strip()
        author_name_exact = f"{author_row[COL_LAST_NAME]}{author_row[COL_FIRST_NAME]}".strip()
        folded[work_id] = {
            "work_id": work_id,
            "title": author_row[COL_TITLE],
            "author": author_name,
            "author_exact": author_name_exact,
            "ndc": author_row[COL_NDC].replace("NDC ", "").strip(),
            "text_url": author_row[COL_TEXT_URL],
            "card_url": f"https://www.aozora.gr.jp/cards/{work_id[:6]}/card{int(work_id)}.html"
            if work_id.isdigit()
            else "",
        }
    return folded


def select_works(
    works: dict[str, dict],
    limit: int,
    ndc_filter: str | None,
    author_filter: str | None,
) -> list[dict]:
    """Sort by work ID ascending, apply filters, then apply --limit."""
    ordered = sorted(works.values(), key=lambda w: w["work_id"])

    if ndc_filter:
        ordered = [w for w in ordered if ndc_filter in w["ndc"]]
    if author_filter:
        ordered = [w for w in ordered if author_filter in w["author"]]

    total = len(ordered)
    if limit > 0:
        ordered = ordered[:limit]
        log(f"==> Selected {len(ordered)} of {total} matching works (--limit {limit})")
    else:
        log(f"==> Selected all {total} matching works (--limit 0)")

    return ordered


def download_body(work: dict, cache_dir: Path, sleep_seconds: float) -> Path | None:
    """Download a work's body-text zip to the cache, skipping if present."""
    dest = cache_dir / f"{work['work_id']}.zip"
    if dest.exists():
        return dest

    try:
        request = urllib.request.Request(work["text_url"], headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=30) as response:
            dest.write_bytes(response.read())
    except Exception as exc:  # noqa: BLE001 - one failed download must not abort the run
        log(f"    warning: failed to download {work['work_id']} ({work['title']}): {exc}")
        return None
    finally:
        time.sleep(sleep_seconds)

    return dest


def strip_header_lines(text: str) -> str:
    """Drop the leading title/author lines for files with no ruler block."""
    lines = text.split("\n")
    non_empty_seen = 0
    for i, line in enumerate(lines):
        if line.strip():
            non_empty_seen += 1
            if non_empty_seen == 2:
                return "\n".join(lines[i + 1 :])
    return text


def clean_body(text: str) -> str:
    """Strip Aozora Bunko's ruby/annotation markup and front/back matter."""
    # The symbol-legend block is fenced by a 55-dash ruler on each side;
    # 3+ pieces means both rulers were found, so the body is the last one.
    parts = text.split(RULER)
    body = parts[-1] if len(parts) >= 3 else strip_header_lines(text)

    for marker in ("底本：", "底本:"):
        if marker in body:
            body = body.split(marker)[0]
            break

    body = re.sub(r"《[^》]*》", "", body)  # ルビ (ruby)
    body = re.sub(r"※?［＃[^］]*］", "", body)  # 入力者注 / 外字注記 (annotations)
    body = body.replace("｜", "")  # ルビ開始位置指定 (ruby-start marker)
    body = re.sub(r"\n{3,}", "\n\n", body)  # collapse long blank runs

    return body.strip()


def extract_text_from_zip(zip_path: Path) -> str | None:
    """Decode the first .txt entry in a downloaded body-text zip."""
    try:
        with zipfile.ZipFile(zip_path) as archive:
            txt_names = [n for n in archive.namelist() if n.lower().endswith(".txt")]
            if not txt_names:
                return None
            # Aozora zips occasionally carry more than one file; the body
            # text is the largest one when that happens.
            txt_name = max(txt_names, key=lambda n: archive.getinfo(n).file_size)
            raw = archive.read(txt_name)
    except zipfile.BadZipFile:
        return None

    # cp932, not shift_jis: Aozora Bunko text uses NEC/IBM extended
    # characters (①, Ⅰ, ㈱, ...) outside strict Shift_JIS.
    text = raw.decode("cp932", errors="replace")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def build_document(work: dict, body: str) -> dict:
    excerpt = body[:200].replace("\n", "")
    return {
        "id": work["work_id"],
        "fields": {
            "title": work["title"],
            "author": work["author"],
            "author_exact": work["author_exact"],
            "body": body,
            "excerpt": excerpt,
            "ndc": work["ndc"],
            "chars": len(body),
            "card_url": work["card_url"],
            # Same text as `title`/`body`, copied into the Hnsw vector
            # fields — the engine embeds it automatically via the
            # schema's `ja_text_embedder` (see schema.toml).
            "title_vec": work["title"],
            "body_vec": body,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--ndc", default=None)
    parser.add_argument("--author", default=None)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--refresh-list", action="store_true")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    if args.limit == 0 and not args.yes:
        log("Warning: --limit 0 will download the body text of every public-domain")
        log("         work on Aozora Bunko (tens of thousands of requests) against a")
        log("         volunteer-run server. Starting in 5 seconds — Ctrl-C to abort,")
        log("         or pass --yes to skip this wait next time.")
        time.sleep(5)

    args.data_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.data_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    csv_path = fetch_work_list(args.data_dir, args.refresh_list)
    rows = load_public_domain_rows(csv_path)
    log(f"==> {len(rows)} public-domain rows with a body text URL")

    works = fold_into_works(rows)
    log(f"==> {len(works)} unique works after folding translator/editor rows")

    selected = select_works(works, args.limit, args.ndc, args.author)
    if not selected:
        sys.exit("Error: no works matched the given filters.")

    log(f"==> Downloading body text (parallel={args.parallel})")
    downloaded = 0
    skipped_download = 0
    skipped_empty = 0
    written = 0

    with args.output.open("w", encoding="utf-8") as out, ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {
            pool.submit(download_body, work, cache_dir, args.sleep): work for work in selected
        }
        for future in as_completed(futures):
            work = futures[future]
            zip_path = future.result()
            if zip_path is None:
                skipped_download += 1
                continue
            downloaded += 1

            text = extract_text_from_zip(zip_path)
            if text is None:
                log(f"    warning: no .txt found in archive for {work['work_id']} ({work['title']})")
                skipped_empty += 1
                continue

            body = clean_body(text)
            if not body:
                skipped_empty += 1
                continue

            out.write(json.dumps(build_document(work, body), ensure_ascii=False) + "\n")
            written += 1

    log(
        f"==> Done. {written} works written to {args.output} "
        f"(downloaded={downloaded}, download_failed={skipped_download}, empty_or_unparsable={skipped_empty})"
    )


if __name__ == "__main__":
    main()
