#!/usr/bin/env bash
# Bump the laurus workspace version across every file that needs it by hand.
#
# What it does:
#   1. Reads the current version from the root Cargo.toml's
#      [workspace.package].version.
#   2. Updates that version and the internal laurus / laurus-mcp /
#      laurus-server dependency pins in [workspace.dependencies].
#   3. Regenerates Cargo.lock. Every other workspace member crate
#      (laurus-cli, laurus-python, laurus-nodejs, laurus-wasm, laurus-ruby,
#      laurus-php) inherits the version via `version = { workspace = true }`,
#      so they update automatically — no per-crate edits needed.
#   4. Updates the documented `laurus = "X.Y"` (major.minor, no patch)
#      install examples in the root/laurus READMEs and the mdBook docs.
#   5. Updates the `VERSION=vX.Y.Z` release-binary download example in the
#      laurus-cli installation docs.
#
# What it does NOT do:
#   * Commit, tag, or push anything — review the diff yourself
#     (`git diff`), then commit and open a PR by hand.
#   * Touch language-binding manifests (pyproject.toml, package.json,
#     *.gemspec, composer.json) — release.yml's publish jobs inject the
#     version dynamically from `cargo metadata` at publish time, so those
#     files intentionally stay at their placeholder values.
#   * Create the release tag — tagging `vX.Y.Z` triggers the release build
#     and crates.io/PyPI/npm/RubyGems publishing, and should only happen
#     after this bump has been reviewed and merged.
#
# Usage:
#   ./scripts/bump-up-version.sh <new-version>
#
# Example:
#   ./scripts/bump-up-version.sh 0.11.0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <new-version>" >&2
  echo "Example: $0 0.11.0" >&2
  exit 1
fi

NEW_VERSION="$1"

# This project's releases are plain major.minor.patch — no pre-release or
# build-metadata suffixes — so a strict X.Y.Z check is enough.
if ! [[ "$NEW_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "Error: '$NEW_VERSION' does not look like a version (expected X.Y.Z)" >&2
  exit 1
fi
NEW_MAJOR_MINOR="${NEW_VERSION%.*}"

CURRENT_VERSION=$(awk -F'"' '/^version = "/{print $2; exit}' Cargo.toml)
if [ -z "$CURRENT_VERSION" ]; then
  echo "Error: could not find [workspace.package].version in Cargo.toml" >&2
  exit 1
fi
if [ "$CURRENT_VERSION" = "$NEW_VERSION" ]; then
  echo "Error: current version is already $NEW_VERSION" >&2
  exit 1
fi
CURRENT_MAJOR_MINOR="${CURRENT_VERSION%.*}"

echo "==> Bumping version: $CURRENT_VERSION -> $NEW_VERSION"

# Portable in-place sed: `-i.bak` (suffix glued directly to `-i`, no space)
# is the one form both GNU sed (Linux) and BSD sed (macOS) accept
# identically; the backup is removed right after.
sed_inplace() {
  sed -i.bak "$@"
}

# --- 1. Cargo.toml: workspace version + internal dependency pins ---
echo "==> Updating Cargo.toml"
sed_inplace \
  -e "s/^version = \"${CURRENT_VERSION}\"/version = \"${NEW_VERSION}\"/" \
  -e "s/{ version = \"${CURRENT_VERSION}\", path = /{ version = \"${NEW_VERSION}\", path = /g" \
  Cargo.toml
rm -f Cargo.toml.bak

# --- 2. Cargo.lock: regenerate so every workspace member reflects it ---
#
# `cargo check -p laurus-cli` is enough: Cargo.lock is resolved for the
# whole workspace regardless of which single member is checked, so every
# local crate's version gets updated in the lockfile without actually
# compiling anything else. Building the *whole* workspace here would also
# try to link laurus-ruby/laurus-php, whose cdylibs only get the linker
# flags they need (dynamic Ruby/PHP symbol lookup) when driven through
# their own tooling (`bundle exec rake compile`, or an explicit RUSTFLAGS
# for PHP) — a bare `cargo build`/`cargo check` on those crates fails.
echo "==> Regenerating Cargo.lock (cargo check -p laurus-cli)"
cargo check -p laurus-cli --no-default-features >/dev/null

# --- 3. Documented install examples (major.minor only, no patch) ---
DOC_FILES=(
  laurus/README.md
  laurus/README_ja.md
  docs/src/laurus.md
  docs/ja/src/laurus.md
  docs/src/getting_started/installation.md
  docs/ja/src/getting_started/installation.md
  docs/src/development/feature_flags.md
  docs/ja/src/development/feature_flags.md
)
echo "==> Updating documented install examples (laurus = \"${CURRENT_MAJOR_MINOR}\" -> \"${NEW_MAJOR_MINOR}\")"
for f in "${DOC_FILES[@]}"; do
  if [ ! -f "$f" ]; then
    echo "Warning: $f not found, skipping" >&2
    continue
  fi
  sed_inplace \
    -e "s/laurus = \"${CURRENT_MAJOR_MINOR}\"/laurus = \"${NEW_MAJOR_MINOR}\"/g" \
    -e "s/laurus = { version = \"${CURRENT_MAJOR_MINOR}\"/laurus = { version = \"${NEW_MAJOR_MINOR}\"/g" \
    "$f"
  rm -f "${f}.bak"
done

# --- 4. CLI release-binary download example (full version, with patch) ---
CLI_INSTALL_FILES=(
  docs/src/laurus-cli/installation.md
  docs/ja/src/laurus-cli/installation.md
)
echo "==> Updating CLI install examples (VERSION=v${CURRENT_VERSION} -> v${NEW_VERSION})"
for f in "${CLI_INSTALL_FILES[@]}"; do
  if [ ! -f "$f" ]; then
    echo "Warning: $f not found, skipping" >&2
    continue
  fi
  sed_inplace "s/VERSION=v${CURRENT_VERSION}/VERSION=v${NEW_VERSION}/g" "$f"
  rm -f "${f}.bak"
done

echo "==> Done."
echo "    Review with 'git diff', then commit and open a PR by hand."
echo "    Not touched (by design): language-binding manifests, release tag."
