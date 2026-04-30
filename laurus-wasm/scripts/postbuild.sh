#!/usr/bin/env bash
# Post-process the wasm-pack output for laurus-wasm.
#
# wasm-pack ships only the generated JS / WASM / d.ts files. This script
# adds the OPFS helper module (./opfs) to the publishable package by
# copying js/opfs.js and js/opfs.d.ts into pkg/ and patching pkg/package.json
# so consumers can `import { downloadDictionary } from "laurus-wasm/opfs"`.
#
# Run after `wasm-pack build laurus-wasm --target web --release`. Idempotent.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRATE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PKG_DIR="${CRATE_DIR}/pkg"
JS_DIR="${CRATE_DIR}/js"

if [[ ! -d "${PKG_DIR}" ]]; then
    echo "error: ${PKG_DIR} does not exist (run wasm-pack build first)" >&2
    exit 1
fi

if [[ ! -f "${JS_DIR}/opfs.js" ]] || [[ ! -f "${JS_DIR}/opfs.d.ts" ]]; then
    echo "error: js/opfs.js or js/opfs.d.ts missing" >&2
    exit 1
fi

cp "${JS_DIR}/opfs.js" "${PKG_DIR}/opfs.js"
cp "${JS_DIR}/opfs.d.ts" "${PKG_DIR}/opfs.d.ts"

# Patch pkg/package.json to expose the opfs subpath and include the new
# files in the publishable file list. Uses Node so we can edit JSON
# safely without external dependencies (Node ships with wasm-pack's
# Rust toolchain prerequisites and is in CI).
node --input-type=module -e '
import { readFileSync, writeFileSync } from "node:fs";
const path = process.argv[1];
const pkg = JSON.parse(readFileSync(path, "utf8"));
const ensure = (arr, val) => arr.includes(val) ? arr : [...arr, val];
pkg.files = ensure(pkg.files ?? [], "opfs.js");
pkg.files = ensure(pkg.files, "opfs.d.ts");
pkg.exports = pkg.exports ?? {};
pkg.exports["."] = pkg.exports["."] ?? {
    types: "./" + (pkg.types ?? "laurus_wasm.d.ts"),
    import: "./" + (pkg.main ?? "laurus_wasm.js"),
};
pkg.exports["./opfs"] = {
    types: "./opfs.d.ts",
    import: "./opfs.js",
};
writeFileSync(path, JSON.stringify(pkg, null, 2) + "\n");
' "${PKG_DIR}/package.json"

echo "postbuild: copied opfs.js / opfs.d.ts and patched package.json"
