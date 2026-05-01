# Laurus WASM — Samples

Each subdirectory under `examples/` is a self-contained
single-page application that boots laurus-wasm directly from
`../pkg/` and runs entirely in the browser. Open
[`examples/index.html`](./index.html) in a local HTTP server for
the landing page, or jump straight to a specific sample.

## Samples

| Sample | What it shows |
| --- | --- |
| [`basic/`](./basic/) | Japanese full-text, vector, and hybrid search with the unified query DSL. Adds documents interactively. |
| [`geo/`](./geo/) | Tokyo points-of-interest on a Leaflet map. Combines a bounding-box constraint (`location:geo_bbox(...)`) drawn from the current viewport with text and vector queries. |
| [`geo3d/`](./geo3d/) | Live aircraft positions on a CesiumJS 3D globe. Demonstrates the `geo3d` field type with `geo3d_bbox(...)` and `geo3d_nearest(...)` queries — true ECEF 3D, including altitude. |

The [`shared/`](./shared/) directory holds assets reused by every
sample (theme stylesheet, logger, dictionary loader, embedder
helper).

## How to run

```bash
cd laurus-wasm
wasm-pack build --target web --dev
./scripts/postbuild.sh
```

The samples expect the UniDic zip (~52 MB) at
`examples/dict/lindera-unidic.zip`. The deploy workflow fetches it
automatically; for local development, download a matching version
from the [Lindera releases][lindera-releases] and drop it under
`examples/dict/`.

```bash
# from the repository root
mkdir -p laurus-wasm/examples/dict
curl -fsSL -o laurus-wasm/examples/dict/lindera-unidic.zip \
  "https://github.com/lindera/lindera/releases/download/v<version>/lindera-unidic-<version>.zip"
```

Then start any HTTP server (WASM cannot be loaded over `file://`):

```bash
# Python
python3 -m http.server 8080
# or Node
npx serve .
```

Open <http://localhost:8080/examples/> in your browser.

[lindera-releases]: https://github.com/lindera/lindera/releases

## Adding a new sample

1. Create `examples/<name>/index.html`. Import laurus-wasm from
   `../../pkg/laurus_wasm.js` and use helpers from
   `../shared/`.
2. Add `examples/<name>/README.md` and `README_ja.md`.
3. Link the new sample from this README and from the landing page
   `examples/index.html`.
4. The deploy workflow ([`.github/workflows/deploy-docs.yml`][deploy])
   copies all of `examples/` into the published Pages site, so no CI
   change is needed for additional samples.

[deploy]: ../../.github/workflows/deploy-docs.yml
