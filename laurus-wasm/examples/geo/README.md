# Geo Search Sample

A single-page application that combines lexical, vector, and
geographic queries on top of laurus-wasm. Tokyo points-of-interest
are plotted on a [Leaflet](https://leafletjs.com/) map; you can
search them with text, embeddings, and a viewport-driven
bounding-box constraint.

See the [examples README](../README.md) for the full list of samples
and the shared build instructions. The shortest path:

```bash
cd laurus-wasm
wasm-pack build --target web --dev
./scripts/postbuild.sh

# Make the UniDic zip available at examples/dict/lindera-unidic.zip,
# then serve from any HTTP server:
python3 -m http.server 8080

# Open http://localhost:8080/examples/geo/ in your browser.
```

## What this sample demonstrates

- An OPFS-persistent geo index (`geo-demo-index`), version-gated on
  load (a stamp mismatch against `version()` wipes and rebuilds it),
  seeded with
  ~14 Tokyo points-of-interest on first visit
- A schema with three Japanese-tokenised text fields
  (`title`, `description`, `category`), one geo field
  (`location` — indexed via the BKD tree) and one HNSW vector field
  (`embedding`, 384-dim multilingual MiniLM)
- A search box that builds a unified DSL string. With *Filter by
  current map view* turned on, the demo emits
  `+(<query>) +location:geo_bbox(min_lat, min_lon, max_lat, max_lon)`
  — both clauses are marked `+` (required) so a document has to
  match the text *and* sit inside the current Leaflet bounds
- Map markers track the search hits — non-matching points are
  removed from the map after each query, so the visible pins always
  mirror the result list
- Panning or zooming the map re-runs the search automatically
  (only while *Filter by current map view* is on), so the bbox
  clause stays in sync with what the user is looking at
- A clickable result list that pans the map and opens the popup of
  the corresponding marker
- A live Debug card that prints the current map center, zoom
  level, the (clamped) bbox values fed into `geo_bbox(...)`, and
  the most recent DSL query the demo sent to `index.search()` —
  handy when you want to copy the query into a CLI or unit test

### Example queries

| Toggle | Query | What it does |
| --- | --- | --- |
| ON | (empty) | Returns every point inside the current viewport. |
| ON | `公園` | Lexical match on `公園` AND inside the viewport. |
| ON | `embedding:"夜景がきれい"` | Semantic match AND inside the viewport. |
| OFF | `title:浅草寺` | Pure lexical match across the entire dataset. |
| OFF | `embedding:"街並みが歴史的"` | Pure semantic match across the entire dataset. |

## Layout

This sample shares the dictionary loader, embedder, log helper,
and theme stylesheet with the other samples through
`examples/shared/`. Leaflet itself is loaded from
[`unpkg.com`][unpkg] with subresource integrity hashes; an offline
build needs to vendor the assets locally.

The OPFS dictionary key (`unidic`) is shared with the basic
sample, so the ~52 MB UniDic zip is downloaded only once across
samples.

[unpkg]: https://unpkg.com/leaflet@1.9.4/

## Caveats

- Map tiles are fetched from OpenStreetMap. The browser must be
  online for tiles to render even though laurus itself runs locally.
- When the user zooms out enough that the visible viewport spans
  more than the world, or wraps across the antimeridian, Leaflet
  reports bounds outside `[-90, 90]` × `[-180, 180]`. Feeding those
  values straight to `geo_bbox(...)` would fail validation and
  silently return zero hits, so the demo clamps the bbox to the
  full valid range in that case (the Debug card flags it). A
  proper antimeridian-aware split-bbox query is out of scope for
  this sample.
