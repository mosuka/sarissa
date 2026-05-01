# Geo3d Search Sample (Aircraft)

A single-page application that demonstrates laurus' 3D geographic
search (`Geo3d` field type, ECEF Cartesian coordinates) on top of a
[CesiumJS](https://cesium.com/platform/cesiumjs/) globe. Live aircraft
positions are pulled from the community ADS-B feed at
[airplanes.live](https://airplanes.live/) on every page load and on
demand via the Refresh button.

See the [examples README](../README.md) for the full list of samples
and the shared build instructions. The shortest path:

```bash
cd laurus-wasm
wasm-pack build --target web --dev
./scripts/postbuild.sh

# Make the UniDic zip available at examples/dict/lindera-unidic.zip,
# then serve from any HTTP server:
python3 -m http.server 8080

# Open http://localhost:8080/examples/geo3d/ in your browser.
```

## What this sample demonstrates

- A volatile OPFS-backed index (`flights3d-demo-index`) that is
  wiped and rebuilt on every load — apt for a Live data demo where
  yesterday's positions have no value
- A schema with five Japanese-tokenised text fields
  (`callsign`, `registration`, `aircraft_type`, `description`,
  `category`), a boolean field (`on_ground`), a float field
  (`altitude_m`), and the headline `position` field of type
  `geo3d` (indexed in a 3D BKD tree)
- A WGS84 → ECEF helper in JS that mirrors `laurus/src/util/ecef.rs`,
  letting the demo feed `{ x, y, z }` objects directly into
  `index.putDocument` (the WASM converter recognises that shape as
  `DataValue::GeoEcef`)
- A search box that builds a unified DSL string. Two mutually
  exclusive 3D constraints can be combined with the free text:
  - **Filter by 3D bbox around camera** — emits
    `+(<text>) +position:geo3d_bbox(minX, minY, minZ, maxX, maxY, maxZ)`
    with the bbox derived from a 200 km AABB centred on the camera's
    ECEF position
  - **Nearest 20 aircraft to camera target** — emits
    `position:geo3d_nearest(targetX, targetY, targetZ, 20)` where the
    target is the screen-centre intersection with the WGS84
    ellipsoid (falls back to the camera position when looking at
    the sky)
- A CesiumJS viewer with no Cesium Ion dependency: imagery is
  fetched from OpenStreetMap and the terrain provider is the simple
  ellipsoid one, so the sample needs no Ion access token
- Cesium entities synced with the search hits — non-matching
  aircraft are removed from the globe after each query so the
  visible markers always mirror the result list
- Camera drag / zoom / rotate re-runs the search automatically
  while a 3D constraint is on, so the spatial clause stays in sync
  with what the user is looking at
- A clickable result list that flies the camera to the corresponding
  entity
- A live Debug card that prints the camera geodetic position
  (lat / lon / altitude), the camera ECEF coordinates, the camera
  target ECEF, the current bbox values and the most recent DSL
  query the demo sent to `index.search()` — handy when you want to
  copy the query into a CLI or unit test

### Example queries

| Filter | Query | What it does |
| --- | --- | --- |
| OFF | (empty) | Empty query — nothing to search for. |
| OFF | `callsign:JAL*` | Lexical prefix match on callsign across the whole snapshot. |
| OFF | `description:Boeing` | Match the type description across the whole snapshot. |
| OFF | `category:heavy` | All wide-bodies in the snapshot. |
| BBOX | (empty) | All aircraft within the 200 km cube around the camera position. |
| BBOX | `aircraft_type:B38M` | 737 MAX 8s within the 3D bbox. |
| NEAREST | (empty) | The 20 aircraft closest to the camera target in 3D. |
| NEAREST | `category:heavy` | The 20 nearest heavies to the camera target. |

## Layout

This sample shares the dictionary loader, log helper and theme
stylesheet with the other samples through `examples/shared/`.
CesiumJS itself is loaded from [unpkg.com][unpkg-cesium] with
subresource integrity hashes; an offline build needs to vendor the
assets locally.

The OPFS dictionary key (`unidic`) is shared with the other samples,
so the ~52 MB UniDic zip is downloaded only once across samples.

[unpkg-cesium]: https://unpkg.com/cesium@1.121.0/Build/Cesium/

## Caveats

- Map imagery is fetched from OpenStreetMap. The browser must be
  online for the globe to render even though laurus itself runs
  locally.
- Aircraft data is fetched from `https://api.airplanes.live/v2/...`
  — a community ADS-B feed. CORS is enabled (`Access-Control-Allow-Origin: *`)
  so browser fetches just work, but the upstream service is
  best-effort and may briefly return zero records or HTTP errors.
  The Refresh button is rate-limited to one request every 5 seconds
  to keep the load reasonable.
- The 3D bounding box is a coarse axis-aligned 200 km cube around
  the camera ECEF position, not an exact frustum of the rendered
  scene. Computing a true camera-frustum 3D bbox is out of scope
  for this sample.
- The default fetch is centred on Japan (`lat=36, lon=138, dist=250nm`).
  Aircraft outside that radius are not included in the snapshot;
  pan the globe to other regions only after pressing Refresh with a
  modified URL if you want global coverage.
- The index is intentionally non-persistent: each page load wipes
  OPFS and rebuilds from a fresh airplanes.live snapshot. Use the
  basic / geo samples if you want to see OPFS persistence in
  action.
