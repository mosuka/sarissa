# Geo3d Search Sample (Satellites)

A single-page application that demonstrates laurus' 3D geographic
search (`Geo3d` field type, ECEF Cartesian coordinates) on top of a
[CesiumJS](https://cesium.com/platform/cesiumjs/) globe. Orbital
element sets are downloaded once per session from
[CelesTrak](https://celestrak.org/) and every satellite position is
propagated **in the browser** with SGP4
([satellite.js](https://github.com/shashwatak/satellite-js)), so the
Refresh button (or the configurable auto-refresh schedule) updates
positions with pure client-side math — no recurring API calls.

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

## How to use it

1. The page loads and drops a yellow pin 📌 at Tokyo, downloads
   the selected satellite group's element sets from CelesTrak
   (Starlink by default), propagates every satellite to the current
   time with SGP4, and shows the 50 satellites 3D-closest to the
   pin as orange markers with a vertical altitude line.
2. **Click anywhere on the globe** to move the pin. The snapshot is
   global, so a click never needs new data — it just re-centres the
   spatial constraint and re-runs the search instantly.
3. Type into the search box (e.g. `STARLINK`, `category:LEO`,
   `category:GEO`) or use the quick-filter chips
   (`LEO` / `GEO` / `Starlink` / `ISS` / `Clear`) to filter by
   text. Pick how many results to display in the **Show:** dropdown
   (10 / 25 / 50 (default) / 100 / 200). Results are always sorted
   by 3D Euclidean distance from the pin — the dropdown selects the
   top N closest matches.
4. Click a result row to fly the camera to that satellite.
5. Use **Refresh positions** to re-propagate to the current time,
   or pick an interval in the **Auto** dropdown
   (5 s / 10 s / 30 s / 60 s) for hands-off updates — satellites
   visibly move between refreshes. Auto-refresh pauses while the
   tab is hidden. Switching the **Group** dropdown fetches that
   group's element sets (once per session) and rebuilds the index.
6. The **↺ Reset view** button (top-right of the globe) flies the
   camera back to the default oblique view of Japan.

### Mouse and touch controls

| Gesture | Action |
| --- | --- |
| Left drag | Rotate / orbit the globe |
| Right drag | Tilt — change camera pitch and heading |
| Middle drag | Tilt (auxiliary) |
| Scroll wheel / pinch | Zoom |
| Left click on the globe | Place the search pin |

## What this sample demonstrates

- A volatile OPFS-backed index (`flights3d-demo-index`) that is
  wiped on every page load — appropriate for Live data where
  yesterday's positions have no value.
- A schema with five Japanese-tokenised text fields
  (`callsign`, `registration`, `satellite_type`, `description`,
  `category`), a float field (`altitude_m`), and the headline
  `position` field of type `geo3d` (indexed in a 3D BKD tree).
- A WGS84 → ECEF helper in JS that mirrors
  `laurus/src/util/ecef.rs`, letting the demo feed `{ x, y, z }`
  objects directly into `index.putDocument` (the WASM converter
  recognises that shape as `DataValue::GeoEcef`).
- A `geo3d_nearest` query against the pin position. The full DSL
  string the demo sends to `index.search()` is shown in the
  collapsible developer-detail card so you can copy it into a CLI
  or unit test verbatim.
- An incremental refresh: instead of clearing the entire index
  every Refresh tick, the demo computes the diff between the old
  snapshot and the new, deletes only the satellites that have left
  the snapshot, and overwrites the rest with `putDocument`. Highlight
  markers reposition in place — no flicker.
- A CesiumJS viewer with no Cesium Ion dependency: imagery is
  served by OpenStreetMap and terrain by the simple ellipsoid
  provider, so no Ion access token is needed.

### Example queries

The pin always exists, so with the spatial constraint enabled the
demo always sends a `geo3d_nearest` clause. Toggle the checkbox off
to drop the constraint and see all matching satellites.

| Spatial limit | Query | What it does |
| --- | --- | --- |
| ON | (empty) | The 50 satellites 3D-closest to the pin. |
| ON | `callsign:STARLINK*` | The closest 50, restricted to Starlink. |
| ON | `category:LEO` | The closest 50 low-Earth-orbit satellites. |
| OFF | `callsign:ISS*` | The ISS modules in the snapshot. |
| OFF | `category:GEO` | Every geostationary satellite in the snapshot. |
| OFF | `category:MEO` | Every medium-Earth-orbit satellite (GNSS…). |

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
- Orbital element sets are fetched from
  `https://celestrak.org/NORAD/elements/gp.php?GROUP=...&FORMAT=json`
  with CORS enabled (`Access-Control-Allow-Origin: *`). CelesTrak
  answers **HTTP 403 per IP per group** until the element set next
  updates (~every 2 hours), so the demo persists each downloaded
  group with the Cache API: reloads inside the window are served
  entirely from cache, and a 403 with a cached copy present simply
  means the copy is still current. A 403 with no cached copy (fresh
  browser behind an IP that already consumed the window) suggests
  switching to another group or retrying after the next update.
- Positions come from SGP4 propagation of the published mean
  elements; expect kilometre-scale differences from precision
  ephemerides. Decayed or malformed element sets are skipped (the
  log shows the skip count).
- Large groups are truncated to the first 500 element sets
  (`MAX_SATELLITES` in `index.html`) to keep Cesium and indexing
  snappy.
- The index is intentionally non-persistent: each page load wipes
  OPFS and rebuilds from a fresh propagation snapshot. Use the
  basic / geo samples if you want to see OPFS persistence in
  action.
