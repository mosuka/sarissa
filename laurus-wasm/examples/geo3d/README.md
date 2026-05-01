# Geo3d Search Sample (Aircraft)

A single-page application that demonstrates laurus' 3D geographic
search (`Geo3d` field type, ECEF Cartesian coordinates) on top of a
[CesiumJS](https://cesium.com/platform/cesiumjs/) globe. Live aircraft
positions are pulled from the community ADS-B feed at
[airplanes.live](https://airplanes.live/) on every page load and on
demand via the Refresh button (or automatically on a configurable
schedule).

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

1. The page loads and drops a yellow pin 📌 at Tokyo. The 50
   aircraft 3D-closest to the pin appear immediately as orange
   markers with a vertical altitude line.
2. **Click anywhere on the globe** to move the pin. The result set
   re-runs with the new search centre.
3. Type into the search box (e.g. `JAL`, `Boeing`,
   `category:heavy`) or use the quick-filter chips
   (`Heavies` / `Helicopters` / `JAL flights` / `ANA flights` /
   `Clear`) to add a text constraint. With the
   `Limit to the 50 aircraft 3D-closest to the pin` checkbox on,
   the demo combines text and spatial:
   `+(text) +position:geo3d_nearest(x, y, z, 50)`.
4. Click a result row to fly the camera to that aircraft.
5. Use **Refresh data** for a manual snapshot, or pick an interval
   in the **Auto** dropdown (10 s / 30 s / 60 s) for hands-off
   updates. Auto-refresh pauses while the tab is hidden.
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
  (`callsign`, `registration`, `aircraft_type`, `description`,
  `category`), a boolean field (`on_ground`), a float field
  (`altitude_m`), and the headline `position` field of type
  `geo3d` (indexed in a 3D BKD tree).
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
  snapshot and the new, deletes only the aircraft that have left
  the feed, and overwrites the rest with `putDocument`. Highlight
  markers reposition in place — no flicker.
- A CesiumJS viewer with no Cesium Ion dependency: imagery is
  served by OpenStreetMap and terrain by the simple ellipsoid
  provider, so no Ion access token is needed.

### Example queries

The pin always exists, so with the spatial constraint enabled the
demo always sends a `geo3d_nearest` clause. Toggle the checkbox off
to drop the constraint and see all matching aircraft.

| Spatial limit | Query | What it does |
| --- | --- | --- |
| ON | (empty) | The 50 aircraft 3D-closest to the pin. |
| ON | `callsign:JAL*` | The closest 50 aircraft, restricted to JAL flights. |
| ON | `category:heavy` | The closest 50 wide-bodies. |
| OFF | `callsign:JAL*` | Every JAL flight in the snapshot. |
| OFF | `description:Boeing` | Every Boeing in the snapshot. |
| OFF | `category:rotorcraft` | Every helicopter in the snapshot. |

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
- Aircraft data is fetched from `https://api.airplanes.live/v2/...`,
  a community ADS-B feed. CORS is enabled
  (`Access-Control-Allow-Origin: *`) so browser fetches just work,
  but the upstream service is best-effort and may briefly return
  zero records or HTTP errors. The manual Refresh button is
  rate-limited to one request every 5 seconds; auto-refresh
  defaults to off and the available cadences (10 / 30 / 60 s) are
  designed to keep the load on the upstream feed reasonable.
- The default fetch is centred on Japan (`lat=36, lon=138,
  dist=1000nm`, a ~1850 km radius). Aircraft outside that radius
  are not included in the snapshot; edit the `AIRPLANES_LIVE_URL`
  constant in `index.html` if you want a different region.
- The index is intentionally non-persistent: each page load wipes
  OPFS and rebuilds from a fresh airplanes.live snapshot. Use the
  basic / geo samples if you want to see OPFS persistence in
  action.
