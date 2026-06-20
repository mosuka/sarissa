# API Reference

## Index

The main entry point for creating and querying search indexes.

### Static Methods

#### `Index.create(schema?)`

Create a new in-memory (ephemeral) index.

- **Parameters:**
  - `schema` (Schema, optional) -- Schema definition.
- **Returns:** `Promise<Index>`

#### `Index.open(name, schema?)`

Open or create a persistent index backed by OPFS.

- **Parameters:**
  - `name` (string) -- Index name (OPFS subdirectory).
  - `schema` (Schema, optional) -- Schema definition.
- **Returns:** `Promise<Index>`

### Instance Methods

#### `putDocument(id, document)`

Replace a document (upsert).

- **Parameters:**
  - `id` (string) -- Document identifier.
  - `document` (object) -- Key-value pairs matching schema fields.
- **Returns:** `Promise<void>`

#### `addDocument(id, document)`

Append a document version (multi-version RAG pattern).

- **Parameters / Returns:** Same as `putDocument`.

#### `getDocuments(id)`

Retrieve all versions of a document.

- **Parameters:**
  - `id` (string)
- **Returns:** `Promise<object[]>`

#### `deleteDocuments(id)`

Delete all versions of a document.

- **Parameters:**
  - `id` (string)
- **Returns:** `Promise<void>`

#### `commit()`

Flush writes and make changes searchable. If opened with
`Index.open()`, data is also persisted to OPFS.

- **Returns:** `Promise<void>`

#### `search(query, limit?, offset?)`

Search using a DSL string query.

- **Parameters:**
  - `query` (string) -- Query DSL (e.g. `"title:hello"`).
  - `limit` (number, default 10)
  - `offset` (number, default 0)
- **Returns:** `Promise<SearchResult[]>`

#### `searchTerm(field, term, limit?, offset?)`

Search for an exact term.

- **Parameters:**
  - `field` (string) -- Field name.
  - `term` (string) -- Exact term.
  - `limit`, `offset` (number, optional)
- **Returns:** `Promise<SearchResult[]>`

#### `searchVector(field, vector, limit?, offset?)`

Search by vector similarity.

- **Parameters:**
  - `field` (string) -- Vector field name.
  - `vector` (number[]) -- Query embedding.
  - `limit`, `offset` (number, optional)
- **Returns:** `Promise<SearchResult[]>`

#### `searchVectorText(field, text, limit?, offset?)`

Search by text (embedded by the registered embedder).

- **Parameters:**
  - `field` (string) -- Vector field name.
  - `text` (string) -- Text to embed.
  - `limit`, `offset` (number, optional)
- **Returns:** `Promise<SearchResult[]>`

#### `searchGeo3dDistance(field, x, y, z, distanceM, limit?, offset?)`

Sphere search over a 3D ECEF point field. Returns documents whose `(x, y, z)`
coordinate is within `distanceM` metres of the centre. See
[Geo3d concepts](../concepts/geo3d.md) for ECEF theory.

- **Parameters:**
  - `field` (string) -- Geo3d field name.
  - `x`, `y`, `z` (number) -- Centre ECEF coordinate (metres).
  - `distanceM` (number) -- Maximum distance from the centre (metres).
  - `limit`, `offset` (number, optional)
- **Returns:** `Promise<SearchResult[]>`

#### `searchGeo3dBoundingBox(field, minX, minY, minZ, maxX, maxY, maxZ, limit?, offset?)`

Axis-aligned 3D bounding-box search over a 3D ECEF point field.

- **Parameters:**
  - `field` (string) -- Geo3d field name.
  - `minX`, `minY`, `minZ`, `maxX`, `maxY`, `maxZ` (number) -- Box bounds (metres).
  - `limit`, `offset` (number, optional)
- **Returns:** `Promise<SearchResult[]>`

#### `searchGeo3dNearest(field, x, y, z, k, limit?, offset?, initialRadiusM?, maxRadiusM?)`

k-nearest-neighbour search over a 3D ECEF point field. Returns the `k`
documents closest to `(x, y, z)`. The optional `initialRadiusM` and
`maxRadiusM` parameters tune the iterative-expansion search cone.

- **Parameters:**
  - `field` (string) -- Geo3d field name.
  - `x`, `y`, `z` (number) -- Centre ECEF coordinate (metres).
  - `k` (number) -- Number of nearest neighbours to return.
  - `limit`, `offset` (number, optional)
  - `initialRadiusM`, `maxRadiusM` (number, optional)
- **Returns:** `Promise<SearchResult[]>`

#### `stats()`

Return index statistics.

- **Returns:** `{ documentCount: number, vectorFields: { [name]: { count, dimension } } }`

## Schema

Builder for defining index fields and embedders.

### Constructor

#### `new Schema()`

Create an empty schema.

### Methods

#### `addTextField(name, stored?, indexed?, termVectors?, analyzer?)`

Add a full-text field. `analyzer` is the name of a parameter-less
built-in (`"standard"`, `"english"`, `"keyword"`, `"simple"`, `"noop"`)
or the name of a runtime analyzer registered via `addAnalyzer()`.

For Japanese morphological analysis, build a `JapaneseAnalyzer` from
raw IPADIC bytes and register it with `addAnalyzer()` first; see
[`JapaneseAnalyzer.fromBytes`](#japaneseanalyzerfrombytesmetadata-dictda--mode)
and [`addAnalyzer`](#addanalyzername-analyzer) below.

#### `addIntegerField(name, stored?, indexed?, multiValued?)`

Add a 64-bit integer field. Pass `multiValued: true` to accept arrays of
integers; range queries then match if any value satisfies the predicate
(Lucene-style "any match" with constant scoring).

#### `addFloatField(name, stored?, indexed?, multiValued?)`

Add a 64-bit float field. Pass `multiValued: true` to accept arrays of
floats; range queries then match if any value satisfies the predicate
(Lucene-style "any match" with constant scoring).

#### `addBooleanField(name, stored?, indexed?)`

Add a boolean field.

#### `addDatetimeField(name, stored?, indexed?)`

Add a date/time field.

#### `addGeoField(name, stored?, indexed?)`

Add a geographic coordinate field.

#### `addGeo3dField(name, stored?, indexed?)`

Add a 3D ECEF Cartesian point field. Values are submitted as a `{ x, y, z }`
object with metres units. See [Geo3d concepts](../concepts/geo3d.md) for
ECEF theory.

The WASM binding does not expose `Geo3dDistanceQuery` / `Geo3dBoundingBoxQuery`
/ `Geo3dNearestQuery` as JS classes (wasm-bindgen cannot expose `dyn Query`
trait objects). Instead, use the `Index.searchGeo3dDistance` /
`Index.searchGeo3dBoundingBox` / `Index.searchGeo3dNearest` methods documented
above.

#### `addBytesField(name, stored?)`

Add a binary data field.

#### `addHnswField(name, dimension, distance?, m?, efConstruction?, embedder?, quantizer?, subvectorCount?, rerankStorage?)`

Add an HNSW vector index field.

- `distance`: `"cosine"` (default), `"euclidean"`, `"dot_product"`,
  `"manhattan"`, `"angular"`
- `m`: Branching factor (default 16)
- `efConstruction`: Build-time expansion (default 200)
- `quantizer`: `"scalar_8bit"` (default) or `"product_quantization"`
  (requires `subvectorCount`)
- `subvectorCount`: number of PQ sub-vectors; must divide `dimension`
- `rerankStorage`: omit (default) or `"f32"` to store a full-precision
  rerank sidecar

#### `addFlatField(name, dimension, distance?, embedder?)`

Add a brute-force vector index field.

#### `addIvfField(name, dimension, distance?, nClusters?, nProbe?, embedder?)`

Add an IVF vector index field.

- `nClusters`: Number of partitioning clusters (default 100)
- `nProbe`: Number of clusters to probe at query time (default 1)

**Vector quantization & rerank storage** (HNSW fields):

- `quantizer` — `"scalar_8bit"` (default, 4× compression) or `"product_quantization"` for higher compression. Product quantization requires `subvectorCount` (must divide `dimension`).
- `rerankStorage` — set to `"f32"` to write a full-precision `*.hnsw.f32` sidecar enabling exact Stage-2 rerank; omit to keep the int8-only segment.

#### `addAnalyzer(name, analyzer)`

Register a pre-built analyzer instance under `name`. Resolved before the
parameter-less built-in names and before `schema.analyzers` definitions
when text fields reference an analyzer by name.

Currently only `JapaneseAnalyzer` instances built via
[`JapaneseAnalyzer.fromBytes`](#japaneseanalyzerfrombytesmetadata-dictda--mode)
are accepted here. The runtime registry is the only practical way to use
the Japanese analyzer in browser WASM, where the
`{ "language": "japanese", "dict": ... }` preset cannot resolve a
filesystem path.

```javascript
import { JapaneseAnalyzer, Schema } from "laurus-wasm";
import { downloadDictionary, loadDictionaryFiles } from "laurus-wasm/opfs";

await downloadDictionary("./dict/lindera-ipadic.zip", "ipadic");
const f = await loadDictionaryFiles("ipadic");
const ja = JapaneseAnalyzer.fromBytes(
  f.metadata, f.dictDa, f.dictVals, f.dictWordsIdx,
  f.dictWords, f.matrixMtx, f.charDef, f.unk, "normal",
);

const schema = new Schema();
schema.addAnalyzer("ja-ipadic", ja);
schema.addTextField("body", undefined, undefined, undefined, "ja-ipadic");
```

#### `addEmbedder(name, config)`

Register a named embedder. WASM supports two `type` values:

- `"precomputed"` — No embedding is performed; vectors are passed directly via
  `putDocument()` / `searchVector()`.
- `"callback"` — Provide a JavaScript callback `embed: (text) => Promise<number[]>`
  that the engine will invoke during ingestion and `searchVectorText()`. This
  enables in-engine auto-embedding using Transformers.js or any other in-browser
  embedding library.

```javascript
// Precomputed embedder
schema.addEmbedder("precomputed-embedder", { type: "precomputed" });

// Callback embedder (e.g. backed by Transformers.js)
schema.addEmbedder("callback-embedder", {
  type: "callback",
  embed: async (text) => {
    const output = await pipeline(text, { pooling: "mean", normalize: true });
    return Array.from(output.data);
  },
});
```

#### `setDefaultFields(fields)`

Set the default search fields.

#### `setDynamicFieldPolicy(policy)`

Set how the engine treats fields that appear in ingested documents but are
absent from the schema. `policy` is one of `"strict"`, `"dynamic"`
(default), or `"ignore"` (case-insensitive). Throws on an invalid value.

- `"strict"` — Reject the document.
- `"dynamic"` — Infer a type for each undeclared field and add it to the
  schema. **Warning**: integer fields silently truncate incoming float
  values (`3.14` → `3`).
- `"ignore"` — Silently drop the undeclared fields.

See [Schema & Fields](../concepts/schema_and_fields.md#dynamic-schema) for
the full behaviour matrix.

#### `dynamicFieldPolicy()`

Returns the current policy as a lowercase string.

#### `fieldNames()`

Returns an array of defined field names.

#### `toString()`

Returns a string representation of the schema (`"Schema(fields=[...])"`).

## SearchResult

```typescript
interface SearchResult {
  id: string;
  score: number;
  document: object | null;
}
```

## Analysis

### JapaneseAnalyzer

Japanese morphological analyzer constructed from raw Lindera dictionary
bytes. Browser WASM has no real filesystem, so the standard
`{ "language": "japanese", "dict": "/path/to/ipadic" }` preset cannot
be used. Instead, fetch a Lindera dictionary archive (typically
`lindera-ipadic-X.Y.Z.zip`), store it in OPFS via the
[OPFS helpers](#opfs-helpers), and pass the eight component byte
arrays to `JapaneseAnalyzer.fromBytes`.

#### `JapaneseAnalyzer.fromBytes(metadata, dictDa, ..., mode?)`

Static factory that builds an analyzer from raw IPADIC bytes.

Arguments (all `Uint8Array` except `mode`):

| Argument | Source file |
| ---- | ---- |
| `metadata` | `metadata.json` |
| `dictDa` | `dict.da` (Double-Array Trie) |
| `dictVals` | `dict.vals` |
| `dictWordsIdx` | `dict.wordsidx` |
| `dictWords` | `dict.words` |
| `matrixMtx` | `matrix.mtx` |
| `charDef` | `char_def.bin` |
| `unk` | `unk.bin` |
| `mode` | `"normal"` (default), `"search"`, or `"decompose"` |

Throws if any component fails to deserialize or the mode string is
invalid.

```javascript
import { JapaneseAnalyzer } from "laurus-wasm";
import { loadDictionaryFiles } from "laurus-wasm/opfs";

const f = await loadDictionaryFiles("ipadic");
const ja = JapaneseAnalyzer.fromBytes(
  f.metadata, f.dictDa, f.dictVals, f.dictWordsIdx,
  f.dictWords, f.matrixMtx, f.charDef, f.unk,
  "normal",
);
```

The pipeline is `NFKC normalization → Japanese iteration mark
normalization → Lindera morphological tokenization → lowercase →
Japanese stop word filter` — identical to the `japanese` preset on the
native side.

### OPFS Helpers

The `laurus-wasm/opfs` subpath bundles helpers for downloading,
storing, and loading Lindera dictionaries from the browser's Origin
Private File System. Used together with `JapaneseAnalyzer.fromBytes`.

```javascript
import {
  downloadDictionary,
  loadDictionaryFiles,
  hasDictionary,
  listDictionaries,
  removeDictionary,
} from "laurus-wasm/opfs";
```

| Function | Description |
| ---- | ---- |
| `downloadDictionary(url, name, options?)` | Fetch a `.zip`, decompress with the Web `DecompressionStream` API, and store the eight Lindera files under `laurus/dictionaries/<name>/` in OPFS. `options.onProgress({ phase, loaded?, total? })` reports progress. |
| `loadDictionaryFiles(name)` | Read the eight files back as a `{ metadata, dictDa, dictVals, dictWordsIdx, dictWords, matrixMtx, charDef, unk }` object suitable for `JapaneseAnalyzer.fromBytes`. |
| `hasDictionary(name)` | `true` if the dictionary directory exists in OPFS. |
| `listDictionaries()` | Return an array of stored dictionary names. |
| `removeDictionary(name)` | Delete the dictionary directory. |

Browser CORS prevents fetching directly from GitHub Releases, so host
the zip on the same origin as your app (the Laurus demo bundles
`./dict/lindera-ipadic.zip` alongside the WASM at deploy time).

### WhitespaceTokenizer

```javascript
const tokenizer = new WhitespaceTokenizer();
const tokens = tokenizer.tokenize("hello world");
// [{ text: "hello", position: 0, ... }, { text: "world", position: 1, ... }]
```

### SynonymDictionary

```javascript
const dict = new SynonymDictionary();
dict.addSynonymGroup(["ml", "machine learning"]);
```

### SynonymGraphFilter

```javascript
new SynonymGraphFilter(dictionary, keepOriginal = true, boost = 1.0)
```

- `dictionary` (`SynonymDictionary`) — Source synonym groups.
- `keepOriginal` (boolean, default `true`) — Keep the original token alongside
  the inserted synonyms.
- `boost` (number, default `1.0`) — Score boost applied to inserted synonym
  tokens.

```javascript
const filter = new SynonymGraphFilter(dict, true, 0.8);
const expanded = filter.apply(tokens);
```
