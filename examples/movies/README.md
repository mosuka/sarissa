# Movies Example

Index and search ~32,000 movies from the [Meilisearch movies dataset](https://github.com/meilisearch/datasets).
Supports both lexical full-text search and multimodal (CLIP) vector search on poster images.

## Prerequisites

- [jq](https://jqlang.github.io/jq/) for JSON processing
- [curl](https://curl.se/) for downloading poster images
- [python3](https://www.python.org/) for binary-to-JSON conversion
- The `embeddings-multimodal` feature must be enabled at build time

## Schema

The [schema.toml](schema.toml) defines the following fields:

| Field | Type | Indexed | Stored | Description |
| ----- | ---- | ------- | ------ | ----------- |
| `title` | Text | Yes | Yes | Movie title |
| `overview` | Text | Yes | Yes | Plot summary |
| `genres` | Text | Yes | Yes | Comma-separated genre list |
| `poster` | Text | No | Yes | Poster image URL |
| `release_date` | Integer | Yes | Yes | Unix timestamp |
| `poster_vec` | Hnsw | Yes | No | CLIP embedding of the poster image (512-dim) |

Default search fields: `title`, `overview`

### Embedder

The schema defines a `clip_embedder` using [CLIP](https://openai.com/index/clip/) (`openai/clip-vit-base-patch32`).
The `poster_vec` field references this embedder so that poster images are automatically
embedded into a 512-dimensional vector space at index time.

## Sample document

`index_movies.sh` doesn't write an intermediate file — it converts each dataset row into a
document JSON object with `jq` and pipes `add doc <id> <document>` commands straight into a
single `laurus repl` process (see [scripts/index_movies.sh](scripts/index_movies.sh)). For
[_The Matrix_](https://www.themoviedb.org/movie/603), the generated document looks like this:

```json
{
  "fields": {
    "title": "The Matrix",
    "overview": "Set in the 22nd century, The Matrix tells the story of a computer hacker who joins a group of underground insurgents fighting the vast and powerful computers who now rule the earth.",
    "genres": "Action, Science Fiction",
    "poster": "https://image.tmdb.org/t/p/w500/f89U3ADr1oiB1s9GkdPOEpXUk5H.jpg",
    "release_date": 922752000,
    "poster_vec": {"data": "/9j/4AAQSkZJRg…", "mime": "image/jpeg"}
  }
}
```

`poster_vec` is an `Hnsw` field backed by a CLIP (multimodal) embedder, which accepts both
text-to-embed and image-bytes-to-decode — a bare base64 string would be ambiguous between the
two, so the explicit `{"data", "mime"}` object is required here (a plain base64 string is only
unambiguous for a declared `Bytes` field, not a multimodal vector field). `poster_vec.data` is
the poster image's raw bytes, base64-encoded; the value above is truncated after its JPEG magic
number (`FF D8 FF E0`) for readability — the real string encodes every byte of the downloaded
file. `poster_vec` is only added once the poster image has been downloaded to
`examples/movies/images/<id>.jpg`; a movie with no poster is indexed without it.

## Usage

### 1. Create the index

```bash
bash examples/movies/scripts/create_index.sh
```

This builds the release binary and creates an empty index at `examples/movies/index/` using the schema.

### 2. Index all movies

```bash
bash examples/movies/scripts/index_movies.sh
```

To index only a subset (e.g. the first 100 movies for a quick test):

```bash
bash examples/movies/scripts/index_movies.sh --limit 100
```

This script:

1. Builds the release binary with the `embeddings-multimodal` feature
2. Downloads poster images from TMDB to `examples/movies/images/` (parallel, idempotent)
3. Converts each movie into a laurus document with lexical fields and poster bytes
4. Pipes all documents into the REPL, committing every 1,000 records
5. The engine automatically embeds poster bytes into 512-dim CLIP vectors

### 3. Run example searches

```bash
bash examples/movies/scripts/search_movies.sh
```

Runs several example queries:

**Lexical searches:**

- `star wars` — full-text search across default fields
- `title:nemo` — field-specific search
- `genres:comedy` — search by genre
- `overview:robot` — search within plot summaries
- JSON output format

**Multimodal (vector) searches:**

- `poster_vec:"space adventure"` — find movies whose poster looks like a space adventure
- `poster_vec:"romantic couple"` — find movies with romantic poster imagery
- `poster_vec:"scary monster horror"` — find movies with horror-style posters

### Manual search

You can also search directly:

```bash
# Lexical search
./target/release/laurus --index-dir examples/movies/index search "title:matrix" --limit 10

# Multimodal vector search (text-to-image)
./target/release/laurus --index-dir examples/movies/index search 'poster_vec:"action hero"' --limit 10
```

Or start an interactive session:

```bash
./target/release/laurus --index-dir examples/movies/index repl
```

## gRPC server and MCP server

Instead of the CLI, you can serve this index over gRPC (with an optional HTTP gateway) and
expose it to an MCP client (e.g. Claude Code).

```bash
# gRPC server (+ HTTP gateway on --http-port) over the already-built movies index.
# The index directory's own schema.toml is used automatically — no extra
# --schema flag exists or is needed.
./target/release/laurus --index-dir examples/movies/index serve --port 50051 --http-port 8080
```

```bash
# HTTP gateway: plain REST/JSON, no gRPC client needed.
curl http://localhost:8080/v1/index
curl -X POST http://localhost:8080/v1/search -H "Content-Type: application/json" -d '{"query":"title:matrix","limit":3}'
```

In another terminal, the MCP server proxies to that same gRPC endpoint over stdio:

```bash
./target/release/laurus mcp --endpoint http://localhost:50051
```

To register it with Claude Code:

```bash
claude mcp add laurus-movies -- ./target/release/laurus mcp --endpoint http://localhost:50051
```

`poster_vec` needs the binary built with `embeddings-multimodal` (already the case if you
followed [Usage](#usage) above) — without it, vector queries against this index fail at request
time even though the server starts fine.

## File structure

```text
examples/movies/
├── README.md
├── README_ja.md
├── schema.toml          # Index schema definition (lexical + vector)
├── scripts/
│   ├── create_index.sh  # Create the index
│   ├── index_movies.sh  # Download images and index the dataset
│   └── search_movies.sh # Example search queries (lexical + multimodal)
├── images/              # Downloaded poster images (git-ignored)
└── index/               # Generated index data (git-ignored)
```
