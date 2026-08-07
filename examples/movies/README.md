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
    "title": {"Text": "The Matrix"},
    "overview": {"Text": "Set in the 22nd century, The Matrix tells the story of a computer hacker who joins a group of underground insurgents fighting the vast and powerful computers who now rule the earth."},
    "genres": {"Text": "Action, Science Fiction"},
    "poster": {"Text": "https://image.tmdb.org/t/p/w500/f89U3ADr1oiB1s9GkdPOEpXUk5H.jpg"},
    "release_date": {"Int64": 922752000},
    "poster_vec": {"Bytes": [[255, 216, 255, 224, "…"], "image/jpeg"]}
  }
}
```

`poster_vec.Bytes` is `[<raw JPEG byte array>, <MIME type>]`; the byte array above is truncated
after its JPEG magic number (`FF D8 FF E0`) for readability — the real array holds every byte of
the downloaded file. `poster_vec` is only added once the poster image has been downloaded to
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
