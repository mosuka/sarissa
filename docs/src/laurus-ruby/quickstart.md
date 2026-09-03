# Quick Start

## 1. Create an index

```ruby
require "laurus"

# In-memory index (ephemeral, useful for prototyping)
index = Laurus::Index.new

# File-based index (persistent)
# Writes `./myindex/schema.toml` and `./myindex/store/` -- the same layout
# `laurus-cli create index --schema` uses, so this directory can also be
# opened with the CLI (or vice versa).
schema = Laurus::Schema.new
schema.add_text_field("title")
schema.add_text_field("body")
index = Laurus::Index.new(path: "./myindex", schema: schema)

# Reopening it later only needs the path -- passing `schema:` again would
# raise, since the schema is already persisted.
index = Laurus::Index.new(path: "./myindex")
```

## 2. Index documents

```ruby
index.put_document("doc1", {
  "title" => "Introduction to Rust",
  "body" => "Rust is a systems programming language focused on safety and performance.",
})
index.put_document("doc2", {
  "title" => "Ruby for Web Development",
  "body" => "Ruby is widely used for web applications and rapid prototyping.",
})
index.commit
```

## 3. Lexical search

```ruby
# DSL string
results = index.search("title:rust", limit: 5)

# Query object
results = index.search(Laurus::TermQuery.new("body", "ruby"), limit: 5)

# Print results
results.each do |r|
  puts "[#{r.id}] score=#{format('%.4f', r.score)}  #{r.document['title']}"
end
```

## 4. Vector search

Vector search requires a schema with a vector field and pre-computed embeddings.

```ruby
require "laurus"

schema = Laurus::Schema.new
schema.add_text_field("title")
schema.add_hnsw_field("embedding", 4)

index = Laurus::Index.new(schema: schema)
index.put_document("doc1", { "title" => "Rust", "embedding" => [0.1, 0.2, 0.3, 0.4] })
index.put_document("doc2", { "title" => "Ruby", "embedding" => [0.9, 0.8, 0.7, 0.6] })
index.commit

query_vec = [0.1, 0.2, 0.3, 0.4]
results = index.search(Laurus::VectorQuery.new("embedding", query_vec), limit: 3)
```

## 5. Hybrid search

```ruby
request = Laurus::SearchRequest.new(
  lexical_query: Laurus::TermQuery.new("title", "rust"),
  vector_query: Laurus::VectorQuery.new("embedding", query_vec),
  fusion: Laurus::RRF.new(k: 60.0),
  limit: 5,
)
results = index.search(request)
```

## 6. Update and delete

```ruby
# Update: put_document replaces all existing versions
index.put_document("doc1", { "title" => "Updated Title", "body" => "New content." })
index.commit

# Append a new version without removing existing ones (RAG chunking pattern)
index.add_document("doc1", { "title" => "Chunk 2", "body" => "Additional chunk." })
index.commit

# Retrieve all versions
docs = index.get_documents("doc1")

# Delete
index.delete_documents("doc1")
index.commit
```

## 7. Schema management

```ruby
schema = Laurus::Schema.new
schema.add_text_field("title")
schema.add_text_field("body")
schema.add_integer_field("year")
schema.add_float_field("score")
schema.add_boolean_field("published")
schema.add_bytes_field("thumbnail")
schema.add_geo_field("location")
schema.add_datetime_field("created_at")
schema.add_hnsw_field("embedding", 384)
schema.add_flat_field("small_vec", 64)
schema.add_ivf_field("ivf_vec", 128, n_clusters: 100)
```

## 8. Index statistics

```ruby
stats = index.stats
puts stats["document_count"]
puts stats["vector_fields"]
```
