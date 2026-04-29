# Quick Start

## 1. Create an index

```python
import laurus

# In-memory index (ephemeral, useful for prototyping)
index = laurus.Index()

# File-based index (persistent)
schema = laurus.Schema()
schema.add_text_field("title")
schema.add_text_field("body")
index = laurus.Index(path="./myindex", schema=schema)
```

## 2. Index documents

```python
index.put_document("doc1", {
    "title": "Introduction to Rust",
    "body": "Rust is a systems programming language focused on safety and performance.",
})
index.put_document("doc2", {
    "title": "Python for Data Science",
    "body": "Python is widely used for data analysis and machine learning.",
})
index.commit()
```

## 3. Lexical search

```python
# DSL string
results = index.search("title:rust", limit=5)

# Query object
results = index.search(laurus.TermQuery("body", "python"), limit=5)

# Print results
for r in results:
    print(f"[{r.id}] score={r.score:.4f}  {r.document['title']}")
```

## 4. Vector search

Vector search requires a schema with a vector field and pre-computed embeddings.

```python
import laurus
import numpy as np

schema = laurus.Schema()
schema.add_text_field("title")
schema.add_hnsw_field("embedding", dimension=4)

index = laurus.Index(schema=schema)
index.put_document("doc1", {"title": "Rust", "embedding": [0.1, 0.2, 0.3, 0.4]})
index.put_document("doc2", {"title": "Python", "embedding": [0.9, 0.8, 0.7, 0.6]})
index.commit()

query_vec = [0.1, 0.2, 0.3, 0.4]
results = index.search(laurus.VectorQuery("embedding", query_vec), limit=3)
```

## 5. Hybrid search

```python
request = laurus.SearchRequest(
    lexical_query=laurus.TermQuery("title", "rust"),
    vector_query=laurus.VectorQuery("embedding", query_vec),
    fusion=laurus.RRF(k=60.0),
    limit=5,
)
results = index.search(request)
```

## 6. Update and delete

```python
# Update: put_document replaces all existing versions
index.put_document("doc1", {"title": "Updated Title", "body": "New content."})
index.commit()

# Append a new version without removing existing ones (RAG chunking pattern)
index.add_document("doc1", {"title": "Chunk 2", "body": "Additional chunk."})
index.commit()

# Retrieve all versions
docs = index.get_documents("doc1")

# Delete
index.delete_documents("doc1")
index.commit()
```

## 7. Schema management

```python
schema = laurus.Schema()
schema.add_text_field("title")
schema.add_text_field("body")
schema.add_integer_field("year")
schema.add_float_field("score")
schema.add_boolean_field("published")
schema.add_bytes_field("thumbnail")
schema.add_geo_field("location")
schema.add_datetime_field("created_at")
schema.add_hnsw_field("embedding", dimension=384)
schema.add_flat_field("small_vec", dimension=64)
schema.add_ivf_field("ivf_vec", dimension=128, n_clusters=100)
```

## 8. Index statistics

```python
stats = index.stats()
print(stats["document_count"])
print(stats["vector_fields"])
```
