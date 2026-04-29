# Quick Start

## 1. Create an index

```php
<?php

use Laurus\Index;
use Laurus\Schema;

// In-memory index (ephemeral, useful for prototyping)
$index = new Index();

// File-based index (persistent)
$schema = new Schema();
$schema->addTextField("title");
$schema->addTextField("body");
$index = new Index("./myindex", $schema);
```

## 2. Index documents

```php
$index->putDocument("doc1", [
    "title" => "Introduction to Rust",
    "body" => "Rust is a systems programming language focused on safety and performance.",
]);
$index->putDocument("doc2", [
    "title" => "PHP for Web Development",
    "body" => "PHP is widely used for web applications and rapid prototyping.",
]);
$index->commit();
```

## 3. Lexical search

```php
// DSL string
$results = $index->search("title:rust", 5);

// Query object
$results = $index->search(new \Laurus\TermQuery("body", "php"), 5);

// Print results
foreach ($results as $r) {
    printf("[%s] score=%.4f  %s\n", $r->getId(), $r->getScore(), $r->getDocument()["title"]);
}
```

## 4. Vector search

Vector search requires a schema with a vector field and pre-computed embeddings.

```php
<?php

use Laurus\Index;
use Laurus\Schema;
use Laurus\VectorQuery;

$schema = new Schema();
$schema->addTextField("title");
$schema->addHnswField("embedding", 4);

$index = new Index(null, $schema);
$index->putDocument("doc1", ["title" => "Rust", "embedding" => [0.1, 0.2, 0.3, 0.4]]);
$index->putDocument("doc2", ["title" => "PHP", "embedding" => [0.9, 0.8, 0.7, 0.6]]);
$index->commit();

$queryVec = [0.1, 0.2, 0.3, 0.4];
$results = $index->search(new VectorQuery("embedding", $queryVec), 3);
```

## 5. Hybrid search

```php
use Laurus\SearchRequest;
use Laurus\TermQuery;
use Laurus\VectorQuery;
use Laurus\RRF;

$request = new SearchRequest(
    query: null,
    lexicalQuery: new TermQuery("title", "rust"),
    vectorQuery: new VectorQuery("embedding", $queryVec),
    filterQuery: null,
    fusion: new RRF(60.0),
    limit: 5,
);
$results = $index->search($request);
```

## 6. Update and delete

```php
// Update: putDocument replaces all existing versions
$index->putDocument("doc1", ["title" => "Updated Title", "body" => "New content."]);
$index->commit();

// Append a new version without removing existing ones (RAG chunking pattern)
$index->addDocument("doc1", ["title" => "Chunk 2", "body" => "Additional chunk."]);
$index->commit();

// Retrieve all versions
$docs = $index->getDocuments("doc1");

// Delete
$index->deleteDocuments("doc1");
$index->commit();
```

## 7. Schema management

```php
$schema = new \Laurus\Schema();
$schema->addTextField("title");
$schema->addTextField("body");
$schema->addIntegerField("year");
$schema->addFloatField("score");
$schema->addBooleanField("published");
$schema->addBytesField("thumbnail");
$schema->addGeoField("location");
$schema->addDatetimeField("created_at");
$schema->addHnswField("embedding", 384);
$schema->addFlatField("small_vec", 64);
$schema->addIvfField("ivf_vec", 128, "cosine", 100, 1);
```

## 8. Index statistics

```php
$stats = $index->stats();
echo $stats["documentCount"];
echo $stats["vectorFields"];
```
