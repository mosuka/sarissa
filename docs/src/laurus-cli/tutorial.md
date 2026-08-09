# Hands-on Tutorial

This tutorial walks you through a complete workflow using the laurus CLI: creating a schema, building an index, adding documents, searching, updating, deleting, and using the interactive REPL.

## Prerequisites

- laurus CLI installed (see [Installation](installation.md))

## Step 1: Create a Schema

First, create a schema file that defines your index structure. You can generate one interactively:

```bash
laurus create schema
```

The interactive wizard guides you through defining fields, their types, and options. For this tutorial, create a schema file manually instead:

```bash
cat > schema.toml << 'EOF'
default_fields = ["title", "body"]

[fields.title.Text]
indexed = true
stored = true
term_vectors = false

[fields.body.Text]
indexed = true
stored = true
term_vectors = false

[fields.category.Text]
indexed = true
stored = true
term_vectors = false
EOF
```

This defines three text fields. The `default_fields` setting means queries without a field prefix will search `title` and `body`.

## Step 2: Create an Index

Create an index using the schema:

```bash
laurus --index-dir ./tutorial_data create index --schema schema.toml
```

Verify the index was created:

```bash
laurus --index-dir ./tutorial_data get stats
```

The output shows the document count is 0.

## Step 3: Add Documents

Add documents to the index. Each document needs an ID and a JSON object of the shape `{"fields": {...}}` with field values:

```bash
laurus --index-dir ./tutorial_data add doc \
  --id doc001 \
  --data '{"fields":{"title":"Introduction to Rust Programming","body":"Rust is a modern systems programming language that focuses on safety, speed, and concurrency.","category":"programming"}}'
```

```bash
laurus --index-dir ./tutorial_data add doc \
  --id doc002 \
  --data '{"fields":{"title":"Web Development with Rust","body":"Building web applications with Rust has become increasingly popular. Frameworks like Actix and Rocket make it easy to create fast and secure web services.","category":"web-development"}}'
```

```bash
laurus --index-dir ./tutorial_data add doc \
  --id doc003 \
  --data '{"fields":{"title":"Python for Data Science","body":"Python is the most popular language for data science and machine learning. Libraries like NumPy and Pandas provide powerful tools for data analysis.","category":"data-science"}}'
```

## Step 4: Commit Changes

Documents are not searchable until committed:

```bash
laurus --index-dir ./tutorial_data commit
```

## Step 5: Search Documents

### Basic Search

Search for documents containing "rust":

```bash
laurus --index-dir ./tutorial_data search "rust"
```

This searches the default fields (`title` and `body`). Results show `doc001` and `doc002`.

### Field-Specific Search

Search only in the `title` field:

```bash
laurus --index-dir ./tutorial_data search "title:python"
```

Only `doc003` is returned.

### Category Search

```bash
laurus --index-dir ./tutorial_data search "category:programming"
```

Only `doc001` is returned.

### Boolean Queries

Combine conditions with `+` (must) and `-` (must not):

```bash
laurus --index-dir ./tutorial_data search "+body:rust -body:web"
```

Only `doc001` is returned (contains "rust" but not "web").

### Phrase Search

Search for an exact phrase:

```bash
laurus --index-dir ./tutorial_data search 'body:"data science"'
```

Only `doc003` is returned.

### Fuzzy Search

Search with typo tolerance using `~`:

```bash
laurus --index-dir ./tutorial_data search "body:programing~1"
```

Matches "programming" despite the typo.

### JSON Output

Get results in JSON format for programmatic use:

```bash
laurus --index-dir ./tutorial_data --format json search "rust"
```

## Step 6: Retrieve a Document

Fetch a specific document by ID:

```bash
laurus --index-dir ./tutorial_data get docs --id doc001
```

## Step 7: Delete a Document

Delete a document and commit the change:

```bash
laurus --index-dir ./tutorial_data delete docs --id doc003
laurus --index-dir ./tutorial_data commit
```

Verify it was deleted:

```bash
laurus --index-dir ./tutorial_data search "python"
```

No results are returned.

## Step 8: Use the REPL

The REPL provides an interactive session for exploring your index:

```bash
laurus --index-dir ./tutorial_data repl
```

Try these commands in the REPL:

```text
> get stats
> search rust
> add doc doc004 {"fields":{"title":"Go Programming","body":"Go is a statically typed language designed for simplicity and efficiency.","category":"programming"}}
> commit
> search programming
> get docs doc004
> delete docs doc004
> commit
> quit
```

The REPL supports command history (Up/Down arrows) and line editing.

## Step 9: Clean Up

Remove the tutorial data:

```bash
rm -rf ./tutorial_data schema.toml
```

## Next Steps

- Learn about the [Schema Format](schema_format.md) for advanced field configurations
- See the full [Commands](commands.md) reference
- Explore the [REPL](repl.md) for interactive usage
- Try the [server tutorial](../laurus-server/tutorial.md) for gRPC/HTTP access
