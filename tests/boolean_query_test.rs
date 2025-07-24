//! Integration tests for BooleanQuery with MUST_NOT support

use sarissa::index::index::IndexConfig;
use sarissa::prelude::*;
use sarissa::query::{BooleanQuery, TermQuery};
use sarissa::schema::{IdField, TextField};
use sarissa::search::{SearchEngine, SearchRequest};
use tempfile::TempDir;

#[test]
fn test_boolean_query_must_not() -> Result<()> {
    // Create temporary directory and schema
    let temp_dir = TempDir::new().unwrap();
    let mut schema = Schema::new();
    schema.add_field("title", Box::new(TextField::new().stored(true).indexed(true)))?;
    schema.add_field("category", Box::new(TextField::new().indexed(true)))?;
    schema.add_field("tags", Box::new(TextField::new().indexed(true)))?;
    schema.add_field("id", Box::new(IdField::new()))?;

    // Create search engine
    let mut engine = SearchEngine::create_in_dir(temp_dir.path(), schema, IndexConfig::default())?;

    // Add test documents
    let documents = vec![
        Document::builder()
            .add_text("title", "Python Programming")
            .add_text("category", "programming")
            .add_text("tags", "python beginner")
            .add_text("id", "doc1")
            .build(),
        Document::builder()
            .add_text("title", "JavaScript Web Development")
            .add_text("category", "programming")
            .add_text("tags", "javascript web")
            .add_text("id", "doc2")
            .build(),
        Document::builder()
            .add_text("title", "Cooking with Python")
            .add_text("category", "cooking")
            .add_text("tags", "python recipes")
            .add_text("id", "doc3")
            .build(),
    ];

    engine.add_documents(documents)?;

    // Test 1: Simple MUST_NOT query
    let mut query = BooleanQuery::new();
    query.add_must_not(Box::new(TermQuery::new("tags", "javascript")));
    let request = SearchRequest::new(Box::new(query)).load_documents(true);
    let results = engine.search_mut(request)?;
    
    assert_eq!(results.total_hits, 2, "Should exclude JavaScript document");
    assert!(results.hits.iter().all(|hit| hit.doc_id != 1));

    // Test 2: MUST with MUST_NOT
    let mut query = BooleanQuery::new();
    query.add_must(Box::new(TermQuery::new("category", "programming")));
    query.add_must_not(Box::new(TermQuery::new("tags", "javascript")));
    let request = SearchRequest::new(Box::new(query)).load_documents(true);
    let results = engine.search_mut(request)?;
    
    assert_eq!(results.total_hits, 1, "Should find only Python programming doc");
    assert_eq!(results.hits[0].doc_id, 0);

    // Test 3: Multiple MUST_NOT clauses
    let mut query = BooleanQuery::new();
    query.add_must_not(Box::new(TermQuery::new("tags", "javascript")));
    query.add_must_not(Box::new(TermQuery::new("category", "cooking")));
    let request = SearchRequest::new(Box::new(query)).load_documents(true);
    let results = engine.search_mut(request)?;
    
    assert_eq!(results.total_hits, 1, "Should exclude both JavaScript and cooking docs");
    assert_eq!(results.hits[0].doc_id, 0);

    // Test 4: Multiple MUST clauses
    let mut query = BooleanQuery::new();
    query.add_must(Box::new(TermQuery::new("tags", "python")));
    query.add_must(Box::new(TermQuery::new("category", "programming")));
    let request = SearchRequest::new(Box::new(query)).load_documents(true);
    let results = engine.search_mut(request)?;
    
    assert_eq!(results.total_hits, 1, "Should find only doc matching both conditions");
    assert_eq!(results.hits[0].doc_id, 0);

    engine.close()?;
    Ok(())
}

#[test]
fn test_boolean_query_complex() -> Result<()> {
    // Create temporary directory and schema
    let temp_dir = TempDir::new().unwrap();
    let mut schema = Schema::new();
    schema.add_field("content", Box::new(TextField::new().indexed(true)))?;
    schema.add_field("type", Box::new(TextField::new().indexed(true)))?;
    schema.add_field("id", Box::new(IdField::new()))?;

    // Create search engine
    let mut engine = SearchEngine::create_in_dir(temp_dir.path(), schema, IndexConfig::default())?;

    // Add test documents
    let documents = vec![
        Document::builder()
            .add_text("content", "rust programming language")
            .add_text("type", "tutorial")
            .add_text("id", "doc1")
            .build(),
        Document::builder()
            .add_text("content", "rust web framework")
            .add_text("type", "library")
            .add_text("id", "doc2")
            .build(),
        Document::builder()
            .add_text("content", "python programming language")
            .add_text("type", "tutorial")
            .add_text("id", "doc3")
            .build(),
        Document::builder()
            .add_text("content", "javascript rust bindings")
            .add_text("type", "library")
            .add_text("id", "doc4")
            .build(),
    ];

    engine.add_documents(documents)?;

    // Test nested boolean query: (rust AND tutorial) OR (library AND NOT javascript)
    let mut rust_tutorial = BooleanQuery::new();
    rust_tutorial.add_must(Box::new(TermQuery::new("content", "rust")));
    rust_tutorial.add_must(Box::new(TermQuery::new("type", "tutorial")));

    let mut library_not_js = BooleanQuery::new();
    library_not_js.add_must(Box::new(TermQuery::new("type", "library")));
    library_not_js.add_must_not(Box::new(TermQuery::new("content", "javascript")));

    let mut main_query = BooleanQuery::new();
    main_query.add_should(Box::new(rust_tutorial));
    main_query.add_should(Box::new(library_not_js));

    let request = SearchRequest::new(Box::new(main_query)).load_documents(true);
    let results = engine.search_mut(request)?;
    
    assert_eq!(results.total_hits, 2, "Should find rust tutorial and rust library (not js)");
    let doc_ids: Vec<u64> = results.hits.iter().map(|h| h.doc_id).collect();
    assert!(doc_ids.contains(&0)); // rust tutorial
    assert!(doc_ids.contains(&1)); // rust library (not js)

    engine.close()?;
    Ok(())
}