use sage::index::index::IndexConfig;
use sage::prelude::*;
use sage::query::TermQuery;
use sage::schema::{IdField, TextField};
use sage::search::{SearchEngine, SearchRequest};
use tempfile::TempDir;

fn main() -> sage::error::Result<()> {
    let temp_dir = TempDir::new().unwrap();
    
    let mut schema = Schema::new();
    schema.add_field("body", Box::new(TextField::new().indexed(true)))?;
    schema.add_field("id", Box::new(IdField::new()))?;

    let mut engine = SearchEngine::create_in_dir(temp_dir.path(), schema, IndexConfig::default())?;

    let documents = vec![
        Document::builder()
            .add_text("body", "Learn advanced Python techniques including decorators, metaclasses, and async programming")
            .add_text("id", "book001")
            .build(),
        Document::builder()
            .add_text("body", "Practical machine learning algorithms implemented in Python")
            .add_text("id", "book003")
            .build(),
    ];

    engine.add_documents(documents)?;

    // Test individual term queries
    println!("=== Individual Term Queries ===");
    
    println!("Searching for 'python' in body:");
    let query = TermQuery::new("body", "python");
    let request = SearchRequest::new(Box::new(query));
    let results = engine.search_mut(request)?;
    println!("Found {} results", results.total_hits);
    for hit in &results.hits {
        println!("  Doc ID: {}", hit.doc_id);
    }

    println!("\nSearching for 'programming' in body:");
    let query = TermQuery::new("body", "programming");
    let request = SearchRequest::new(Box::new(query));
    let results = engine.search_mut(request)?;
    println!("Found {} results", results.total_hits);
    for hit in &results.hits {
        println!("  Doc ID: {}", hit.doc_id);
    }

    Ok(())
}