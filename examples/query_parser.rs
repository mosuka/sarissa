//! Query parser example - demonstrates direct use of query parser for boolean queries.

use sarissa::index::index::IndexConfig;
use sarissa::prelude::*;
use sarissa::schema::{IdField, TextField};
use sarissa::search::{SearchEngine, SearchRequest};
use tempfile::TempDir;

fn main() -> Result<()> {
    println!("=== Query Parser Example - Boolean Query Parsing ===\n");

    // Create a temporary directory for the index
    let temp_dir = TempDir::new().unwrap();
    println!("Creating index in: {:?}", temp_dir.path());

    // Create a schema
    let mut schema = Schema::new();
    schema.add_field(
        "title",
        Box::new(TextField::new().stored(true).indexed(true)),
    )?;
    schema.add_field("body", Box::new(TextField::new().indexed(true)))?;
    schema.add_field("author", Box::new(IdField::new()))?;

    // Create a search engine
    let mut engine = SearchEngine::create_in_dir(temp_dir.path(), schema, IndexConfig::default())?;

    // Add sample documents
    let documents = vec![
        Document::builder()
            .add_text("title", "The Great Gatsby")
            .add_text("body", "In my younger and more vulnerable years my father gave me some advice")
            .add_text("author", "F. Scott Fitzgerald")
            .build(),
        Document::builder()
            .add_text("title", "To Kill a Mockingbird")
            .add_text("body", "When I was almost six years old, I heard my brother arguing with my father")
            .add_text("author", "Harper Lee")
            .build(),
        Document::builder()
            .add_text("title", "1984")
            .add_text("body", "It was a bright cold day in April, and the clocks were striking thirteen")
            .add_text("author", "George Orwell")
            .build(),
        Document::builder()
            .add_text("title", "Pride and Prejudice")
            .add_text("body", "It is a truth universally acknowledged, that a single man in possession of a good fortune")
            .add_text("author", "Jane Austen")
            .build(),
        Document::builder()
            .add_text("title", "The Catcher in the Rye")
            .add_text("body", "If you really want to hear about it, the first thing you'll probably want to know")
            .add_text("author", "J.D. Salinger")
            .build(),
    ];

    println!("Adding {} documents to the index...", documents.len());
    engine.add_documents(documents)?;
    engine.commit()?;

    println!("\n=== Query Parser Examples ===\n");

    // Example 1: Simple OR query (with correct case)
    println!("1. Simple OR query (Mockingbird OR Gatsby):");
    let parser = engine.query_parser_with_default("title");
    let query = parser.parse("Mockingbird OR Gatsby")?;
    println!("   Parsed query: {}", query.description());

    let request = SearchRequest::new(query).load_documents(true);
    let results = engine.search_mut(request)?;
    println!("   Found {} results", results.total_hits);
    for (i, hit) in results.hits.iter().enumerate() {
        println!(
            "   {}. Score: {:.4}, Doc ID: {}",
            i + 1,
            hit.score,
            hit.doc_id
        );
        if let Some(doc) = &hit.document {
            if let Some(field_value) = doc.get_field("title") {
                if let Some(title) = field_value.as_text() {
                    println!("      Title: {}", title);
                }
            }
        }
    }

    // Example 2: AND query with field specification
    println!("\n2. AND query with field specification (title:Pride AND body:truth):");
    let query = parser.parse("title:Pride AND body:truth")?;
    println!("   Parsed query: {}", query.description());

    let request = SearchRequest::new(query).load_documents(true);
    let results = engine.search_mut(request)?;
    println!("   Found {} results", results.total_hits);
    for (i, hit) in results.hits.iter().enumerate() {
        println!(
            "   {}. Score: {:.4}, Doc ID: {}",
            i + 1,
            hit.score,
            hit.doc_id
        );
        if let Some(doc) = &hit.document {
            if let Some(field_value) = doc.get_field("title") {
                if let Some(title) = field_value.as_text() {
                    println!("      Title: {}", title);
                }
            }
            if let Some(field_value) = doc.get_field("author") {
                if let Some(author) = field_value.as_text() {
                    println!("      Author: {}", author);
                }
            }
        }
    }

    // Example 3: Field-specific search
    println!("\n3. Field-specific search (body:younger):");
    let query = parser.parse("body:younger")?;
    println!("   Parsed query: {}", query.description());

    let request = SearchRequest::new(query).load_documents(true);
    let results = engine.search_mut(request)?;
    println!("   Found {} results", results.total_hits);
    for (i, hit) in results.hits.iter().enumerate() {
        println!(
            "   {}. Score: {:.4}, Doc ID: {}",
            i + 1,
            hit.score,
            hit.doc_id
        );
        if let Some(doc) = &hit.document {
            if let Some(field_value) = doc.get_field("title") {
                if let Some(title) = field_value.as_text() {
                    println!("      Title: {}", title);
                }
            }
        }
    }

    // Example 4: Grouped query with parentheses
    println!("\n4. Grouped query ((Gatsby OR Catcher) AND body:you):");
    let query = parser.parse("(Gatsby OR Catcher) AND body:you")?;
    println!("   Parsed query: {}", query.description());

    let request = SearchRequest::new(query).load_documents(true);
    let results = engine.search_mut(request)?;
    println!("   Found {} results", results.total_hits);
    for (i, hit) in results.hits.iter().enumerate() {
        println!(
            "   {}. Score: {:.4}, Doc ID: {}",
            i + 1,
            hit.score,
            hit.doc_id
        );
        if let Some(doc) = &hit.document {
            if let Some(field_value) = doc.get_field("title") {
                if let Some(title) = field_value.as_text() {
                    println!("      Title: {}", title);
                }
            }
        }
    }

    // Example 5: Body field search
    println!("\n5. Body field search (body:father OR body:brother):");
    let query = parser.parse("body:father OR body:brother")?;
    println!("   Parsed query: {}", query.description());

    let request = SearchRequest::new(query).load_documents(true);
    let results = engine.search_mut(request)?;
    println!("   Found {} results", results.total_hits);
    for (i, hit) in results.hits.iter().enumerate() {
        println!(
            "   {}. Score: {:.4}, Doc ID: {}",
            i + 1,
            hit.score,
            hit.doc_id
        );
        if let Some(doc) = &hit.document {
            if let Some(field_value) = doc.get_field("title") {
                if let Some(title) = field_value.as_text() {
                    println!("      Title: {}", title);
                }
            }
        }
    }

    // Example 6: Different default field
    println!("\n6. Using 'body' as default field (father OR brother):");
    let body_parser = engine.query_parser_with_default("body");
    let query = body_parser.parse("father OR brother")?;
    println!("   Parsed query: {}", query.description());

    let request = SearchRequest::new(query).load_documents(true);
    let results = engine.search_mut(request)?;
    println!("   Found {} results", results.total_hits);
    for (i, hit) in results.hits.iter().enumerate() {
        println!(
            "   {}. Score: {:.4}, Doc ID: {}",
            i + 1,
            hit.score,
            hit.doc_id
        );
        if let Some(doc) = &hit.document {
            if let Some(field_value) = doc.get_field("title") {
                if let Some(title) = field_value.as_text() {
                    println!("      Title: {}", title);
                }
            }
        }
    }

    engine.close()?;
    println!("\nQuery parser example completed successfully!");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_parser_example() {
        let result = main();
        assert!(result.is_ok());
    }
}
