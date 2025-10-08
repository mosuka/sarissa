//! FuzzyQuery example - demonstrates approximate string matching with edit distance.

use sarissa::full_text::index::IndexConfig;
use sarissa::full_text_search::SearchEngine;
use sarissa::full_text_search::SearchRequest;
use sarissa::prelude::*;
use sarissa::query::FuzzyQuery;
use tempfile::TempDir;

fn main() -> Result<()> {
    println!("=== FuzzyQuery Example - Approximate String Matching ===\n");

    // Create a temporary directory for the index
    let temp_dir = TempDir::new().unwrap();
    println!("Creating index in: {:?}", temp_dir.path());

    // Create a search engine
    let mut engine = SearchEngine::create_in_dir(temp_dir.path(), IndexConfig::default())?;

    // Add documents with various spellings and terms for fuzzy matching
    let documents = vec![
        Document::builder()
            .add_text("title", "JavaScript Programming Guide")
            .add_text(
                "body",
                "Comprehensive guide to JavaScript development and programming techniques",
            )
            .add_text("author", "John Smith")
            .add_text("tags", "javascript programming tutorial")
            .add_text("id", "doc001")
            .build(),
        Document::builder()
            .add_text("title", "Python Programming Fundamentals")
            .add_text(
                "body",
                "Learn Python programming language from scratch with practical examples",
            )
            .add_text("author", "Alice Johnson")
            .add_text("tags", "python programming beginner")
            .add_text("id", "doc002")
            .build(),
        Document::builder()
            .add_text("title", "Machine Learning Algorithms")
            .add_text(
                "body",
                "Understanding algorithms used in machine learning and artificial intelligence",
            )
            .add_text("author", "Bob Wilson")
            .add_text("tags", "machine-learning algorithms ai")
            .add_text("id", "doc003")
            .build(),
        Document::builder()
            .add_text("title", "Database Management Systems")
            .add_text(
                "body",
                "Introduction to database systems, SQL, and data management principles",
            )
            .add_text("author", "Carol Davis")
            .add_text("tags", "database sql management")
            .add_text("id", "doc004")
            .build(),
        Document::builder()
            .add_text("title", "Web Development with React")
            .add_text(
                "body",
                "Building modern web applications using React framework and components",
            )
            .add_text("author", "David Brown")
            .add_text("tags", "react web-development frontend")
            .add_text("id", "doc005")
            .build(),
        Document::builder()
            .add_text("title", "Artificial Intelligence Overview")
            .add_text(
                "body",
                "Introduction to artificial intelligence concepts, applications, and algorithms",
            )
            .add_text("author", "Eva Martinez")
            .add_text("tags", "artificial-intelligence overview concepts")
            .add_text("id", "doc006")
            .build(),
        Document::builder()
            .add_text("title", "Software Engineering Principles")
            .add_text(
                "body",
                "Best practices in software engineering, design patterns, and development",
            )
            .add_text("author", "Frank Miller")
            .add_text("tags", "software engineering principles")
            .add_text("id", "doc007")
            .build(),
    ];

    println!("Adding {} documents to the index...", documents.len());
    engine.add_documents(documents)?;

    println!("\n=== FuzzyQuery Examples ===\n");

    // Example 1: Simple fuzzy search with small edit distance
    println!("1. Fuzzy search for 'javascritp' (typo for 'javascript') with edit distance 1:");
    let query = FuzzyQuery::new("body", "javascritp").max_edits(1);
    let request = SearchRequest::new(Box::new(query)).load_documents(true);
    let results = engine.search(request)?;

    println!("   Found {} results", results.total_hits);
    for (i, hit) in results.hits.iter().enumerate() {
        println!(
            "   {}. Score: {:.4}, Doc ID: {}",
            i + 1,
            hit.score,
            hit.doc_id
        );
        if let Some(doc) = &hit.document
            && let Some(field_value) = doc.get_field("title")
            && let Some(title) = field_value.as_text()
        {
            println!("      Title: {title}");
        }
    }

    // Example 2: Fuzzy search with higher edit distance
    println!("\n2. Fuzzy search for 'programing' (missing 'm') with edit distance 2:");
    let query = FuzzyQuery::new("body", "programing").max_edits(2);
    let request = SearchRequest::new(Box::new(query)).load_documents(true);
    let results = engine.search(request)?;

    println!("   Found {} results", results.total_hits);
    for (i, hit) in results.hits.iter().enumerate() {
        println!(
            "   {}. Score: {:.4}, Doc ID: {}",
            i + 1,
            hit.score,
            hit.doc_id
        );
        if let Some(doc) = &hit.document
            && let Some(field_value) = doc.get_field("title")
            && let Some(title) = field_value.as_text()
        {
            println!("      Title: {title}");
        }
    }

    // Example 3: Fuzzy search in title field
    println!("\n3. Fuzzy search for 'machne' (missing 'i') in title with edit distance 1:");
    let query = FuzzyQuery::new("title", "machne").max_edits(1);
    let request = SearchRequest::new(Box::new(query)).load_documents(true);
    let results = engine.search(request)?;

    println!("   Found {} results", results.total_hits);
    for (i, hit) in results.hits.iter().enumerate() {
        println!(
            "   {}. Score: {:.4}, Doc ID: {}",
            i + 1,
            hit.score,
            hit.doc_id
        );
        if let Some(doc) = &hit.document
            && let Some(field_value) = doc.get_field("title")
            && let Some(title) = field_value.as_text()
        {
            println!("      Title: {title}");
        }
    }

    // Example 4: Fuzzy search for author names
    println!("\n4. Fuzzy search for 'Jon' (should match 'John') in author with edit distance 1:");
    let query = FuzzyQuery::new("author", "Jon").max_edits(1);
    let request = SearchRequest::new(Box::new(query)).load_documents(true);
    let results = engine.search(request)?;

    println!("   Found {} results", results.total_hits);
    for (i, hit) in results.hits.iter().enumerate() {
        println!(
            "   {}. Score: {:.4}, Doc ID: {}",
            i + 1,
            hit.score,
            hit.doc_id
        );
        if let Some(doc) = &hit.document {
            if let Some(field_value) = doc.get_field("title")
                && let Some(title) = field_value.as_text()
            {
                println!("      Title: {title}");
            }
            if let Some(field_value) = doc.get_field("author")
                && let Some(author) = field_value.as_text()
            {
                println!("      Author: {author}");
            }
        }
    }

    // Example 5: Fuzzy search with various misspellings
    println!("\n5. Fuzzy search for 'algoritm' (missing 'h') with edit distance 2:");
    let query = FuzzyQuery::new("body", "algoritm").max_edits(2);
    let request = SearchRequest::new(Box::new(query)).load_documents(true);
    let results = engine.search(request)?;

    println!("   Found {} results", results.total_hits);
    for (i, hit) in results.hits.iter().enumerate() {
        println!(
            "   {}. Score: {:.4}, Doc ID: {}",
            i + 1,
            hit.score,
            hit.doc_id
        );
        if let Some(doc) = &hit.document
            && let Some(field_value) = doc.get_field("title")
            && let Some(title) = field_value.as_text()
        {
            println!("      Title: {title}");
        }
    }

    // Example 6: Fuzzy search in tags
    println!("\n6. Fuzzy search for 'artifical' (missing 'i') in tags with edit distance 1:");
    let query = FuzzyQuery::new("tags", "artifical").max_edits(1);
    let request = SearchRequest::new(Box::new(query)).load_documents(true);
    let results = engine.search(request)?;

    println!("   Found {} results", results.total_hits);
    for (i, hit) in results.hits.iter().enumerate() {
        println!(
            "   {}. Score: {:.4}, Doc ID: {}",
            i + 1,
            hit.score,
            hit.doc_id
        );
        if let Some(doc) = &hit.document
            && let Some(field_value) = doc.get_field("title")
            && let Some(title) = field_value.as_text()
        {
            println!("      Title: {title}");
        }
    }

    // Example 7: Fuzzy search with exact match (edit distance 0)
    println!("\n7. Fuzzy search for exact 'python' with edit distance 0:");
    let query = FuzzyQuery::new("body", "python").max_edits(0);
    let request = SearchRequest::new(Box::new(query)).load_documents(true);
    let results = engine.search(request)?;

    println!("   Found {} results", results.total_hits);
    for (i, hit) in results.hits.iter().enumerate() {
        println!(
            "   {}. Score: {:.4}, Doc ID: {}",
            i + 1,
            hit.score,
            hit.doc_id
        );
        if let Some(doc) = &hit.document
            && let Some(field_value) = doc.get_field("title")
            && let Some(title) = field_value.as_text()
        {
            println!("      Title: {title}");
        }
    }

    // Example 8: Fuzzy search with high edit distance (more permissive)
    println!("\n8. Fuzzy search for 'databse' (missing 'a') with edit distance 3:");
    let query = FuzzyQuery::new("body", "databse").max_edits(3);
    let request = SearchRequest::new(Box::new(query)).load_documents(true);
    let results = engine.search(request)?;

    println!("   Found {} results", results.total_hits);
    for (i, hit) in results.hits.iter().enumerate() {
        println!(
            "   {}. Score: {:.4}, Doc ID: {}",
            i + 1,
            hit.score,
            hit.doc_id
        );
        if let Some(doc) = &hit.document
            && let Some(field_value) = doc.get_field("title")
            && let Some(title) = field_value.as_text()
        {
            println!("      Title: {title}");
        }
    }

    // Example 9: No fuzzy matches found
    println!("\n9. Fuzzy search for 'xyz123' (no similar terms) with edit distance 2:");
    let query = FuzzyQuery::new("body", "xyz123").max_edits(2);
    let request = SearchRequest::new(Box::new(query));
    let results = engine.search(request)?;

    println!("   Found {} results", results.total_hits);

    // Example 10: Count fuzzy matches
    println!("\n10. Counting documents with fuzzy match for 'developement' (extra 'e'):");
    let query = FuzzyQuery::new("body", "developement").max_edits(2);
    let count = engine.count(Box::new(query))?;
    println!("    Count: {count} documents");

    engine.close()?;
    println!("\nFuzzyQuery example completed successfully!");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fuzzy_query_example() {
        let result = main();
        assert!(result.is_ok());
    }
}
