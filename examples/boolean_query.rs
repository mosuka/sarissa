//! BooleanQuery example - demonstrates complex boolean logic with AND, OR, NOT operations.

use sarissa::full_text::index::IndexConfig;
use sarissa::prelude::*;
use sarissa::query::{BooleanQuery, NumericRangeQuery, PhraseQuery, TermQuery};
use sarissa::full_text_search::SearchEngine;
use sarissa::full_text_search::SearchRequest;
use tempfile::TempDir;

fn main() -> Result<()> {
    println!("=== BooleanQuery Example - Complex Boolean Logic ===\n");

    // Create a temporary directory for the index
    let temp_dir = TempDir::new().unwrap();
    println!("Creating index in: {:?}", temp_dir.path());

    // Create a search engine
    let mut engine = SearchEngine::create_in_dir(temp_dir.path(), IndexConfig::default())?;

    // Add documents for testing boolean queries
    let documents = vec![
        Document::builder()
            .add_text("title", "Advanced Python Programming")
            .add_text("body", "Learn advanced Python techniques including decorators, metaclasses, and async programming")
            .add_text("author", "Alice Johnson")
            .add_text("category", "programming")
            .add_float("price", 59.99)
            .add_float("rating", 4.7)
            .add_text("tags", "python advanced programming")
            .add_text("id", "book001")
            .build(),
        Document::builder()
            .add_text("title", "JavaScript for Web Development")
            .add_text("body", "Modern JavaScript techniques for frontend and backend web development")
            .add_text("author", "Bob Smith")
            .add_text("category", "web-development")
            .add_float("price", 45.50)
            .add_float("rating", 4.3)
            .add_text("tags", "javascript web frontend backend")
            .add_text("id", "book002")
            .build(),
        Document::builder()
            .add_text("title", "Machine Learning with Python")
            .add_text("body", "Practical machine learning algorithms implemented in Python")
            .add_text("author", "Carol Davis")
            .add_text("category", "data-science")
            .add_float("price", 72.99)
            .add_float("rating", 4.8)
            .add_text("tags", "python machine-learning data-science")
            .add_text("id", "book003")
            .build(),
        Document::builder()
            .add_text("title", "Web Design Fundamentals")
            .add_text("body", "Learn the basics of web design including HTML, CSS, and responsive design")
            .add_text("author", "David Brown")
            .add_text("category", "web-development")
            .add_float("price", 39.99)
            .add_float("rating", 4.1)
            .add_text("tags", "web design html css")
            .add_text("id", "book004")
            .build(),
        Document::builder()
            .add_text("title", "Data Science with R")
            .add_text("body", "Statistical computing and data analysis using the R programming language")
            .add_text("author", "Eva Wilson")
            .add_text("category", "data-science")
            .add_float("price", 65.00)
            .add_float("rating", 4.5)
            .add_text("tags", "r data-science statistics")
            .add_text("id", "book005")
            .build(),
        Document::builder()
            .add_text("title", "Advanced JavaScript Patterns")
            .add_text("body", "Design patterns and advanced programming techniques in JavaScript")
            .add_text("author", "Frank Miller")
            .add_text("category", "programming")
            .add_float("price", 54.99)
            .add_float("rating", 4.6)
            .add_text("tags", "javascript advanced patterns")
            .add_text("id", "book006")
            .build(),
    ];

    println!("Adding {} documents to the index...", documents.len());
    engine.add_documents(documents)?;
    engine.commit()?;

    println!("\n=== BooleanQuery Examples ===\n");

    // Example 1: Simple AND query
    // Note: Using lowercase terms because StandardAnalyzer normalizes text
    println!("1. Books about Python AND programming:");
    let mut query = BooleanQuery::new();
    query.add_must(Box::new(TermQuery::new("body", "python")));
    query.add_must(Box::new(TermQuery::new("body", "programming")));
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
            if let Some(field_value) = doc.get_field("title") {
                if let Some(title) = field_value.as_text() {
                    println!("      Title: {title}");
                }
            }
        }
    }

    // Example 2: Simple OR query
    println!("\n2. Books about Python OR JavaScript:");
    let mut query = BooleanQuery::new();
    query.add_should(Box::new(TermQuery::new("body", "python")));
    query.add_should(Box::new(TermQuery::new("body", "javascript")));
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
            if let Some(field_value) = doc.get_field("title") {
                if let Some(title) = field_value.as_text() {
                    println!("      Title: {title}");
                }
            }
        }
    }

    // Example 3: NOT query (must not contain)
    println!("\n3. Programming books that are NOT about JavaScript:");
    let mut query = BooleanQuery::new();
    query.add_must(Box::new(TermQuery::new("category", "programming")));
    query.add_must_not(Box::new(TermQuery::new("body", "javascript")));
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
            if let Some(field_value) = doc.get_field("title") {
                if let Some(title) = field_value.as_text() {
                    println!("      Title: {title}");
                }
            }
        }
    }

    // Example 4: Complex boolean query with multiple conditions
    println!("\n4. Web development books with high rating (>= 4.2) and reasonable price (<= $50):");
    let mut query = BooleanQuery::new();
    query.add_must(Box::new(TermQuery::new("category", "web-development")));
    query.add_must(Box::new(NumericRangeQuery::f64_range(
        "rating",
        Some(4.2),
        None,
    )));
    query.add_must(Box::new(NumericRangeQuery::f64_range(
        "price",
        None,
        Some(50.0),
    )));
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
            if let Some(field_value) = doc.get_field("title") {
                if let Some(title) = field_value.as_text() {
                    println!("      Title: {title}");
                }
            }
            if let Some(field_value) = doc.get_field("price") {
                if let sarissa::document::FieldValue::Float(price) = field_value {
                    println!("      Price: ${price:.2}");
                }
            }
            if let Some(field_value) = doc.get_field("rating") {
                if let sarissa::document::FieldValue::Float(rating) = field_value {
                    println!("      Rating: {rating:.1}");
                }
            }
        }
    }

    // Example 5: Phrase query in boolean context
    println!("\n5. Data science books that contain 'machine learning' phrase:");
    let mut query = BooleanQuery::new();
    query.add_must(Box::new(TermQuery::new("category", "data-science")));
    query.add_must(Box::new(PhraseQuery::new(
        "body",
        vec!["machine".to_string(), "learning".to_string()],
    )));
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
            if let Some(field_value) = doc.get_field("title") {
                if let Some(title) = field_value.as_text() {
                    println!("      Title: {title}");
                }
            }
        }
    }

    // Example 6: Multiple OR conditions with AND
    println!("\n6. Advanced books about either Python OR JavaScript:");
    let mut language_query = BooleanQuery::new();
    language_query.add_should(Box::new(TermQuery::new("body", "python")));
    language_query.add_should(Box::new(TermQuery::new("body", "javascript")));

    let mut query = BooleanQuery::new();
    query.add_must(Box::new(TermQuery::new("tags", "advanced")));
    query.add_must(Box::new(language_query));

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
            if let Some(field_value) = doc.get_field("title") {
                if let Some(title) = field_value.as_text() {
                    println!("      Title: {title}");
                }
            }
        }
    }

    // Example 7: Exclude expensive books
    println!("\n7. Books under $60 that are NOT about web design:");
    let mut query = BooleanQuery::new();
    query.add_must(Box::new(NumericRangeQuery::f64_range(
        "price",
        None,
        Some(60.0),
    )));
    query.add_must_not(Box::new(TermQuery::new("body", "design")));
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
            if let Some(field_value) = doc.get_field("title") {
                if let Some(title) = field_value.as_text() {
                    println!("      Title: {title}");
                }
            }
            if let Some(field_value) = doc.get_field("price") {
                if let sarissa::document::FieldValue::Float(price) = field_value {
                    println!("      Price: ${price:.2}");
                }
            }
        }
    }

    // Example 8: Optional conditions (SHOULD clauses)
    println!("\n8. Programming books, preferably about Python or with high rating:");
    let mut query = BooleanQuery::new();
    query.add_must(Box::new(TermQuery::new("category", "programming")));
    query.add_should(Box::new(TermQuery::new("body", "python")));
    query.add_should(Box::new(NumericRangeQuery::f64_range(
        "rating",
        Some(4.5),
        None,
    )));
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
            if let Some(field_value) = doc.get_field("title") {
                if let Some(title) = field_value.as_text() {
                    println!("      Title: {title}");
                }
            }
            if let Some(field_value) = doc.get_field("rating") {
                if let sarissa::document::FieldValue::Float(rating) = field_value {
                    println!("      Rating: {rating:.1}");
                }
            }
        }
    }

    // Example 9: Nested boolean queries - Complex logic
    println!(
        "\n9. Nested boolean queries - (Python OR JavaScript) AND (advanced OR high-rating) AND NOT expensive:"
    );

    // First nested query: (Python OR JavaScript)
    let mut language_query = BooleanQuery::new();
    language_query.add_should(Box::new(TermQuery::new("body", "python")));
    language_query.add_should(Box::new(TermQuery::new("body", "javascript")));

    // Second nested query: (advanced OR high-rating)
    let mut quality_query = BooleanQuery::new();
    quality_query.add_should(Box::new(TermQuery::new("tags", "advanced")));
    quality_query.add_should(Box::new(NumericRangeQuery::f64_range(
        "rating",
        Some(4.5),
        None,
    )));

    // Main query combining all conditions
    let mut main_query = BooleanQuery::new();
    main_query.add_must(Box::new(language_query)); // Must match (Python OR JavaScript)
    main_query.add_must(Box::new(quality_query)); // Must match (advanced OR high-rating)
    main_query.add_must_not(Box::new(NumericRangeQuery::f64_range(
        // Must NOT be expensive
        "price",
        Some(70.0),
        None,
    )));

    let request = SearchRequest::new(Box::new(main_query)).load_documents(true);
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
            if let Some(field_value) = doc.get_field("title") {
                if let Some(title) = field_value.as_text() {
                    println!("      Title: {title}");
                }
            }
            if let Some(field_value) = doc.get_field("price") {
                if let sarissa::document::FieldValue::Float(price) = field_value {
                    println!("      Price: ${price:.2}");
                }
            }
            if let Some(field_value) = doc.get_field("rating") {
                if let sarissa::document::FieldValue::Float(rating) = field_value {
                    println!("      Rating: {rating:.1}");
                }
            }
        }
    }

    // Example 10: Triple-nested boolean query - More complex logic
    println!(
        "\n10. Triple-nested query - ((Python AND advanced) OR (JavaScript AND web)) AND price < $60:"
    );

    // First nested sub-query: (Python AND advanced)
    let mut python_advanced = BooleanQuery::new();
    python_advanced.add_must(Box::new(TermQuery::new("body", "python")));
    python_advanced.add_must(Box::new(TermQuery::new("tags", "advanced")));

    // Second nested sub-query: (JavaScript AND web)
    let mut javascript_web = BooleanQuery::new();
    javascript_web.add_must(Box::new(TermQuery::new("body", "javascript")));
    javascript_web.add_must(Box::new(TermQuery::new("tags", "web")));

    // Combine the two sub-queries with OR
    let mut combined_query = BooleanQuery::new();
    combined_query.add_should(Box::new(python_advanced));
    combined_query.add_should(Box::new(javascript_web));

    // Final query with price constraint
    let mut final_query = BooleanQuery::new();
    final_query.add_must(Box::new(combined_query));
    final_query.add_must(Box::new(NumericRangeQuery::f64_range(
        "price",
        None,
        Some(60.0),
    )));

    let request = SearchRequest::new(Box::new(final_query)).load_documents(true);
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
            if let Some(field_value) = doc.get_field("title") {
                if let Some(title) = field_value.as_text() {
                    println!("      Title: {title}");
                }
            }
            if let Some(field_value) = doc.get_field("price") {
                if let sarissa::document::FieldValue::Float(price) = field_value {
                    println!("      Price: ${price:.2}");
                }
            }
        }
    }

    // Example 11: Count matching documents
    println!("\n11. Counting books about either data science OR web development:");
    let mut query = BooleanQuery::new();
    query.add_should(Box::new(TermQuery::new("category", "data-science")));
    query.add_should(Box::new(TermQuery::new("category", "web-development")));
    let count = engine.count(Box::new(query))?;
    println!("   Count: {count} books");

    engine.close()?;
    println!("\nBooleanQuery example completed successfully!");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boolean_query_example() {
        let result = main();
        assert!(result.is_ok());
    }
}
