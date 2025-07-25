//! RangeQuery example - demonstrates range search for numeric and date values.

use sarissa::index::index::IndexConfig;
use sarissa::prelude::*;
use sarissa::query::NumericRangeQuery;
use sarissa::schema::{IdField, NumericField, TextField};
use sarissa::search::SearchEngine;
use sarissa::search::SearchRequest;
use tempfile::TempDir;

fn main() -> Result<()> {
    println!("=== RangeQuery Example - Numeric and Date Range Search ===\n");

    // Create a temporary directory for the index
    let temp_dir = TempDir::new().unwrap();
    println!("Creating index in: {:?}", temp_dir.path());

    // Create a schema with numeric fields
    let mut schema = Schema::new();
    schema.add_field(
        "title",
        Box::new(TextField::new().stored(true).indexed(true)),
    )?;
    schema.add_field("description", Box::new(TextField::new().indexed(true)))?;
    schema.add_field("price", Box::new(NumericField::f64().indexed(true)))?;
    schema.add_field("rating", Box::new(NumericField::f64().indexed(true)))?;
    schema.add_field("year", Box::new(NumericField::u64().indexed(true)))?;
    schema.add_field("pages", Box::new(NumericField::u64().indexed(true)))?;
    schema.add_field("id", Box::new(IdField::new()))?;

    // Create a search engine
    let mut engine = SearchEngine::create_in_dir(temp_dir.path(), schema, IndexConfig::default())?;

    // Add documents with various numeric values
    let documents = vec![
        Document::builder()
            .add_text("title", "Introduction to Algorithms")
            .add_text(
                "description",
                "Comprehensive guide to algorithms and data structures",
            )
            .add_float("price", 89.99)
            .add_float("rating", 4.8)
            .add_integer("year", 2009)
            .add_integer("pages", 1312)
            .add_text("id", "book001")
            .build(),
        Document::builder()
            .add_text("title", "Clean Code")
            .add_text("description", "A handbook of agile software craftsmanship")
            .add_float("price", 45.50)
            .add_float("rating", 4.6)
            .add_integer("year", 2008)
            .add_integer("pages", 464)
            .add_text("id", "book002")
            .build(),
        Document::builder()
            .add_text("title", "Design Patterns")
            .add_text(
                "description",
                "Elements of reusable object-oriented software",
            )
            .add_float("price", 62.95)
            .add_float("rating", 4.5)
            .add_integer("year", 1994)
            .add_integer("pages", 395)
            .add_text("id", "book003")
            .build(),
        Document::builder()
            .add_text("title", "The Pragmatic Programmer")
            .add_text("description", "Your journey to mastery")
            .add_float("price", 52.99)
            .add_float("rating", 4.7)
            .add_integer("year", 2019)
            .add_integer("pages", 352)
            .add_text("id", "book004")
            .build(),
        Document::builder()
            .add_text("title", "Refactoring")
            .add_text("description", "Improving the design of existing code")
            .add_float("price", 58.75)
            .add_float("rating", 4.4)
            .add_integer("year", 2018)
            .add_integer("pages", 448)
            .add_text("id", "book005")
            .build(),
        Document::builder()
            .add_text("title", "Code Complete")
            .add_text(
                "description",
                "A practical handbook of software construction",
            )
            .add_float("price", 73.99)
            .add_float("rating", 4.9)
            .add_integer("year", 2004)
            .add_integer("pages", 914)
            .add_text("id", "book006")
            .build(),
    ];

    println!("Adding {} documents to the index...", documents.len());
    engine.add_documents(documents)?;

    println!("\n=== RangeQuery Examples ===\n");

    // Example 1: Price range query
    println!("1. Books with price between $50.00 and $70.00:");
    let query = NumericRangeQuery::f64_range("price", Some(50.0), Some(70.0));
    let request = SearchRequest::new(Box::new(query)).load_documents(true);
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
            if let Some(field_value) = doc.get_field("price") {
                if let sarissa::schema::FieldValue::Float(price) = field_value {
                    println!("      Price: ${:.2}", price);
                }
            }
        }
    }

    // Example 2: Rating range query (high-rated books)
    println!("\n2. Books with rating 4.5 or higher:");
    let query = NumericRangeQuery::f64_range("rating", Some(4.5), None);
    let request = SearchRequest::new(Box::new(query)).load_documents(true);
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
            if let Some(field_value) = doc.get_field("rating") {
                if let sarissa::schema::FieldValue::Float(rating) = field_value {
                    println!("      Rating: {:.1}", rating);
                }
            }
        }
    }

    // Example 3: Year range query (recent books)
    println!("\n3. Books published after 2010:");
    let query = NumericRangeQuery::i64_range("year", Some(2010), None);
    let request = SearchRequest::new(Box::new(query)).load_documents(true);
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
            if let Some(field_value) = doc.get_field("year") {
                if let sarissa::schema::FieldValue::Integer(year) = field_value {
                    println!("      Year: {}", year);
                }
            }
        }
    }

    // Example 4: Page count range query (shorter books)
    println!("\n4. Books with 400 pages or fewer:");
    let query = NumericRangeQuery::i64_range("pages", None, Some(400));
    let request = SearchRequest::new(Box::new(query)).load_documents(true);
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
            if let Some(field_value) = doc.get_field("pages") {
                if let sarissa::schema::FieldValue::Integer(pages) = field_value {
                    println!("      Pages: {}", pages);
                }
            }
        }
    }

    // Example 5: Exact year range (books from 2008-2009)
    println!("\n5. Books published between 2008 and 2009:");
    let query = NumericRangeQuery::i64_range("year", Some(2008), Some(2009));
    let request = SearchRequest::new(Box::new(query)).load_documents(true);
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
            if let Some(field_value) = doc.get_field("year") {
                if let sarissa::schema::FieldValue::Integer(year) = field_value {
                    println!("      Year: {}", year);
                }
            }
        }
    }

    // Example 6: Budget-friendly books (price under $50)
    println!("\n6. Budget-friendly books (price under $50.00):");
    let query = NumericRangeQuery::f64_range_exclusive_upper("price", None, Some(50.0));
    let request = SearchRequest::new(Box::new(query)).load_documents(true);
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
            if let Some(field_value) = doc.get_field("price") {
                if let sarissa::schema::FieldValue::Float(price) = field_value {
                    println!("      Price: ${:.2}", price);
                }
            }
        }
    }

    // Example 7: Large books (more than 500 pages)
    println!("\n7. Large books (more than 500 pages):");
    let query = NumericRangeQuery::i64_range("pages", Some(500), None);
    let request = SearchRequest::new(Box::new(query)).load_documents(true);
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
            if let Some(field_value) = doc.get_field("pages") {
                if let sarissa::schema::FieldValue::Integer(pages) = field_value {
                    println!("      Pages: {}", pages);
                }
            }
        }
    }

    // Example 8: Count books in price range
    println!("\n8. Counting books with price between $40.00 and $80.00:");
    let query = NumericRangeQuery::f64_range("price", Some(40.0), Some(80.0));
    let count = engine.count_mut(Box::new(query))?;
    println!("   Count: {} books", count);

    // Example 9: Empty range (no results expected)
    println!("\n9. Books with impossible price range ($200-$300):");
    let query = NumericRangeQuery::f64_range("price", Some(200.0), Some(300.0));
    let request = SearchRequest::new(Box::new(query));
    let results = engine.search_mut(request)?;

    println!("   Found {} results", results.total_hits);

    engine.close()?;
    println!("\nRangeQuery example completed successfully!");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_range_query_example() {
        let result = main();
        assert!(result.is_ok());
    }
}
