//! RangeQuery example - demonstrates range search for numeric and date values.

use std::sync::Arc;

use tempfile::TempDir;

use yatagarasu::document::document::Document;
use yatagarasu::error::Result;
use yatagarasu::lexical::engine::LexicalEngine;
use yatagarasu::lexical::index::config::LexicalIndexConfig;
use yatagarasu::lexical::index::factory::LexicalIndexFactory;
use yatagarasu::lexical::search::searcher::LexicalSearchRequest;
use yatagarasu::lexical::index::inverted::query::Query;
use yatagarasu::lexical::index::inverted::query::range::NumericRangeQuery;
use yatagarasu::storage::file::FileStorage;
use yatagarasu::storage::file::FileStorageConfig;

fn main() -> Result<()> {
    println!("=== RangeQuery Example - Numeric and Date Range Search ===\n");

    // Create a temporary directory for the index
    let temp_dir = TempDir::new().unwrap();
    println!("Creating index in: {:?}", temp_dir.path());

    // Create a schema with numeric fields

    // Create a search engine
    let config = LexicalIndexConfig::default();
    let storage = Arc::new(FileStorage::new(
        temp_dir.path(),
        FileStorageConfig::new(temp_dir.path()),
    )?);
    let index = LexicalIndexFactory::create(storage, config)?;
    let mut engine = LexicalEngine::new(index)?;

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
    engine.commit()?;

    println!("\n=== RangeQuery Examples ===\n");

    // Example 1: Price range query
    println!("1. Books with price between $50.00 and $70.00:");
    let query = NumericRangeQuery::f64_range("price", Some(50.0), Some(70.0));
    let request = LexicalSearchRequest::new(Box::new(query) as Box<dyn Query>).load_documents(true);
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
            if let Some(yatagarasu::document::field_value::FieldValue::Float(price)) =
                doc.get_field("price")
            {
                println!("      Price: ${price:.2}");
            }
        }
    }

    // Example 2: Rating range query (high-rated books)
    println!("\n2. Books with rating 4.5 or higher:");
    let query = NumericRangeQuery::f64_range("rating", Some(4.5), None);
    let request = LexicalSearchRequest::new(Box::new(query) as Box<dyn Query>).load_documents(true);
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
            if let Some(yatagarasu::document::field_value::FieldValue::Float(rating)) =
                doc.get_field("rating")
            {
                println!("      Rating: {rating:.1}");
            }
        }
    }

    // Example 3: Year range query (recent books)
    println!("\n3. Books published after 2010:");
    let query = NumericRangeQuery::i64_range("year", Some(2010), None);
    let request = LexicalSearchRequest::new(Box::new(query) as Box<dyn Query>).load_documents(true);
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
            if let Some(yatagarasu::document::field_value::FieldValue::Integer(year)) =
                doc.get_field("year")
            {
                println!("      Year: {year}");
            }
        }
    }

    // Example 4: Page count range query (shorter books)
    println!("\n4. Books with 400 pages or fewer:");
    let query = NumericRangeQuery::i64_range("pages", None, Some(400));
    let request = LexicalSearchRequest::new(Box::new(query) as Box<dyn Query>).load_documents(true);
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
            if let Some(yatagarasu::document::field_value::FieldValue::Integer(pages)) =
                doc.get_field("pages")
            {
                println!("      Pages: {pages}");
            }
        }
    }

    // Example 5: Exact year range (books from 2008-2009)
    println!("\n5. Books published between 2008 and 2009:");
    let query = NumericRangeQuery::i64_range("year", Some(2008), Some(2009));
    let request = LexicalSearchRequest::new(Box::new(query) as Box<dyn Query>).load_documents(true);
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
            if let Some(yatagarasu::document::field_value::FieldValue::Integer(year)) =
                doc.get_field("year")
            {
                println!("      Year: {year}");
            }
        }
    }

    // Example 6: Budget-friendly books (price under $50)
    println!("\n6. Budget-friendly books (price under $50.00):");
    let query = NumericRangeQuery::f64_range_exclusive_upper("price", None, Some(50.0));
    let request = LexicalSearchRequest::new(Box::new(query) as Box<dyn Query>).load_documents(true);
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
            if let Some(yatagarasu::document::field_value::FieldValue::Float(price)) =
                doc.get_field("price")
            {
                println!("      Price: ${price:.2}");
            }
        }
    }

    // Example 7: Large books (more than 500 pages)
    println!("\n7. Large books (more than 500 pages):");
    let query = NumericRangeQuery::i64_range("pages", Some(500), None);
    let request = LexicalSearchRequest::new(Box::new(query) as Box<dyn Query>).load_documents(true);
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
            if let Some(yatagarasu::document::field_value::FieldValue::Integer(pages)) =
                doc.get_field("pages")
            {
                println!("      Pages: {pages}");
            }
        }
    }

    // Example 8: Count books in price range
    println!("\n8. Counting books with price between $40.00 and $80.00:");
    let query = NumericRangeQuery::f64_range("price", Some(40.0), Some(80.0));
    let count = engine.count(Box::new(query) as Box<dyn Query>)?;
    println!("   Count: {count} books");

    // Example 9: Empty range (no results expected)
    println!("\n9. Books with impossible price range ($200-$300):");
    let query = NumericRangeQuery::f64_range("price", Some(200.0), Some(300.0));
    let request = LexicalSearchRequest::new(Box::new(query) as Box<dyn Query>);
    let results = engine.search(request)?;

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
