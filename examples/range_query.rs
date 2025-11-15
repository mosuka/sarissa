//! RangeQuery example - demonstrates range search for numeric and date values.

use std::sync::Arc;

use tempfile::TempDir;

use yatagarasu::analysis::analyzer::analyzer::Analyzer;
use yatagarasu::analysis::analyzer::keyword::KeywordAnalyzer;
use yatagarasu::analysis::analyzer::per_field::PerFieldAnalyzer;
use yatagarasu::analysis::analyzer::standard::StandardAnalyzer;
use yatagarasu::document::document::Document;
use yatagarasu::document::field::{FloatOption, IntegerOption, TextOption};
use yatagarasu::error::Result;
use yatagarasu::lexical::engine::LexicalEngine;
use yatagarasu::lexical::index::config::InvertedIndexConfig;
use yatagarasu::lexical::index::config::LexicalIndexConfig;
use yatagarasu::lexical::index::factory::LexicalIndexFactory;
use yatagarasu::lexical::index::inverted::query::Query;
use yatagarasu::lexical::index::inverted::query::range::NumericRangeQuery;
use yatagarasu::lexical::search::searcher::LexicalSearchRequest;
use yatagarasu::storage::StorageConfig;
use yatagarasu::storage::StorageFactory;
use yatagarasu::storage::file::FileStorageConfig;

fn main() -> Result<()> {
    println!("=== RangeQuery Example - Numeric and Date Range Search ===\n");

    // Create a storage backend
    let temp_dir = TempDir::new().unwrap();
    let storage =
        StorageFactory::create(StorageConfig::File(FileStorageConfig::new(temp_dir.path())))?;

    // Create an analyzer
    let standard_analyzer: Arc<dyn Analyzer> = Arc::new(StandardAnalyzer::new()?);
    let keyword_analyzer: Arc<dyn Analyzer> = Arc::new(KeywordAnalyzer::new());
    let mut per_field_analyzer = PerFieldAnalyzer::new(Arc::clone(&standard_analyzer));
    per_field_analyzer.add_analyzer("id", Arc::clone(&keyword_analyzer));

    // Create a lexical index
    let lexical_index_config = LexicalIndexConfig::Inverted(InvertedIndexConfig {
        analyzer: Arc::new(per_field_analyzer.clone()),
        ..InvertedIndexConfig::default()
    });
    let lexical_index = LexicalIndexFactory::create(storage, lexical_index_config)?;

    // Create a lexical engine
    let mut lexical_engine = LexicalEngine::new(lexical_index)?;

    // Add documents with various numeric values
    let documents = vec![
        Document::builder()
            .add_text("title", "Introduction to Algorithms", TextOption::default())
            .add_text(
                "description",
                "Comprehensive guide to algorithms and data structures",
                TextOption::default(),
            )
            .add_float("price", 89.99, FloatOption::default())
            .add_float("rating", 4.8, FloatOption::default())
            .add_integer("year", 2009, IntegerOption::default())
            .add_integer("pages", 1312, IntegerOption::default())
            .add_text("id", "book001", TextOption::default())
            .build(),
        Document::builder()
            .add_text("title", "Clean Code", TextOption::default())
            .add_text(
                "description",
                "A handbook of agile software craftsmanship",
                TextOption::default(),
            )
            .add_float("price", 45.50, FloatOption::default())
            .add_float("rating", 4.6, FloatOption::default())
            .add_integer("year", 2008, IntegerOption::default())
            .add_integer("pages", 464, IntegerOption::default())
            .add_text("id", "book002", TextOption::default())
            .build(),
        Document::builder()
            .add_text("title", "Design Patterns", TextOption::default())
            .add_text(
                "description",
                "Elements of reusable object-oriented software",
                TextOption::default(),
            )
            .add_float("price", 62.95, FloatOption::default())
            .add_float("rating", 4.5, FloatOption::default())
            .add_integer("year", 1994, IntegerOption::default())
            .add_integer("pages", 395, IntegerOption::default())
            .add_text("id", "book003", TextOption::default())
            .build(),
        Document::builder()
            .add_text("title", "The Pragmatic Programmer", TextOption::default())
            .add_text(
                "description",
                "Your journey to mastery",
                TextOption::default(),
            )
            .add_float("price", 52.99, FloatOption::default())
            .add_float("rating", 4.7, FloatOption::default())
            .add_integer("year", 2019, IntegerOption::default())
            .add_integer("pages", 352, IntegerOption::default())
            .add_text("id", "book004", TextOption::default())
            .build(),
        Document::builder()
            .add_text("title", "Refactoring", TextOption::default())
            .add_text(
                "description",
                "Improving the design of existing code",
                TextOption::default(),
            )
            .add_float("price", 58.75, FloatOption::default())
            .add_float("rating", 4.4, FloatOption::default())
            .add_integer("year", 2018, IntegerOption::default())
            .add_integer("pages", 448, IntegerOption::default())
            .add_text("id", "book005", TextOption::default())
            .build(),
        Document::builder()
            .add_text("title", "Code Complete", TextOption::default())
            .add_text(
                "description",
                "A practical handbook of software construction",
                TextOption::default(),
            )
            .add_float("price", 73.99, FloatOption::default())
            .add_float("rating", 4.9, FloatOption::default())
            .add_integer("year", 2004, IntegerOption::default())
            .add_integer("pages", 914, IntegerOption::default())
            .add_text("id", "book006", TextOption::default())
            .build(),
    ];

    println!("Adding {} documents to the index...", documents.len());
    for doc in documents {
        lexical_engine.add_document(doc)?;
    }
    lexical_engine.commit()?;

    println!("\n=== RangeQuery Examples ===\n");

    // Example 1: Price range query
    println!("1. Books with price between $50.00 and $70.00:");
    let query = NumericRangeQuery::f64_range("price", Some(50.0), Some(70.0));
    let request = LexicalSearchRequest::new(Box::new(query) as Box<dyn Query>).load_documents(true);
    let results = lexical_engine.search(request)?;

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
                && let Some(title) = field_value.value.as_text()
            {
                println!("      Title: {title}");
            }
            if let Some(field) = doc.get_field("price")
                && let yatagarasu::document::field::FieldValue::Float(price) = &field.value
            {
                println!("      Price: ${price:.2}");
            }
        }
    }

    // Example 2: Rating range query (high-rated books)
    println!("\n2. Books with rating 4.5 or higher:");
    let query = NumericRangeQuery::f64_range("rating", Some(4.5), None);
    let request = LexicalSearchRequest::new(Box::new(query) as Box<dyn Query>).load_documents(true);
    let results = lexical_engine.search(request)?;

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
                && let Some(title) = field_value.value.as_text()
            {
                println!("      Title: {title}");
            }
            if let Some(field) = doc.get_field("rating")
                && let yatagarasu::document::field::FieldValue::Float(rating) = &field.value
            {
                println!("      Rating: {rating:.1}");
            }
        }
    }

    // Example 3: Year range query (recent books)
    println!("\n3. Books published after 2010:");
    let query = NumericRangeQuery::i64_range("year", Some(2010), None);
    let request = LexicalSearchRequest::new(Box::new(query) as Box<dyn Query>).load_documents(true);
    let results = lexical_engine.search(request)?;

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
                && let Some(title) = field_value.value.as_text()
            {
                println!("      Title: {title}");
            }
            if let Some(field) = doc.get_field("year")
                && let yatagarasu::document::field::FieldValue::Integer(year) = &field.value
            {
                println!("      Year: {year}");
            }
        }
    }

    // Example 4: Page count range query (shorter books)
    println!("\n4. Books with 400 pages or fewer:");
    let query = NumericRangeQuery::i64_range("pages", None, Some(400));
    let request = LexicalSearchRequest::new(Box::new(query) as Box<dyn Query>).load_documents(true);
    let results = lexical_engine.search(request)?;

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
                && let Some(title) = field_value.value.as_text()
            {
                println!("      Title: {title}");
            }
            if let Some(field) = doc.get_field("pages")
                && let yatagarasu::document::field::FieldValue::Integer(pages) = &field.value
            {
                println!("      Pages: {pages}");
            }
        }
    }

    // Example 5: Exact year range (books from 2008-2009)
    println!("\n5. Books published between 2008 and 2009:");
    let query = NumericRangeQuery::i64_range("year", Some(2008), Some(2009));
    let request = LexicalSearchRequest::new(Box::new(query) as Box<dyn Query>).load_documents(true);
    let results = lexical_engine.search(request)?;

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
                && let Some(title) = field_value.value.as_text()
            {
                println!("      Title: {title}");
            }
            if let Some(field) = doc.get_field("year")
                && let yatagarasu::document::field::FieldValue::Integer(year) = &field.value
            {
                println!("      Year: {year}");
            }
        }
    }

    // Example 6: Budget-friendly books (price under $50)
    println!("\n6. Budget-friendly books (price under $50.00):");
    let query = NumericRangeQuery::f64_range_exclusive_upper("price", None, Some(50.0));
    let request = LexicalSearchRequest::new(Box::new(query) as Box<dyn Query>).load_documents(true);
    let results = lexical_engine.search(request)?;

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
                && let Some(title) = field_value.value.as_text()
            {
                println!("      Title: {title}");
            }
            if let Some(field) = doc.get_field("price")
                && let yatagarasu::document::field::FieldValue::Float(price) = &field.value
            {
                println!("      Price: ${price:.2}");
            }
        }
    }

    // Example 7: Large books (more than 500 pages)
    println!("\n7. Large books (more than 500 pages):");
    let query = NumericRangeQuery::i64_range("pages", Some(500), None);
    let request = LexicalSearchRequest::new(Box::new(query) as Box<dyn Query>).load_documents(true);
    let results = lexical_engine.search(request)?;

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
                && let Some(title) = field_value.value.as_text()
            {
                println!("      Title: {title}");
            }
            if let Some(field) = doc.get_field("pages")
                && let yatagarasu::document::field::FieldValue::Integer(pages) = &field.value
            {
                println!("      Pages: {pages}");
            }
        }
    }

    // Example 8: Count books in price range
    println!("\n8. Counting books with price between $40.00 and $80.00:");
    let query = NumericRangeQuery::f64_range("price", Some(40.0), Some(80.0));
    let count = lexical_engine.count(Box::new(query) as Box<dyn Query>)?;
    println!("   Count: {count} books");

    // Example 9: Empty range (no results expected)
    println!("\n9. Books with impossible price range ($200-$300):");
    let query = NumericRangeQuery::f64_range("price", Some(200.0), Some(300.0));
    let request = LexicalSearchRequest::new(Box::new(query) as Box<dyn Query>);
    let results = lexical_engine.search(request)?;

    println!("   Found {} results", results.total_hits);

    lexical_engine.close()?;
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
