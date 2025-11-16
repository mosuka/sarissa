//! GeoQuery example - demonstrates geographic location-based searches.

use std::sync::Arc;

use tempfile::TempDir;

use yatagarasu::analysis::analyzer::analyzer::Analyzer;
use yatagarasu::analysis::analyzer::keyword::KeywordAnalyzer;
use yatagarasu::analysis::analyzer::per_field::PerFieldAnalyzer;
use yatagarasu::analysis::analyzer::standard::StandardAnalyzer;
use yatagarasu::document::document::Document;
use yatagarasu::document::field::{GeoOption, TextOption};
use yatagarasu::error::Result;
use yatagarasu::lexical::engine::LexicalEngine;
use yatagarasu::lexical::index::config::InvertedIndexConfig;
use yatagarasu::lexical::index::config::LexicalIndexConfig;
use yatagarasu::lexical::index::factory::LexicalIndexFactory;
use yatagarasu::lexical::index::inverted::query::Query;
use yatagarasu::lexical::index::inverted::query::geo::GeoQuery;
use yatagarasu::lexical::search::searcher::LexicalSearchRequest;
use yatagarasu::storage::StorageConfig;
use yatagarasu::storage::StorageFactory;
use yatagarasu::storage::file::FileStorageConfig;

fn main() -> Result<()> {
    println!("=== GeoQuery Example - Geographic Location-Based Search ===\n");

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

    // Add documents with geographic coordinates
    // Using famous locations around the world
    let documents = vec![
        Document::builder()
            .add_text("name", "Central Park", TextOption::default())
            .add_text(
                "description",
                "Large public park in Manhattan, New York City",
                TextOption::default(),
            )
            .add_text("category", "park", TextOption::default())
            .add_geo("location", 40.7829, -73.9654, GeoOption::default()) // Central Park, NYC
            .add_text("city", "New York", TextOption::default())
            .add_text("id", "loc001", TextOption::default())
            .build(),
        Document::builder()
            .add_text("name", "Statue of Liberty", TextOption::default())
            .add_text(
                "description",
                "Iconic statue on Liberty Island",
                TextOption::default(),
            )
            .add_text("category", "monument", TextOption::default())
            .add_geo("location", 40.6892, -74.0445, GeoOption::default()) // Statue of Liberty, NYC
            .add_text("city", "New York", TextOption::default())
            .add_text("id", "loc002", TextOption::default())
            .build(),
        Document::builder()
            .add_text("name", "Golden Gate Bridge", TextOption::default())
            .add_text(
                "description",
                "Suspension bridge in San Francisco",
                TextOption::default(),
            )
            .add_text("category", "bridge", TextOption::default())
            .add_geo("location", 37.8199, -122.4783, GeoOption::default()) // Golden Gate Bridge, SF
            .add_text("city", "San Francisco", TextOption::default())
            .add_text("id", "loc003", TextOption::default())
            .build(),
        Document::builder()
            .add_text("name", "Alcatraz Island", TextOption::default())
            .add_text(
                "description",
                "Former federal prison on island in San Francisco Bay",
                TextOption::default(),
            )
            .add_text("category", "historical", TextOption::default())
            .add_geo("location", 37.8267, -122.4233, GeoOption::default()) // Alcatraz Island, SF
            .add_text("city", "San Francisco", TextOption::default())
            .add_text("id", "loc004", TextOption::default())
            .build(),
        Document::builder()
            .add_text("name", "Hollywood Sign", TextOption::default())
            .add_text(
                "description",
                "Landmark sign in Hollywood Hills",
                TextOption::default(),
            )
            .add_text("category", "landmark", TextOption::default())
            .add_geo("location", 34.1341, -118.3215, GeoOption::default()) // Hollywood Sign, LA
            .add_text("city", "Los Angeles", TextOption::default())
            .add_text("id", "loc005", TextOption::default())
            .build(),
        Document::builder()
            .add_text("name", "Santa Monica Pier", TextOption::default())
            .add_text(
                "description",
                "Amusement park and pier on Santa Monica Beach",
                TextOption::default(),
            )
            .add_text("category", "entertainment", TextOption::default())
            .add_geo("location", 34.0084, -118.4966, GeoOption::default()) // Santa Monica Pier, LA
            .add_text("city", "Los Angeles", TextOption::default())
            .add_text("id", "loc006", TextOption::default())
            .build(),
        Document::builder()
            .add_text("name", "Space Needle", TextOption::default())
            .add_text(
                "description",
                "Observation tower in Seattle Center",
                TextOption::default(),
            )
            .add_text("category", "tower", TextOption::default())
            .add_geo("location", 47.6205, -122.3493, GeoOption::default()) // Space Needle, Seattle
            .add_text("city", "Seattle", TextOption::default())
            .add_text("id", "loc007", TextOption::default())
            .build(),
        Document::builder()
            .add_text("name", "Pike Place Market", TextOption::default())
            .add_text(
                "description",
                "Public market overlooking Elliott Bay",
                TextOption::default(),
            )
            .add_text("category", "market", TextOption::default())
            .add_geo("location", 47.6101, -122.3421, GeoOption::default()) // Pike Place Market, Seattle
            .add_text("city", "Seattle", TextOption::default())
            .add_text("id", "loc008", TextOption::default())
            .build(),
    ];

    println!("Adding {} documents to the index...", documents.len());
    for doc in documents {
        lexical_engine.add_document(doc)?;
    }

    // Commit changes to engine
    lexical_engine.commit()?;

    println!("\n=== GeoQuery Examples ===\n");

    // Example 1: Find locations within radius of Times Square, NYC
    println!("1. Locations within 5km of Times Square (40.7580° N, 73.9855° W):");
    let query = GeoQuery::within_radius("location", 40.7580, -73.9855, 5.0)?;
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
            if let Some(field_value) = doc.get_field("name")
                && let Some(name) = field_value.value.as_text()
            {
                println!("      Name: {name}");
            }
            if let Some(field_value) = doc.get_field("city")
                && let Some(city) = field_value.value.as_text()
            {
                println!("      City: {city}");
            }
        }
    }

    // Example 2: Find locations within radius of downtown San Francisco
    println!("\n2. Locations within 10km of downtown San Francisco (37.7749° N, 122.4194° W):");
    let query = GeoQuery::within_radius("location", 37.7749, -122.4194, 10.0)?;
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
            if let Some(field_value) = doc.get_field("name")
                && let Some(name) = field_value.value.as_text()
            {
                println!("      Name: {name}");
            }
            if let Some(field_value) = doc.get_field("description")
                && let Some(description) = field_value.value.as_text()
            {
                println!("      Description: {description}");
            }
        }
    }

    // Example 3: Find locations within a bounding box (Los Angeles area)
    println!("\n3. Locations within bounding box of Los Angeles area:");
    println!("   (33.9° N, 118.6° W) to (34.3° N, 118.1° W)");
    let query = GeoQuery::within_bounding_box("location", 33.9, -118.6, 34.3, -118.1)?;
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
            if let Some(field_value) = doc.get_field("name")
                && let Some(name) = field_value.value.as_text()
            {
                println!("      Name: {name}");
            }
            if let Some(field_value) = doc.get_field("category")
                && let Some(category) = field_value.value.as_text()
            {
                println!("      Category: {category}");
            }
        }
    }

    // Example 4: Find locations within a large radius to include multiple cities
    println!("\n4. All West Coast locations within 1000km of San Francisco:");
    let query = GeoQuery::within_radius("location", 37.7749, -122.4194, 1000.0)?;
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
            if let Some(field_value) = doc.get_field("name")
                && let Some(name) = field_value.value.as_text()
            {
                println!("      Name: {name}");
            }
            if let Some(field_value) = doc.get_field("city")
                && let Some(city) = field_value.value.as_text()
            {
                println!("      City: {city}");
            }
        }
    }

    // Example 5: Find locations within radius of Seattle
    println!("\n5. Locations within 2km of downtown Seattle (47.6062° N, 122.3321° W):");
    let query = GeoQuery::within_radius("location", 47.6062, -122.3321, 2.0)?;
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
            if let Some(field_value) = doc.get_field("name")
                && let Some(name) = field_value.value.as_text()
            {
                println!("      Name: {name}");
            }
            if let Some(field_value) = doc.get_field("description")
                && let Some(description) = field_value.value.as_text()
            {
                println!("      Description: {description}");
            }
        }
    }

    // Example 6: Find locations within a tight radius (should find few/no results)
    println!("\n6. Locations within 1km of a specific point in the ocean:");
    let query = GeoQuery::within_radius("location", 36.0, -125.0, 1.0)?; // Pacific Ocean
    let request = LexicalSearchRequest::new(Box::new(query) as Box<dyn Query>).load_documents(true);
    let results = lexical_engine.search(request)?;

    println!("   Found {} results", results.total_hits);

    // Example 7: Bounding box covering the entire continental US (wide search)
    println!("\n7. All locations within US continental bounding box:");
    println!("   (25° N, 125° W) to (49° N, 66° W)");
    let query = GeoQuery::within_bounding_box("location", 25.0, -125.0, 49.0, -66.0)?;
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
            if let Some(field_value) = doc.get_field("name")
                && let Some(name) = field_value.value.as_text()
            {
                println!("      Name: {name}");
            }
            if let Some(field_value) = doc.get_field("city")
                && let Some(city) = field_value.value.as_text()
            {
                println!("      City: {city}");
            }
        }
    }

    // Example 8: Count locations within a specific area
    println!("\n8. Counting locations within 50km of Los Angeles center:");
    let query = GeoQuery::within_radius("location", 34.0522, -118.2437, 50.0)?;
    let count = lexical_engine.count(Box::new(query) as Box<dyn Query>)?;
    println!("   Count: {count} locations");

    lexical_engine.close()?;
    println!("\nGeoQuery example completed successfully!");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geo_query_example() {
        let result = main();
        assert!(result.is_ok());
    }
}
