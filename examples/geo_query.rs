//! GeoQuery example - demonstrates geographic location-based searches.

use sarissa::index::index::IndexConfig;
use sarissa::prelude::*;
use sarissa::query::GeoQuery;
use sarissa::schema::{GeoField, IdField, TextField};
use sarissa::search::SearchEngine;
use sarissa::search::SearchRequest;
use tempfile::TempDir;

fn main() -> Result<()> {
    println!("=== GeoQuery Example - Geographic Location-Based Search ===\n");

    // Create a temporary directory for the index
    let temp_dir = TempDir::new().unwrap();
    println!("Creating index in: {:?}", temp_dir.path());

    // Create a schema with geographic fields
    let mut schema = Schema::new();
    schema.add_field(
        "name",
        Box::new(TextField::new().stored(true).indexed(true)),
    )?;
    schema.add_field("description", Box::new(TextField::new().indexed(true)))?;
    schema.add_field("category", Box::new(TextField::new().indexed(true)))?;
    schema.add_field("location", Box::new(GeoField::new().indexed(true)))?;
    schema.add_field("city", Box::new(TextField::new().indexed(true)))?;
    schema.add_field("id", Box::new(IdField::new()))?;

    // Create a search engine
    let mut engine = SearchEngine::create_in_dir(temp_dir.path(), schema, IndexConfig::default())?;

    // Add documents with geographic coordinates
    // Using famous locations around the world
    let documents = vec![
        Document::builder()
            .add_text("name", "Central Park")
            .add_text(
                "description",
                "Large public park in Manhattan, New York City",
            )
            .add_text("category", "park")
            .add_geo("location", 40.7829, -73.9654) // Central Park, NYC
            .add_text("city", "New York")
            .add_text("id", "loc001")
            .build(),
        Document::builder()
            .add_text("name", "Statue of Liberty")
            .add_text("description", "Iconic statue on Liberty Island")
            .add_text("category", "monument")
            .add_geo("location", 40.6892, -74.0445) // Statue of Liberty, NYC
            .add_text("city", "New York")
            .add_text("id", "loc002")
            .build(),
        Document::builder()
            .add_text("name", "Golden Gate Bridge")
            .add_text("description", "Suspension bridge in San Francisco")
            .add_text("category", "bridge")
            .add_geo("location", 37.8199, -122.4783) // Golden Gate Bridge, SF
            .add_text("city", "San Francisco")
            .add_text("id", "loc003")
            .build(),
        Document::builder()
            .add_text("name", "Alcatraz Island")
            .add_text(
                "description",
                "Former federal prison on island in San Francisco Bay",
            )
            .add_text("category", "historical")
            .add_geo("location", 37.8267, -122.4233) // Alcatraz Island, SF
            .add_text("city", "San Francisco")
            .add_text("id", "loc004")
            .build(),
        Document::builder()
            .add_text("name", "Hollywood Sign")
            .add_text("description", "Landmark sign in Hollywood Hills")
            .add_text("category", "landmark")
            .add_geo("location", 34.1341, -118.3215) // Hollywood Sign, LA
            .add_text("city", "Los Angeles")
            .add_text("id", "loc005")
            .build(),
        Document::builder()
            .add_text("name", "Santa Monica Pier")
            .add_text(
                "description",
                "Amusement park and pier on Santa Monica Beach",
            )
            .add_text("category", "entertainment")
            .add_geo("location", 34.0084, -118.4966) // Santa Monica Pier, LA
            .add_text("city", "Los Angeles")
            .add_text("id", "loc006")
            .build(),
        Document::builder()
            .add_text("name", "Space Needle")
            .add_text("description", "Observation tower in Seattle Center")
            .add_text("category", "tower")
            .add_geo("location", 47.6205, -122.3493) // Space Needle, Seattle
            .add_text("city", "Seattle")
            .add_text("id", "loc007")
            .build(),
        Document::builder()
            .add_text("name", "Pike Place Market")
            .add_text("description", "Public market overlooking Elliott Bay")
            .add_text("category", "market")
            .add_geo("location", 47.6101, -122.3421) // Pike Place Market, Seattle
            .add_text("city", "Seattle")
            .add_text("id", "loc008")
            .build(),
    ];

    println!("Adding {} documents to the index...", documents.len());
    engine.add_documents(documents)?;

    println!("\n=== GeoQuery Examples ===\n");

    // Example 1: Find locations within radius of Times Square, NYC
    println!("1. Locations within 5km of Times Square (40.7580° N, 73.9855° W):");
    let query = GeoQuery::within_radius("location", 40.7580, -73.9855, 5.0)?;
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
            if let Some(field_value) = doc.get_field("name") {
                if let Some(name) = field_value.as_text() {
                    println!("      Name: {}", name);
                }
            }
            if let Some(field_value) = doc.get_field("city") {
                if let Some(city) = field_value.as_text() {
                    println!("      City: {}", city);
                }
            }
        }
    }

    // Example 2: Find locations within radius of downtown San Francisco
    println!("\n2. Locations within 10km of downtown San Francisco (37.7749° N, 122.4194° W):");
    let query = GeoQuery::within_radius("location", 37.7749, -122.4194, 10.0)?;
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
            if let Some(field_value) = doc.get_field("name") {
                if let Some(name) = field_value.as_text() {
                    println!("      Name: {}", name);
                }
            }
            if let Some(field_value) = doc.get_field("description") {
                if let Some(description) = field_value.as_text() {
                    println!("      Description: {}", description);
                }
            }
        }
    }

    // Example 3: Find locations within a bounding box (Los Angeles area)
    println!("\n3. Locations within bounding box of Los Angeles area:");
    println!("   (33.9° N, 118.6° W) to (34.3° N, 118.1° W)");
    let query = GeoQuery::within_bounding_box("location", 33.9, -118.6, 34.3, -118.1)?;
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
            if let Some(field_value) = doc.get_field("name") {
                if let Some(name) = field_value.as_text() {
                    println!("      Name: {}", name);
                }
            }
            if let Some(field_value) = doc.get_field("category") {
                if let Some(category) = field_value.as_text() {
                    println!("      Category: {}", category);
                }
            }
        }
    }

    // Example 4: Find locations within a large radius to include multiple cities
    println!("\n4. All West Coast locations within 1000km of San Francisco:");
    let query = GeoQuery::within_radius("location", 37.7749, -122.4194, 1000.0)?;
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
            if let Some(field_value) = doc.get_field("name") {
                if let Some(name) = field_value.as_text() {
                    println!("      Name: {}", name);
                }
            }
            if let Some(field_value) = doc.get_field("city") {
                if let Some(city) = field_value.as_text() {
                    println!("      City: {}", city);
                }
            }
        }
    }

    // Example 5: Find locations within radius of Seattle
    println!("\n5. Locations within 2km of downtown Seattle (47.6062° N, 122.3321° W):");
    let query = GeoQuery::within_radius("location", 47.6062, -122.3321, 2.0)?;
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
            if let Some(field_value) = doc.get_field("name") {
                if let Some(name) = field_value.as_text() {
                    println!("      Name: {}", name);
                }
            }
            if let Some(field_value) = doc.get_field("description") {
                if let Some(description) = field_value.as_text() {
                    println!("      Description: {}", description);
                }
            }
        }
    }

    // Example 6: Find locations within a tight radius (should find few/no results)
    println!("\n6. Locations within 1km of a specific point in the ocean:");
    let query = GeoQuery::within_radius("location", 36.0, -125.0, 1.0)?; // Pacific Ocean
    let request = SearchRequest::new(Box::new(query));
    let results = engine.search_mut(request)?;

    println!("   Found {} results", results.total_hits);

    // Example 7: Bounding box covering the entire continental US (wide search)
    println!("\n7. All locations within US continental bounding box:");
    println!("   (25° N, 125° W) to (49° N, 66° W)");
    let query = GeoQuery::within_bounding_box("location", 25.0, -125.0, 49.0, -66.0)?;
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
            if let Some(field_value) = doc.get_field("name") {
                if let Some(name) = field_value.as_text() {
                    println!("      Name: {}", name);
                }
            }
            if let Some(field_value) = doc.get_field("city") {
                if let Some(city) = field_value.as_text() {
                    println!("      City: {}", city);
                }
            }
        }
    }

    // Example 8: Count locations within a specific area
    println!("\n8. Counting locations within 50km of Los Angeles center:");
    let query = GeoQuery::within_radius("location", 34.0522, -118.2437, 50.0)?;
    let count = engine.count_mut(Box::new(query))?;
    println!("   Count: {} locations", count);

    println!("\n=== Geographic Coordinate Examples ===");
    println!("Major US Cities Coordinates:");
    println!("• New York City: 40.7128° N, 74.0060° W");
    println!("• San Francisco: 37.7749° N, 122.4194° W");
    println!("• Los Angeles: 34.0522° N, 118.2437° W");
    println!("• Seattle: 47.6062° N, 122.3321° W");
    println!("• Chicago: 41.8781° N, 87.6298° W");
    println!("• Miami: 25.7617° N, 80.1918° W");

    println!("\n=== Distance Examples ===");
    println!("Approximate distances:");
    println!("• 1 km ≈ 0.62 miles");
    println!("• 5 km ≈ 3.1 miles (walking distance)");
    println!("• 10 km ≈ 6.2 miles (cycling distance)");
    println!("• 50 km ≈ 31 miles (suburban area)");
    println!("• 100 km ≈ 62 miles (metropolitan area)");

    println!("\n=== GeoQuery Key Features ===");
    println!("• Radius-based searches (within_radius)");
    println!("• Bounding box searches (within_bounding_box)");
    println!("• Distance calculations using great circle distance");
    println!("• Works with latitude/longitude coordinates");
    println!("• Efficient spatial indexing");
    println!("• Distance-based relevance scoring");

    println!("\n=== Use Cases ===");
    println!("• Location-based services: 'Find restaurants near me'");
    println!("• Real estate: 'Homes within 10km of downtown'");
    println!("• Logistics: 'Warehouses within delivery radius'");
    println!("• Tourism: 'Attractions within walking distance'");
    println!("• Emergency services: 'Nearest hospitals'");
    println!("• Social media: 'Posts from this area'");
    println!("• Weather services: 'Stations in region'");

    println!("\n=== Coordinate System Notes ===");
    println!("• Uses WGS84 coordinate system (GPS standard)");
    println!("• Latitude: -90° (South Pole) to +90° (North Pole)");
    println!("• Longitude: -180° (West) to +180° (East)");
    println!("• Distance calculations account for Earth's curvature");
    println!("• Precision: ~11 meters at equator per 0.0001°");

    engine.close()?;
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
