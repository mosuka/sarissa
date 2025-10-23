//! Comprehensive query parser example.
//!
//! This example demonstrates all query parser features including:
//! - Basic queries and field-specific queries
//! - Boolean operators (AND, OR, +, -)
//! - Phrase queries with proximity search
//! - Fuzzy and wildcard queries
//! - Range queries (numeric and text)
//! - Boost factors
//! - Grouped queries
//! - Actual search engine integration

use tempfile::TempDir;

use sage::document::document::Document;
use sage::error::Result;
use sage::lexical::engine::LexicalEngine;
use sage::lexical::inverted_index::IndexConfig;
use sage::lexical::types::SearchRequest;
use sage::query::parser::QueryParser;

fn main() -> Result<()> {
    println!("=== Query Parser - Complete Feature Demonstration ===\n");

    // Create a temporary directory for the index
    let temp_dir = TempDir::new().unwrap();
    println!("Creating index in: {:?}\n", temp_dir.path());

    // Create a search engine
    let mut engine = LexicalEngine::create_in_dir(temp_dir.path(), IndexConfig::default())?;

    // Add sample documents
    let documents = vec![
        Document::builder()
            .add_text("title", "The Great Gatsby")
            .add_text("body", "In my younger and more vulnerable years my father gave me some advice")
            .add_text("author", "F. Scott Fitzgerald")
            .add_numeric("year", 1925.0)
            .build(),
        Document::builder()
            .add_text("title", "To Kill a Mockingbird")
            .add_text("body", "When I was almost six years old, I heard my brother arguing with my father")
            .add_text("author", "Harper Lee")
            .add_numeric("year", 1960.0)
            .build(),
        Document::builder()
            .add_text("title", "1984")
            .add_text("body", "It was a bright cold day in April, and the clocks were striking thirteen")
            .add_text("author", "George Orwell")
            .add_numeric("year", 1949.0)
            .build(),
        Document::builder()
            .add_text("title", "Pride and Prejudice")
            .add_text("body", "It is a truth universally acknowledged, that a single man in possession of a good fortune")
            .add_text("author", "Jane Austen")
            .add_numeric("year", 1813.0)
            .build(),
        Document::builder()
            .add_text("title", "The Catcher in the Rye")
            .add_text("body", "If you really want to hear about it, the first thing you'll probably want to know")
            .add_text("author", "J.D. Salinger")
            .add_numeric("year", 1951.0)
            .build(),
    ];

    println!("Adding {} documents to the index...", documents.len());
    engine.add_documents(documents)?;
    engine.commit()?;

    // Create parser with standard analyzer
    let parser = QueryParser::with_standard_analyzer()?.with_default_field("title");

    println!("\n{}", "=".repeat(80));
    println!("PART 1: Basic Query Syntax (No Search)");
    println!("{}", "=".repeat(80));

    demo_syntax(&parser);

    println!("\n{}", "=".repeat(80));
    println!("PART 2: Search Examples (With Results)");
    println!("{}", "=".repeat(80));

    demo_search(&mut engine, &parser)?;

    println!("\n{}", "=".repeat(80));
    println!("PART 3: Nested Boolean Queries");
    println!("{}", "=".repeat(80));

    demo_nested_queries(&parser);

    engine.close()?;
    println!("\n✓ Example completed successfully!");

    Ok(())
}

fn demo_syntax(parser: &QueryParser) {
    println!("\n## 1. Basic Queries");
    println!("{}", "-".repeat(80));
    parse_and_show(
        parser,
        vec![
            ("hello", "Simple term"),
            ("title:hello", "Field-specific term"),
        ],
    );

    println!("\n## 2. Boolean Queries");
    println!("{}", "-".repeat(80));
    parse_and_show(
        parser,
        vec![
            ("hello AND world", "AND operator"),
            ("hello OR world", "OR operator"),
            ("+hello world", "Required term + optional"),
            ("hello -world", "Exclude term"),
            ("+hello +world", "Both required"),
        ],
    );

    println!("\n## 3. Phrase Queries");
    println!("{}", "-".repeat(80));
    parse_and_show(
        parser,
        vec![
            ("\"hello world\"", "Exact phrase"),
            ("\"hello world\"~10", "Proximity search (slop=10)"),
        ],
    );

    println!("\n## 4. Fuzzy & Wildcard Queries");
    println!("{}", "-".repeat(80));
    parse_and_show(
        parser,
        vec![
            ("hello~2", "Fuzzy search (edit distance=2)"),
            ("hel*", "Wildcard: zero or more chars"),
            ("te?t", "Wildcard: exactly one char"),
        ],
    );

    println!("\n## 5. Range Queries");
    println!("{}", "-".repeat(80));
    parse_and_show(
        parser,
        vec![
            ("year:[1900 TO 2000]", "Numeric inclusive range"),
            ("year:{1900 TO 2000}", "Numeric exclusive range"),
            ("age:[18 TO 65]", "Age range"),
        ],
    );

    println!("\n## 6. Boost Queries");
    println!("{}", "-".repeat(80));
    parse_and_show(
        parser,
        vec![
            ("hello^2", "Boost term (factor=2)"),
            ("title:hello^3", "Boost field-specific term"),
            ("\"hello world\"^1.5", "Boost phrase"),
            ("(hello OR world)^2", "Boost grouped query"),
        ],
    );

    println!("\n## 7. Complex Combined Queries");
    println!("{}", "-".repeat(80));
    parse_and_show(
        parser,
        vec![
            ("title:hello^2 AND year:[1900 TO 2000]", "Boost + range"),
            (
                "(title:important^3 OR body:urgent) AND year:[1900 TO 2000]",
                "Full featured",
            ),
            (
                "+title:\"breaking news\"^2 -category:spam",
                "Required phrase + exclusion",
            ),
        ],
    );
}

fn parse_and_show(parser: &QueryParser, queries: Vec<(&str, &str)>) {
    for (query_str, description) in queries {
        print!("  {:50} ", format!("\"{}\"", query_str));
        match parser.parse(query_str) {
            Ok(query) => {
                let debug_str = format!("{query:?}");
                let query_type = get_query_type(&debug_str);
                print!("✓ {query_type} ");
                if query.boost() != 1.0 {
                    print!("(boost: {}) ", query.boost());
                }
                println!("← {description}");
            }
            Err(e) => {
                println!("✗ Error: {e}");
            }
        }
    }
}

fn demo_search(engine: &mut LexicalEngine, parser: &QueryParser) -> Result<()> {
    println!("\n## 1. Simple OR Query");
    println!("{}", "-".repeat(80));
    execute_search(engine, parser, "Mockingbird OR Gatsby")?;

    println!("\n## 2. AND Query with Field Specification");
    println!("{}", "-".repeat(80));
    execute_search(engine, parser, "title:Pride AND body:truth")?;

    println!("\n## 3. Required and Prohibited Terms");
    println!("{}", "-".repeat(80));
    execute_search(engine, parser, "+title:Catcher -Gatsby")?;

    println!("\n## 4. Phrase Search");
    println!("{}", "-".repeat(80));
    execute_search(engine, parser, "\"Great Gatsby\"")?;

    println!("\n## 5. Range Query (Years 1900-1960)");
    println!("{}", "-".repeat(80));
    execute_search(engine, parser, "year:[1900 TO 1960]")?;

    println!("\n## 6. Wildcard Search");
    println!("{}", "-".repeat(80));
    execute_search(engine, parser, "title:Mock*")?;

    Ok(())
}

fn execute_search(engine: &mut LexicalEngine, parser: &QueryParser, query_str: &str) -> Result<()> {
    println!("Query: {query_str}");

    let query = parser.parse(query_str)?;
    println!("Parsed: {}", query.description());

    let request = SearchRequest::new(query).load_documents(true);
    let results = engine.search(request)?;

    println!("Results: {} hits", results.total_hits);
    for (i, hit) in results.hits.iter().enumerate() {
        print!("  {}. Score: {:.4} ", i + 1, hit.score);
        if let Some(doc) = &hit.document {
            if let Some(field_value) = doc.get_field("title")
                && let Some(title) = field_value.as_text()
            {
                print!("- {title}");
            }
            if let Some(field_value) = doc.get_field("year")
                && let Some(year) = field_value.as_numeric()
                && let Ok(year_num) = year.parse::<f64>()
            {
                print!(" ({})", year_num as i32);
            }
        }
        println!();
    }

    Ok(())
}

fn demo_nested_queries(parser: &QueryParser) {
    let nested_queries = vec![
        // Level 1: Simple nesting
        ("(apple OR banana) AND cherry", "Simple: OR inside AND"),
        ("apple AND (banana OR cherry)", "Simple: AND with OR group"),
        (
            "(apple AND banana) OR (cherry AND date)",
            "Two AND groups with OR",
        ),
        // Level 2: Deeper nesting
        ("((apple OR banana) AND cherry) OR date", "Double nesting"),
        (
            "apple AND (banana OR (cherry AND date))",
            "Nested OR with AND inside",
        ),
        (
            "(apple OR (banana AND cherry)) AND date",
            "Nested AND inside OR",
        ),
        // Level 3: Complex nesting
        (
            "((apple AND banana) OR (cherry AND date)) AND egg",
            "Two groups AND-ed",
        ),
        (
            "(apple OR banana) AND (cherry OR date) AND egg",
            "Multiple OR groups",
        ),
        (
            "((apple OR banana) AND (cherry OR date)) OR ((egg AND fig) AND grape)",
            "Complex nested",
        ),
        // With field specifications
        (
            "(title:hello OR body:world) AND author:test",
            "Fields in nested",
        ),
        // With operators
        ("+(apple OR banana) -cherry", "Required group + exclusion"),
        (
            "+((apple AND banana) OR cherry) -(date OR egg)",
            "Complex required/excluded groups",
        ),
        // With boost
        ("(apple OR banana)^2 AND cherry", "Boosted group"),
        ("((apple OR banana)^2 AND cherry)^1.5", "Double boost"),
        // With phrases
        (
            "(\"hello world\" OR \"foo bar\") AND title:test",
            "Phrases in groups",
        ),
        // Very deep nesting (4 levels)
        (
            "(((apple OR banana) AND (cherry OR date)) OR ((egg AND fig) OR (grape AND honey))) AND item",
            "Very deep nesting",
        ),
        // Mixed features
        (
            "(title:important^2 OR body:urgent) AND year:[2020 TO 2025]",
            "Nested with boost and range",
        ),
        (
            "+(\"breaking news\"^2 OR title:urgent^3) -category:spam",
            "Nested phrases with boost and exclusion",
        ),
    ];

    for (query_str, description) in nested_queries {
        println!("\n  Query: {query_str}");
        println!("  Description: {description}");

        match parser.parse(query_str) {
            Ok(query) => {
                println!("  ✓ Successfully parsed");
                let desc = query.description();
                // Truncate if too long
                if desc.len() > 100 {
                    println!("    Structure: {}...", &desc[..97]);
                } else {
                    println!("    Structure: {desc}");
                }
                if query.boost() != 1.0 {
                    println!("    Boost: {}", query.boost());
                }
            }
            Err(e) => {
                println!("  ✗ Parse error: {e}");
            }
        }
    }
}

fn get_query_type(debug_str: &str) -> &str {
    if debug_str.starts_with("TermQuery") {
        "TermQuery"
    } else if debug_str.starts_with("BooleanQuery") {
        "BooleanQuery"
    } else if debug_str.starts_with("PhraseQuery") {
        "PhraseQuery"
    } else if debug_str.starts_with("FuzzyQuery") {
        "FuzzyQuery"
    } else if debug_str.starts_with("WildcardQuery") {
        "WildcardQuery"
    } else if debug_str.starts_with("NumericRangeQuery") {
        "NumericRangeQuery"
    } else {
        "Query"
    }
}
